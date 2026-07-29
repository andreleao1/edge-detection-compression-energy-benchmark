"""
Pruning nao-estrutural (unstructured L1) + Fine-tuning para multiplos modelos.

Suporta:
  - faster_rcnn : Faster R-CNN ResNet-50 FPN (.pth)
  - retinanet   : RetinaNet ResNet-50 FPN (.pth)
  - mobilenet   : SSDLite320-MobileNetV3 (.pth)
  - yolo        : YOLOv10 variantes n/s/m/b/l/x (Ultralytics .pt)

Fluxo por modelo:
  1. Carrega modelo treinado
  2. Aplica pruning nao-estrutural global L1 (mascaras ativas)
  3. Fine-tuning de --ft_epochs epocas COM mascaras ativas
  4. Remove mascaras (pruning permanente — zeros gravados no weight)
  5. Salva modelo podado

--------------------------------------------------------------------
MODO: TODOS OS MODELOS
--------------------------------------------------------------------
  python prune_models.py --all_models --pruning_percentage 50

  Itera sobre faster_rcnn / retinanet / mobilenet / todos os YOLO
  nos datasets oxford e caviar, usando os best.pth/.pt de:
    runs/faster_rcnn/epochs_100/{dataset}/best.pth
    runs/retinanet/epochs_100/{dataset}/best.pth
    runs/mobilenet/epochs_100/{dataset}/best.pth
    runs/train/yolov10{v}_{dataset}_finetune/weights/best.pt

  Salva em:
    runs/faster_rcnn/pruned/{pct}/{dataset}/best_pruned.pth
    runs/retinanet/pruned/{pct}/{dataset}/best_pruned.pth
    runs/mobilenet/pruned/{pct}/{dataset}/best_pruned.pth
    runs/train/pruned/{pct}/{dataset}/yolov10{v}_pruned.pt

--------------------------------------------------------------------
MODO: MODELO UNICO
--------------------------------------------------------------------
  python prune_models.py \\
    --model_type faster_rcnn \\
    --model_path runs/faster_rcnn/epochs_100/oxford/best.pth \\
    --dataset_dir oxford \\
    --pruning_percentage 50 \\
    --output_path runs/faster_rcnn/pruned/50/oxford/best_pruned.pth
"""

import argparse
import logging
import traceback
from functools import partial
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection._utils import retrieve_out_channels

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

BASE_DIR    = Path("C:/workspace/mestrado/novo_teste")
CLASS_NAMES = ["pessoa"]
NUM_CLASSES = len(CLASS_NAMES) + 1   # +1 para background

DATASETS = ["oxford", "caviar"]
YOLO_VARIANTS = ["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x"]

# Configuracoes padrao por modelo
DEFAULTS = {
    "faster_rcnn": {"imgsz": 800, "batch": 2,  "lr": 0.005, "workers": 4, "square": False},
    "retinanet":   {"imgsz": 800, "batch": 2,  "lr": 0.001, "workers": 4, "square": False},
    "mobilenet":   {"imgsz": 320, "batch": 4,  "lr": 0.005, "workers": 4, "square": True},
    "yolo":        {"imgsz": 640, "batch": -1, "lr": 1e-3,  "workers": 8, "square": False},
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_file_logging(log_path: Path):
    """Adiciona um FileHandler para gravar todo o output em arquivo."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info(f"Log gravado em: {log_path}")


# ---------------------------------------------------------------------------
# Dataset compartilhado (modelos torchvision)
# ---------------------------------------------------------------------------

class YOLODetectionDataset(Dataset):
    def __init__(self, images_dir: Path, labels_dir: Path, imgsz: int, square: bool = False):
        self.labels_dir = Path(labels_dir)
        self.imgsz      = imgsz
        self.square     = square
        self.samples    = [
            p for p in sorted(Path(images_dir).glob("*.jpg"))
                      + sorted(Path(images_dir).glob("*.png"))
            if (self.labels_dir / (p.stem + ".txt")).exists()
        ]
        if not self.samples:
            raise RuntimeError(f"Nenhuma imagem com label em: {images_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path   = self.samples[idx]
        label_path = self.labels_dir / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if self.square:
            img     = cv2.resize(img, (self.imgsz, self.imgsz))
            scale_x = self.imgsz / w
            scale_y = self.imgsz / h
            new_w = new_h = self.imgsz
        else:
            scale   = self.imgsz / max(h, w)
            new_w   = int(w * scale)
            new_h   = int(h * scale)
            img     = cv2.resize(img, (new_w, new_h))
            scale_x = scale_y = scale

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        boxes, labels = [], []
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = max(0.0, (xc - bw / 2) * w * scale_x)
            y1 = max(0.0, (yc - bh / 2) * h * scale_y)
            x2 = min(float(new_w), (xc + bw / 2) * w * scale_x)
            y2 = min(float(new_h), (yc + bh / 2) * h * scale_y)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(cls + 1)

        boxes_t  = torch.tensor(boxes,  dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros((0,),   dtype=torch.int64)

        return img_tensor, {"boxes": boxes_t, "labels": labels_t}


def collate_fn(batch):
    return tuple(zip(*batch))


def make_dataloaders(dataset_dir: Path, imgsz: int, batch_size: int,
                     num_workers: int = 4, square: bool = False):
    train_loader = DataLoader(
        YOLODetectionDataset(
            dataset_dir / "datasets" / "images" / "train",
            dataset_dir / "datasets" / "labels" / "train",
            imgsz=imgsz, square=square,
        ),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        YOLODetectionDataset(
            dataset_dir / "datasets" / "images" / "val",
            dataset_dir / "datasets" / "labels" / "val",
            imgsz=imgsz, square=square,
        ),
        batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    logger.info(f"  Train: {len(train_loader.dataset)} imgs  |  Val: {len(val_loader.dataset)} imgs")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Construtores de modelos torchvision
# ---------------------------------------------------------------------------

def _build_faster_rcnn(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _build_retinanet(num_classes):
    model = retinanet_resnet50_fpn(weights=None)
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256, num_anchors=num_anchors, num_classes=num_classes,
        norm_layer=partial(nn.GroupNorm, 32),
    )
    return model


def _build_mobilenet(num_classes, imgsz=320):
    # weights_backbone=None reproduz o reduce_tail=True usado em mobilenet_train.py
    # (que carrega weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT, o que
    # internamente zera weights_backbone); sem isso o backbone reconstruido aqui
    # fica com o dobro dos canais e o load_state_dict falha por size mismatch.
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
    in_channels = retrieve_out_channels(model.backbone, (imgsz, imgsz))
    num_anchors  = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels=in_channels, num_anchors=num_anchors,
        num_classes=num_classes, norm_layer=nn.BatchNorm2d,
    )
    return model


def load_torchvision_model(model_type, model_path, num_classes, device, imgsz=800):
    builders = {
        "faster_rcnn": lambda: _build_faster_rcnn(num_classes),
        "retinanet":   lambda: _build_retinanet(num_classes),
        "mobilenet":   lambda: _build_mobilenet(num_classes, imgsz),
    }
    model = builders[model_type]()
    ckpt  = torch.load(model_path, map_location=device, weights_only=True)
    sd    = ckpt.get("model", ckpt) if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    return model.to(device)


# ---------------------------------------------------------------------------
# Pruning utilitarios
# ---------------------------------------------------------------------------

def apply_global_unstructured_pruning(model, sparsity, prune_conv=True, prune_linear=True):
    params = []
    for m in model.modules():
        if prune_conv   and isinstance(m, nn.Conv2d):
            params.append((m, "weight"))
        if prune_linear and isinstance(m, nn.Linear):
            params.append((m, "weight"))
    if not params:
        raise RuntimeError("Nenhum parametro elegivel para pruning.")
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=sparsity)
    logger.info(f"  {len(params)} tensores incluidos no pruning global L1")


def collect_masks_cpu(model, prune_conv=True, prune_linear=True):
    masks = {}
    for name, m in model.named_modules():
        if prune_conv   and isinstance(m, nn.Conv2d)  and hasattr(m, "weight_mask"):
            masks[name] = m.weight_mask.data.clone().cpu()
        if prune_linear and isinstance(m, nn.Linear)  and hasattr(m, "weight_mask"):
            masks[name] = m.weight_mask.data.clone().cpu()
    return masks


def remove_pruning_hooks(model, prune_conv=True, prune_linear=True):
    for m in model.modules():
        if prune_conv   and isinstance(m, nn.Conv2d):
            try: prune.remove(m, "weight")
            except ValueError: pass
        if prune_linear and isinstance(m, nn.Linear):
            try: prune.remove(m, "weight")
            except ValueError: pass


def apply_masks_inplace(model, masks_cpu):
    with torch.no_grad():
        nm = dict(model.named_modules())
        for name, mask in masks_cpu.items():
            mod = nm.get(name)
            if mod is not None and hasattr(mod, "weight"):
                mod.weight.data.mul_(mask.to(mod.weight.device))


def count_params(model):
    total   = sum(p.numel() for p in model.parameters())
    nonzero = sum(int((p != 0).sum()) for p in model.parameters())
    return total, nonzero


# ---------------------------------------------------------------------------
# Loop de treinamento (torchvision)
# ---------------------------------------------------------------------------

def _train_one_epoch(model, optimizer, loader, device, scaler, epoch):
    model.train()
    total, n = 0.0, len(loader)
    for bi, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        valid   = [(img, tgt) for img, tgt in zip(images, targets) if tgt["boxes"].numel() > 0]
        if not valid:
            continue
        images, targets = zip(*valid)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda"):
            loss = sum(model(list(images), list(targets)).values())

        if not torch.isfinite(loss):
            logger.warning(f"  Loss NaN/Inf epoch {epoch} batch {bi} — ignorado")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / max(n, 1)


@torch.no_grad()
def _val_loss(model, loader, device):
    model.train()
    total = 0.0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.amp.autocast(device_type="cuda"):
            total += sum(model(images, targets).values()).item()
    return total / max(len(loader), 1)


def finetune_torchvision(model, train_loader, val_loader,
                          device, epochs, lr, run_dir):
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=0.9, weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.amp.GradScaler()

    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt  = run_dir / "best_pruned_ft.pth"
    best_loss  = float("inf")

    for epoch in range(1, epochs + 1):
        tl = _train_one_epoch(model, optimizer, train_loader, device, scaler, epoch)
        scheduler.step()
        vl = _val_loss(model, val_loader, device)
        logger.info(f"  Epoch {epoch:02d}/{epochs}  train={tl:.4f}  val={vl:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")
        if vl < best_loss:
            best_loss = vl
            torch.save(model.state_dict(), best_ckpt)
            logger.info(f"  >> checkpoint salvo (val={best_loss:.4f})")

    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    return model


# ---------------------------------------------------------------------------
# Pipeline torchvision
# ---------------------------------------------------------------------------

def run_torchvision_pipeline(model_type, model_path, dataset_dir,
                              output_path, sparsity, ft_epochs,
                              device, prune_conv=True, prune_linear=True,
                              lr=None, batch_size=None):
    cfg   = DEFAULTS[model_type]
    imgsz = cfg["imgsz"]
    batch = batch_size or cfg["batch"]
    lr    = lr         or cfg["lr"]

    logger.info(f"\nCarregando {model_type} de: {model_path}")
    model = load_torchvision_model(model_type, model_path, NUM_CLASSES, device, imgsz)

    t0, nz0 = count_params(model)
    logger.info(f"  Params totais: {t0:,}  |  nao-zero: {nz0:,}")

    logger.info(f"Aplicando pruning L1 (sparsity={sparsity * 100:.1f}%)...")
    apply_global_unstructured_pruning(model, sparsity, prune_conv, prune_linear)
    _, nzm = count_params(model)
    logger.info(f"  Esparsidade apos mascaramento: {100.0*(1-nzm/t0):.2f}%")

    train_loader, val_loader = make_dataloaders(
        Path(dataset_dir), imgsz, batch, cfg["workers"], cfg["square"]
    )

    logger.info(f"Fine-tuning por {ft_epochs} epocas (mascaras ativas, lr={lr})...")
    run_dir = Path(output_path).parent / "_ft_tmp"
    model   = finetune_torchvision(model, train_loader, val_loader,
                                    device, ft_epochs, lr, run_dir)

    logger.info("Removendo mascaras (pruning permanente)...")
    remove_pruning_hooks(model, prune_conv, prune_linear)

    tf, nzf = count_params(model)
    sparsity_f = 100.0 * (1 - nzf / tf)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out))
    disk_mb = out.stat().st_size / 1e6

    logger.info(f"  Esparsidade final : {sparsity_f:.2f}%")
    logger.info(f"  Reducao params    : {100.0*(1-nzf/nz0):.2f}%  ({nz0:,} → {nzf:,})")
    logger.info(f"  Tamanho no disco  : {disk_mb:.1f} MB")
    logger.info(f"  Salvo em          : {out}")


# ---------------------------------------------------------------------------
# Pipeline YOLO
# ---------------------------------------------------------------------------

def run_yolo_pipeline(model_path, data_yaml, output_path, sparsity, ft_epochs,
                       device, prune_conv=True, prune_linear=True, lr=None, batch_size=None):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("Ultralytics nao encontrada. Execute: pip install ultralytics")

    cfg   = DEFAULTS["yolo"]
    lr    = lr         or cfg["lr"]
    batch = batch_size or cfg["batch"]

    logger.info(f"\nCarregando YOLO de: {model_path}")
    yolo        = YOLO(str(model_path))
    torch_model = yolo.model

    t0, nz0 = count_params(torch_model)
    logger.info(f"  Params totais: {t0:,}  |  nao-zero: {nz0:,}")

    logger.info(f"Aplicando pruning L1 (sparsity={sparsity * 100:.1f}%)...")
    apply_global_unstructured_pruning(torch_model, sparsity, prune_conv, prune_linear)
    masks_cpu = collect_masks_cpu(torch_model, prune_conv, prune_linear)
    remove_pruning_hooks(torch_model, prune_conv, prune_linear)
    _, nzm = count_params(torch_model)
    logger.info(f"  Esparsidade apos pruning: {100.0*(1-nzm/t0):.2f}%  ({len(masks_cpu)} mascaras)")

    def _zero_pruned(trainer):
        apply_masks_inplace(trainer.model, masks_cpu)

    yolo.add_callback("on_train_batch_end", _zero_pruned)

    logger.info(f"Fine-tuning YOLO por {ft_epochs} epocas (lr={lr})...")
    out       = Path(output_path)
    runs_dir  = out.parent / "_ft_yolo_tmp"

    yolo.train(
        data=str(data_yaml),
        epochs=ft_epochs,
        imgsz=cfg["imgsz"],
        batch=batch,
        device=device.index if device.type == "cuda" else "cpu",
        workers=cfg["workers"],
        lr0=lr,
        amp=True,
        project=str(runs_dir),
        name="pruned_ft",
        exist_ok=True,
        patience=0,
        verbose=False,
    )

    apply_masks_inplace(yolo.model, masks_cpu)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(yolo.model.state_dict(), str(out))
    disk_mb = out.stat().st_size / 1e6

    tf, nzf = count_params(yolo.model)
    logger.info(f"  Esparsidade final : {100.0*(1-nzf/tf):.2f}%")
    logger.info(f"  Reducao params    : {100.0*(1-nzf/nz0):.2f}%  ({nz0:,} → {nzf:,})")
    logger.info(f"  Tamanho no disco  : {disk_mb:.1f} MB")
    logger.info(f"  Salvo em          : {out}")


# ---------------------------------------------------------------------------
# Catalogo de todos os modelos (modo --all_models)
# ---------------------------------------------------------------------------

def build_all_models_catalog(pct: int) -> list[dict]:
    """
    Retorna lista de dicts com as configuracoes de cada job de pruning.
    pct: percentual inteiro, ex: 50 para 50%.
    """
    jobs = []

    # --- Torchvision ---
    tv_models = {
        "faster_rcnn": ("runs/faster_rcnn/epochs_100", "best.pth"),
        "retinanet":   ("runs/retinanet/epochs_100",   "best.pth"),
        "mobilenet":   ("runs/mobilenet/epochs_100",   "best.pth"),
    }
    for model_type, (run_subdir, fname) in tv_models.items():
        for dataset in DATASETS:
            model_path  = BASE_DIR / run_subdir / dataset / fname
            output_path = BASE_DIR / "runs" / model_type / "pruned" / str(pct) / dataset / f"best_pruned.pth"
            jobs.append({
                "type":         model_type,
                "model_path":   model_path,
                "dataset_dir":  BASE_DIR / dataset,
                "data_yaml":    None,
                "output_path":  output_path,
                "label":        f"{model_type} | {dataset} | {pct}%",
            })

    # --- YOLO ---
    for variant in YOLO_VARIANTS:
        for dataset in DATASETS:
            model_path  = BASE_DIR / "runs" / "train" / f"{variant}_{dataset}_finetune" / "weights" / "best.pt"
            output_path = BASE_DIR / "runs" / "train" / "pruned" / str(pct) / dataset / f"{variant}_pruned.pt"
            jobs.append({
                "type":        "yolo",
                "model_path":  model_path,
                "dataset_dir": BASE_DIR / dataset,
                "data_yaml":   BASE_DIR / dataset / "data.yaml",
                "output_path": output_path,
                "label":       f"{variant} | {dataset} | {pct}%",
            })

    return jobs


# ---------------------------------------------------------------------------
# Orquestrador modo --all_models
# ---------------------------------------------------------------------------

def run_all_models(pct: int, ft_epochs: int, device: torch.device,
                   prune_conv: bool, prune_linear: bool,
                   skip_existing: bool = True, log_path: Path = None):
    if log_path:
        setup_file_logging(log_path)

    sparsity = pct / 100.0
    jobs     = build_all_models_catalog(pct)

    results = {"ok": [], "skip": [], "erro": []}   # erro: lista de (label, traceback_str)

    logger.info(f"\n{'#'*64}")
    logger.info(f"  MODO ALL_MODELS  |  pruning={pct}%  |  ft_epochs={ft_epochs}")
    logger.info(f"  Total de jobs: {len(jobs)}")
    logger.info(f"{'#'*64}")

    for i, job in enumerate(jobs, 1):
        label       = job["label"]
        model_path  = job["model_path"]
        output_path = job["output_path"]

        logger.info(f"\n[{i}/{len(jobs)}] {label}")

        if not model_path.exists():
            logger.warning(f"  PULADO — modelo nao encontrado: {model_path}")
            results["skip"].append(label)
            continue

        if skip_existing and output_path.exists():
            logger.info(f"  PULADO — ja existe: {output_path}")
            results["skip"].append(label)
            continue

        try:
            if job["type"] == "yolo":
                run_yolo_pipeline(
                    model_path  = model_path,
                    data_yaml   = job["data_yaml"],
                    output_path = output_path,
                    sparsity    = sparsity,
                    ft_epochs   = ft_epochs,
                    device      = device,
                    prune_conv  = prune_conv,
                    prune_linear= prune_linear,
                )
            else:
                run_torchvision_pipeline(
                    model_type  = job["type"],
                    model_path  = model_path,
                    dataset_dir = job["dataset_dir"],
                    output_path = output_path,
                    sparsity    = sparsity,
                    ft_epochs   = ft_epochs,
                    device      = device,
                    prune_conv  = prune_conv,
                    prune_linear= prune_linear,
                )
            results["ok"].append(label)

        except Exception:
            tb = traceback.format_exc()
            logger.error(f"  ERRO em '{label}':\n{tb}")
            results["erro"].append((label, tb))

    # Resumo final
    logger.info(f"\n{'='*64}")
    logger.info("  RESUMO FINAL")
    logger.info(f"{'='*64}")
    logger.info(f"  Concluidos : {len(results['ok'])}")
    logger.info(f"  Pulados    : {len(results['skip'])}")
    logger.info(f"  Com erro   : {len(results['erro'])}")

    if results["erro"]:
        logger.info(f"\n{'='*64}")
        logger.info("  DETALHES DOS ERROS")
        logger.info(f"{'='*64}")
        for label, tb in results["erro"]:
            logger.info(f"\n  >>> {label}")
            # Exibe apenas a ultima linha do traceback (mensagem da excecao)
            last_line = [l.strip() for l in tb.strip().splitlines() if l.strip()][-1]
            logger.info(f"  Causa: {last_line}")
            logger.info("  Traceback completo:")
            for line in tb.strip().splitlines():
                logger.info(f"    {line}")
            logger.info("")

    logger.info(f"{'='*64}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Pruning nao-estrutural L1 + Fine-tuning de modelos de deteccao",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Modo
    mode = p.add_argument_group("Modo de execucao")
    mode.add_argument("--all_models", action="store_true",
                      help="Pruna todos os modelos (faster_rcnn, retinanet, mobilenet, yolo x6) "
                           "nos datasets oxford e caviar")
    mode.add_argument("--model_type", choices=["yolo", "faster_rcnn", "retinanet", "mobilenet"],
                      help="[modo unico] Tipo do modelo")
    mode.add_argument("--model_path",
                      help="[modo unico] Caminho para o modelo treinado")
    mode.add_argument("--output_path",
                      help="[modo unico] Caminho de saida para o modelo podado")

    # Dataset (modo unico)
    ds = p.add_argument_group("Dataset (modo unico)")
    ds.add_argument("--dataset_dir",
                    help="Diretorio do dataset (obrigatorio para modelos torchvision). "
                         "Deve conter datasets/images/{train,val} e datasets/labels/{train,val}")
    ds.add_argument("--data_yaml",
                    help="Caminho para data.yaml (obrigatorio para YOLO)")

    # Pruning
    pr = p.add_argument_group("Pruning")
    pr.add_argument("--pruning_percentage", type=int, default=50,
                    help="Percentual de pesos zerados, ex: 50 para 50%%  (1-99)")
    pr.add_argument("--no_prune_conv",   action="store_true", help="Nao podar camadas Conv2d")
    pr.add_argument("--no_prune_linear", action="store_true", help="Nao podar camadas Linear")

    # Fine-tuning
    ft = p.add_argument_group("Fine-tuning")
    ft.add_argument("--ft_epochs",   type=int,   default=10,  help="Epocas de fine-tuning pos-pruning")
    ft.add_argument("--lr",          type=float, default=0.0, help="Learning rate (0 = padrao do modelo)")
    ft.add_argument("--batch_size",  type=int,   default=0,   help="Batch size (0 = padrao do modelo)")

    # Comportamento
    p.add_argument("--overwrite", action="store_true",
                   help="[all_models] Reprocessa mesmo que o arquivo de saida ja exista")

    args = p.parse_args()

    if not 1 <= args.pruning_percentage <= 99:
        p.error("--pruning_percentage deve ser um inteiro entre 1 e 99")

    if not args.all_models:
        if not args.model_type:
            p.error("--model_type e obrigatorio no modo unico (ou use --all_models)")
        if not args.model_path:
            p.error("--model_path e obrigatorio no modo unico")
        if not args.output_path:
            p.error("--output_path e obrigatorio no modo unico")

    args.prune_conv   = not args.no_prune_conv
    args.prune_linear = not args.no_prune_linear

    return args


def main():
    args    = parse_args()
    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sparsity = args.pruning_percentage / 100.0

    logger.info("=" * 64)
    logger.info("  PRUNING NAO-ESTRUTURAL L1 + FINE-TUNING")
    logger.info("=" * 64)
    logger.info(f"  Percentual de pruning : {args.pruning_percentage}%")
    logger.info(f"  Epocas fine-tuning    : {args.ft_epochs}")
    logger.info(f"  Prune Conv2d          : {args.prune_conv}")
    logger.info(f"  Prune Linear          : {args.prune_linear}")
    logger.info(f"  Dispositivo           : {device}")

    if args.all_models:
        log_path = BASE_DIR / "runs" / "pruned_logs" / f"prune_{args.pruning_percentage}pct.log"
        run_all_models(
            pct          = args.pruning_percentage,
            ft_epochs    = args.ft_epochs,
            device       = device,
            prune_conv   = args.prune_conv,
            prune_linear = args.prune_linear,
            skip_existing= not args.overwrite,
            log_path     = log_path,
        )
    elif args.model_type == "yolo":
        if not args.data_yaml:
            raise SystemExit("--data_yaml e obrigatorio para modelo YOLO.")
        run_yolo_pipeline(
            model_path  = Path(args.model_path),
            data_yaml   = Path(args.data_yaml),
            output_path = Path(args.output_path),
            sparsity    = sparsity,
            ft_epochs   = args.ft_epochs,
            device      = device,
            prune_conv  = args.prune_conv,
            prune_linear= args.prune_linear,
            lr          = args.lr   or None,
            batch_size  = args.batch_size or None,
        )
    else:
        if not args.dataset_dir:
            raise SystemExit("--dataset_dir e obrigatorio para modelos torchvision.")
        run_torchvision_pipeline(
            model_type  = args.model_type,
            model_path  = Path(args.model_path),
            dataset_dir = Path(args.dataset_dir),
            output_path = Path(args.output_path),
            sparsity    = sparsity,
            ft_epochs   = args.ft_epochs,
            device      = device,
            prune_conv  = args.prune_conv,
            prune_linear= args.prune_linear,
            lr          = args.lr   or None,
            batch_size  = args.batch_size or None,
        )


if __name__ == "__main__":
    main()
