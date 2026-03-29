"""
Avaliacao de todos os modelos treinados (YOLO, Faster R-CNN, RetinaNet)
nas bases oxford e caviar.

Gera planilha Excel: resultados_avaliacao.xlsx
  - Uma aba por dataset
  - Uma linha por modelo com as metricas: Params, mAP50, mAP50-95,
    Precision, Recall, F1, Inferencia(ms), FPS, FLOPS(G)

Dependencias extras:
    pip install openpyxl torchmetrics thop
"""

import time
import logging
from functools import partial
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from ultralytics import YOLO

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
    retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from torchmetrics.detection.mean_ap import MeanAveragePrecision

try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

# ---------------------------------------------------------------------------
# Configuracoes
# ---------------------------------------------------------------------------

BASE_DIR    = Path("C:/workspace/mestrado/novo_teste")
NUM_EPOCHS  = 100
MODELS_DIR  = BASE_DIR / "models" / f"epochs_{NUM_EPOCHS}"
IMGSZ       = 800
BATCH_SIZE  = 1     # inferencia imagem a imagem para medir latencia real
NUM_WORKERS = 4
CLASS_NAMES = ["pessoa"]

DATASETS = {
    "oxford": BASE_DIR / "oxford",
    "caviar": BASE_DIR / "caviar",
}

# Modelos YOLO disponiveis por dataset
YOLO_STEMS = ["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset para modelos Torchvision
# ---------------------------------------------------------------------------

class YOLOTestDataset(Dataset):
    def __init__(self, images_dir: Path, labels_dir: Path, imgsz: int = 800):
        self.labels_dir = labels_dir
        self.imgsz      = imgsz
        self.samples    = [
            p for p in sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
            if (labels_dir / (p.stem + ".txt")).exists()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path   = self.samples[idx]
        label_path = self.labels_dir / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        scale = self.imgsz / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        boxes, labels = [], []
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = max(0.0, (xc - bw / 2) * new_w)
            y1 = max(0.0, (yc - bh / 2) * new_h)
            x2 = min(float(new_w), (xc + bw / 2) * new_w)
            y2 = min(float(new_h), (yc + bh / 2) * new_h)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(cls + 1)

        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)

        return img_tensor, {"boxes": boxes_t, "labels": labels_t}


def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Contagem de parametros e FLOPS
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> float:
    """Retorna numero de parametros em milhoes."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def estimate_flops(model: nn.Module, device: torch.device, imgsz: int = 800) -> float:
    """Estima GFLOPs com thop. Retorna None se thop nao estiver instalado."""
    if not HAS_THOP:
        return None
    dummy = torch.zeros(1, 3, imgsz, imgsz).to(device)
    try:
        model.eval()
        flops, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        return flops / 1e9
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Avaliacao de modelos YOLO
# ---------------------------------------------------------------------------

def evaluate_yolo(model_path: Path, data_yaml: str, model_label: str, device: int, imgsz: int = IMGSZ) -> dict:
    logger.info(f"  Avaliando YOLO: {model_label}")
    model = YOLO(str(model_path))

    # Metricas no conjunto de teste
    results = model.val(data=data_yaml, split="test", device=device, verbose=False)
    box     = results.box

    map50    = float(box.map50)
    map50_95 = float(box.map)
    precision = float(box.mp)
    recall    = float(box.mr)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    # Parametros
    params_m = count_params(model.model)

    # FLOPs via thop (mesmo metodo dos modelos Torchvision)
    device = next(model.model.parameters()).device
    flops_g = estimate_flops(model.model, device, imgsz)

    # Inferencia: mede tempo medio por imagem
    inf_ms = _yolo_inference_time(model, data_yaml, device)

    return {
        "Modelo":            model_label,
        "Compressao":        "Nenhum (baseline)",
        "Params (M)":        round(params_m, 2),
        "Taxa compressao (%)": 0.0,
        "mAP50":             round(map50, 4),
        "mAP50-95":          round(map50_95, 4),
        "Precision":         round(precision, 4),
        "Recall":            round(recall, 4),
        "F1":                round(f1, 4),
        "Inferencia (ms)":   round(inf_ms, 2),
        "FPS":               round(1000 / inf_ms, 1) if inf_ms else None,
        "FLOPS (G)":         round(flops_g, 2) if flops_g else None,
    }


def _yolo_inference_time(model, data_yaml: str, device: int, n_warmup: int = 10) -> float:
    """Mede latencia media de inferencia em ms usando o conjunto de teste."""
    results = model.val(data=data_yaml, split="test", device=device, verbose=False)
    # speed retorna dict com 'inference' em ms
    return float(results.speed.get("inference", 0))


# ---------------------------------------------------------------------------
# Precision / Recall por IoU matching (para modelos Torchvision)
# ---------------------------------------------------------------------------

def _compute_precision_recall(all_preds: list, all_targets: list, iou_thresh: float = 0.5):
    """
    Calcula Precision e Recall globais usando matching greedy por IoU.
    all_preds:   lista de dicts {"boxes": Tensor[N,4], "scores": Tensor[N], "labels": Tensor[N]}
    all_targets: lista de dicts {"boxes": Tensor[M,4], "labels": Tensor[M]}
    """
    tp = fp = fn = 0

    for pred, target in zip(all_preds, all_targets):
        pred_boxes   = pred["boxes"]
        target_boxes = target["boxes"]

        matched_gt = set()

        for pb in pred_boxes:
            if len(target_boxes) == 0:
                fp += 1
                continue

            # Calcula IoU entre pb e todas as GT boxes
            x1 = torch.max(pb[0], target_boxes[:, 0])
            y1 = torch.max(pb[1], target_boxes[:, 1])
            x2 = torch.min(pb[2], target_boxes[:, 2])
            y2 = torch.min(pb[3], target_boxes[:, 3])
            inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
            area_pb = (pb[2] - pb[0]) * (pb[3] - pb[1])
            area_gt = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
            iou = inter / (area_pb + area_gt - inter + 1e-9)

            best_iou, best_idx = iou.max(0)
            best_idx = best_idx.item()

            if best_iou >= iou_thresh and best_idx not in matched_gt:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1

        fn += len(target_boxes) - len(matched_gt)

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    return precision, recall


# ---------------------------------------------------------------------------
# Avaliacao de modelos Torchvision (FasterRCNN / RetinaNet)
# ---------------------------------------------------------------------------

def _load_faster_rcnn(weights_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device).eval()


def _load_retinanet(weights_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = retinanet_resnet50_fpn(weights=None)
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(nn.GroupNorm, 32),
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device).eval()


def evaluate_torchvision(
    model: nn.Module,
    model_label: str,
    dataset_dir: Path,
    device: torch.device,
) -> dict:
    logger.info(f"  Avaliando: {model_label}")

    num_classes = len(CLASS_NAMES) + 1
    images_dir  = dataset_dir / "datasets" / "images" / "test"
    labels_dir  = dataset_dir / "datasets" / "labels" / "test"
    loader      = DataLoader(
        YOLOTestDataset(images_dir, labels_dir, imgsz=IMGSZ),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    metric    = MeanAveragePrecision(iou_type="bbox")
    times     = []
    all_preds = []
    all_tgts  = []

    with torch.no_grad():
        for images, targets in loader:
            images_gpu = [img.to(device) for img in images]

            # Aquece a GPU nas primeiras imagens
            if len(times) == 0:
                for _ in range(5):
                    _ = model(images_gpu)
                torch.cuda.synchronize()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            preds = model(images_gpu)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

            preds_fmt   = [{"boxes": p["boxes"].cpu(), "scores": p["scores"].cpu(), "labels": p["labels"].cpu()} for p in preds]
            targets_fmt = [{"boxes": t["boxes"],       "labels": t["labels"]}  for t in targets]
            metric.update(preds_fmt, targets_fmt)
            all_preds.extend(preds_fmt)
            all_tgts.extend(targets_fmt)

    results  = metric.compute()
    map50    = float(results["map_50"])
    map50_95 = float(results["map"])

    # Precision e Recall calculados por IoU matching direto (evita -1 do torchmetrics)
    precision, recall = _compute_precision_recall(all_preds, all_tgts, iou_thresh=0.5)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    inf_ms    = float(np.mean(times[1:]) if len(times) > 1 else times[0])
    params_m  = count_params(model)
    flops_g   = estimate_flops(model, device, IMGSZ)

    return {
        "Modelo":              model_label,
        "Compressao":          "Nenhum (baseline)",
        "Params (M)":          round(params_m, 2),
        "Taxa compressao (%)": 0.0,
        "mAP50":               round(map50, 4),
        "mAP50-95":            round(map50_95, 4),
        "Precision":           round(precision, 4),
        "Recall":              round(recall, 4),
        "F1":                  round(f1, 4),
        "Inferencia (ms)":     round(inf_ms, 2),
        "FPS":                 round(1000 / inf_ms, 1),
        "FLOPS (G)":           round(flops_g, 2) if flops_g else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("GPU nao encontrada.")

    device     = torch.device("cuda:0")
    device_int = 0
    num_classes = len(CLASS_NAMES) + 1

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    if not HAS_THOP:
        logger.warning("thop nao instalado — FLOPS sera None para modelos Torchvision. Instale com: pip install thop")

    writer = pd.ExcelWriter(BASE_DIR / "resultados_avaliacao.xlsx", engine="openpyxl")

    for dataset_name, dataset_dir in DATASETS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset_name.upper()}")
        logger.info(f"{'='*60}")

        data_yaml = str(dataset_dir / "data.yaml")
        rows      = []

        models_dataset_dir = MODELS_DIR / dataset_name

        # ---- YOLO --------------------------------------------------------
        for stem in YOLO_STEMS:
            model_path = models_dataset_dir / f"{stem}_finetuned.pt"
            if not model_path.exists():
                logger.warning(f"Nao encontrado: {model_path}, pulando.")
                continue
            row = evaluate_yolo(model_path, data_yaml, stem.upper(), device_int)
            rows.append(row)

        # ---- Faster R-CNN ------------------------------------------------
        frcnn_path = models_dataset_dir / "faster_rcnn_finetuned.pth"
        if frcnn_path.exists():
            frcnn = _load_faster_rcnn(frcnn_path, num_classes, device)
            row   = evaluate_torchvision(frcnn, "FasterRCNN", dataset_dir, device)
            rows.append(row)
            del frcnn
            torch.cuda.empty_cache()
        else:
            logger.warning(f"Nao encontrado: faster_rcnn_finetuned.pth em {models_dataset_dir}")

        # ---- RetinaNet ---------------------------------------------------
        retina_path = models_dataset_dir / "retinanet_finetuned.pth"
        if retina_path.exists():
            retina = _load_retinanet(retina_path, num_classes, device)
            row    = evaluate_torchvision(retina, "RetinaNet", dataset_dir, device)
            rows.append(row)
            del retina
            torch.cuda.empty_cache()
        else:
            logger.warning(f"Nao encontrado: retinanet_finetuned.pth em {models_dataset_dir}")

        # ---- Salva aba do dataset ----------------------------------------
        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name=dataset_name, index=False)
        logger.info(f"Aba '{dataset_name}' salva com {len(rows)} modelos.")

    writer.close()
    output = BASE_DIR / "resultados_avaliacao.xlsx"
    logger.info(f"\nPlanilha salva em: {output}")
