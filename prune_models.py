"""
Pruning (nao-estrutural L1 ou estrutural via torch-pruning) + Fine-tuning
(opcionalmente com knowledge distillation) para multiplos modelos.

Suporta:
  - faster_rcnn : Faster R-CNN ResNet-50 FPN (.pth)
  - retinanet   : RetinaNet ResNet-50 FPN (.pth)
  - mobilenet   : SSDLite320-MobileNetV3 (.pth)
  - yolo        : YOLOv10 variantes n/s/m/b/l/x (Ultralytics .pt)

--pruning_mode unstructured (padrao, comportamento original):
  1. Carrega modelo treinado
  2. Aplica pruning nao-estrutural global L1 (mascaras ativas) — zera pesos mas
     mantem a forma dos tensores (Params/FLOPS praticamente inalterados)
  3. Fine-tuning de --ft_epochs epocas COM mascaras ativas
  4. Remove mascaras (pruning permanente — zeros gravados no weight)
  5. Salva modelo podado

--pruning_mode structured (poda de canais/filtros via torch-pruning):
  1. Carrega modelo treinado
  2. Constroi um torch_pruning.DependencyGraph e remove canais/filtros de fato
     (ignorando a cabeca de predicao e, nos modelos torchvision, a FPN/backbone
     de extracao de features inteira — ver _torchvision_ignored_modules) —
     reduz Params/FLOPS de verdade, produz um nn.Module menor
  3. Fine-tuning de --ft_epochs epocas (sem mascaras — o modelo ja e menor)
  4. Salva modelo podado

--use_kd (qualquer um dos dois modos acima): durante o fine-tuning pos-poda,
soma-se uma loss de knowledge distillation contra --teacher_path (o modelo
original, sem poda, da mesma arquitetura/variante — "self-distillation" para
recuperar acuracia perdida na poda). Para YOLO usa o suporte nativo
distill_model=/dis= do ultralytics.YOLO.train(); para os modelos torchvision
usa hooks nas cabecas de predicao (ver _register_kd_hooks/_compute_kd_loss).

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
import torch.nn.functional as F
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
    """Carrega um checkpoint torchvision. Aceita tanto um state_dict puro
    (formato usado pelo baseline e pela poda nao-estrutural — arquitetura
    identica ao build padrao, load_state_dict(strict=True)) quanto um
    nn.Module inteiro ja pickled (formato usado pela poda estrutural, cuja
    arquitetura — numero de canais — difere do build padrao e por isso nao
    pode ser reconstruida a partir de um state_dict).

    IMPORTANTE: sempre desserializa com map_location="cpu" e so' move pro
    device final via model.to(device) explicito. Alguns modulos (ex.:
    DefaultBoxGenerator do SSDLite/MobileNet) guardam caches internos como
    atributo Python comum (lista de tensores), nao como buffer registrado —
    nn.Module.to(device) nao alcanca esses atributos, entao eles ficam para
    sempre no device em que foram criados. Um build fresco (branch do
    state_dict abaixo) sempre os cria em CPU; mas se desserializassemos o
    nn.Module inteiro (poda estrutural) direto com map_location=device, esses
    mesmos caches viriam remapeados pro device tambem (map_location com um
    torch.device forca TODO tensor do arquivo, buffer ou nao) — e o forward()
    desses modulos assume que esses caches estao em CPU (constroi outros
    tensores auxiliares em CPU e da' cat() neles), causando um
    "Expected all tensors to be on the same device" na avaliacao."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, nn.Module):
        return ckpt.to(device)

    builders = {
        "faster_rcnn": lambda: _build_faster_rcnn(num_classes),
        "retinanet":   lambda: _build_retinanet(num_classes),
        "mobilenet":   lambda: _build_mobilenet(num_classes, imgsz),
    }
    model = builders[model_type]()
    sd    = ckpt.get("model", ckpt) if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    return model.to(device)


# ---------------------------------------------------------------------------
# Pruning utilitarios
# ---------------------------------------------------------------------------

def apply_global_unstructured_pruning(model, sparsity, prune_conv=True, prune_linear=True,
                                       exclude_modules=None):
    """exclude_modules: modulos cujas camadas (e submodulos) ficam fora do
    pruning global (ex.: a cabeca de deteccao do YOLO)."""
    excluded = set()
    for ex in (exclude_modules or []):
        excluded.update(id(m) for m in ex.modules())

    params = []
    for m in model.modules():
        if id(m) in excluded:
            continue
        if prune_conv   and isinstance(m, nn.Conv2d):
            params.append((m, "weight"))
        if prune_linear and isinstance(m, nn.Linear):
            params.append((m, "weight"))
    if not params:
        raise RuntimeError("Nenhum parametro elegivel para pruning.")
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=sparsity)
    logger.info(f"  {len(params)} tensores incluidos no pruning global L1"
                f"{f' ({len(excluded)} modulos excluidos)' if excluded else ''}")


def log_layer_sparsity(model, title="", warn_threshold=0.95):
    """Loga a esparsidade (% de zeros) de cada Conv2d/Linear. Camadas com
    esparsidade >= warn_threshold sao sinalizadas — uma camada quase inteira
    zerada geralmente mata a rede (diagnostico de colapso pos-pruning).
    Funciona com mascaras ativas (weight_orig/weight_mask) ou removidas."""
    logger.info(f"  Esparsidade por camada {title}".rstrip())
    n_warn = 0
    for name, m in model.named_modules():
        if not isinstance(m, (nn.Conv2d, nn.Linear)):
            continue
        w = m.weight
        sp = float((w == 0).sum()) / max(w.numel(), 1)
        flag = ""
        if sp >= warn_threshold:
            flag = "  <<< CAMADA QUASE TODA ZERADA"
            n_warn += 1
        logger.info(f"    {name:<48s} {tuple(w.shape)!s:<22s} {100*sp:6.2f}%{flag}")
    if n_warn:
        logger.warning(f"  {n_warn} camada(s) com esparsidade >= {100*warn_threshold:.0f}% — "
                       f"risco de colapso da rede")


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
# Pruning estrutural (poda de canais/filtros via torch-pruning)
# ---------------------------------------------------------------------------
#
# Ao contrario do pruning nao-estrutural acima (zera pesos, mantem a forma dos
# tensores), aqui removemos canais/filtros de verdade — o modelo resultante e
# um nn.Module menor, com reducao real de Params e FLOPS. Isso exige nao podar
# camadas cujo numero de canais e fixo pela tarefa (cabeca de predicao, presa a
# num_classes/num_anchors) nem camadas que quebrariam sob poda de canais
# (blocos de atencao do YOLOv10, que fazem view/split de canais nao reconhecido
# automaticamente pelo DependencyGraph do torch-pruning).
#
# NOTA: esta poda ainda nao foi validada com uma execucao real (ver aviso no
# plano de implementacao) — rode um smoke test (--pruning_mode structured,
# --ft_epochs 1, um modelo pequeno) antes de disparar a matriz completa.

def _yolo_ignored_layers(model):
    """Camadas do YOLOv10 que a poda estrutural deve ignorar: a cabeca de
    deteccao inteira (model.model[-1], v10Detect — canais de saida presos a
    nc/anchors) e todo modulo de atencao (bloco PSA), cujo qkv/proj fazem
    view/split de canais que o torch-pruning nao rastreia corretamente."""
    ignored = [model.model[-1]]
    for m in model.modules():
        if m.__class__.__name__ == "Attention":
            for attr in ("qkv", "proj", "pe"):
                sub = getattr(m, attr, None)
                if sub is not None:
                    ignored.append(sub)
    return ignored


def apply_structured_pruning_yolo(torch_model, ratio, imgsz=640, device="cpu"):
    """Poda estrutural (canais) de um yolo.model (nn.Module puro) via
    torch-pruning. Retorna o proprio modulo, agora menor."""
    try:
        import torch_pruning as tp
    except ImportError:
        raise RuntimeError(
            "torch-pruning nao encontrada. Execute: pip install torch-pruning"
        )

    torch_model.eval()
    example_inputs = torch.randn(1, 3, imgsz, imgsz, device=device)
    ignored_layers = _yolo_ignored_layers(torch_model)

    imp = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        torch_model,
        example_inputs,
        importance=imp,
        pruning_ratio=ratio,
        ignored_layers=ignored_layers,
        global_pruning=True,
    )
    pruner.step()
    torch_model.train()
    return torch_model


def _torchvision_ignored_modules(model, model_type):
    """Modulos que a poda estrutural deve ignorar nos modelos torchvision:
    a cabeca de predicao (canais presos a num_classes/num_anchors) e a FPN
    (ou, no mobilenet, a ausencia dela) inteira — mantendo a largura de saida
    do extrator de features fixa, o que tambem e o que permite o KD baseado em
    hooks (ver _register_kd_hooks) funcionar sem adaptador de canal entre
    professor e aluno."""
    ignored = []
    if model_type == "faster_rcnn":
        ignored.append(model.backbone.fpn)
        ignored.append(model.rpn.head)
        ignored.append(model.roi_heads.box_predictor)
    elif model_type == "retinanet":
        ignored.append(model.backbone.fpn)
        ignored.append(model.head.classification_head)
        ignored.append(model.head.regression_head)
    elif model_type == "mobilenet":
        ignored.append(model.head.classification_head)
        ignored.append(model.head.regression_head)
    else:
        raise ValueError(f"model_type desconhecido: {model_type}")
    return ignored


def _swap_frozen_bn_for_tracing(model):
    """torch-pruning so reconhece nn.modules.batchnorm._BatchNorm como modulo
    acoplado a canais podados (reduz weight/bias/running_mean/running_var em
    lockstep com a conv anterior). O FrozenBatchNorm2d do torchvision (usado
    no backbone ResNet-50 do Faster R-CNN/RetinaNet) e uma classe propria que
    NAO herda de _BatchNorm — a poda estrutural o ignora silenciosamente,
    podando a conv anterior mas deixando o FrozenBatchNorm2d com o tamanho
    antigo (RuntimeError de shape mismatch no primeiro forward apos a poda).
    Troca cada FrozenBatchNorm2d por um nn.BatchNorm2d equivalente (mesmos
    valores, eval() fixo) antes da poda; ver _restore_frozen_bn para reverter
    depois preservando o comportamento "estatisticas congeladas" original."""
    from torchvision.ops.misc import FrozenBatchNorm2d

    swapped = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, FrozenBatchNorm2d):
                bn = nn.BatchNorm2d(child.weight.numel(), eps=child.eps,
                                     device=child.weight.device, dtype=child.weight.dtype)
                with torch.no_grad():
                    bn.weight.copy_(child.weight)
                    bn.bias.copy_(child.bias)
                    bn.running_mean.copy_(child.running_mean)
                    bn.running_var.copy_(child.running_var)
                bn.eval()
                full_name = f"{name}.{child_name}" if name else child_name
                setattr(module, child_name, bn)
                swapped.append(full_name)
    return swapped


def _restore_frozen_bn(model, swapped_names):
    """Reverte _swap_frozen_bn_for_tracing apos a poda, reconstruindo o
    FrozenBatchNorm2d com o novo numero de canais a partir dos valores (ja
    podados) do nn.BatchNorm2d correspondente."""
    from torchvision.ops.misc import FrozenBatchNorm2d

    named = dict(model.named_modules())
    for full_name in swapped_names:
        if "." in full_name:
            parent_name, child_name = full_name.rsplit(".", 1)
            parent = named[parent_name]
        else:
            parent, child_name = model, full_name
        bn = getattr(parent, child_name)
        frozen = FrozenBatchNorm2d(bn.num_features, eps=bn.eps).to(
            device=bn.weight.device, dtype=bn.weight.dtype)
        with torch.no_grad():
            frozen.weight.copy_(bn.weight)
            frozen.bias.copy_(bn.bias)
            frozen.running_mean.copy_(bn.running_mean)
            frozen.running_var.copy_(bn.running_var)
        setattr(parent, child_name, frozen)


def apply_structured_pruning_torchvision(model, model_type, ratio, imgsz, device="cpu"):
    """Poda estrutural (canais) de um modelo torchvision via torch-pruning,
    restrita ao extrator de features (backbone.body no faster_rcnn/retinanet,
    backbone inteiro no mobilenet) — FPN e cabeca de predicao ficam com a
    largura congelada (ver _torchvision_ignored_modules). Retorna o proprio
    modelo, agora menor."""
    try:
        import torch_pruning as tp
    except ImportError:
        raise RuntimeError(
            "torch-pruning nao encontrada. Execute: pip install torch-pruning"
        )

    # torch-pruning rastreia dependencias via grad_fn (backward autograd) — o
    # backbone do faster_rcnn/retinanet vem com o stem e a layer1 congelados
    # por padrao (trainable_backbone_layers do resnet_fpn_backbone), entao
    # sem isso o rastreamento simplesmente nao enxerga essa regiao da rede
    # (mesmo problema, mesma causa, do requires_grad=False do YOLO — ver
    # run_yolo_pipeline). Habilita grad em TODOS os parametros so' durante o
    # rastreamento/poda; restaura o requires_grad original de cada parametro
    # depois, pra nao alterar quais camadas o fine-tuning pos-poda de fato
    # treina (o congelamento do stem/layer1 e' proposital — parte da receita
    # de treino original, nao algo pra desfazer por acidente aqui).
    original_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    model.requires_grad_(True)
    model.eval()
    swapped_bn = _swap_frozen_bn_for_tracing(model)
    example_inputs = [torch.rand(3, imgsz, imgsz, device=device)]
    ignored_layers = _torchvision_ignored_modules(model, model_type)

    imp = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=ratio,
        ignored_layers=ignored_layers,
        global_pruning=True,
    )
    try:
        pruner.step()
    finally:
        _restore_frozen_bn(model, swapped_bn)
        # Os tensores de peso sao objetos novos apos a poda (canais
        # removidos) — restaura requires_grad pelo NOME do parametro (estavel
        # antes/depois, ja que a poda so' encolhe tensores existentes, nao
        # renomeia/adiciona/remove camadas), nao pelo objeto tensor antigo.
        for name, p in model.named_parameters():
            if name in original_requires_grad:
                p.requires_grad_(original_requires_grad[name])
    model.train()
    return model


# ---------------------------------------------------------------------------
# Knowledge distillation (modelos torchvision) — hooks nas cabecas densas
# ---------------------------------------------------------------------------
#
# Para YOLO, o KD e feito via o suporte nativo distill_model=/dis= do
# ultralytics.YOLO.train() (ver run_yolo_pipeline) — nenhuma logica de loss
# customizada e necessaria. Para os modelos torchvision nao ha suporte nativo,
# entao usamos KD por resposta: hooks nas camadas de predicao densas (antes do
# NMS/pos-processamento), cujo formato de saida independe do professor ser
# podado ou nao (a FPN e a cabeca ficam com largura fixa mesmo apos poda
# estrutural — ver _torchvision_ignored_modules), entao nao e preciso nenhum
# adaptador de canal entre professor e aluno.

def _register_kd_hooks(model, model_type):
    """Registra forward hooks nas camadas de predicao densas e retorna
    (handles, captured) — captured e populado a cada forward com a saida
    bruta dessas camadas (objectness/bbox do RPN no faster_rcnn; logits de
    classificacao/regressao no retinanet e mobilenet)."""
    captured = {}

    def _make_hook(key):
        def _hook(module, inputs, output):
            captured[key] = output
        return _hook

    handles = []
    if model_type == "faster_rcnn":
        handles.append(model.rpn.head.register_forward_hook(_make_hook("rpn")))
    elif model_type in ("retinanet", "mobilenet"):
        handles.append(model.head.classification_head.register_forward_hook(_make_hook("cls")))
        handles.append(model.head.regression_head.register_forward_hook(_make_hook("reg")))
    else:
        raise ValueError(f"model_type desconhecido: {model_type}")
    return handles, captured


def _remove_kd_hooks(handles):
    for h in handles:
        h.remove()


def _compute_kd_loss(model_type, student_captured, teacher_captured, temperature=2.0):
    """Combina MSE (regressao de caixas) + KLDiv com temperatura (classificacao)
    entre as saidas capturadas do aluno e do professor (professor destacado do
    grafo — ja vem de um forward em torch.no_grad())."""
    if model_type == "faster_rcnn":
        s_obj, s_reg = student_captured["rpn"]
        t_obj, t_reg = teacher_captured["rpn"]
        cls_loss = sum(
            F.kl_div(
                F.logsigmoid(s / temperature), torch.sigmoid(t.detach() / temperature),
                reduction="batchmean",
            ) * (temperature ** 2)
            for s, t in zip(s_obj, t_obj)
        ) / max(len(s_obj), 1)
        reg_loss = sum(
            F.mse_loss(s, t.detach()) for s, t in zip(s_reg, t_reg)
        ) / max(len(s_reg), 1)
        return cls_loss + reg_loss

    # retinanet / mobilenet: classification_head retorna (N, HWA, num_classes)
    # logits; regression_head retorna (N, HWA, 4).
    s_cls, t_cls = student_captured["cls"], teacher_captured["cls"]
    s_reg, t_reg = student_captured["reg"], teacher_captured["reg"]
    cls_loss = F.kl_div(
        F.log_softmax(s_cls / temperature, dim=-1),
        F.softmax(t_cls.detach() / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
    reg_loss = F.mse_loss(s_reg, t_reg.detach())
    return cls_loss + reg_loss


# ---------------------------------------------------------------------------
# Loop de treinamento (torchvision)
# ---------------------------------------------------------------------------

def _train_one_epoch(model, optimizer, loader, device, scaler, epoch,
                      teacher=None, model_type=None, kd_weight=0.0):
    model.train()
    total, n = 0.0, len(loader)

    student_handles = student_captured = teacher_handles = teacher_captured = None
    if teacher is not None:
        student_handles, student_captured = _register_kd_hooks(model, model_type)
        teacher_handles, teacher_captured = _register_kd_hooks(teacher, model_type)

    try:
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

                if teacher is not None:
                    with torch.no_grad():
                        teacher(list(images))
                    kd_loss = _compute_kd_loss(model_type, student_captured, teacher_captured)
                    loss = loss + kd_weight * kd_loss

            if not torch.isfinite(loss):
                logger.warning(f"  Loss NaN/Inf epoch {epoch} batch {bi} — ignorado")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
    finally:
        if student_handles is not None:
            _remove_kd_hooks(student_handles)
            _remove_kd_hooks(teacher_handles)

    return total / max(n, 1)


@torch.no_grad()
def _val_loss(model, loader, device):
    # model.train() e' necessario so' para o forward dos modelos torchvision
    # retornar o dict de losses (o branch de eval() retorna deteccoes
    # pos-processadas); mas isso tambem poe as camadas BatchNorm em modo de
    # treino, e o val_loader usa batch_size=1 — em mapas de feature que
    # colapsam para 1x1 (ex.: blocos extra do SSDLite/MobileNet em imgsz
    # baixo), BatchNorm com 1 valor por canal falha
    # ("Expected more than 1 value per channel"). Mantemos BatchNorm em
    # eval() (usa estatisticas ja acumuladas) enquanto o resto do modelo fica
    # em train() so' para habilitar o calculo da loss.
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()

    total = 0.0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.amp.autocast(device_type="cuda"):
            total += sum(model(images, targets).values()).item()
    return total / max(len(loader), 1)


def finetune_torchvision(model, train_loader, val_loader,
                          device, epochs, lr, run_dir,
                          teacher=None, model_type=None, kd_weight=1.0):
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=0.9, weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.amp.GradScaler()

    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt  = run_dir / "best_pruned_ft.pth"
    best_loss  = float("inf")

    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        logger.info(f"  Knowledge distillation ativa (professor={model_type}, kd_weight={kd_weight})")

    for epoch in range(1, epochs + 1):
        tl = _train_one_epoch(model, optimizer, train_loader, device, scaler, epoch,
                               teacher=teacher, model_type=model_type, kd_weight=kd_weight)
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
                              lr=None, batch_size=None, pruning_mode="unstructured",
                              use_kd=False, teacher_path=None, kd_weight=1.0):
    cfg   = DEFAULTS[model_type]
    imgsz = cfg["imgsz"]
    batch = batch_size or cfg["batch"]
    lr    = lr         or cfg["lr"]

    logger.info(f"\nCarregando {model_type} de: {model_path}")
    model = load_torchvision_model(model_type, model_path, NUM_CLASSES, device, imgsz)

    t0, nz0 = count_params(model)
    logger.info(f"  Params totais: {t0:,}  |  nao-zero: {nz0:,}")

    if pruning_mode == "structured":
        logger.info(f"Aplicando pruning ESTRUTURAL (ratio={sparsity * 100:.1f}%)...")
        apply_structured_pruning_torchvision(model, model_type, sparsity, imgsz, device=device)
        model = model.to(device)
        _, nzm = count_params(model)
        logger.info(f"  Params apos poda estrutural: {nzm:,}  ({100.0*(1-nzm/t0):.2f}% removidos)")
    else:
        logger.info(f"Aplicando pruning L1 nao-estrutural (sparsity={sparsity * 100:.1f}%)...")
        apply_global_unstructured_pruning(model, sparsity, prune_conv, prune_linear)
        _, nzm = count_params(model)
        logger.info(f"  Esparsidade apos mascaramento: {100.0*(1-nzm/t0):.2f}%")

    train_loader, val_loader = make_dataloaders(
        Path(dataset_dir), imgsz, batch, cfg["workers"], cfg["square"]
    )

    teacher = None
    if use_kd:
        if teacher_path is None:
            raise ValueError("use_kd=True exige teacher_path.")
        logger.info(f"  Carregando professor (KD) de: {teacher_path}")
        teacher = load_torchvision_model(model_type, teacher_path, NUM_CLASSES, device, imgsz)

    logger.info(f"Fine-tuning por {ft_epochs} epocas (lr={lr}"
                f"{', com KD' if use_kd else ''})...")
    run_dir = Path(output_path).parent / "_ft_tmp"
    model   = finetune_torchvision(model, train_loader, val_loader,
                                    device, ft_epochs, lr, run_dir,
                                    teacher=teacher, model_type=model_type, kd_weight=kd_weight)

    if pruning_mode == "unstructured":
        logger.info("Removendo mascaras (pruning permanente)...")
        remove_pruning_hooks(model, prune_conv, prune_linear)

    tf, nzf = count_params(model)
    sparsity_f = 100.0 * (1 - nzf / tf)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if pruning_mode == "structured":
        # Arquitetura (numero de canais) difere do build padrao — precisa
        # salvar o nn.Module inteiro, nao so o state_dict (ver
        # load_torchvision_model, que reconhece os dois formatos).
        torch.save(model, str(out))
    else:
        torch.save(model.state_dict(), str(out))
    disk_mb = out.stat().st_size / 1e6

    logger.info(f"  Esparsidade/reducao final : {sparsity_f:.2f}%")
    logger.info(f"  Reducao params    : {100.0*(1-nzf/nz0):.2f}%  ({nz0:,} → {nzf:,})")
    logger.info(f"  Tamanho no disco  : {disk_mb:.1f} MB")
    logger.info(f"  Salvo em          : {out}")


# ---------------------------------------------------------------------------
# Pipeline YOLO
# ---------------------------------------------------------------------------

def run_yolo_pipeline(model_path, data_yaml, output_path, sparsity, ft_epochs,
                       device, prune_conv=True, prune_linear=True, lr=None, batch_size=None,
                       pruning_mode="unstructured", use_kd=False, teacher_path=None, kd_weight=0.1,
                       skip_head=False, optimizer="SGD", warmup_epochs=0):
    if pruning_mode == "structured":
        raise NotImplementedError(
            "Poda estrutural nao e suportada para YOLO: os blocos C2f/C2fCIB/PSA do "
            "YOLOv10 fazem chunk()/split interno de canais que o torch-pruning nao "
            "consegue rastrear corretamente (produz IndexError ou poda 0% dos canais "
            "silenciosamente, dependendo de requires_grad). A solucao oficial do "
            "torch-pruning para isso (exemplo yolov8_pruning.py) reescreve o bloco C2f "
            "trocando o split por duas convolucoes separadas — nao implementado aqui "
            "por decisao explicita do usuario (custo/risco vs beneficio). Use "
            "pruning_mode='unstructured' para YOLO."
        )
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

    masks_cpu = None
    if pruning_mode == "structured":
        logger.info(f"Aplicando pruning ESTRUTURAL (ratio={sparsity * 100:.1f}%)...")
        param_device = next(torch_model.parameters()).device
        apply_structured_pruning_yolo(torch_model, sparsity, imgsz=cfg["imgsz"], device=param_device)
        _, nzm = count_params(torch_model)
        logger.info(f"  Params apos poda estrutural: {nzm:,}  ({100.0*(1-nzm/t0):.2f}% removidos)")
    else:
        logger.info(f"Aplicando pruning L1 nao-estrutural (sparsity={sparsity * 100:.1f}%)...")
        exclude = [torch_model.model[-1]] if skip_head else None
        if skip_head:
            logger.info("  Cabeca de deteccao (model.model[-1]) excluida do pruning")
        apply_global_unstructured_pruning(torch_model, sparsity, prune_conv, prune_linear,
                                          exclude_modules=exclude)
        masks_cpu = collect_masks_cpu(torch_model, prune_conv, prune_linear)
        remove_pruning_hooks(torch_model, prune_conv, prune_linear)
        _, nzm = count_params(torch_model)
        logger.info(f"  Esparsidade apos pruning: {100.0*(1-nzm/t0):.2f}%  ({len(masks_cpu)} mascaras)")
        log_layer_sparsity(torch_model, "(antes do fine-tuning)")

        def _zero_pruned(trainer):
            apply_masks_inplace(trainer.model, masks_cpu)

        yolo.add_callback("on_train_batch_end", _zero_pruned)

    logger.info(f"Fine-tuning YOLO por {ft_epochs} epocas (lr={lr}"
                f"{', com KD' if use_kd else ''})...")
    out       = Path(output_path)
    runs_dir  = out.parent / "_ft_yolo_tmp"

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=ft_epochs,
        imgsz=cfg["imgsz"],
        batch=batch,
        device=device.index if device.type == "cuda" else "cpu",
        workers=cfg["workers"],
        lr0=lr,
        # optimizer="auto" (padrao) ignora lr0 e escolhe AdamW com LR proprio
        # quando ha poucas iteracoes; warmup sobre pesos ja podados desestabiliza.
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        amp=True,
        project=str(runs_dir),
        name="pruned_ft",
        exist_ok=True,
        patience=0,
        verbose=False,
    )
    if use_kd:
        if teacher_path is None:
            raise ValueError("use_kd=True exige teacher_path.")
        # Suporte nativo de knowledge distillation do ultralytics instalado
        # (mesmo mecanismo ja usado e validado em kd_test/kd.py): professor
        # congelado internamente, sem necessidade de loss customizada.
        train_kwargs["distill_model"] = str(teacher_path)
        train_kwargs["dis"] = kd_weight

    yolo.train(**train_kwargs)

    out.parent.mkdir(parents=True, exist_ok=True)
    if pruning_mode == "structured":
        # yolo.train() acima re-treina o proprio objeto `yolo` in-place;
        # yolo.save() grava um checkpoint completo (arquitetura + pesos),
        # carregavel depois via YOLO(path) sem precisar de um template —
        # necessario porque a arquitetura (numero de canais) difere do
        # modelo original.
        yolo.save(str(out))
    else:
        apply_masks_inplace(yolo.model, masks_cpu)
        torch.save(yolo.model.state_dict(), str(out))
        log_layer_sparsity(yolo.model, "(modelo final salvo)")
    disk_mb = out.stat().st_size / 1e6

    tf, nzf = count_params(yolo.model)
    logger.info(f"  Esparsidade/reducao final : {100.0*(1-nzf/tf):.2f}%")
    logger.info(f"  Reducao params    : {100.0*(1-nzf/nz0):.2f}%  ({nz0:,} → {nzf:,})")
    logger.info(f"  Tamanho no disco  : {disk_mb:.1f} MB")
    logger.info(f"  Salvo em          : {out}")


# ---------------------------------------------------------------------------
# Catalogo de todos os modelos (modo --all_models)
# ---------------------------------------------------------------------------

def build_all_models_catalog(pct: int, mode: str = "unstructured") -> list[dict]:
    """
    Retorna lista de dicts com as configuracoes de cada job de pruning.
    pct: percentual inteiro, ex: 50 para 50%.
    mode: "unstructured" (padrao, saida em runs/*/pruned/) ou "structured"
    (saida em runs/*/pruned_structured/ — caminho paralelo, nunca sobrescreve
    os pesos nao-estruturados ja existentes). KD nao e coberto por este
    catalogo/modo --all_models — ver run_compression_pipeline.py.
    """
    jobs = []
    pruned_dirname = "pruned" if mode == "unstructured" else "pruned_structured"

    # --- Torchvision ---
    tv_models = {
        "faster_rcnn": ("runs/faster_rcnn/epochs_100", "best.pth"),
        "retinanet":   ("runs/retinanet/epochs_100",   "best.pth"),
        "mobilenet":   ("runs/mobilenet/epochs_100",   "best.pth"),
    }
    for model_type, (run_subdir, fname) in tv_models.items():
        for dataset in DATASETS:
            model_path  = BASE_DIR / run_subdir / dataset / fname
            output_path = BASE_DIR / "runs" / model_type / pruned_dirname / str(pct) / dataset / f"best_pruned.pth"
            jobs.append({
                "type":         model_type,
                "model_path":   model_path,
                "dataset_dir":  BASE_DIR / dataset,
                "data_yaml":    None,
                "output_path":  output_path,
                "label":        f"{model_type} | {dataset} | {pct}% | {mode}",
            })

    # --- YOLO ---
    for variant in YOLO_VARIANTS:
        for dataset in DATASETS:
            model_path  = BASE_DIR / "runs" / "train" / f"{variant}_{dataset}_finetune" / "weights" / "best.pt"
            output_path = BASE_DIR / "runs" / "train" / pruned_dirname / str(pct) / dataset / f"{variant}_pruned.pt"
            jobs.append({
                "type":        "yolo",
                "model_path":  model_path,
                "dataset_dir": BASE_DIR / dataset,
                "data_yaml":   BASE_DIR / dataset / "data.yaml",
                "output_path": output_path,
                "label":       f"{variant} | {dataset} | {pct}% | {mode}",
            })

    return jobs


# ---------------------------------------------------------------------------
# Orquestrador modo --all_models
# ---------------------------------------------------------------------------

def run_all_models(pct: int, ft_epochs: int, device: torch.device,
                   prune_conv: bool, prune_linear: bool,
                   skip_existing: bool = True, log_path: Path = None,
                   pruning_mode: str = "unstructured"):
    if log_path:
        setup_file_logging(log_path)

    sparsity = pct / 100.0
    jobs     = build_all_models_catalog(pct, mode=pruning_mode)

    results = {"ok": [], "skip": [], "erro": []}   # erro: lista de (label, traceback_str)

    logger.info(f"\n{'#'*64}")
    logger.info(f"  MODO ALL_MODELS  |  pruning={pct}% ({pruning_mode})  |  ft_epochs={ft_epochs}")
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
                    model_path   = model_path,
                    data_yaml    = job["data_yaml"],
                    output_path  = output_path,
                    sparsity     = sparsity,
                    ft_epochs    = ft_epochs,
                    device       = device,
                    prune_conv   = prune_conv,
                    prune_linear = prune_linear,
                    pruning_mode = pruning_mode,
                )
            else:
                run_torchvision_pipeline(
                    model_type   = job["type"],
                    model_path   = model_path,
                    dataset_dir  = job["dataset_dir"],
                    output_path  = output_path,
                    sparsity     = sparsity,
                    ft_epochs    = ft_epochs,
                    device       = device,
                    prune_conv   = prune_conv,
                    prune_linear = prune_linear,
                    pruning_mode = pruning_mode,
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
        description="Pruning (nao-estrutural L1 ou estrutural) + Fine-tuning "
                     "(opcionalmente com KD) de modelos de deteccao",
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
                    help="Percentual de pesos zerados/canais removidos, ex: 50 para 50%%  (1-99)")
    pr.add_argument("--pruning_mode", choices=["unstructured", "structured"], default="unstructured",
                    help="unstructured = L1 global (mascaras, Params/FLOPS quase nao mudam); "
                         "structured = poda de canais/filtros via torch-pruning (Params/FLOPS "
                         "reduzem de verdade). Saida de 'structured' vai para um caminho paralelo "
                         "(pruned_structured/), nunca sobrescreve os pesos unstructured existentes.")
    pr.add_argument("--no_prune_conv",   action="store_true", help="Nao podar camadas Conv2d")
    pr.add_argument("--no_prune_linear", action="store_true", help="Nao podar camadas Linear")
    pr.add_argument("--skip_head", action="store_true",
                    help="[yolo, unstructured] Exclui a cabeca de deteccao (v10Detect) do pruning global")
    pr.add_argument("--yolo_optimizer", default="SGD",
                    help="[yolo] Otimizador do fine-tuning ('auto' ignora --lr e escolhe o proprio LR)")
    pr.add_argument("--yolo_warmup_epochs", type=float, default=0,
                    help="[yolo] Epocas de warmup do fine-tuning")

    # Fine-tuning
    ft = p.add_argument_group("Fine-tuning")
    ft.add_argument("--ft_epochs",   type=int,   default=10,  help="Epocas de fine-tuning pos-pruning")
    ft.add_argument("--lr",          type=float, default=0.0, help="Learning rate (0 = padrao do modelo)")
    ft.add_argument("--batch_size",  type=int,   default=0,   help="Batch size (0 = padrao do modelo)")

    # Knowledge distillation (modo unico apenas — --all_models nao cobre KD,
    # ver run_compression_pipeline.py)
    kd = p.add_argument_group("Knowledge distillation (modo unico)")
    kd.add_argument("--use_kd", action="store_true",
                    help="Ativa KD durante o fine-tuning pos-poda (professor = --teacher_path)")
    kd.add_argument("--teacher_path",
                    help="[--use_kd] Caminho do modelo professor (mesma arquitetura, sem poda)")
    kd.add_argument("--kd_weight", type=float, default=None,
                    help="[--use_kd] Peso da loss de KD (default: 0.1 para YOLO, 1.0 para torchvision)")

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
        if args.use_kd and not args.teacher_path:
            p.error("--use_kd exige --teacher_path")
    elif args.use_kd:
        p.error("--use_kd nao e suportado em --all_models — use run_compression_pipeline.py")

    args.prune_conv   = not args.no_prune_conv
    args.prune_linear = not args.no_prune_linear

    return args


def main():
    args    = parse_args()
    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sparsity = args.pruning_percentage / 100.0

    logger.info("=" * 64)
    logger.info(f"  PRUNING {args.pruning_mode.upper()} + FINE-TUNING")
    logger.info("=" * 64)
    logger.info(f"  Percentual de pruning : {args.pruning_percentage}%")
    logger.info(f"  Modo                  : {args.pruning_mode}")
    logger.info(f"  Epocas fine-tuning    : {args.ft_epochs}")
    logger.info(f"  Prune Conv2d          : {args.prune_conv}")
    logger.info(f"  Prune Linear          : {args.prune_linear}")
    logger.info(f"  Knowledge distillation: {args.use_kd}")
    logger.info(f"  Dispositivo           : {device}")

    if args.all_models:
        log_path = BASE_DIR / "runs" / "pruned_logs" / f"prune_{args.pruning_mode}_{args.pruning_percentage}pct.log"
        run_all_models(
            pct          = args.pruning_percentage,
            ft_epochs    = args.ft_epochs,
            device       = device,
            prune_conv   = args.prune_conv,
            prune_linear = args.prune_linear,
            skip_existing= not args.overwrite,
            log_path     = log_path,
            pruning_mode = args.pruning_mode,
        )
    elif args.model_type == "yolo":
        if not args.data_yaml:
            raise SystemExit("--data_yaml e obrigatorio para modelo YOLO.")
        run_yolo_pipeline(
            model_path   = Path(args.model_path),
            data_yaml    = Path(args.data_yaml),
            output_path  = Path(args.output_path),
            sparsity     = sparsity,
            ft_epochs    = args.ft_epochs,
            device       = device,
            prune_conv   = args.prune_conv,
            prune_linear = args.prune_linear,
            lr           = args.lr   or None,
            batch_size   = args.batch_size or None,
            pruning_mode = args.pruning_mode,
            use_kd       = args.use_kd,
            teacher_path = Path(args.teacher_path) if args.teacher_path else None,
            kd_weight    = args.kd_weight if args.kd_weight is not None else 0.1,
            skip_head    = args.skip_head,
            optimizer    = args.yolo_optimizer,
            warmup_epochs= args.yolo_warmup_epochs,
        )
    else:
        if not args.dataset_dir:
            raise SystemExit("--dataset_dir e obrigatorio para modelos torchvision.")
        run_torchvision_pipeline(
            model_type   = args.model_type,
            model_path   = Path(args.model_path),
            dataset_dir  = Path(args.dataset_dir),
            output_path  = Path(args.output_path),
            sparsity     = sparsity,
            ft_epochs    = args.ft_epochs,
            device       = device,
            prune_conv   = args.prune_conv,
            prune_linear = args.prune_linear,
            lr           = args.lr   or None,
            batch_size   = args.batch_size or None,
            pruning_mode = args.pruning_mode,
            use_kd       = args.use_kd,
            teacher_path = Path(args.teacher_path) if args.teacher_path else None,
            kd_weight    = args.kd_weight if args.kd_weight is not None else 1.0,
        )


if __name__ == "__main__":
    main()
