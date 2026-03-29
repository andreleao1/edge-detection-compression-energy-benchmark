"""
Treinamento de SSDLite320 com backbone MobileNetV3-Large (Torchvision)
- 100 epocas em oxford e caviar
- Anotacoes no formato YOLO convertidas automaticamente
- Modelos salvos em:
    C:/workspace/mestrado/novo_teste/models/epochs_100/oxford/mobilenet_finetuned.pth
    C:/workspace/mestrado/novo_teste/models/epochs_100/caviar/mobilenet_finetuned.pth

Dependencias (ja instaladas com PyTorch):
    pip install torchvision
"""

import shutil
import logging
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection._utils import retrieve_out_channels

# ---------------------------------------------------------------------------
# Configuracoes globais
# ---------------------------------------------------------------------------

BASE_DIR      = Path("C:/workspace/mestrado/novo_teste")
NUM_EPOCHS    = 100
MODELS_OUT    = BASE_DIR / "models" / f"epochs_{NUM_EPOCHS}"
BATCH_SIZE    = 4       # SSDLite é leve, suporta batch maior
NUM_WORKERS   = 4
IMGSZ         = 320     # tamanho fixo do SSDLite320
LR            = 0.005
MOMENTUM      = 0.9
WEIGHT_DECAY  = 0.0005
LR_MILESTONES = [65, 85]
LR_GAMMA      = 0.1
GRAD_CLIP     = 1.0
VAL_INTERVAL  = 10

DATASETS = {
    "oxford": BASE_DIR / "oxford",
    "caviar": BASE_DIR / "caviar",
}

CLASS_NAMES = ["pessoa"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class YOLODetectionDataset(Dataset):
    """Carrega imagens e labels no formato YOLO para uso com SSDLite."""

    def __init__(self, images_dir: Path, labels_dir: Path, imgsz: int = 320):
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

        # SSDLite320 exige entrada quadrada 320x320
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        scale_x = self.imgsz / w
        scale_y = self.imgsz / h

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        boxes, labels = [], []
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            x1 = max(0.0, (xc - bw / 2) * w * scale_x)
            y1 = max(0.0, (yc - bh / 2) * h * scale_y)
            x2 = min(float(self.imgsz), (xc + bw / 2) * w * scale_x)
            y2 = min(float(self.imgsz), (yc + bh / 2) * h * scale_y)

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(cls + 1)   # 0 = background

        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)

        return img_tensor, {"boxes": boxes_t, "labels": labels_t}


def collate_fn(batch):
    return tuple(zip(*batch))


def make_dataloader(dataset_dir: Path, split: str, shuffle: bool) -> DataLoader:
    images_dir = dataset_dir / "datasets" / "images" / split
    labels_dir = dataset_dir / "datasets" / "labels" / split
    dataset    = YOLODetectionDataset(images_dir, labels_dir, imgsz=IMGSZ)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def build_model(num_classes: int) -> nn.Module:
    """
    SSDLite320 com backbone MobileNetV3-Large pre-treinado no COCO.
    Substitui apenas o cabecalho de classificacao para o numero de classes do dataset.
    """
    model = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    )

    in_channels  = retrieve_out_channels(model.backbone, (IMGSZ, IMGSZ))
    num_anchors  = model.anchor_generator.num_anchors_per_location()

    model.head.classification_head = SSDLiteClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=nn.BatchNorm2d,
    )
    return model


# ---------------------------------------------------------------------------
# Treinamento / Validacao
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, dataloader, device, scaler, epoch):
    model.train()
    total_loss = 0.0

    for images, targets in dataloader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda"):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        if not torch.isfinite(loss):
            logger.warning(f"  Loss NaN/Inf no epoch {epoch}, batch ignorado")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    logger.info(f"  Epoch {epoch:03d} | train_loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, device, epoch):
    model.train()   # modo train para calcular losses na validacao
    total_loss = 0.0

    for images, targets in dataloader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast(device_type="cuda"):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    logger.info(f"  Epoch {epoch:03d} | val_loss:   {avg_loss:.4f}")
    return avg_loss


# ---------------------------------------------------------------------------
# Loop principal por dataset
# ---------------------------------------------------------------------------

def train_on_dataset(dataset_name: str, dataset_dir: Path, device: torch.device):
    dst_dir = MODELS_OUT / dataset_name
    dst     = dst_dir / "mobilenet_finetuned.pth"
    if dst.exists():
        logger.info(f"Ja treinado, pulando: {dataset_name}")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"Treinando SSDLite320-MobileNetV3 — dataset: {dataset_name}")
    logger.info(f"{'='*60}")

    train_loader = make_dataloader(dataset_dir, "train", shuffle=True)
    val_loader   = make_dataloader(dataset_dir, "val",   shuffle=False)
    logger.info(f"Train: {len(train_loader.dataset)} imagens | Val: {len(val_loader.dataset)} imagens")

    num_classes = len(CLASS_NAMES) + 1   # +1 para background
    model = build_model(num_classes).to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=LR_MILESTONES, gamma=LR_GAMMA
    )
    scaler = torch.amp.GradScaler()

    output_dir = BASE_DIR / "runs" / "mobilenet" / f"epochs_{NUM_EPOCHS}" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    dst_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_ckpt     = output_dir / "best.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        train_one_epoch(model, optimizer, train_loader, device, scaler, epoch)
        scheduler.step()

        if epoch % VAL_INTERVAL == 0 or epoch == NUM_EPOCHS:
            val_loss = validate(model, val_loader, device, epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_ckpt)
                logger.info(f"  Melhor modelo salvo (val_loss={best_val_loss:.4f})")

        if epoch % 50 == 0:
            torch.save(model.state_dict(), output_dir / f"epoch_{epoch:03d}.pth")

    if not best_ckpt.exists():
        logger.warning("best.pth nao encontrado; salvando ultimo estado do modelo.")
        torch.save(model.state_dict(), best_ckpt)

    shutil.copy2(best_ckpt, dst)
    logger.info(f"Modelo final salvo em: {dst}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("GPU nao encontrada. Instale o PyTorch com suporte a CUDA.")

    device = torch.device("cuda:0")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    for dataset_name, dataset_dir in DATASETS.items():
        train_on_dataset(dataset_name, dataset_dir, device)

    logger.info("\nTreinamento SSDLite320-MobileNetV3 finalizado para todos os datasets.")
