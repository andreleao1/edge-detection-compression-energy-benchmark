"""
Treinamento de RetinaNet (ResNet-50 FPN) com Torchvision
- 200 epocas em oxford e caviar
- Anotacoes no formato YOLO convertidas automaticamente
- Modelos salvos em:
    C:/workspace/mestrado/novo_teste/oxford/retinanet_finetuned.pth
    C:/workspace/mestrado/novo_teste/caviar/retinanet_finetuned.pth

Dependencias (ja instaladas com PyTorch):
    pip install torchvision
"""

import shutil
import logging
from functools import partial
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

# ---------------------------------------------------------------------------
# Configuracoes globais
# ---------------------------------------------------------------------------

BASE_DIR      = Path("C:/workspace/mestrado/novo_teste")
NUM_EPOCHS    = 100
MODELS_OUT    = BASE_DIR / "models" / f"epochs_{NUM_EPOCHS}"
BATCH_SIZE    = 2
NUM_WORKERS   = 4
IMGSZ         = 800
LR            = 0.001
MOMENTUM      = 0.9
WEIGHT_DECAY  = 0.0001
GRAD_CLIP     = 1.0   # max norm para gradient clipping
LR_MILESTONES = [65, 85]     # epocas onde o LR decai (proporcional a 100 epocas)
LR_GAMMA      = 0.1
VAL_INTERVAL  = 10           # avalia a cada N epocas

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
    """Carrega imagens e labels no formato YOLO para uso com RetinaNet."""

    def __init__(self, images_dir: Path, labels_dir: Path, imgsz: int = 800):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.imgsz      = imgsz
        self.samples    = self._load_samples()

    def _load_samples(self):
        samples = []
        for img_path in sorted(self.images_dir.glob("*.jpg")) + sorted(self.images_dir.glob("*.png")):
            label_path = self.labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                samples.append((img_path, label_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        scale  = self.imgsz / max(h, w)
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        img    = cv2.resize(img, (new_w, new_h))

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        boxes, labels = [], []
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            x1 = (xc - bw / 2) * new_w
            y1 = (yc - bh / 2) * new_h
            x2 = (xc + bw / 2) * new_w
            y2 = (yc + bh / 2) * new_h

            x1, x2 = max(0.0, x1), min(float(new_w), x2)
            y1, y2 = max(0.0, y1), min(float(new_h), y2)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(cls + 1)   # 0 reservado para background

        if boxes:
            boxes_tensor  = torch.tensor(boxes,  dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor  = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,),   dtype=torch.int64)

        target = {"boxes": boxes_tensor, "labels": labels_tensor}
        return img_tensor, target


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
    RetinaNet ResNet-50 FPN pre-treinado no COCO.
    Substitui a cabeca de classificacao para o numero de classes do dataset.
    num_classes deve incluir o background: len(CLASS_NAMES) + 1
    """
    model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)

    num_anchors = model.head.classification_head.num_anchors
    in_channels = 256   # saida padrao do FPN

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(nn.GroupNorm, 32),
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
    dst     = dst_dir / "retinanet_finetuned.pth"
    if dst.exists():
        logger.info(f"Ja treinado, pulando: {dataset_name}")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"Treinando RetinaNet — dataset: {dataset_name}")
    logger.info(f"{'='*60}")

    train_loader = make_dataloader(dataset_dir, "train", shuffle=True)
    val_loader   = make_dataloader(dataset_dir, "val",   shuffle=False)
    logger.info(f"Train: {len(train_loader.dataset)} imagens | Val: {len(val_loader.dataset)} imagens")

    num_classes = len(CLASS_NAMES) + 1   # +1 para background (igual ao FasterRCNN no torchvision)
    model = build_model(num_classes).to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=LR_MILESTONES, gamma=LR_GAMMA
    )
    scaler = torch.amp.GradScaler()

    output_dir = BASE_DIR / "runs" / "retinanet" / f"epochs_{NUM_EPOCHS}" / dataset_name
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

    logger.info("\nTreinamento RetinaNet finalizado para todos os datasets.")
