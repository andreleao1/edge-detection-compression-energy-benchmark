"""
Treinamento de Faster R-CNN (ResNet-50 FPN) com Torchvision
- 100 epocas em oxford e caviar
- Anotacoes no formato YOLO convertidas automaticamente
- Modelos salvos em:
    C:/workspace/mestrado/novo_teste/models/epochs_100/oxford/faster_rcnn_finetuned.pth
    C:/workspace/mestrado/novo_teste/models/epochs_100/caviar/faster_rcnn_finetuned.pth

Dependencias (ja instaladas com PyTorch):
    pip install torchvision
"""

import shutil
import logging
from pathlib import Path

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ---------------------------------------------------------------------------
# Configuracoes globais
# ---------------------------------------------------------------------------

BASE_DIR      = Path("C:/workspace/mestrado/novo_teste")
NUM_EPOCHS    = 100
MODELS_OUT    = BASE_DIR / "models" / f"epochs_{NUM_EPOCHS}"
BATCH_SIZE    = 2
NUM_WORKERS   = 4
IMGSZ         = 800
LR            = 0.005
MOMENTUM      = 0.9
WEIGHT_DECAY  = 0.0005
# Decai o LR nos epochs 65 e 85 (proporcional a 100 epocas)
LR_MILESTONES = [65, 85]
LR_GAMMA      = 0.1
# Avalia no conjunto de validacao a cada N epocas
VAL_INTERVAL  = 10

DATASETS = {
    "oxford": BASE_DIR / "oxford",
    "caviar": BASE_DIR / "caviar",
}

CLASS_NAMES = ["pessoa"]   # classe 0 = pessoa (background e tratado internamente pelo modelo)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class YOLODetectionDataset(Dataset):
    """Carrega imagens e labels no formato YOLO para uso com Faster R-CNN."""

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

        # Redimensiona mantendo proporcao
        scale = self.imgsz / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))

        # Normaliza e converte para tensor [C, H, W] float32 em [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        boxes, labels = [], []
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # YOLO (normalizado) -> pixels na imagem redimensionada
            x1 = (xc - bw / 2) * new_w
            y1 = (yc - bh / 2) * new_h
            x2 = (xc + bw / 2) * new_w
            y2 = (yc + bh / 2) * new_h

            # Garante que a caixa e valida
            x1, x2 = max(0, x1), min(new_w, x2)
            y1, y2 = max(0, y1), min(new_h, y2)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(cls + 1)   # +1 porque 0 e reservado para background

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

def build_model(num_classes: int) -> torch.nn.Module:
    """Faster R-CNN ResNet-50 FPN pre-treinado no COCO com cabeca substituida."""
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # num_classes inclui background: 1 (pessoa) + 1 (background) = 2
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Treinamento / Validacao
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, dataloader, device, scaler, epoch):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda"):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    logger.info(f"  Epoch {epoch:03d} | train_loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, device, epoch):
    model.train()   # modo train para obter losses na validacao
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
    dst     = dst_dir / "faster_rcnn_finetuned.pth"
    if dst.exists():
        logger.info(f"Ja treinado, pulando: {dataset_name}")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"Treinando Faster R-CNN — dataset: {dataset_name}")
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

    output_dir = BASE_DIR / "runs" / "faster_rcnn" / f"epochs_{NUM_EPOCHS}" / dataset_name
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

        # Checkpoint a cada 50 epocas
        if epoch % 50 == 0:
            torch.save(model.state_dict(), output_dir / f"epoch_{epoch:03d}.pth")

    # Copia melhor checkpoint para o diretorio do dataset
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

    logger.info("\nTreinamento Faster R-CNN finalizado para todos os datasets.")
