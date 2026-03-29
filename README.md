# Person Detection — Object Detection Benchmark

Benchmark comparing multiple object detection architectures fine-tuned for **person detection** on two surveillance datasets: **Oxford Town Centre** and **CAVIAR**.

## Models

| Model | Framework |
|---|---|
| YOLOv10n / s / m / b / l / x | Ultralytics |
| Faster R-CNN (ResNet-50 FPN) | Torchvision |
| RetinaNet (ResNet-50 FPN) | Torchvision |
| SSDLite320 (MobileNetV3-Large) | Torchvision |

## Project Structure

```
novo_teste/
├── base_models/                  # Pre-trained YOLOv10 weights (.pt)
│   ├── yolov10n.pt
│   ├── yolov10s.pt
│   ├── yolov10m.pt
│   ├── yolov10b.pt
│   ├── yolov10l.pt
│   └── yolov10x.pt
│
├── oxford/                       # Oxford Town Centre dataset
│   ├── data.yaml                 # YOLO dataset config
│   └── datasets/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
│
├── caviar/                       # CAVIAR dataset
│   ├── data.yaml
│   └── datasets/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
│
├── models/                       # Fine-tuned model weights (not versioned)
│   └── epochs_100/
│       ├── oxford/
│       └── caviar/
│
├── runs/                         # Training logs and checkpoints (not versioned)
│
├── base_model_fine_tunning.py    # YOLOv10 fine-tuning script
├── faster_rcnn_train.py          # Faster R-CNN fine-tuning script
├── retinanet_train.py            # RetinaNet fine-tuning script
├── mobilenet_train.py            # SSDLite320-MobileNetV3 fine-tuning script
├── evaluate_models.py            # Evaluation script — generates resultados_avaliacao.xlsx
├── requirements.txt
└── .gitignore
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (required — all scripts raise `RuntimeError` without a GPU)

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 2. Install PyTorch with CUDA support (adjust the CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt
```

## Running Locally

### 1. Download base YOLOv10 weights

Place the `.pt` files in the `base_models/` directory. Official weights can be downloaded from the [YOLOv10 repository](https://github.com/THU-MIG/yolov10).

### 2. Prepare datasets

Datasets must follow the YOLO annotation format (`.txt` files with normalized bounding boxes) organized into `train/`, `val/`, and `test/` splits under `oxford/datasets/` and `caviar/datasets/`.

Update the absolute paths in `oxford/data.yaml` and `caviar/data.yaml` to match your local environment.

### 3. Fine-tune the models

Run each training script independently. They skip already-trained models, so they are safe to re-run.

```bash
# YOLOv10 (all variants, both datasets)
python base_model_fine_tunning.py

# Faster R-CNN
python faster_rcnn_train.py

# RetinaNet
python retinanet_train.py

# SSDLite320-MobileNetV3
python mobilenet_train.py
```

Fine-tuned weights are saved to `models/epochs_100/<dataset>/`.

### 4. Evaluate all models

```bash
python evaluate_models.py
```

Generates `resultados_avaliacao.xlsx` with one sheet per dataset containing: Params (M), mAP50, mAP50-95, Precision, Recall, F1, Inference (ms), FPS, and FLOPs (G).

## Training Hyperparameters

| Parameter | YOLOv10 | Faster R-CNN / RetinaNet / SSDLite |
|---|---|---|
| Epochs | 100 | 100 |
| Image size | 640 | 800 (320 for SSDLite) |
| Optimizer | SGD | SGD |
| Initial LR | 1e-3 | 5e-3 (1e-3 for RetinaNet) |
| Mixed precision | Yes (AMP) | Yes (AMP) |
| Early stopping patience | 30 | — |
