# Person Detection — Object Detection Benchmark

Benchmark comparing multiple object detection architectures fine-tuned for **person detection** on two surveillance datasets: **Oxford Town Centre** and **CAVIAR**.

The project covers the full research pipeline: model training on a GPU workstation, energy-aware inference evaluation on a **Raspberry Pi**, and metric collection via a **Prometheus + PostgreSQL** observability stack.

## Models

| Model | Framework |
|---|---|
| YOLOv10n / s / m / b / l / x | Ultralytics |
| Faster R-CNN (ResNet-50 FPN) | Torchvision |
| RetinaNet (ResNet-50 FPN) | Torchvision |
| SSDLite320 (MobileNetV3-Large) | Torchvision |

## Datasets

| Dataset | Description |
|---|---|
| Oxford Town Centre | Outdoor pedestrian surveillance footage |
| CAVIAR | Indoor surveillance scenarios |

## Project Structure

```
edge-detection-compression-energy-benchmark/
│
├── base_model_fine_tunning.py    # YOLOv10 fine-tuning script (all variants)
├── faster_rcnn_train.py          # Faster R-CNN fine-tuning script
├── retinanet_train.py            # RetinaNet fine-tuning script
├── mobilenet_train.py            # SSDLite320-MobileNetV3 fine-tuning script
├── evaluate_models.py            # Offline evaluation → resultados_avaliacao.xlsx
├── requirements.txt              # Training dependencies
│
├── oxford/                       # Oxford Town Centre dataset config
│   └── data.yaml
│
├── caviar/                       # CAVIAR dataset config
│   └── data.yaml
│
├── rasp_monitor/                 # Go exporter — Raspberry Pi system metrics
│   ├── main.go                   # Prometheus exporter (CPU, RAM, temperature)
│   ├── go.mod
│   └── go.sum
│
├── arduino/                      # Energy monitoring subsystem
│   ├── main.py                   # Serial → Prometheus /metrics bridge (INA219)
│   ├── docker-compose.yaml       # Prometheus + Grafana + PostgreSQL stack
│   ├── prometheus/               # Prometheus config and alert rules
│   ├── grafana/                  # Grafana provisioning (dashboards + datasources)
│   ├── ina219_without_ethernet/
│   │   └── ina219_without_ethernet.ino  # Arduino sketch
│   └── README.md
│
└── rasp_evaluate_models/         # Automated evaluation orchestrator (Raspberry Pi)
    ├── main.py                   # Orchestrator entry point
    ├── database.py               # SQLAlchemy ORM + migration runner
    ├── prometheus_client.py      # Prometheus HTTP API wrapper
    ├── settings.example.yaml     # Config template (commit this)
    ├── settings.yaml             # Local config — git-ignored
    ├── requirements.txt          # Evaluation dependencies
    ├── migrations/               # yoyo SQL migration scripts
    └── README.md
```

## Research Pipeline

```
[GPU workstation]                        [Raspberry Pi]
  Training scripts      →  .pt/.pth  →    rasp_evaluate_models/
  (fine-tune models)                       │
                                           ├── runs YOLO inference
                                           ├── records timestamps
                                           └── queries Prometheus

[Monitoring stack — docker-compose]
  Arduino + INA219  →  energia_potencia
  rasp_monitor Go   →  cpu / ram / temp
       ↓
  Prometheus :9090  →  rasp_evaluate_models/prometheus_client.py
  PostgreSQL :5432  →  experiment_results table
  Grafana    :3000  →  dashboards
```

## Quick Start

### 1. Training (GPU workstation)

```bash
# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Fine-tune all models on both datasets
python base_model_fine_tunning.py
python faster_rcnn_train.py
python retinanet_train.py
python mobilenet_train.py
```

Fine-tuned weights are saved to `models/epochs_100/<dataset>/`.

### 2. Offline evaluation (GPU workstation)

```bash
python evaluate_models.py
```

Generates `resultados_avaliacao.xlsx` with: mAP50, mAP50-95, Precision, Recall, F1, Inference (ms), FPS, FLOPs (G).

### 3. On-device energy benchmark (Raspberry Pi)

Start the monitoring stack and run the orchestrator — see [rasp_evaluate_models/README.md](rasp_evaluate_models/README.md) for the full setup guide.

```bash
# Start Prometheus + Grafana + PostgreSQL
cd arduino/ && docker compose up -d

# Run the benchmark pipeline
cd rasp_evaluate_models/
cp settings.example.yaml settings.yaml   # fill in your paths and credentials
pip install -r requirements.txt
python main.py
```

Results are persisted to the `experiment_results` table in PostgreSQL.

## Energy Monitoring (Arduino)

The `arduino/` subsystem measures the **power draw of the Raspberry Pi** using an **Arduino UNO** and an **INA219** current/power sensor. The Arduino bridge script streams wattage readings to Prometheus, which are then correlated with inference windows by the orchestrator.

See [arduino/README.md](arduino/README.md) for wiring instructions and setup steps.

## System Metrics (Go exporter)

The `rasp_monitor/` Go service exposes Raspberry Pi system metrics to Prometheus:

| Metric | Description |
|---|---|
| `rasp_cpu_usage_percent` | CPU usage per core + aggregate |
| `rasp_memory_used_percent` | RAM usage percentage |
| `rasp_cpu_temperature_celsius` | CPU temperature from thermal zone |

```bash
cd rasp_monitor/
go run main.go   # exposes :9100/metrics
```

## Training Hyperparameters

| Parameter | YOLOv10 | Faster R-CNN / RetinaNet / SSDLite |
|---|---|---|
| Epochs | 100 | 100 |
| Image size | 640 | 800 (320 for SSDLite) |
| Optimizer | SGD | SGD |
| Initial LR | 1e-3 | 5e-3 (1e-3 for RetinaNet) |
| Mixed precision | Yes (AMP) | Yes (AMP) |
| Early stopping patience | 30 | — |

## Requirements

| Environment | Requirement |
|---|---|
| Training | Python 3.10+, NVIDIA GPU with CUDA |
| On-device inference | Python 3.10+, Raspberry Pi, Docker |
| Go exporter | Go 1.21+ |
| Arduino bridge | Arduino UNO, INA219 sensor |
