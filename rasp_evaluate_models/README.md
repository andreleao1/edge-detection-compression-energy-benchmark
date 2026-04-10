# rasp_evaluate_models

Automated orchestration script that benchmarks computer-vision models on a Raspberry Pi, collecting energy consumption and system resource metrics from a Prometheus stack, and persisting consolidated results in PostgreSQL.

## Architecture

```
Raspberry Pi (inference)
│
├── main.py  ──────────────────────── orchestrator
│   ├── loads settings.yaml
│   ├── iterates models × datasets
│   ├── runs YOLO inference
│   ├── waits for Prometheus scrape window
│   ├── queries Prometheus HTTP API
│   └── persists results to PostgreSQL
│
├── database.py  ─────────────────── SQLAlchemy + yoyo migrations
├── prometheus_client.py  ────────── Prometheus HTTP API wrapper
└── migrations/  ─────────────────── SQL migration scripts (yoyo)

Monitoring stack (docker-compose in arduino/)
├── Prometheus :9090
├── Grafana    :3000
└── PostgreSQL :5432
```

### Prometheus metrics consumed

| Metric | Source | Description |
|---|---|---|
| `energia_potencia` | `arduino/main.py` | Power draw in watts (INA219 sensor) |
| `rasp_cpu_usage_percent` | `rasp_monitor/main.go` | CPU usage per core + aggregate |
| `rasp_memory_used_percent` | `rasp_monitor/main.go` | RAM usage percentage |
| `rasp_cpu_temperature_celsius` | `rasp_monitor/main.go` | CPU temperature |

### Database schema

Table `experiment_results`:

| Column | Type | Description |
|---|---|---|
| `id` | serial | Primary key |
| `nome_modelo` | varchar | Model name from settings |
| `nome_dataset` | varchar | Dataset name from settings |
| `avg_watt` | float | Average power (W) during the run |
| `max_watt` | float | Peak power (W) during the run |
| `avg_cpu` | float | Average CPU usage (%) |
| `avg_mem` | float | Average RAM usage (%) |
| `avg_temp` | float | Average CPU temperature (°C) |
| `data_execucao` | timestamptz | Run start time (UTC) |
| `duracao_total` | float | Total inference duration (s) |
| `erro` | text | Error message, `NULL` on success |

## Requirements

- Python 3.10+
- Raspberry Pi with internet access (to reach Prometheus and PostgreSQL)
- Monitoring stack running (`arduino/docker-compose.yaml`)
- `rasp_monitor` Go exporter running on the Raspberry Pi

## Setup

### 1. Start the monitoring stack

```bash
cd arduino/
docker compose up -d
```

This starts Prometheus (`:9090`), Grafana (`:3000`), and PostgreSQL (`:5432`).

### 2. Start the system metrics exporter on the Raspberry Pi

```bash
cd rasp_monitor/
go run main.go          # default port 9100
```

### 3. Start the energy bridge (Arduino + INA219)

```bash
cd arduino/
python main.py --port /dev/ttyUSB0 --baud 9600
```

### 4. Install Python dependencies

```bash
cd rasp_evaluate_models/
pip install -r requirements.txt
```

> **PyTorch** must be installed separately. On a Raspberry Pi (ARM, CPU-only):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

### 5. Configure settings

```bash
cp settings.example.yaml settings.yaml
```

Edit `settings.yaml` with your environment's values:

- `postgres` — host/port/credentials for the PostgreSQL container.
- `prometheus` — host/port for the Prometheus container.
- `models` — list of `{name, path}` entries pointing to the fine-tuned model files (`.pt` / `.pth`).
- `datasets` — list of `{name, path}` entries pointing to image folders.
- `experiment` — timing and inference parameters.

> `settings.yaml` is git-ignored. **Never commit it** — it may contain credentials.  
> Commit `settings.example.yaml` as the shared template.

## Running

```bash
python main.py
# or explicitly:
python main.py --settings settings.yaml
```

The script logs to both stdout and `evaluation.log`.

### Execution flow per test

```
1. Load model  →  run YOLO.predict()
2. Record start_time / end_time (Unix timestamp)
3. Sleep wait_after_run seconds  (Prometheus scrape window)
4. Query Prometheus:
     avg_over_time(metric[Xs]) @ end_time
     max_over_time(metric[Xs]) @ end_time
5. INSERT into experiment_results (or INSERT error row)
6. Sleep cooldown_between_tests seconds  (thermal + resource cooldown)
```

### PromQL strategy

Each metric is queried as an instant query evaluated at `end_time` using a
lookback window equal to `(end_time − start_time)` seconds. This scopes the
aggregation exactly to the inference window without manual range-query math.

## Configuration reference (`settings.yaml`)

```yaml
postgres:
  host: localhost       # PostgreSQL host
  port: 5432
  database: benchmark_results
  user: benchmark
  password: benchmark

prometheus:
  host: localhost       # Prometheus host
  port: 9090

models:
  - name: yolov10n                        # display name saved to DB
    path: /abs/path/to/yolov10n.pt        # must be absolute

datasets:
  - name: CAVIAR
    path: /abs/path/to/caviar/images      # folder of images

experiment:
  wait_after_run: 30          # seconds to wait for Prometheus to scrape after inference
  cooldown_between_tests: 15  # seconds between consecutive tests
  imgsz: 640                  # inference image size
  conf: 0.3                   # confidence threshold
  device: cpu                 # 'cpu' for Raspberry Pi
```

## Project structure

```
rasp_evaluate_models/
├── main.py                  # Orchestrator entry point
├── database.py              # SQLAlchemy ORM + yoyo migration runner
├── prometheus_client.py     # Prometheus HTTP API wrapper
├── settings.yaml            # Local config — git-ignored
├── settings.example.yaml    # Template — committed to version control
├── requirements.txt         # Python dependencies
└── migrations/
    ├── 0001_create_experiment_results.sql          # DDL apply
    └── 0001_create_experiment_results.rollback.sql # DDL rollback
```
