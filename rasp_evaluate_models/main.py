"""
Orchestration script for the edge-model energy benchmark.

Flow for each (model × dataset) pair:
  1. Load model and run inference.
  2. Record start/end Unix timestamps.
  3. Wait for Prometheus to scrape the metrics (configurable cooldown).
  4. Query Prometheus avg_over_time / max_over_time for the execution window.
  5. Persist results (or error) to PostgreSQL.
  6. Wait the cooldown period before the next test.

Usage:
    python main.py [--settings settings.yaml]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

from database import init_db, save_error, save_result
from prometheus_client import PrometheusClient

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _setup_logging(log_file: str = "evaluation.log") -> None:
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def load_settings(path: str) -> dict:
    settings_path = Path(path)
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path.resolve()}")

    with settings_path.open("r") as fh:
        cfg = yaml.safe_load(fh)

    _validate_settings(cfg)
    return cfg


def _validate_settings(cfg: dict) -> None:
    required_top = {"postgres", "prometheus", "models", "datasets", "experiment"}
    missing = required_top - cfg.keys()
    if missing:
        raise ValueError(f"settings.yaml is missing required keys: {missing}")

    if not cfg["models"]:
        raise ValueError("'models' list is empty in settings.yaml.")
    if not cfg["datasets"]:
        raise ValueError("'datasets' list is empty in settings.yaml.")

    for m in cfg["models"]:
        if "name" not in m or "path" not in m:
            raise ValueError(f"Each model entry must have 'name' and 'path'. Got: {m}")
    for d in cfg["datasets"]:
        if "name" not in d or "path" not in d:
            raise ValueError(f"Each dataset entry must have 'name' and 'path'. Got: {d}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model_cfg: dict,
    dataset_cfg: dict,
    experiment_cfg: dict,
) -> tuple[float, float]:
    """
    Load the model, run inference over the dataset, and return
    (start_time, end_time) as Unix timestamps.

    Raises any exception produced by the model so the orchestrator can
    catch it and record the failure.
    """
    model_path = model_cfg["path"]
    dataset_path = dataset_cfg["path"]

    logger.info("Loading model '%s' from: %s", model_cfg["name"], model_path)
    model = YOLO(model_path)

    logger.info(
        "Starting inference — dataset='%s' path=%s",
        dataset_cfg["name"],
        dataset_path,
    )

    start_time = time.time()
    model.predict(
        source=dataset_path,
        imgsz=experiment_cfg.get("imgsz", 640),
        conf=experiment_cfg.get("conf", 0.3),
        device=experiment_cfg.get("device", "cpu"),
        save=False,
        verbose=False,
    )
    end_time = time.time()

    logger.info(
        "Inference complete — duration=%.2fs",
        end_time - start_time,
    )
    return start_time, end_time


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(settings: dict) -> None:
    models: list[dict] = settings["models"]
    datasets: list[dict] = settings["datasets"]
    experiment_cfg: dict = settings.get("experiment", {})

    wait_after_run: int = experiment_cfg.get("wait_after_run", 30)
    cooldown: int = experiment_cfg.get("cooldown_between_tests", 15)

    prometheus = PrometheusClient(
        host=settings["prometheus"]["host"],
        port=settings["prometheus"]["port"],
    )

    init_db(settings["postgres"])

    total = len(models) * len(datasets)
    completed = 0
    errors = 0

    logger.info(
        "Pipeline started — %d model(s) × %d dataset(s) = %d test(s).",
        len(models),
        len(datasets),
        total,
    )

    for model_cfg in models:
        for dataset_cfg in datasets:
            completed += 1
            test_label = f"{model_cfg['name']} × {dataset_cfg['name']}"
            logger.info("─" * 60)
            logger.info("[%d/%d] %s", completed, total, test_label)

            try:
                start_time, end_time = run_inference(
                    model_cfg, dataset_cfg, experiment_cfg
                )

                logger.info(
                    "Waiting %ds for Prometheus scrape window...", wait_after_run
                )
                time.sleep(wait_after_run)

                logger.info("Querying Prometheus metrics...")
                metrics = prometheus.get_metrics(start_time, end_time)

                save_result(
                    model_name=model_cfg["name"],
                    dataset_name=dataset_cfg["name"],
                    metrics=metrics,
                    start_time=start_time,
                    end_time=end_time,
                )

                logger.info(
                    "  avg_watt=%.4f  max_watt=%.4f  avg_cpu=%.2f%%"
                    "  avg_mem=%.2f%%  avg_temp=%.2f°C",
                    metrics.get("avg_watt") or 0.0,
                    metrics.get("max_watt") or 0.0,
                    metrics.get("avg_cpu") or 0.0,
                    metrics.get("avg_mem") or 0.0,
                    metrics.get("avg_temp") or 0.0,
                )

            except Exception as exc:  # noqa: BLE001 — intentional broad catch
                errors += 1
                logger.error(
                    "FAILED: %s — %s", test_label, exc, exc_info=True
                )
                save_error(
                    model_name=model_cfg["name"],
                    dataset_name=dataset_cfg["name"],
                    error_message=str(exc),
                )
                logger.info("Error recorded. Continuing pipeline...")

            # Cooldown between consecutive tests (skip after the last one).
            if completed < total:
                logger.info("Cooling down for %ds before next test...", cooldown)
                time.sleep(cooldown)

    logger.info("─" * 60)
    logger.info(
        "Pipeline finished — %d/%d tests completed, %d error(s).",
        total - errors,
        total,
        errors,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edge-model energy benchmark orchestrator."
    )
    parser.add_argument(
        "--settings",
        default="settings.yaml",
        metavar="FILE",
        help="Path to the YAML settings file (default: settings.yaml).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _setup_logging()
    args = _parse_args()

    try:
        settings = load_settings(args.settings)
    except (FileNotFoundError, ValueError) as exc:
        logger.critical("Configuration error: %s", exc)
        sys.exit(1)

    run_pipeline(settings)
