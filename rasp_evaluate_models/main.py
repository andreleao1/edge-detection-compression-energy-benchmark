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
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from database import (
    get_stuck_energy_results,
    has_successful_result,
    init_db,
    save_error,
    save_result,
    update_energy_metrics,
)
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

    dataset_names = {d.get("name") for d in cfg["datasets"]}

    for m in cfg["models"]:
        if "name" not in m or "path" not in m:
            raise ValueError(f"Each model entry must have 'name' and 'path'. Got: {m}")
        if m.get("compression_type") == "pruning" and not m.get("template_path"):
            raise ValueError(
                f"Model '{m['name']}' has compression_type=pruning but no 'template_path'. Got: {m}"
            )
        if m.get("dataset") and m["dataset"] not in dataset_names:
            raise ValueError(
                f"Model '{m['name']}' has dataset='{m['dataset']}', but no dataset with that "
                f"name exists in 'datasets'. Available: {sorted(dataset_names)}"
            )
    for d in cfg["datasets"]:
        if "name" not in d or "path" not in d:
            raise ValueError(f"Each dataset entry must have 'name' and 'path'. Got: {d}")


# ---------------------------------------------------------------------------
# Inference — ONNX Runtime (modelos quantizados gerados por quantize_models.py)
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _get_onnx_input_spec(session) -> tuple[str, bool, int, bool]:
    """
    Introspecta o unico input do grafo ONNX para descobrir como pre-processar
    as imagens, sem precisar de configuracao extra por modelo:
      - has_batch: 4 dims (ex. YOLO, [1,3,640,640]) vs 3 dims (torchvision, [3,H,W])
      - imgsz/square: dims H/W fixas (int)  -> resize quadrado para esse tamanho
                      dims H/W dinamicas    -> resize preservando aspecto (lado
                                               maior = imgsz, padrao usado por
                                               Faster R-CNN / RetinaNet)
    """
    inp = session.get_inputs()[0]
    shape = inp.shape
    has_batch = len(shape) == 4
    h_dim, w_dim = shape[-2], shape[-1]

    if isinstance(h_dim, int) and isinstance(w_dim, int):
        imgsz, square = h_dim, True
    else:
        imgsz, square = 800, False

    return inp.name, has_batch, imgsz, square


def _preprocess_image(img_bgr: np.ndarray, imgsz: int, square: bool) -> np.ndarray:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    if square:
        img = cv2.resize(img, (imgsz, imgsz))
    else:
        scale = imgsz / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    arr = img.astype(np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray(arr)


def run_onnx_inference(model_path: str, dataset_path: str) -> tuple[float, float, int]:
    """Roda inferencia de um modelo .onnx (quantizado ou nao) sobre todas as
    imagens de dataset_path, usando so onnxruntime — sem torch/ultralytics."""
    import onnxruntime as ort

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name, has_batch, imgsz, square = _get_onnx_input_spec(session)
    logger.info(
        "  ONNX Runtime — input='%s' imgsz=%d square=%s batched=%s",
        input_name, imgsz, square, has_batch,
    )

    image_paths = [
        p for p in sorted(Path(dataset_path).iterdir())
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_paths:
        raise RuntimeError(f"No images found in dataset path: {dataset_path}")

    start_time = time.time()
    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        arr = _preprocess_image(img_bgr, imgsz, square)
        if has_batch:
            arr = arr[None, ...]
        session.run(None, {input_name: arr})
    end_time = time.time()

    return start_time, end_time, len(image_paths)


# ---------------------------------------------------------------------------
# Inference — Ultralytics (modelos .pt nao quantizados)
# ---------------------------------------------------------------------------

def run_yolo_inference(
    model: "YOLO", dataset_path: str, experiment_cfg: dict
) -> tuple[float, float, int]:
    start_time = time.time()
    results = model.predict(
        source=dataset_path,
        imgsz=experiment_cfg.get("imgsz", 640),
        conf=experiment_cfg.get("conf", 0.3),
        device=experiment_cfg.get("device", "cpu"),
        save=False,
        verbose=False,
    )
    end_time = time.time()
    return start_time, end_time, len(results)


def load_yolo_model(model_cfg: dict) -> "YOLO":
    """
    Carrega um modelo YOLO ".pt". Dois casos:
      - baseline (compression_type != "pruning"): o proprio arquivo ja e um
        checkpoint completo da ultralytics, carrega direto.
      - podado (compression_type == "pruning"): prune_models.py salva so o
        state_dict() bruto do modelo (nao o checkpoint completo que a
        ultralytics espera), entao carregamos a arquitetura a partir do
        'template_path' (um checkpoint fine-tuned valido) e sobrescrevemos
        os pesos com o state_dict podado.
    """
    from ultralytics import YOLO

    if model_cfg.get("compression_type") == "pruning":
        template_path = model_cfg["template_path"]
        logger.info("  Pruned model — loading architecture from template: %s", template_path)
        model = YOLO(template_path)

        import torch
        state_dict = torch.load(model_cfg["path"], map_location="cpu")
        model.model.load_state_dict(state_dict)
        logger.info("  Pruned weights loaded from: %s", model_cfg["path"])
        return model

    return YOLO(model_cfg["path"])


# ---------------------------------------------------------------------------
# Inference — dispatcher
# ---------------------------------------------------------------------------

def run_inference(
    model_cfg: dict,
    dataset_cfg: dict,
    experiment_cfg: dict,
) -> tuple[float, float, int]:
    """
    Load the model, run inference over the dataset, and return
    (start_time, end_time, image_count).

    Modelos ".onnx" (gerados por quantize_models.py) rodam via ONNX Runtime;
    os demais (".pt") rodam via ultralytics.YOLO() — modelos com
    compression_type=pruning carregam a arquitetura de 'template_path' e
    aplicam por cima o state_dict podado (veja load_yolo_model()).

    Raises any exception produced by the model so the orchestrator can
    catch it and record the failure.
    """
    model_path = model_cfg["path"]
    dataset_path = dataset_cfg["path"]

    logger.info("Loading model '%s' from: %s", model_cfg["name"], model_path)
    logger.info(
        "Starting inference — dataset='%s' path=%s",
        dataset_cfg["name"],
        dataset_path,
    )

    if str(model_path).lower().endswith(".onnx"):
        start_time, end_time, image_count = run_onnx_inference(model_path, dataset_path)
    else:
        model = load_yolo_model(model_cfg)
        start_time, end_time, image_count = run_yolo_inference(model, dataset_path, experiment_cfg)

    logger.info(
        "Inference complete — duration=%.2fs  images=%d",
        end_time - start_time,
        image_count,
    )
    return start_time, end_time, image_count


# ---------------------------------------------------------------------------
# Seguranca termica
# ---------------------------------------------------------------------------
# Le a temperatura direto do sensor via vcgencmd (nao depende do Prometheus —
# uma checagem de seguranca nao deveria falhar junto com o que ela protege).
# Chamada antes de cada teste em run_pipeline(): se a CPU estiver quente
# demais, espera esfriar (soft_limit_c); se nao esfriar a tempo ou atingir o
# limite critico (hard_limit_c), aborta o pipeline inteiro.

_TEMP_RE = re.compile(r"temp=([\d.]+)")


def read_cpu_temperature() -> float | None:
    """Le a temperatura da CPU em graus Celsius via `vcgencmd measure_temp`.
    Retorna None se o comando nao existir/falhar (ex: rodando fora de uma
    Raspberry Pi) — nesse caso a checagem termica nao bloqueia o pipeline."""
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True, text=True, timeout=5, check=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        logger.warning("Nao foi possivel ler a temperatura via vcgencmd: %s", exc)
        return None

    match = _TEMP_RE.search(result.stdout)
    if not match:
        logger.warning("Saida inesperada de vcgencmd measure_temp: %r", result.stdout)
        return None
    return float(match.group(1))


def wait_for_safe_temperature(thermal_cfg: dict) -> bool:
    """
    Garante que a temperatura esta em nivel seguro antes de iniciar um teste.

    Retorna True se seguro para prosseguir (temperatura abaixo de
    soft_limit_c, ou sensor indisponivel). Retorna False se o limite critico
    (hard_limit_c) foi atingido, ou se a temperatura nao caiu abaixo de
    soft_limit_c dentro de max_wait_s — sinal para abortar o pipeline
    inteiro (cooling insuficiente para a carga, ex: fan travado).
    """
    soft_limit = thermal_cfg.get("soft_limit_c", 75.0)
    hard_limit = thermal_cfg.get("hard_limit_c", 83.0)
    poll_interval = thermal_cfg.get("poll_interval_s", 15)
    max_wait = thermal_cfg.get("max_wait_s", 300)

    waited = 0
    while True:
        temp = read_cpu_temperature()
        if temp is None:
            return True

        if temp < soft_limit:
            if waited > 0:
                logger.info("  Temperatura OK (%.1f°C) — prosseguindo.", temp)
            return True

        if temp >= hard_limit:
            logger.critical(
                "Temperatura da CPU em %.1f°C >= limite critico (%.1f°C) — "
                "abortando pipeline por seguranca termica.",
                temp, hard_limit,
            )
            return False

        if waited >= max_wait:
            logger.critical(
                "Temperatura nao caiu abaixo de %.1f°C apos %ds de espera "
                "(ainda em %.1f°C) — abortando pipeline por seguranca termica.",
                soft_limit, max_wait, temp,
            )
            return False

        logger.warning(
            "  Temperatura da CPU em %.1f°C (limite=%.1f°C) — aguardando %ds "
            "antes de tentar de novo...",
            temp, soft_limit, poll_interval,
        )
        time.sleep(poll_interval)
        waited += poll_interval


# ---------------------------------------------------------------------------
# Reprocessamento de leituras de energia travadas
# ---------------------------------------------------------------------------
# Uma leitura "travada" (sensor INA219/Arduino preso, mandando sempre o mesmo
# valor) fica marcada no banco como avg_watt == max_watt: a janela inteira do
# Prometheus tem amostras identicas, o que praticamente nao acontece numa
# leitura real de varios segundos. Re-consultar essa mesma janela no
# Prometheus nao ajuda — ela ja ficou gravada no passado com o valor
# travado. A unica forma de conseguir uma leitura valida e rodar a
# inferencia de novo (idealmente com o Arduino ja resetado) e sobrescrever a
# linha existente com a nova janela.

def reprocess_stuck_energy(settings: dict) -> None:
    models: list[dict] = settings["models"]
    datasets: list[dict] = settings["datasets"]
    experiment_cfg: dict = settings.get("experiment", {})

    wait_after_run: int = experiment_cfg.get("wait_after_run", 30)
    cooldown: int = experiment_cfg.get("cooldown_between_tests", 15)
    thermal_cfg: dict = experiment_cfg.get("thermal_safety", {})
    thermal_enabled: bool = thermal_cfg.get("enabled", True)

    prometheus = PrometheusClient(
        host=settings["prometheus"]["host"],
        port=settings["prometheus"]["port"],
    )

    init_db(settings["postgres"])

    stuck_rows = get_stuck_energy_results()
    if not stuck_rows:
        logger.info("Nenhuma leitura de energia travada encontrada.")
        return

    model_by_name = {m["name"]: m for m in models}
    dataset_by_name = {d["name"]: d for d in datasets}

    logger.info(
        "%d resultado(s) com leitura de energia travada — reprocessando...",
        len(stuck_rows),
    )

    for i, row in enumerate(stuck_rows, 1):
        test_label = f"{row['nome_modelo']} × {row['nome_dataset']}"
        logger.info("─" * 60)
        logger.info(
            "[%d/%d] %s (id=%d, valor travado=%.4fW)",
            i, len(stuck_rows), test_label, row["id"], row["avg_watt"],
        )

        model_cfg = model_by_name.get(row["nome_modelo"])
        dataset_cfg = dataset_by_name.get(row["nome_dataset"])
        if model_cfg is None or dataset_cfg is None:
            logger.warning(
                "  Pulando — modelo/dataset nao encontrado no settings.yaml atual "
                "(modelo '%s' presente=%s, dataset '%s' presente=%s).",
                row["nome_modelo"], model_cfg is not None,
                row["nome_dataset"], dataset_cfg is not None,
            )
            continue

        if thermal_enabled and not wait_for_safe_temperature(thermal_cfg):
            logger.critical(
                "Reprocessamento abortado por seguranca termica antes de: %s",
                test_label,
            )
            break

        try:
            start_time, end_time, image_count = run_inference(
                model_cfg, dataset_cfg, experiment_cfg
            )

            logger.info(
                "Waiting %ds for Prometheus scrape window...", wait_after_run
            )
            time.sleep(wait_after_run)

            logger.info("Querying Prometheus metrics...")
            metrics = prometheus.get_metrics(start_time, end_time)

            new_avg = metrics.get("avg_watt")
            new_max = metrics.get("max_watt")
            if new_avg is not None and new_avg == new_max:
                logger.warning(
                    "  Ainda travado apos reprocessar (avg_watt == max_watt == "
                    "%.4fW) — sensor/Arduino pode continuar preso. Gravando "
                    "mesmo assim (e a melhor leitura disponivel agora).",
                    new_avg,
                )

            update_energy_metrics(
                record_id=row["id"],
                metrics=metrics,
                start_time=start_time,
                end_time=end_time,
                image_count=image_count,
            )

            logger.info(
                "  avg_watt=%.4f  max_watt=%.4f  avg_cpu=%.2f%%"
                "  avg_mem=%.2f%%  avg_temp=%.2f°C",
                new_avg or 0.0,
                new_max or 0.0,
                metrics.get("avg_cpu") or 0.0,
                metrics.get("avg_mem") or 0.0,
                metrics.get("avg_temp") or 0.0,
            )

        except Exception as exc:  # noqa: BLE001 — intentional broad catch
            logger.error(
                "FAILED reprocessing: %s — %s", test_label, exc, exc_info=True
            )

        if i < len(stuck_rows):
            logger.info("Cooling down for %ds before next reprocess...", cooldown)
            time.sleep(cooldown)

    logger.info("─" * 60)
    logger.info("Reprocessamento concluido.")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(settings: dict) -> None:
    models: list[dict] = settings["models"]
    datasets: list[dict] = settings["datasets"]
    experiment_cfg: dict = settings.get("experiment", {})

    wait_after_run: int = experiment_cfg.get("wait_after_run", 30)
    cooldown: int = experiment_cfg.get("cooldown_between_tests", 15)
    thermal_cfg: dict = experiment_cfg.get("thermal_safety", {})
    thermal_enabled: bool = thermal_cfg.get("enabled", True)

    prometheus = PrometheusClient(
        host=settings["prometheus"]["host"],
        port=settings["prometheus"]["port"],
    )

    init_db(settings["postgres"])

    # Um modelo com 'dataset' definido roda so contra esse dataset (util para
    # modelos quantizados/calibrados especificamente para um dataset, onde
    # rodar cross-dataset pode nao ser desejado ou ate crashar o ONNX Runtime
    # em casos extremos de forma dinamica).
    pairs = [
        (model_cfg, dataset_cfg)
        for model_cfg in models
        for dataset_cfg in datasets
        if not model_cfg.get("dataset") or model_cfg["dataset"] == dataset_cfg["name"]
    ]

    total = len(pairs)
    completed = 0
    errors = 0
    skipped = 0

    logger.info(
        "Pipeline started — %d test(s) (%d model(s) × %d dataset(s), "
        "after per-model 'dataset' restriction).",
        total,
        len(models),
        len(datasets),
    )

    for model_cfg, dataset_cfg in pairs:
        completed += 1
        test_label = f"{model_cfg['name']} × {dataset_cfg['name']}"
        logger.info("─" * 60)
        logger.info("[%d/%d] %s", completed, total, test_label)

        if has_successful_result(model_cfg["name"], dataset_cfg["name"]):
            skipped += 1
            logger.info("  Already completed successfully — skipping (resume mode).")
            continue

        if thermal_enabled and not wait_for_safe_temperature(thermal_cfg):
            completed -= 1  # este teste nao foi de fato tentado
            logger.critical(
                "Pipeline abortado por seguranca termica antes de: %s", test_label
            )
            break

        try:
            start_time, end_time, image_count = run_inference(
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
                image_count=image_count,
            )

            avg_watt = metrics.get("avg_watt") or 0.0
            joules_por_imagem = (
                (avg_watt * (end_time - start_time)) / image_count if image_count else 0.0
            )
            logger.info(
                "  avg_watt=%.4f  max_watt=%.4f  avg_cpu=%.2f%%"
                "  avg_mem=%.2f%%  avg_temp=%.2f°C  joules/img=%.4f (n=%d)",
                avg_watt,
                metrics.get("max_watt") or 0.0,
                metrics.get("avg_cpu") or 0.0,
                metrics.get("avg_mem") or 0.0,
                metrics.get("avg_temp") or 0.0,
                joules_por_imagem,
                image_count,
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

    not_attempted = total - completed

    logger.info("─" * 60)
    logger.info(
        "Pipeline finished — %d/%d tests completed, %d error(s), %d skipped (already done)"
        "%s.",
        completed - errors - skipped,
        total,
        errors,
        skipped,
        f", {not_attempted} not attempted (thermal abort)" if not_attempted else "",
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
    parser.add_argument(
        "--only-onnx",
        action="store_true",
        help="Run only models whose path ends in .onnx (quantized models), "
             "skipping .pt baselines/pruned entries.",
    )
    parser.add_argument(
        "--reprocess-stuck-energy",
        action="store_true",
        help="Instead of running the full pipeline, re-run inference only for "
             "existing results whose energy reading looks frozen (avg_watt == "
             "max_watt, e.g. a stuck INA219/Arduino reading), overwriting the "
             "row with a fresh measurement.",
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

    if args.reprocess_stuck_energy:
        reprocess_stuck_energy(settings)
        sys.exit(0)

    if args.only_onnx:
        before = len(settings["models"])
        settings["models"] = [
            m for m in settings["models"] if str(m["path"]).lower().endswith(".onnx")
        ]
        logger.info(
            "--only-onnx: filtered %d model(s) down to %d quantized (.onnx) model(s).",
            before,
            len(settings["models"]),
        )
        if not settings["models"]:
            logger.critical("No .onnx models found in settings.yaml after filtering.")
            sys.exit(1)

    run_pipeline(settings)
