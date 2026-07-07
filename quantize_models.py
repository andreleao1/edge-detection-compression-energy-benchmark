"""
Quantizacao pos-treinamento (PTQ estatica INT8) via exportacao ONNX para multiplos modelos.

Suporta:
  - faster_rcnn : Faster R-CNN ResNet-50 FPN (.pth)
  - retinanet   : RetinaNet ResNet-50 FPN (.pth)
  - mobilenet   : SSDLite320-MobileNetV3 (.pth)
  - yolo        : YOLOv10 variantes n/s/m/b/l/x (Ultralytics .pt)

Fluxo por modelo:
  1. Carrega modelo treinado (reaproveita os builders de prune_models.py)
  2. Exporta para ONNX (opset 17)
  3. Pre-processa o grafo (shape inference / otimizacao)
  4. Calibra com imagens do conjunto de treino (MinMax)
  5. Converte pesos para INT8 per-channel e ativacoes para UINT8 (QDQ)
  6. Salva o modelo .onnx quantizado

Nos nao suportados para quantizacao (RoiAlign, NonMaxSuppression, redimensionamento
interno etc.) permanecem em float32 automaticamente — o onnxruntime so quantiza os
nos que reconhece (Conv, Gemm, MatMul, Add, ...), o que torna essa abordagem robusta
mesmo para detectores com controle de fluxo complexo (Faster R-CNN / RetinaNet).

--------------------------------------------------------------------
MODO: TODOS OS MODELOS
--------------------------------------------------------------------
  python quantize_models.py --all_models

  Itera sobre faster_rcnn / retinanet / mobilenet / todos os YOLO
  nos datasets oxford e caviar, cobrindo dois grupos de modelos:

  1) Baseline (100 epocas, sem poda):
       runs/faster_rcnn/epochs_100/{dataset}/best.pth
       runs/retinanet/epochs_100/{dataset}/best.pth
       runs/mobilenet/epochs_100/{dataset}/best.pth
       runs/train/yolov10{v}_{dataset}_finetune/weights/best.pt
     Salva em:
       runs/faster_rcnn/quantized/{dataset}/best_quantized.onnx
       runs/retinanet/quantized/{dataset}/best_quantized.onnx
       runs/mobilenet/quantized/{dataset}/best_quantized.onnx
       runs/train/quantized/{dataset}/yolov10{v}_quantized.onnx

  2) Podados (qualquer percentual ja gerado por prune_models.py em runs/,
     descoberto dinamicamente — hoje ha apenas 80%, mas outros percentuais
     sao detectados automaticamente):
       runs/faster_rcnn/pruned/{pct}/{dataset}/best_pruned.pth
       runs/retinanet/pruned/{pct}/{dataset}/best_pruned.pth
       runs/mobilenet/pruned/{pct}/{dataset}/best_pruned.pth
       runs/train/pruned/{pct}/{dataset}/yolov10{v}_pruned.pt
     Salva em:
       runs/faster_rcnn/quantized/pruned/{pct}/{dataset}/best_quantized.onnx
       runs/retinanet/quantized/pruned/{pct}/{dataset}/best_quantized.onnx
       runs/mobilenet/quantized/pruned/{pct}/{dataset}/best_quantized.onnx
       runs/train/quantized/pruned/{pct}/{dataset}/yolov10{v}_quantized.onnx

  Modelos sem arquivo correspondente (ex.: mobilenet ainda nao foi podado)
  sao pulados automaticamente.

--------------------------------------------------------------------
MODO: MODELO UNICO
--------------------------------------------------------------------
  python quantize_models.py \\
    --model_type faster_rcnn \\
    --model_path runs/faster_rcnn/epochs_100/oxford/best.pth \\
    --dataset_dir oxford \\
    --output_path runs/faster_rcnn/quantized/oxford/best_quantized.onnx

Dependencias extras:
    pip install onnx onnxruntime
"""

import argparse
import logging
import os
import shutil
import tempfile
import traceback
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from prune_models import (
    BASE_DIR,
    NUM_CLASSES,
    DATASETS,
    YOLO_VARIANTS,
    DEFAULTS,
    load_torchvision_model,
    setup_file_logging,
)

OPSET_VERSION = 17

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibracao
# ---------------------------------------------------------------------------

def load_calibration_arrays(images_dir: Path, imgsz: int, square: bool,
                             n_samples: int, add_batch_dim: bool) -> list[np.ndarray]:
    """Carrega ate n_samples imagens do dataset e aplica o mesmo pre-processamento
    usado no treinamento/pruning (resize preservando aspecto ou stretch quadrado,
    normalizacao 0-1, sem subtracao de media/desvio)."""
    paths = sorted(Path(images_dir).glob("*.jpg")) + sorted(Path(images_dir).glob("*.png"))
    if not paths:
        raise RuntimeError(f"Nenhuma imagem de calibracao em: {images_dir}")
    paths = paths[:n_samples]

    arrays = []
    for p in paths:
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if square:
            img = cv2.resize(img, (imgsz, imgsz))
        else:
            scale = imgsz / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        arr = img.astype(np.float32).transpose(2, 0, 1) / 255.0
        if add_batch_dim:
            arr = arr[None, ...]
        arrays.append(np.ascontiguousarray(arr))

    logger.info(f"  {len(arrays)} imagens de calibracao carregadas de {images_dir}")
    return arrays


class ArrayCalibrationReader(CalibrationDataReader):
    def __init__(self, arrays: list[np.ndarray], input_name: str):
        self._arrays = arrays
        self._input_name = input_name
        self._idx = 0

    def get_next(self):
        if self._idx >= len(self._arrays):
            return None
        arr = self._arrays[self._idx]
        self._idx += 1
        return {self._input_name: arr}

    def rewind(self):
        self._idx = 0


# ---------------------------------------------------------------------------
# Exportacao ONNX
# ---------------------------------------------------------------------------

def export_torchvision_onnx(model: torch.nn.Module, imgsz: int, square: bool, onnx_path: Path):
    model.eval()
    dummy = [torch.rand(3, imgsz, imgsz)]
    dynamic_axes = None if square else {"images": {1: "height", 2: "width"}}

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model, (dummy,), str(onnx_path),
            input_names=["images"], output_names=["boxes", "labels", "scores"],
            dynamic_axes=dynamic_axes, opset_version=OPSET_VERSION, dynamo=False,
        )


def export_yolo_onnx(model_path: Path, imgsz: int, onnx_path: Path):
    # Este processo ja importou onnxruntime (para a etapa de quantizacao), entao suas
    # DLLs estao carregadas em memoria. O Ultralytics, ao exportar com simplify=True,
    # tenta reinstalar onnxslim/onnxruntime-gpu via pip — no Windows isso falha com
    # "Acesso negado" porque o pip nao consegue sobrescrever DLLs em uso, desperdicando
    # ~1min por modelo. Como simplificar o grafo e so um refinamento (nao afeta a
    # quantizacao), desativamos essa checagem/auto-instalacao.
    os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("Ultralytics nao encontrada. Execute: pip install ultralytics")

    yolo = YOLO(str(model_path))
    exported = yolo.export(
        format="onnx", imgsz=imgsz, dynamic=False,
        simplify=True, opset=OPSET_VERSION, nms=False, verbose=False,
    )
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), str(onnx_path))


# ---------------------------------------------------------------------------
# Quantizacao estatica (ONNX Runtime)
# ---------------------------------------------------------------------------

def quantize_onnx_static(fp32_path: Path, quant_path: Path,
                          calibration_reader: CalibrationDataReader, per_channel: bool = True):
    quant_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        pre_path = Path(td) / "preprocessed.onnx"
        quant_pre_process(str(fp32_path), str(pre_path), skip_symbolic_shape=True)

        logging.disable(logging.WARNING)   # silencia avisos internos do quantizador (nao-fatais)
        try:
            quantize_static(
                model_input=str(pre_path),
                model_output=str(quant_path),
                calibration_data_reader=calibration_reader,
                quant_format=QuantFormat.QDQ,
                per_channel=per_channel,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8,
                calibrate_method=CalibrationMethod.MinMax,
            )
        finally:
            logging.disable(logging.NOTSET)


def _log_size_reduction(fp32_path: Path, quant_path: Path):
    fp32_mb = fp32_path.stat().st_size / 1e6
    int8_mb = quant_path.stat().st_size / 1e6
    logger.info(f"  Tamanho FP32 : {fp32_mb:.1f} MB")
    logger.info(f"  Tamanho INT8 : {int8_mb:.1f} MB  ({100 * (1 - int8_mb / fp32_mb):.1f}% menor)")
    logger.info(f"  Salvo em     : {quant_path}")


# ---------------------------------------------------------------------------
# Pipelines por modelo
# ---------------------------------------------------------------------------

def run_torchvision_quant_pipeline(model_type, model_path, dataset_dir, output_path,
                                    n_calib=100, per_channel=True):
    cfg = DEFAULTS[model_type]
    imgsz, square = cfg["imgsz"], cfg["square"]
    output_path = Path(output_path)

    logger.info(f"\nCarregando {model_type} de: {model_path}")
    model = load_torchvision_model(model_type, model_path, NUM_CLASSES, torch.device("cpu"), imgsz)

    with tempfile.TemporaryDirectory() as td:
        onnx_fp32 = Path(td) / "model_fp32.onnx"
        logger.info("Exportando para ONNX...")
        export_torchvision_onnx(model, imgsz, square, onnx_fp32)

        images_dir = Path(dataset_dir) / "datasets" / "images" / "train"
        calib_arrays = load_calibration_arrays(images_dir, imgsz, square, n_calib, add_batch_dim=False)
        reader = ArrayCalibrationReader(calib_arrays, input_name="images")

        logger.info(f"Quantizando (INT8 estatico, per_channel={per_channel})...")
        quantize_onnx_static(onnx_fp32, output_path, reader, per_channel)
        _log_size_reduction(onnx_fp32, output_path)


def run_yolo_quant_pipeline(model_path, dataset_dir, output_path, n_calib=100, per_channel=True):
    cfg = DEFAULTS["yolo"]
    imgsz = cfg["imgsz"]
    output_path = Path(output_path)

    with tempfile.TemporaryDirectory() as td:
        onnx_fp32 = Path(td) / "model_fp32.onnx"
        logger.info(f"\nExportando YOLO para ONNX: {model_path}")
        export_yolo_onnx(Path(model_path), imgsz, onnx_fp32)

        images_dir = Path(dataset_dir) / "datasets" / "images" / "train"
        calib_arrays = load_calibration_arrays(images_dir, imgsz, square=True,
                                                n_samples=n_calib, add_batch_dim=True)
        reader = ArrayCalibrationReader(calib_arrays, input_name="images")

        logger.info(f"Quantizando (INT8 estatico, per_channel={per_channel})...")
        quantize_onnx_static(onnx_fp32, output_path, reader, per_channel)
        _log_size_reduction(onnx_fp32, output_path)


# ---------------------------------------------------------------------------
# Catalogo de todos os modelos (modo --all_models)
# ---------------------------------------------------------------------------

def _find_pruned_percentages(pruned_root: Path) -> list[str]:
    """Descobre os percentuais de poda existentes (nomes das subpastas de runs/.../pruned/)."""
    if not pruned_root.exists():
        return []
    pcts = [p.name for p in pruned_root.iterdir() if p.is_dir()]
    return sorted(pcts, key=lambda x: int(x) if x.isdigit() else x)


def build_all_models_catalog() -> list[dict]:
    """
    Monta o catalogo de jobs de quantizacao a partir de dois grupos de modelos
    ja existentes em runs/:
      - baseline    : runs/<model>/epochs_100/{dataset}/best.pth (100 epocas, sem poda)
      - pruned      : runs/<model>/pruned/<pct>/{dataset}/best_pruned.pth (qualquer
                      percentual de poda ja presente em runs/, descoberto dinamicamente)
    """
    jobs = []

    tv_models = {
        "faster_rcnn": ("runs/faster_rcnn/epochs_100", "best.pth"),
        "retinanet":   ("runs/retinanet/epochs_100",   "best.pth"),
        "mobilenet":   ("runs/mobilenet/epochs_100",   "best.pth"),
    }

    # --- Baseline (100 epocas, sem poda) ---
    for model_type, (run_subdir, fname) in tv_models.items():
        for dataset in DATASETS:
            model_path  = BASE_DIR / run_subdir / dataset / fname
            output_path = BASE_DIR / "runs" / model_type / "quantized" / dataset / "best_quantized.onnx"
            jobs.append({
                "type":        model_type,
                "model_path":  model_path,
                "dataset_dir": BASE_DIR / dataset,
                "output_path": output_path,
                "label":       f"{model_type} | {dataset} | baseline",
            })

    for variant in YOLO_VARIANTS:
        for dataset in DATASETS:
            model_path  = BASE_DIR / "runs" / "train" / f"{variant}_{dataset}_finetune" / "weights" / "best.pt"
            output_path = BASE_DIR / "runs" / "train" / "quantized" / dataset / f"{variant}_quantized.onnx"
            jobs.append({
                "type":        "yolo",
                "model_path":  model_path,
                "dataset_dir": BASE_DIR / dataset,
                "output_path": output_path,
                "label":       f"{variant} | {dataset} | baseline",
            })

    # --- Podados (qualquer percentual ja presente em runs/<model>/pruned/) ---
    for model_type in tv_models:
        pruned_root = BASE_DIR / "runs" / model_type / "pruned"
        for pct in _find_pruned_percentages(pruned_root):
            for dataset in DATASETS:
                model_path  = pruned_root / pct / dataset / "best_pruned.pth"
                output_path = BASE_DIR / "runs" / model_type / "quantized" / "pruned" / pct / dataset / "best_quantized.onnx"
                jobs.append({
                    "type":        model_type,
                    "model_path":  model_path,
                    "dataset_dir": BASE_DIR / dataset,
                    "output_path": output_path,
                    "label":       f"{model_type} | {dataset} | pruned {pct}%",
                })

    yolo_pruned_root = BASE_DIR / "runs" / "train" / "pruned"
    for pct in _find_pruned_percentages(yolo_pruned_root):
        for variant in YOLO_VARIANTS:
            for dataset in DATASETS:
                model_path  = yolo_pruned_root / pct / dataset / f"{variant}_pruned.pt"
                output_path = BASE_DIR / "runs" / "train" / "quantized" / "pruned" / pct / dataset / f"{variant}_quantized.onnx"
                jobs.append({
                    "type":        "yolo",
                    "model_path":  model_path,
                    "dataset_dir": BASE_DIR / dataset,
                    "output_path": output_path,
                    "label":       f"{variant} | {dataset} | pruned {pct}%",
                })

    return jobs


# ---------------------------------------------------------------------------
# Orquestrador modo --all_models
# ---------------------------------------------------------------------------

def run_all_models(n_calib: int, per_channel: bool, skip_existing: bool = True, log_path: Path = None):
    if log_path:
        setup_file_logging(log_path)

    jobs = build_all_models_catalog()
    results = {"ok": [], "skip": [], "erro": []}

    logger.info(f"\n{'#' * 64}")
    logger.info(f"  MODO ALL_MODELS  |  quantizacao INT8 (ONNX)  |  n_calib={n_calib}")
    logger.info(f"  Total de jobs: {len(jobs)}")
    logger.info(f"{'#' * 64}")

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
                run_yolo_quant_pipeline(
                    model_path  = model_path,
                    dataset_dir = job["dataset_dir"],
                    output_path = output_path,
                    n_calib     = n_calib,
                    per_channel = per_channel,
                )
            else:
                run_torchvision_quant_pipeline(
                    model_type  = job["type"],
                    model_path  = model_path,
                    dataset_dir = job["dataset_dir"],
                    output_path = output_path,
                    n_calib     = n_calib,
                    per_channel = per_channel,
                )
            results["ok"].append(label)

        except Exception:
            tb = traceback.format_exc()
            logger.error(f"  ERRO em '{label}':\n{tb}")
            results["erro"].append((label, tb))

    logger.info(f"\n{'=' * 64}")
    logger.info("  RESUMO FINAL")
    logger.info(f"{'=' * 64}")
    logger.info(f"  Concluidos : {len(results['ok'])}")
    logger.info(f"  Pulados    : {len(results['skip'])}")
    logger.info(f"  Com erro   : {len(results['erro'])}")

    if results["erro"]:
        logger.info(f"\n{'=' * 64}")
        logger.info("  DETALHES DOS ERROS")
        logger.info(f"{'=' * 64}")
        for label, tb in results["erro"]:
            logger.info(f"\n  >>> {label}")
            last_line = [l.strip() for l in tb.strip().splitlines() if l.strip()][-1]
            logger.info(f"  Causa: {last_line}")
            logger.info("  Traceback completo:")
            for line in tb.strip().splitlines():
                logger.info(f"    {line}")
            logger.info("")

    logger.info(f"{'=' * 64}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Quantizacao PTQ estatica (INT8) via exportacao ONNX de modelos de deteccao",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = p.add_argument_group("Modo de execucao")
    mode.add_argument("--all_models", action="store_true",
                      help="Quantiza todos os modelos (faster_rcnn, retinanet, mobilenet, yolo x6) "
                           "nos datasets oxford e caviar")
    mode.add_argument("--model_type", choices=["yolo", "faster_rcnn", "retinanet", "mobilenet"],
                      help="[modo unico] Tipo do modelo")
    mode.add_argument("--model_path",
                      help="[modo unico] Caminho para o modelo treinado (.pth ou .pt)")
    mode.add_argument("--output_path",
                      help="[modo unico] Caminho de saida para o modelo .onnx quantizado")

    ds = p.add_argument_group("Dataset (modo unico)")
    ds.add_argument("--dataset_dir",
                    help="Diretorio do dataset, usado para calibracao. "
                         "Deve conter datasets/images/train")

    q = p.add_argument_group("Quantizacao")
    q.add_argument("--n_calib_images", type=int, default=100,
                  help="Numero de imagens de treino usadas para calibracao")
    q.add_argument("--no_per_channel", action="store_true",
                  help="Desativa quantizacao per-channel dos pesos (usa per-tensor)")

    p.add_argument("--overwrite", action="store_true",
                   help="[all_models] Reprocessa mesmo que o arquivo de saida ja exista")

    args = p.parse_args()

    if not args.all_models:
        if not args.model_type:
            p.error("--model_type e obrigatorio no modo unico (ou use --all_models)")
        if not args.model_path:
            p.error("--model_path e obrigatorio no modo unico")
        if not args.output_path:
            p.error("--output_path e obrigatorio no modo unico")
        if not args.dataset_dir:
            p.error("--dataset_dir e obrigatorio no modo unico (usado para calibracao)")

    args.per_channel = not args.no_per_channel

    return args


def main():
    args = parse_args()

    logger.info("=" * 64)
    logger.info("  QUANTIZACAO PTQ ESTATICA (INT8) VIA EXPORTACAO ONNX")
    logger.info("=" * 64)
    logger.info(f"  Imagens de calibracao : {args.n_calib_images}")
    logger.info(f"  Per-channel           : {args.per_channel}")

    if args.all_models:
        log_path = BASE_DIR / "runs" / "quantized_logs" / "quantize.log"
        run_all_models(
            n_calib       = args.n_calib_images,
            per_channel   = args.per_channel,
            skip_existing = not args.overwrite,
            log_path      = log_path,
        )
    elif args.model_type == "yolo":
        run_yolo_quant_pipeline(
            model_path  = Path(args.model_path),
            dataset_dir = Path(args.dataset_dir),
            output_path = Path(args.output_path),
            n_calib     = args.n_calib_images,
            per_channel = args.per_channel,
        )
    else:
        run_torchvision_quant_pipeline(
            model_type  = args.model_type,
            model_path  = Path(args.model_path),
            dataset_dir = Path(args.dataset_dir),
            output_path = Path(args.output_path),
            n_calib     = args.n_calib_images,
            per_channel = args.per_channel,
        )


if __name__ == "__main__":
    main()
