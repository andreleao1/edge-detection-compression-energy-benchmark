"""
Avaliacao de todos os modelos treinados (YOLO, Faster R-CNN, RetinaNet, MobileNet)
nas bases oxford e caviar — baseline, podado, quantizado, podado+quantizado, e os
modelos de knowledge distillation (YOLOv10n) treinados em kd_test/.

Gera:
  - Planilha Excel: resultados_avaliacao.xlsx (uma aba por dataset)
  - Linhas na tabela Postgres model_accuracy_results (accuracy_db.py) — mesmo
    banco usado pelo benchmark de energia da rasp, tabela propria e independente
    (nao mexe em rasp_evaluate_models/database.py nem nas migrations de la).

Metricas por linha: Params, Tamanho (MB), mAP50, mAP50-95, mAP70, Precision,
Recall, F1, Inferencia(ms), FPS, FLOPS(G).

RetinaNet quantizado (INT8) e' pulado propositalmente — trava o ONNX Runtime
com segmentation fault em qualquer session.run() (ver quantize_models.py,
comentario acima de quantize_onnx_static).

Dependencias extras:
    pip install openpyxl torchmetrics thop sqlalchemy psycopg2-binary pyyaml
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ultralytics import YOLO

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from prune_models import DEFAULTS, NUM_CLASSES, YOLODetectionDataset, load_torchvision_model

import accuracy_db

try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

# ---------------------------------------------------------------------------
# Configuracoes
# ---------------------------------------------------------------------------

BASE_DIR    = Path("C:/workspace/mestrado/novo_teste")
RUNS_DIR    = BASE_DIR / "runs"
NUM_EPOCHS  = 100
MODELS_DIR  = BASE_DIR / "models" / f"epochs_{NUM_EPOCHS}"
BATCH_SIZE  = 1     # inferencia imagem a imagem para medir latencia real
NUM_WORKERS = 4
CLASS_NAMES = ["pessoa"]

DATASETS = {
    "oxford": BASE_DIR / "oxford",
    "caviar": BASE_DIR / "caviar",
}
DATASET_DB_NAME = {"oxford": "OXFORD_TOWER", "caviar": "CAVIAR"}

YOLO_STEMS = ["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x"]
TORCHVISION_TYPES = ["faster_rcnn", "retinanet", "mobilenet"]

# RetinaNet quantizado trava o ONNX Runtime com segmentation fault em
# qualquer session.run() (ver quantize_models.py, comentario acima de
# quantize_onnx_static) — nunca avaliar as variantes quantizadas dele.
SKIP_QUANTIZED = {"retinanet"}

# Modelos YOLOv10n destilados (knowledge distillation), treinados fora deste
# repo em kd_test/. Sem variante podada/quantizada por enquanto.
KD_MODELS = {
    "caviar": Path(r"C:\workspace\kd_test\runs\train\yolov10n_caviar_distill\weights\best.pt"),
    "oxford": Path(r"C:\workspace\kd_test\runs\train\yolov10n_oxford_distill\weights\best.pt"),
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Contagem de parametros e FLOPS
# ---------------------------------------------------------------------------

def count_params(model) -> float:
    """Retorna numero de parametros em milhoes."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def estimate_flops(model, device, imgsz: int) -> float | None:
    """Estima GFLOPs com thop. Retorna None se thop nao estiver instalado."""
    if not HAS_THOP:
        return None
    dummy = torch.zeros(1, 3, imgsz, imgsz).to(device)
    try:
        model.eval()
        flops, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        return flops / 1e9
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Catalogo de jobs de avaliacao (baseline / podado / quantizado / KD)
# ---------------------------------------------------------------------------

def _find_pruned_percentages(pruned_root: Path) -> list[str]:
    """Descobre os percentuais de poda existentes (nomes das subpastas de
    runs/.../pruned/), na ordem numerica."""
    if not pruned_root.exists():
        return []
    return sorted(
        (p.name for p in pruned_root.iterdir() if p.is_dir()),
        key=lambda x: int(x) if x.isdigit() else x,
    )


def build_evaluation_catalog(dataset_key: str) -> list[dict]:
    """
    Monta a lista de jobs de avaliacao para um dataset, cobrindo baseline,
    knowledge distillation, podado, quantizado e podado+quantizado — as
    mesmas variantes usadas no benchmark de energia da rasp
    (rasp_evaluate_models/settings.yaml), com nomes compativeis para permitir
    JOIN futuro por (nome_modelo, nome_dataset).
    """
    jobs: list[dict] = []
    models_dataset_dir = MODELS_DIR / dataset_key
    yolo_pruned_pcts = _find_pruned_percentages(RUNS_DIR / "train" / "pruned")

    # ---- Baseline ----------------------------------------------------------
    for stem in YOLO_STEMS:
        jobs.append({
            "eval_type": "yolo",
            "label": stem,
            "compressao": "Nenhum (baseline)",
            "pruning_rate": None,
            "model_path": models_dataset_dir / f"{stem}_finetuned.pt",
        })

    for arch in TORCHVISION_TYPES:
        jobs.append({
            "eval_type": "torchvision",
            "arch": arch,
            "label": arch,
            "compressao": "Nenhum (baseline)",
            "pruning_rate": None,
            "model_path": models_dataset_dir / f"{arch}_finetuned.pth",
        })

    # ---- Knowledge distillation (YOLOv10n) ---------------------------------
    kd_path = KD_MODELS.get(dataset_key)
    if kd_path is not None:
        jobs.append({
            "eval_type": "yolo",
            "label": "yolov10n_distill",
            "compressao": "Destilacao (KD)",
            "pruning_rate": None,
            "model_path": kd_path,
        })

    # ---- Podado (YOLO) ------------------------------------------------------
    for pct in yolo_pruned_pcts:
        for variant in YOLO_STEMS:
            jobs.append({
                "eval_type": "yolo",
                "label": f"{variant}_pruned_p{pct}_{dataset_key}",
                "compressao": f"Podado ({pct}%)",
                "pruning_rate": float(pct),
                "model_path": RUNS_DIR / "train" / "pruned" / pct / dataset_key / f"{variant}_pruned.pt",
                "template_path": RUNS_DIR / "train" / f"{variant}_{dataset_key}_finetune" / "weights" / "best.pt",
            })

    # ---- Podado (torchvision) ------------------------------------------------
    for arch in TORCHVISION_TYPES:
        for pct in _find_pruned_percentages(RUNS_DIR / arch / "pruned"):
            jobs.append({
                "eval_type": "torchvision",
                "arch": arch,
                "label": f"{arch}_pruned_p{pct}_{dataset_key}",
                "compressao": f"Podado ({pct}%)",
                "pruning_rate": float(pct),
                "model_path": RUNS_DIR / arch / "pruned" / pct / dataset_key / "best_pruned.pth",
            })

    # ---- Quantizado (YOLO) ----------------------------------------------------
    for variant in YOLO_STEMS:
        jobs.append({
            "eval_type": "yolo",
            "label": f"{variant}_quantized_{dataset_key}",
            "compressao": "Quantizado (INT8)",
            "pruning_rate": None,
            "model_path": RUNS_DIR / "train" / "quantized" / dataset_key / f"{variant}_quantized.onnx",
        })

    for pct in yolo_pruned_pcts:
        for variant in YOLO_STEMS:
            jobs.append({
                "eval_type": "yolo",
                "label": f"{variant}_pruned_p{pct}_quantized_{dataset_key}",
                "compressao": f"Podado ({pct}%) + Quantizado",
                "pruning_rate": float(pct),
                "model_path": RUNS_DIR / "train" / "quantized" / "pruned" / pct / dataset_key / f"{variant}_quantized.onnx",
            })

    # ---- Quantizado (torchvision) ----------------------------------------------
    for arch in TORCHVISION_TYPES:
        if arch in SKIP_QUANTIZED:
            continue

        jobs.append({
            "eval_type": "torchvision",
            "arch": arch,
            "label": f"{arch}_quantized_{dataset_key}",
            "compressao": "Quantizado (INT8)",
            "pruning_rate": None,
            "model_path": RUNS_DIR / arch / "quantized" / dataset_key / "best_quantized.onnx",
        })

        for pct in _find_pruned_percentages(RUNS_DIR / arch / "pruned"):
            jobs.append({
                "eval_type": "torchvision",
                "arch": arch,
                "label": f"{arch}_pruned_p{pct}_quantized_{dataset_key}",
                "compressao": f"Podado ({pct}%) + Quantizado",
                "pruning_rate": float(pct),
                "model_path": RUNS_DIR / arch / "quantized" / "pruned" / pct / dataset_key / "best_quantized.onnx",
            })

    return jobs


# ---------------------------------------------------------------------------
# Avaliacao de modelos YOLO
# ---------------------------------------------------------------------------

def _load_yolo_model(model_path: Path, template_path: Path | None) -> YOLO:
    """Carrega um modelo YOLO. Se template_path for informado, model_path e'
    tratado como state_dict puro de um modelo podado (prune_models.py salva
    so os pesos, nao o checkpoint completo) — mesma logica de
    rasp_evaluate_models/main.py::load_yolo_model()."""
    if template_path is not None:
        model = YOLO(str(template_path))
        state_dict = torch.load(model_path, map_location="cpu")
        model.model.load_state_dict(state_dict)
        return model
    return YOLO(str(model_path))


def evaluate_yolo(
    model_path: Path,
    data_yaml: str,
    model_label: str,
    device: int,
    compressao: str = "Nenhum (baseline)",
    pruning_rate: float | None = None,
    template_path: Path | None = None,
) -> dict:
    logger.info(f"  Avaliando YOLO: {model_label}")
    is_onnx = model_path.suffix.lower() == ".onnx"
    model = _load_yolo_model(model_path, template_path)

    # Modelos .onnx: forca CPU. O onnxruntime instalado aqui tem suporte a
    # CUDA mas exige CUDA 13/cuDNN 9 (o torch usa CUDA 11.8) — o
    # CUDAExecutionProvider falha ao criar e cai pra CPU silenciosamente,
    # mas o ultralytics (que recebeu device=0) ainda tenta vincular o input
    # na GPU, causando mismatch de device no IOBinding. Forcar CPU evita essa
    # inconsistencia (mesma abordagem ja usada em evaluate_torchvision_onnx).
    val_device = "cpu" if is_onnx else device
    results = model.val(data=data_yaml, split="test", device=val_device, verbose=False)
    box = results.box

    map50 = float(box.map50)
    map50_95 = float(box.map)
    try:
        map70 = float(box.all_ap[:, 4].mean()) if len(box.all_ap) else None
    except (AttributeError, IndexError):
        logger.warning(f"  Nao consegui calcular mAP70 para {model_label} (box.all_ap indisponivel)")
        map70 = None
    precision = float(box.mp)
    recall = float(box.mr)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    if is_onnx:
        # AutoBackend do ultralytics envolve uma sessao onnxruntime aqui, nao
        # um nn.Module real — nao ha parametros/FLOPs pra medir.
        params_m = None
        flops_g = None
    else:
        params_m = count_params(model.model)
        param_device = next(model.model.parameters()).device
        flops_g = estimate_flops(model.model, param_device, DEFAULTS["yolo"]["imgsz"])

    # results.speed ja vem do .val() acima — nao roda uma segunda validacao so
    # pra medir latencia (o codigo original chamava model.val() duas vezes).
    inf_ms = float(results.speed.get("inference", 0))

    return {
        "Modelo": model_label,
        "Compressao": compressao,
        "Pruning rate (%)": pruning_rate,
        "Tamanho (MB)": round(model_path.stat().st_size / 1e6, 2),
        "Params (M)": round(params_m, 2) if params_m is not None else None,
        "mAP50": round(map50, 4),
        "mAP50-95": round(map50_95, 4),
        "mAP70": round(map70, 4) if map70 is not None else None,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "Inferencia (ms)": round(inf_ms, 2),
        "FPS": round(1000 / inf_ms, 1) if inf_ms else None,
        "FLOPS (G)": round(flops_g, 2) if flops_g else None,
    }


# ---------------------------------------------------------------------------
# Precision / Recall por IoU matching (para modelos Torchvision)
# ---------------------------------------------------------------------------

def _compute_precision_recall(all_preds: list, all_targets: list, iou_thresh: float = 0.5):
    """
    Calcula Precision e Recall globais usando matching greedy por IoU.
    all_preds:   lista de dicts {"boxes": Tensor[N,4], "scores": Tensor[N], "labels": Tensor[N]}
    all_targets: lista de dicts {"boxes": Tensor[M,4], "labels": Tensor[M]}
    """
    tp = fp = fn = 0

    for pred, target in zip(all_preds, all_targets):
        pred_boxes   = pred["boxes"]
        target_boxes = target["boxes"]

        matched_gt = set()

        for pb in pred_boxes:
            if len(target_boxes) == 0:
                fp += 1
                continue

            x1 = torch.max(pb[0], target_boxes[:, 0])
            y1 = torch.max(pb[1], target_boxes[:, 1])
            x2 = torch.min(pb[2], target_boxes[:, 2])
            y2 = torch.min(pb[3], target_boxes[:, 3])
            inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
            area_pb = (pb[2] - pb[0]) * (pb[3] - pb[1])
            area_gt = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
            iou = inter / (area_pb + area_gt - inter + 1e-9)

            best_iou, best_idx = iou.max(0)
            best_idx = best_idx.item()

            if best_iou >= iou_thresh and best_idx not in matched_gt:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1

        fn += len(target_boxes) - len(matched_gt)

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    return precision, recall


# ---------------------------------------------------------------------------
# Avaliacao de modelos Torchvision (FasterRCNN / RetinaNet / MobileNet)
# ---------------------------------------------------------------------------

def evaluate_torchvision(
    model,
    model_label: str,
    dataset_dir: Path,
    device: torch.device,
    imgsz: int,
    square: bool,
    model_path: Path,
    compressao: str = "Nenhum (baseline)",
    pruning_rate: float | None = None,
) -> dict:
    logger.info(f"  Avaliando: {model_label}")

    images_dir = dataset_dir / "datasets" / "images" / "test"
    labels_dir = dataset_dir / "datasets" / "labels" / "test"
    loader = DataLoader(
        YOLODetectionDataset(images_dir, labels_dir, imgsz=imgsz, square=square),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    metric    = MeanAveragePrecision(iou_type="bbox")
    metric_70 = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.70])
    times     = []
    all_preds = []
    all_tgts  = []

    with torch.no_grad():
        for images, targets in loader:
            images_gpu = [img.to(device) for img in images]

            if len(times) == 0:
                for _ in range(5):
                    _ = model(images_gpu)
                torch.cuda.synchronize()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            preds = model(images_gpu)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

            preds_fmt   = [{"boxes": p["boxes"].cpu(), "scores": p["scores"].cpu(), "labels": p["labels"].cpu()} for p in preds]
            targets_fmt = [{"boxes": t["boxes"],       "labels": t["labels"]}  for t in targets]
            metric.update(preds_fmt, targets_fmt)
            metric_70.update(preds_fmt, targets_fmt)
            all_preds.extend(preds_fmt)
            all_tgts.extend(targets_fmt)

    results  = metric.compute()
    map50    = float(results["map_50"])
    map50_95 = float(results["map"])
    map70    = float(metric_70.compute()["map"])

    precision, recall = _compute_precision_recall(all_preds, all_tgts, iou_thresh=0.5)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    inf_ms   = float(np.mean(times[1:]) if len(times) > 1 else times[0])
    params_m = count_params(model)
    flops_g  = estimate_flops(model, device, imgsz)

    return {
        "Modelo":            model_label,
        "Compressao":        compressao,
        "Pruning rate (%)":  pruning_rate,
        "Tamanho (MB)":      round(model_path.stat().st_size / 1e6, 2),
        "Params (M)":        round(params_m, 2),
        "mAP50":             round(map50, 4),
        "mAP50-95":          round(map50_95, 4),
        "mAP70":             round(map70, 4),
        "Precision":         round(precision, 4),
        "Recall":            round(recall, 4),
        "F1":                round(f1, 4),
        "Inferencia (ms)":   round(inf_ms, 2),
        "FPS":               round(1000 / inf_ms, 1),
        "FLOPS (G)":         round(flops_g, 2) if flops_g else None,
    }


def evaluate_torchvision_onnx(
    onnx_path: Path,
    model_label: str,
    dataset_dir: Path,
    imgsz: int,
    square: bool,
    compressao: str,
    pruning_rate: float | None = None,
) -> dict:
    """Avalia um Faster R-CNN / MobileNet(SSDLite) quantizado (.onnx) via
    ONNX Runtime. O grafo exportado por quantize_models.py::export_torchvision_onnx
    ja inclui RoIAlign/NMS internos (nao quantizados), entao a saida ja e'
    deteccoes pos-NMS prontas pro MeanAveragePrecision — sem pos-processamento
    extra. RetinaNet quantizado nunca chega aqui (ver SKIP_QUANTIZED)."""
    logger.info(f"  Avaliando (ONNX): {model_label}")
    import onnxruntime as ort

    session    = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    images_dir = dataset_dir / "datasets" / "images" / "test"
    labels_dir = dataset_dir / "datasets" / "labels" / "test"
    loader = DataLoader(
        YOLODetectionDataset(images_dir, labels_dir, imgsz=imgsz, square=square),
        batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
    )

    metric    = MeanAveragePrecision(iou_type="bbox")
    metric_70 = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.70])
    times     = []
    all_preds = []
    all_tgts  = []

    for images, targets in loader:
        arr = np.ascontiguousarray(images[0].numpy().astype(np.float32))

        if len(times) == 0:
            for _ in range(5):
                session.run(None, {input_name: arr})

        t0 = time.perf_counter()
        boxes, labels, scores = session.run(["boxes", "labels", "scores"], {input_name: arr})
        times.append((time.perf_counter() - t0) * 1000)

        pred_fmt = {
            "boxes":  torch.from_numpy(boxes).float(),
            "scores": torch.from_numpy(scores).float(),
            "labels": torch.from_numpy(labels).long(),
        }
        target_fmt = {"boxes": targets[0]["boxes"], "labels": targets[0]["labels"]}
        metric.update([pred_fmt], [target_fmt])
        metric_70.update([pred_fmt], [target_fmt])
        all_preds.append(pred_fmt)
        all_tgts.append(target_fmt)

    results  = metric.compute()
    map50    = float(results["map_50"])
    map50_95 = float(results["map"])
    map70    = float(metric_70.compute()["map"])

    precision, recall = _compute_precision_recall(all_preds, all_tgts, iou_thresh=0.5)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    inf_ms = float(np.mean(times[1:]) if len(times) > 1 else times[0])

    return {
        "Modelo":            model_label,
        "Compressao":        compressao,
        "Pruning rate (%)":  pruning_rate,
        "Tamanho (MB)":      round(onnx_path.stat().st_size / 1e6, 2),
        "Params (M)":        None,
        "mAP50":             round(map50, 4),
        "mAP50-95":          round(map50_95, 4),
        "mAP70":             round(map70, 4),
        "Precision":         round(precision, 4),
        "Recall":            round(recall, 4),
        "F1":                round(f1, 4),
        "Inferencia (ms)":   round(inf_ms, 2),
        "FPS":               round(1000 / inf_ms, 1) if inf_ms else None,
        "FLOPS (G)":         None,
    }


# ---------------------------------------------------------------------------
# Dispatcher — despacha cada job do catalogo pro avaliador certo
# ---------------------------------------------------------------------------

def evaluate_job(job: dict, dataset_dir: Path, data_yaml: str, device: torch.device, device_int: int) -> dict | None:
    model_path = job["model_path"]
    if not model_path.exists():
        logger.warning(f"Nao encontrado: {model_path} — pulando {job['label']}.")
        return None

    if job["eval_type"] == "yolo":
        template_path = job.get("template_path")
        if template_path is not None and not template_path.exists():
            logger.warning(f"Template nao encontrado: {template_path} — pulando {job['label']}.")
            return None

        return evaluate_yolo(
            model_path, data_yaml, job["label"], device_int,
            compressao=job["compressao"], pruning_rate=job.get("pruning_rate"),
            template_path=template_path,
        )

    # torchvision
    arch = job["arch"]
    cfg = DEFAULTS[arch]
    imgsz, square = cfg["imgsz"], cfg["square"]

    if model_path.suffix.lower() == ".onnx":
        return evaluate_torchvision_onnx(
            model_path, job["label"], dataset_dir, imgsz, square,
            compressao=job["compressao"], pruning_rate=job.get("pruning_rate"),
        )

    model = load_torchvision_model(arch, model_path, NUM_CLASSES, device, imgsz).eval()
    try:
        return evaluate_torchvision(
            model, job["label"], dataset_dir, device, imgsz, square, model_path,
            compressao=job["compressao"], pruning_rate=job.get("pruning_rate"),
        )
    finally:
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("GPU nao encontrada.")

    device     = torch.device("cuda:0")
    device_int = 0

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    if not HAS_THOP:
        logger.warning("thop nao instalado — FLOPS sera None para modelos Torchvision. Instale com: pip install thop")

    accuracy_db.init_db()

    writer = pd.ExcelWriter(BASE_DIR / "resultados_avaliacao.xlsx", engine="openpyxl")

    for dataset_key, dataset_dir in DATASETS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset_key.upper()}")
        logger.info(f"{'='*60}")

        data_yaml       = str(dataset_dir / "data.yaml")
        dataset_db_name = DATASET_DB_NAME[dataset_key]
        jobs            = build_evaluation_catalog(dataset_key)
        rows            = []

        logger.info(f"{len(jobs)} job(s) no catalogo para {dataset_key}.")

        for job in jobs:
            existing = accuracy_db.get_result(job["label"], dataset_db_name)
            if existing is not None:
                logger.info(f"  Ja avaliado — pulando {job['label']} (resume mode).")
                rows.append(existing)
                continue

            try:
                row = evaluate_job(job, dataset_dir, data_yaml, device, device_int)
            except Exception:
                logger.error(f"  ERRO avaliando {job['label']} — pulando.", exc_info=True)
                continue

            if row is None:
                continue

            rows.append(row)
            try:
                accuracy_db.save_accuracy_result(row, dataset_db_name)
            except Exception:
                logger.error(f"  ERRO gravando no banco: {job['label']}", exc_info=True)

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name=dataset_key, index=False)
        logger.info(f"Aba '{dataset_key}' salva com {len(rows)} modelos.")

    writer.close()
    output = BASE_DIR / "resultados_avaliacao.xlsx"
    logger.info(f"\nPlanilha salva em: {output}")
