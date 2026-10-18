"""
Compara o YOLOv10n fine-tuned (treinado neste repo) com o YOLOv10n destilado
via knowledge distillation (treinado em kd_test/), nas bases oxford e caviar.

Gera: comparacao_finetune_vs_distill.xlsx, com uma aba por dataset e uma
coluna extra "Delta" mostrando a diferenca (distill - finetune) em cada
metrica, para facilitar ver se a destilacao trouxe alguma melhora.

Reaproveita evaluate_yolo() de evaluate_models.py — mesma logica de metricas
(mAP50, mAP50-95, mAP70, Precision, Recall, F1, Inferencia, FPS, FLOPS) usada
no restante do pipeline de avaliacao.

Dependencias: as mesmas de evaluate_models.py
    pip install openpyxl torchmetrics thop pyyaml
"""

import logging
from pathlib import Path

import pandas as pd
import torch

from evaluate_models import evaluate_yolo, HAS_THOP

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path("C:/workspace/mestrado/novo_teste")

DATASETS = {
    "oxford": BASE_DIR / "oxford" / "data.yaml",
    "caviar": BASE_DIR / "caviar" / "data.yaml",
}

# (label, compressao, model_path) por dataset
MODELS = {
    "oxford": [
        ("yolov10n_finetune", "Nenhum (baseline)", BASE_DIR / "runs" / "train" / "yolov10n_oxford_finetune" / "weights" / "best.pt"),
        ("yolov10n_distill",  "Destilacao (KD)",   Path(r"C:\workspace\kd_test\runs\train\yolov10n_oxford_distill\weights\best.pt")),
    ],
    "caviar": [
        ("yolov10n_finetune", "Nenhum (baseline)", BASE_DIR / "runs" / "train" / "yolov10n_caviar_finetune" / "weights" / "best.pt"),
        ("yolov10n_distill",  "Destilacao (KD)",   Path(r"C:\workspace\kd_test\runs\train\yolov10n_caviar_distill\weights\best.pt")),
    ],
}

NUMERIC_COLS = [
    "Tamanho (MB)", "Params (M)", "mAP50", "mAP50-95", "mAP70",
    "Precision", "Recall", "F1", "Inferencia (ms)", "FPS", "FLOPS (G)",
]


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("GPU nao encontrada.")

    device_int = 0
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    if not HAS_THOP:
        logger.warning("thop nao instalado — FLOPS sera None. Instale com: pip install thop")

    output_path = BASE_DIR / "comparacao_finetune_vs_distill.xlsx"
    writer = pd.ExcelWriter(output_path, engine="openpyxl")

    for dataset_key, data_yaml in DATASETS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset_key.upper()}")
        logger.info(f"{'='*60}")

        rows = []
        for label, compressao, model_path in MODELS[dataset_key]:
            if not model_path.exists():
                logger.warning(f"Nao encontrado: {model_path} — pulando {label}.")
                continue
            try:
                row = evaluate_yolo(
                    model_path, str(data_yaml), label, device_int,
                    compressao=compressao,
                )
            except Exception:
                logger.error(f"  ERRO avaliando {label} — pulando.", exc_info=True)
                continue
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Modelo")

        if len(rows) == 2:
            delta = df.loc["yolov10n_distill", NUMERIC_COLS] - df.loc["yolov10n_finetune", NUMERIC_COLS]
            delta.name = "Delta (distill - finetune)"
            df = pd.concat([df, delta.to_frame().T])

        df.to_excel(writer, sheet_name=dataset_key)
        logger.info(f"Aba '{dataset_key}' salva com {len(rows)} modelo(s).")

    writer.close()
    logger.info(f"\nPlanilha salva em: {output_path}")
