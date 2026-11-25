#!/usr/bin/env python3
"""
Varredura da resolucao de inferencia (imgsz) nos modelos de
experiments/top_models, nas duas bases.

O que varia e' o tamanho de entrada do MODELO (imgsz), nao a resolucao do
video: o YOLO redimensiona qualquer imagem para imgsz antes de inferir, entao
e' o imgsz que muda o custo computacional (e a energia).

Por tipo de modelo:
  - .pt (KD): o mesmo checkpoint, inferido com imgsz diferente;
  - ONNX quantizado: o tamanho de entrada e' fixo no grafo, entao para cada
    imgsz o checkpoint PyTorch de origem e' exportado e quantizado de novo,
    com o mesmo procedimento de quantize_models.py (INT8 estatico, 100
    imagens de calibracao do treino, cabeca do YOLO em FP32). Em 640 usa o
    proprio ONNX ja medido na Raspberry.

Etapas (--stage):
  build    gera os ONNX quantizados por imgsz em
           experiments/top_models/<modelo>/variants/imgsz<N>/<base>/
  eval     mAP no PC: Caviar nas sequencias extras do CAVIAR (sem vazamento,
           ver prepare_caviar_cross_video.py) e Oxford no split de teste.
           Saida: experiments/top_models/resolution_sweep_map.csv
  deploy   copia os modelos de cada imgsz para
           models_to_deploy/<base>/<arch>/<variante>_imgsz<N>/ — o
           main_models.py da Raspberry usa o imgsz do nome da pasta
           (imgsz<N>) e rotula a medicao com ele. 640 nao e' copiado (ja
           medido).
  all      build + eval + deploy (padrao)

Uso:
    venv/bin/python resolution_sweep.py
    venv/bin/python resolution_sweep.py --stage eval --imgsz 640 416
"""
import argparse
import csv
import json
import logging
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOP_DIR = ROOT / "experiments" / "top_models"
CROSS_DIR = ROOT / "caviar" / "datasets" / "cross_video"
DEPLOY_DIR = ROOT / "models_to_deploy"
OXFORD_YAML = ROOT / "runs" / "yaml" / "oxford.yaml"
DEFAULT_IMGSZ = [640, 512, 416, 320]
DATASETS = ("caviar", "oxford")
N_CALIB = 100

logger = logging.getLogger("resolution_sweep")


def is_quantized(entry: dict) -> bool:
    return entry["requantize_after_modifying"]


def model_file(entry: dict, dataset: str, imgsz: int) -> Path:
    """Arquivo a usar para (modelo, base, imgsz)."""
    files = entry["files"][dataset]
    if not is_quantized(entry):
        return TOP_DIR / files["source"]
    if imgsz == 640:
        return TOP_DIR / files["reference"]
    name = Path(files["reference"]).name
    return TOP_DIR / entry["dir"] / "variants" / f"imgsz{imgsz}" / dataset / name


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(manifest: dict, sizes: list[int]) -> None:
    import quantize_models as qm

    for entry in manifest["models"]:
        if not is_quantized(entry):
            continue
        for ds in DATASETS:
            src = TOP_DIR / entry["files"][ds]["source"]
            for imgsz in sizes:
                if imgsz == 640:
                    continue
                out = model_file(entry, ds, imgsz)
                if out.exists():
                    logger.info(f"  ja existe: {out.relative_to(ROOT)}")
                    continue
                logger.info(f"  {entry['dir']} {ds} imgsz={imgsz}: exportando e quantizando...")
                with tempfile.TemporaryDirectory() as td:
                    fp32 = Path(td) / "model_fp32.onnx"
                    qm.export_yolo_onnx(src, imgsz, fp32)
                    calib = qm.load_calibration_arrays(ROOT / ds / "datasets" / "images" / "train", imgsz,
                                                       square=True, n_samples=N_CALIB, add_batch_dim=True)
                    reader = qm.ArrayCalibrationReader(calib, input_name="images")
                    qm.quantize_onnx_static(fp32, out, reader, per_channel=True, exclude_yolo_head=True)
                logger.info(f"    -> {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def eval_sets(dataset: str) -> list[tuple[str, Path]]:
    if dataset == "oxford":
        return [("test", OXFORD_YAML)]
    return [(d.name, d / "data.yaml") for d in sorted(CROSS_DIR.iterdir()) if (d / "data.yaml").exists()]


def evaluate(manifest: dict, sizes: list[int], device: int, out_csv: Path) -> None:
    from ultralytics import YOLO

    done = set()
    rows = []
    if out_csv.exists():  # retoma: nao reavalia o que ja esta' no CSV
        rows = list(csv.DictReader(out_csv.open()))
        done = {(r["dir"], r["dataset"], r["set"], int(r["imgsz"])) for r in rows}
    fields = ["rank", "dir", "label", "dataset", "set", "imgsz", "file", "mAP50", "mAP50-95", "Precision", "Recall"]

    for entry in manifest["models"]:
        for ds in DATASETS:
            for imgsz in sizes:
                path = model_file(entry, ds, imgsz)
                if not path.exists():
                    logger.warning(f"  falta {path.relative_to(ROOT)} — rode --stage build")
                    continue
                onnx = path.suffix == ".onnx"
                for set_name, yaml_path in eval_sets(ds):
                    key = (entry["dir"], ds, set_name, imgsz)
                    if key in done:
                        continue
                    model = YOLO(str(path), task="detect")
                    r = model.val(data=str(yaml_path), split="test", imgsz=imgsz, batch=1 if onnx else 16,
                                  device="cpu" if onnx else device, verbose=False, plots=False)
                    row = {"rank": entry["rank"], "dir": entry["dir"], "label": entry["label"], "dataset": ds,
                           "set": set_name, "imgsz": imgsz, "file": str(path.relative_to(ROOT)),
                           "mAP50": round(float(r.box.map50), 4), "mAP50-95": round(float(r.box.map), 4),
                           "Precision": round(float(r.box.mp), 4), "Recall": round(float(r.box.mr), 4)}
                    rows.append(row)
                    logger.info(f"  {entry['dir']:22s} {ds:7s} {set_name:20s} imgsz {imgsz}: "
                                f"mAP50 {row['mAP50']:.3f}  mAP50-95 {row['mAP50-95']:.3f}")
                    with out_csv.open("w", newline="") as fh:  # grava a cada resultado (retomavel)
                        wr = csv.DictWriter(fh, fieldnames=fields)
                        wr.writeheader()
                        wr.writerows(rows)
    logger.info(f"CSV: {out_csv.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# deploy
# ---------------------------------------------------------------------------

def deploy(manifest: dict, sizes: list[int]) -> None:
    n = 0
    for entry in manifest["models"]:
        for ds in DATASETS:
            for imgsz in sizes:
                if imgsz == 640:
                    continue
                src = model_file(entry, ds, imgsz)
                if not src.exists():
                    logger.warning(f"  falta {src.relative_to(ROOT)}")
                    continue
                dst = DEPLOY_DIR / ds / entry["arch"] / f"{entry['variant']}_imgsz{imgsz}" / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not (dst.exists() and dst.stat().st_size == src.stat().st_size):
                    shutil.copy2(src, dst)
                n += 1
    logger.info(f"{n} modelo(s) em models_to_deploy/<base>/<arch>/<variante>_imgsz<N>/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("build", "eval", "deploy", "all"), default="all")
    ap.add_argument("--imgsz", type=int, nargs="+", default=DEFAULT_IMGSZ)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", type=Path, default=TOP_DIR / "resolution_sweep_map.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("ultralytics", "quantize_models", "root"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    bad = [s for s in args.imgsz if s % 32]
    if bad:
        ap.error(f"imgsz precisa ser multiplo de 32: {bad}")
    manifest = json.loads((TOP_DIR / "manifest.json").read_text())

    if args.stage in ("build", "all"):
        build(manifest, args.imgsz)
    if args.stage in ("eval", "all"):
        evaluate(manifest, args.imgsz, args.device, args.out)
    if args.stage in ("deploy", "all"):
        deploy(manifest, args.imgsz)


if __name__ == "__main__":
    main()
