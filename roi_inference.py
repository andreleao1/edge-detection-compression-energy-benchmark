#!/usr/bin/env python3
"""
Inferencia guiada por movimento: o detector roda so' nas regioes do quadro
que mudaram (camera fixa), em vez do quadro inteiro.

Pipeline por quadro:
  1. Subtracao de fundo (MOG2 do OpenCV, sombra = fundo) numa versao reduzida
     do quadro (lado maior <= 640) -> mascara do que mudou.
  2. Limpeza (abertura/dilatacao), componentes conectados, descarte de
     regioes pequenas, margem em volta e fusao de regioes que se sobrepoem.
  3. Memoria: as deteccoes do quadro anterior (conf >= 0.3) tambem viram
     regioes — sem isso, uma pessoa parada "some" do fundo e deixa de ser
     detectada.
  4. O detector roda em cada recorte, na MESMA escala do quadro inteiro
     (ex.: Caviar 384 -> 640 = 1,67x): a pessoa aparece do tamanho que o
     modelo viu no treino, e o custo cai com a area processada. Modelos ONNX
     (tamanho fixo) usam o menor tamanho disponivel que comporta o recorte
     (320/416/512/640, da varredura de resolucao).
  5. Caixas voltam para o quadro inteiro; NMS entre recortes sobrepostos.

Quadros sem regiao nenhuma nao rodam o detector (= quadro pulado). Nos
primeiros --warmup quadros (fundo ainda sendo aprendido) roda o quadro
inteiro, como faria um sistema real ao ligar.

Para cada (modelo, sequencia) roda tambem o modo "full" (quadro inteiro em
todos os quadros) com o MESMO calculo de mAP, para comparacao direta.

Avaliacao:
  - Caviar: sequencias extras do CAVIAR (sem vazamento), todos os quadros.
  - Oxford: o video inteiro e' processado em ordem (o fundo precisa de
    continuidade), o mAP e' medido nos quadros de teste anotados.

Saidas (experiments/top_models/roi/):
  roi_results.csv          mAP50, mAP50-95, fracao de quadros com inferencia,
                           recortes por quadro e fracao de pixels de entrada
                           do modelo em relacao ao quadro inteiro
  costs/<...>.json         por quadro, a lista de recortes inferidos como
                           [imgsz, area de entrada do modelo em pixels] —
                           entrada do roi_energy_model.py
  samples/<...>.jpg        quadros de exemplo com mascara, regioes e deteccoes

Uso:
    venv/bin/python roi_inference.py --models 01_yolov10n_kd --datasets caviar
    venv/bin/python roi_inference.py                       # top 5, duas bases
"""
import argparse
import csv
import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
TOP_DIR = ROOT / "experiments" / "top_models"
OUT_DIR = TOP_DIR / "roi"
CROSS_DIR = ROOT / "caviar" / "datasets" / "cross_video"
OXFORD_VIDEO = ROOT / "oxfordtower.mp4"
OXFORD_DS = ROOT / "oxford" / "datasets"
ONNX_SIZES = (320, 416, 512, 640)
FULL_IMGSZ = 640
CONF_EVAL = 0.001      # todas as deteccoes entram no mAP (como o val do ultralytics)
CONF_KEEP = 0.3        # deteccoes "confirmadas": memoria e exemplos (mesmo limiar da Raspberry)

logger = logging.getLogger("roi")


# ---------------------------------------------------------------------------
# Dados: quadros do video + ground truth nos quadros avaliados
# ---------------------------------------------------------------------------

def yolo_labels(path: Path, w: int, h: int) -> np.ndarray:
    rows = []
    if path.exists():
        for ln in path.read_text().split("\n"):
            if ln.strip():
                _, xc, yc, bw, bh = map(float, ln.split()[:5])
                rows.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
    return np.array(rows, dtype=np.float32).reshape(-1, 4)


def sequences(dataset: str) -> list[dict]:
    """[{name, video, gt: {quadro: caixas xyxy}}]"""
    if dataset == "caviar":
        out = []
        for d in sorted(CROSS_DIR.iterdir()):
            if not (d / "data.yaml").exists():
                continue
            labels = sorted((d / "labels" / "test").glob("*.txt"))
            out.append({"name": d.name, "video": next(d.glob("*.mpg")),
                        "gt": {int(p.stem): yolo_labels(p, 384, 288) for p in labels}})
        return out
    labels = sorted((OXFORD_DS / "labels" / "test").glob("*.txt"))
    return [{"name": "OxfordTownCentre_test", "video": OXFORD_VIDEO,
             "gt": {int(p.stem): yolo_labels(p, 1920, 1080) for p in labels}}]


# ---------------------------------------------------------------------------
# Regioes de movimento
# ---------------------------------------------------------------------------

def expand(box, frac: float, pad: float, w: int, h: int):
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * frac + pad, (y2 - y1) * frac + pad
    return [max(0.0, x1 - mx), max(0.0, y1 - my), min(float(w), x2 + mx), min(float(h), y2 + my)]


def merge_boxes(boxes: list) -> list:
    """Funde caixas que se sobrepoem ate nao sobrar sobreposicao."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed and len(boxes) > 1:
        changed = False
        out = []
        while boxes:
            a = boxes.pop()
            i = 0
            while i < len(boxes):
                b = boxes[i]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    a = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(i)
                    changed = True
                else:
                    i += 1
            out.append(a)
        boxes = out
    return boxes


class MotionRegions:
    def __init__(self, w: int, h: int, min_area_frac: float, margin: float):
        self.w, self.h = w, h
        self.s = min(1.0, 640 / max(w, h))            # mascara calculada em resolucao reduzida
        self.sw, self.sh = int(round(w * self.s)), int(round(h * self.s))
        self.bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.min_area = min_area_frac * self.sw * self.sh
        self.margin = margin
        self.mask = None

    def __call__(self, frame) -> list:
        small = cv2.resize(frame, (self.sw, self.sh), interpolation=cv2.INTER_AREA) if self.s < 1 else frame
        m = self.bg.apply(small)
        m = np.where(m == 255, 255, 0).astype(np.uint8)              # 127 = sombra -> fundo
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.k_open)
        m = cv2.dilate(m, self.k_dil, iterations=2)
        self.mask = m
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        boxes = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < self.min_area:
                continue
            box = [x / self.s, y / self.s, (x + bw) / self.s, (y + bh) / self.s]
            boxes.append(expand(box, self.margin, 4 / self.s, self.w, self.h))
        return boxes


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class Detector:
    """Envolve .pt (imgsz livre) ou ONNX (um arquivo por tamanho fixo)."""

    def __init__(self, entry: dict, dataset: str, device):
        from ultralytics import YOLO

        self.device = device
        files = entry["files"][dataset]
        self.onnx = entry["requantize_after_modifying"]
        if self.onnx:
            name = Path(files["reference"]).name
            paths = {640: TOP_DIR / files["reference"]}
            for s in ONNX_SIZES[:-1]:
                p = TOP_DIR / entry["dir"] / "variants" / f"imgsz{s}" / dataset / name
                if p.exists():
                    paths[s] = p
            self.models = {s: YOLO(str(p), task="detect") for s, p in sorted(paths.items())}
            self.sizes = sorted(self.models)
            self.device = "cpu"
        else:
            self.model = YOLO(str(TOP_DIR / files["source"]), task="detect")

    def input_size(self, cw: float, ch: float, scale: float) -> tuple[int, int]:
        """(lado do tamanho de inferencia, area de entrada do modelo em pixels)."""
        need = max(cw, ch) * scale
        if self.onnx:
            s = next((s for s in self.sizes if s >= need), self.sizes[-1])
            return s, s * s                                            # ONNX: entrada quadrada fixa
        s = int(min(FULL_IMGSZ, max(64, math.ceil(need / 32) * 32)))
        r = s / max(cw, ch)                                            # letterbox minimo (multiplo de 32)
        return s, math.ceil(cw * r / 32) * 32 * math.ceil(ch * r / 32) * 32

    def __call__(self, img, imgsz: int) -> np.ndarray:
        model = self.models[imgsz] if self.onnx else self.model
        r = model.predict(img, imgsz=imgsz, conf=CONF_EVAL, device=self.device, verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Execucao de uma sequencia
# ---------------------------------------------------------------------------

def run_sequence(seq: dict, det: Detector, mode: str, args, sample_dir: Path | None, tag: str) -> dict:
    # importados aqui: MotionRegions (subtracao de fundo) e' reutilizado por
    # roi_energy_model.py --bench-bg na Raspberry, que pode nao ter torchmetrics
    from torchmetrics.detection import MeanAveragePrecision
    from torchvision.ops import nms

    cap = cv2.VideoCapture(str(seq["video"]))
    w, h = int(cap.get(3)), int(cap.get(4))
    scale = FULL_IMGSZ / max(w, h)                                     # escala do quadro inteiro
    full_area = det.input_size(w, h, scale)[1]
    last_eval = max(seq["gt"])
    motion = MotionRegions(w, h, args.min_area, args.margin) if mode == "roi" else None
    metric = MeanAveragePrecision(iou_type="bbox")
    prev_keep = np.zeros((0, 4), dtype=np.float32)
    n = n_infer = n_crops = 0
    pixels = 0.0
    costs = []
    sample_at = set(args.sample_frames)

    while n <= last_eval:
        ok, frame = cap.read()
        if not ok:
            break
        if mode == "full" or n < args.warmup:
            regions = [[0.0, 0.0, float(w), float(h)]]
            motion and motion(frame)                                   # continua aprendendo o fundo
        else:
            regions = motion(frame) + [expand(b, args.margin, 4, w, h) for b in prev_keep]
            regions = merge_boxes(regions)

        preds, sizes = [], []
        for x1, y1, x2, y2 in regions:
            x1, y1, x2, y2 = int(x1), int(y1), int(math.ceil(x2)), int(math.ceil(y2))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            imgsz, area = det.input_size(x2 - x1, y2 - y1, scale)
            p = det(frame[y1:y2, x1:x2], imgsz)
            if len(p):
                p[:, [0, 2]] += x1
                p[:, [1, 3]] += y1
                preds.append(p)
            sizes.append([imgsz, area])
            pixels += area
        n_crops += len(sizes)
        n_infer += bool(sizes)
        costs.append(sizes)

        pred = np.concatenate(preds) if preds else np.zeros((0, 5), dtype=np.float32)
        if len(regions) > 1 and len(pred):
            keep = nms(torch.from_numpy(pred[:, :4]), torch.from_numpy(pred[:, 4]), 0.5).numpy()
            pred = pred[keep]
        prev_keep = pred[pred[:, 4] >= CONF_KEEP, :4]

        if n in seq["gt"]:
            gt = seq["gt"][n]
            metric.update([{"boxes": torch.from_numpy(pred[:, :4]), "scores": torch.from_numpy(pred[:, 4]),
                            "labels": torch.zeros(len(pred), dtype=torch.int64)}],
                          [{"boxes": torch.from_numpy(gt), "labels": torch.zeros(len(gt), dtype=torch.int64)}])
            if sample_dir is not None and n in sample_at:
                save_sample(sample_dir / f"{tag}_{n:05d}.jpg", frame, motion, regions, pred, gt)
        n += 1

    res = metric.compute()
    evaluated = [k for k in seq["gt"] if k < n]
    return {"frames": n, "eval_frames": len(evaluated), "mAP50": float(res["map_50"]),
            "mAP50-95": float(res["map"]), "infer_frac": n_infer / n, "crops_per_frame": n_crops / n,
            "pixel_frac": pixels / (full_area * n), "costs": costs}


def save_sample(path: Path, frame, motion, regions, pred, gt) -> None:
    img = frame.copy()
    if motion is not None and motion.mask is not None:
        m = cv2.resize(motion.mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        tint = img.copy()
        tint[m > 0] = (0.5 * tint[m > 0] + 0.5 * np.array([214, 120, 42])).astype(np.uint8)
        img = tint
    th = max(1, img.shape[1] // 400)
    for x1, y1, x2, y2 in regions:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), th, cv2.LINE_AA)
    for g in gt:
        cv2.rectangle(img, tuple(map(int, g[:2])), tuple(map(int, g[2:])), (0, 161, 237), th, cv2.LINE_AA)
    for p in pred[pred[:, 4] >= CONF_KEEP]:
        cv2.rectangle(img, tuple(map(int, p[:2])), tuple(map(int, p[2:4])), (72, 73, 227), th * 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None, help="pastas de experiments/top_models (padrao: todas)")
    ap.add_argument("--datasets", nargs="*", default=["caviar", "oxford"], choices=["caviar", "oxford"])
    ap.add_argument("--modes", nargs="*", default=["full", "roi"], choices=["full", "roi"])
    ap.add_argument("--warmup", type=int, default=50, help="quadros iniciais com quadro inteiro (padrao 50)")
    ap.add_argument("--min-area", type=float, default=0.0005, help="area minima da regiao (fracao do quadro)")
    ap.add_argument("--margin", type=float, default=0.15, help="margem em volta da regiao (fracao do lado)")
    ap.add_argument("--sample-frames", type=int, nargs="*", default=[300, 600, 900, 1200, 2600, 3200])
    ap.add_argument("--device", default="0")
    ap.add_argument("--tag", default="", help="sufixo no CSV (para comparar configuracoes)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    device = int(args.device) if args.device.isdigit() else args.device
    manifest = json.loads((TOP_DIR / "manifest.json").read_text())
    entries = [e for e in manifest["models"] if not args.models or e["dir"] in args.models]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"roi_results{args.tag}.csv"
    fields = ["dir", "label", "dataset", "sequence", "mode", "frames", "eval_frames", "mAP50", "mAP50-95",
              "infer_frac", "crops_per_frame", "pixel_frac", "warmup", "min_area", "margin"]
    rows = list(csv.DictReader(out_csv.open())) if out_csv.exists() else []
    done = {(r["dir"], r["sequence"], r["mode"]) for r in rows}

    for entry in entries:
        for ds in args.datasets:
            det = Detector(entry, ds, device)
            for seq in sequences(ds):
                for mode in args.modes:
                    if (entry["dir"], seq["name"], mode) in done:
                        continue
                    tag = f"{entry['dir']}_{seq['name']}_{mode}"
                    r = run_sequence(seq, det, mode, args, OUT_DIR / "samples", tag)
                    (OUT_DIR / "costs").mkdir(exist_ok=True)
                    (OUT_DIR / "costs" / f"{tag}{args.tag}.json").write_text(json.dumps(r.pop("costs")))
                    row = {"dir": entry["dir"], "label": entry["label"], "dataset": ds, "sequence": seq["name"],
                           "mode": mode, **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()},
                           "warmup": args.warmup, "min_area": args.min_area, "margin": args.margin}
                    rows.append(row)
                    logger.info(f"{entry['dir']:22s} {seq['name']:22s} {mode:4s} mAP50 {row['mAP50']:.3f} "
                                f"mAP50-95 {row['mAP50-95']:.3f} | quadros com inferencia {row['infer_frac']:.0%} "
                                f"| recortes/quadro {row['crops_per_frame']:.2f} | pixels {row['pixel_frac']:.0%}")
                    with out_csv.open("w", newline="") as fh:
                        wr = csv.DictWriter(fh, fieldnames=fields)
                        wr.writeheader()
                        wr.writerows(rows)
    logger.info(f"CSV: {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
