#!/usr/bin/env python3
"""
Figuras da avaliacao da Caviar sem vazamento (sequencias extras do CAVIAR),
para a dissertacao. Usa experiments/top_models/manifest.json e o CSV gerado
por evaluate_cross_video.py.

Saidas em --out (padrao figures/), textos em ingles:

  fig5_caviar_leakage.pdf/.png
      mAP50-95 e mAP50 de cada modelo do top 5: teste original da Caviar
      (quadros sorteados do proprio video de treino) contra as outras
      sequencias do mesmo corredor (media e faixa min-max). Mostra o quanto
      o numero original estava inflado.

  fig6_inference_<Sequencia>.pdf/.png
      Um quadro de cada sequencia: ground truth e a deteccao de cada modelo
      do top 5, com acertos (TP), falsos positivos (FP) e perdas (FN) a
      IoU 0.5.

  inference/<Sequencia>_<quadro>_<modelo>.png
      As mesmas imagens, uma por arquivo e sem moldura, para montar figuras
      a mao.

Uso:
    venv/bin/python plot_cross_video.py
    venv/bin/python plot_cross_video.py --frames TwoEnterShop1cor=500 OneShopOneWait2cor=700
"""
import argparse
import csv
import json
import logging
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from plot_dissertation_figures import (  # noqa: E402
    AXIS, INK, INK_2, MUTED, SERIES, SURFACE, _short, br, br_formatter, header, save, setup_style)

ROOT = Path(__file__).resolve().parent
TOP_DIR = ROOT / "experiments" / "top_models"
CROSS_DIR = ROOT / "caviar" / "datasets" / "cross_video"
LEAKY_SET = "teste atual (vazamento)"
# quadros escolhidos: oclusao/grupos (ThreePastShop1), pessoa parada (OneShopOneWait2),
# pessoas entrando/saindo de cena (TwoEnterShop1)
DEFAULT_FRAMES = {"ThreePastShop1cor": 600, "OneShopOneWait2cor": 700, "TwoEnterShop1cor": 500}
CONF = 0.3          # mesmo limiar usado na Raspberry
IOU_MATCH = 0.5

GT_BGR = (0, 161, 237)       # amarelo da paleta (#eda100) em BGR
PRED_BGR = (214, 120, 42)    # azul da paleta (#2a78d6) em BGR
FP_BGR = (72, 73, 227)       # vermelho da paleta (#e34948) em BGR


def load_results(csv_path: Path):
    rows = list(csv.DictReader(csv_path.open()))
    sets = list(dict.fromkeys(r["set"] for r in rows))
    by_rank = {}
    for r in rows:
        by_rank.setdefault(int(r["rank"]), {})[r["set"]] = {k: float(r[k]) for k in ("mAP50", "mAP50-95")}
    return by_rank, [s for s in sets if s != LEAKY_SET]


# ---------------------------------------------------------------------------
# Figura 5 — com x sem vazamento
# ---------------------------------------------------------------------------

def fig_leakage(manifest: dict, by_rank: dict, seqs: list, out: Path) -> None:
    models = manifest["models"]
    labels = [f"{m['rank']}. {_short(m)}" for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    blue, orange = SERIES[0], SERIES[1]
    for ax, key, title in ((axes[0], "mAP50-95", "mAP50-95"), (axes[1], "mAP50", "mAP50")):
        for i, m in enumerate(models):
            y = len(models) - 1 - i
            res = by_rank[m["rank"]]
            leak = res[LEAKY_SET][key]
            cross = [res[s][key] for s in seqs]
            mean = sum(cross) / len(cross)
            ax.plot([mean, leak], [y, y], color=AXIS, linewidth=1.2, zorder=1)
            ax.plot([min(cross), max(cross)], [y, y], color=blue, linewidth=3.0, alpha=0.25,
                    solid_capstyle="round", zorder=2)
            ax.scatter([leak], [y], s=40, marker="s", color=orange, edgecolors=SURFACE, linewidths=1.0, zorder=3)
            ax.scatter([mean], [y], s=46, color=blue, edgecolors=SURFACE, linewidths=1.0, zorder=4)
            ax.annotate(f"{mean - leak:+.2f}", ((mean + leak) / 2, y), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=6.5, color=INK_2)
        ax.set_title(title, loc="left")
        ax.set_xlim(0.6, 1.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_major_formatter(br_formatter(1))
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(models) - 0.4)
    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels(labels[::-1])
    axes[0].tick_params(axis="y", length=0)
    handles = [
        Line2D([], [], linestyle="", marker="s", color=orange, markeredgecolor=SURFACE, markersize=6,
               label="Original test split (frames of the training video)"),
        Line2D([], [], linestyle="", marker="o", color=blue, markeredgecolor=SURFACE, markersize=6,
               label=f"Other CAVIAR sequences (mean of {len(seqs)})"),
        Line2D([], [], color=blue, alpha=0.25, linewidth=4, label="Min–max across sequences"),
    ]
    header(fig, "CAVIAR: detection on unseen sequences vs. the original test split",
           "The original test frames sit 1–3 frames from training frames; other sequences of the same corridor "
           "measure real generalization. Numbers: change in the mean.", handles, ncol=3, w_pad=1.5)
    save(fig, out, "fig5_caviar_leakage")


# ---------------------------------------------------------------------------
# Figura 6 — exemplos de inferencia
# ---------------------------------------------------------------------------

def read_gt(seq: str, frame: int, w: int, h: int) -> list:
    out = []
    for ln in (CROSS_DIR / seq / "labels" / "test" / f"{frame:04d}.txt").read_text().split("\n"):
        if ln.strip():
            _, xc, yc, bw, bh = map(float, ln.split())
            out.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
    return out


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def match(preds: list, gts: list):
    """Casamento guloso por confianca a IoU 0.5 -> (indices TP das preds, n FP, n FN)."""
    used, tp = set(), set()
    for pi in sorted(range(len(preds)), key=lambda i: -preds[i][4]):
        best, bj = 0.0, None
        for j, g in enumerate(gts):
            if j not in used and (v := iou(preds[pi], g)) > best:
                best, bj = v, j
        if bj is not None and best >= IOU_MATCH:
            used.add(bj)
            tp.add(pi)
    return tp, len(preds) - len(tp), len(gts) - len(used)


def draw(img, gts, preds=None, tp=None):
    im = img.copy()
    for g in gts:
        cv2.rectangle(im, tuple(map(int, g[:2])), tuple(map(int, g[2:4])), GT_BGR, 1, cv2.LINE_AA)
    for i, p in enumerate(preds or []):
        color = PRED_BGR if i in tp else FP_BGR
        cv2.rectangle(im, tuple(map(int, p[:2])), tuple(map(int, p[2:4])), color, 2, cv2.LINE_AA)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def fig_inference(manifest: dict, frames: dict, out: Path) -> None:
    from ultralytics import YOLO

    (out / "inference").mkdir(parents=True, exist_ok=True)
    models = [(m, YOLO(str(TOP_DIR / (m["files"]["caviar"].get("reference") or m["files"]["caviar"]["source"])),
                       task="detect")) for m in manifest["models"]]
    for seq, frame in frames.items():
        img = cv2.imread(str(CROSS_DIR / seq / "images" / "test" / f"{frame:04d}.jpg"))
        h, w = img.shape[:2]
        gts = read_gt(seq, frame, w, h)
        tiles = [("Ground truth", f"{len(gts)} people", draw(img, gts))]
        cv2.imwrite(str(out / "inference" / f"{seq}_{frame:04d}_ground_truth.png"),
                    cv2.cvtColor(tiles[0][2], cv2.COLOR_RGB2BGR))
        for m, model in models:
            r = model.predict(img, conf=CONF, imgsz=640, verbose=False)[0]
            preds = [[*b.xyxy[0].tolist(), float(b.conf)] for b in r.boxes]
            tp, fp, fn = match(preds, gts)
            tile = draw(img, gts, preds, tp)
            tiles.append((f"{m['rank']}. {_short(m)}", f"TP {len(tp)} · FP {fp} · FN {fn}", tile))
            cv2.imwrite(str(out / "inference" / f"{seq}_{frame:04d}_{m['dir']}.png"),
                        cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

        fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.0))
        for ax, (title, sub, tile) in zip(axes.flat, tiles):
            ax.imshow(tile)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{title}\n", loc="left", fontsize=7.5)
            ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=6.8, color=INK_2, va="bottom")
        handles = [Line2D([], [], color="#eda100", linewidth=1.2, label="Ground truth"),
                   Line2D([], [], color=SERIES[0], linewidth=2, label=f"Correct detection (IoU ≥ {IOU_MATCH})"),
                   Line2D([], [], color="#e34948", linewidth=2, label="No matching ground-truth box (FP)")]
        header(fig, f"Detections on an unseen CAVIAR sequence — {seq}, frame {frame}",
               f"Caviar models of the top 5 at confidence ≥ {CONF}; FN = people missed. Red boxes are often real people "
               "the ground truth leaves unannotated (background).",
               handles, ncol=3, h_pad=2.2, w_pad=0.4)
        save(fig, out, f"fig6_inference_{seq}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--csv", type=Path, default=TOP_DIR / "cross_video_caviar.csv")
    ap.add_argument("--frames", nargs="*", default=None, metavar="SEQ=QUADRO",
                    help="quadros a ilustrar (padrao: " + ", ".join(f"{k}={v}" for k, v in DEFAULT_FRAMES.items()) + ")")
    args = ap.parse_args()

    logging.disable(logging.WARNING)
    setup_style()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((TOP_DIR / "manifest.json").read_text())
    by_rank, seqs = load_results(args.csv)
    frames = dict(DEFAULT_FRAMES)
    if args.frames:
        frames = {k: int(v) for k, v in (s.split("=") for s in args.frames)}

    print(f"Figuras em {args.out}/:")
    fig_leakage(manifest, by_rank, seqs, args.out)
    fig_inference(manifest, frames, args.out)
    print(f"  inference/ ({len(frames) * (len(manifest['models']) + 1)} imagens individuais)")


if __name__ == "__main__":
    main()
