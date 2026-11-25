#!/usr/bin/env python3
"""
Energia x deteccao por resolucao de inferencia (imgsz) dos 5 modelos
selecionados, nas duas bases.

Junta:
  - mAP: experiments/top_models/resolution_sweep_map.csv (resolution_sweep.py;
    CAVIAR = media das tres sequencias de teste, Oxford = conjunto de teste);
  - energia: banco PostgreSQL (main_models.py na Raspberry Pi). imgsz 640 = a
    medicao original do modelo (<arch>__<variante>__<stem>); as demais =
    <arch>__<variante>_imgsz<N>__<stem>.

Saidas:
  experiments/top_models/resolution_sweep_energy.csv  tabela completa
  figures/fig7_resolution_energy_map.pdf/.png          um painel por base:
      J/imagem (log) x mAP50-95, uma linha por modelo de 320 a 640 (marcador
      maior = resolucao maior), fronteira de Pareto da base em tracejado.
Tambem imprime as linhas da tabela LaTeX e quais pontos ficam na fronteira.

Uso:
    venv/bin/python plot_resolution_energy.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from plot_dissertation_figures import (AXIS, GRID, INK, INK_2, MUTED, SERIES, SURFACE,  # noqa: E402
                                       header, log_formatter, save, setup_style)
from rasp_sheet_map import DEFAULT_PG, ROOT, fetch_db_results  # noqa: E402

TOP = ROOT / "experiments" / "top_models"
SIZES = [320, 416, 512, 640]
DB_DATASET = {"caviar": "CAVIAR", "oxford": "OXFORD_TOWER"}
DS_LABEL = {"caviar": "CAVIAR (3 test sequences)", "oxford": "Oxford Town Centre"}
SHORT = {"01_yolov10n_kd": "YOLOv10n KD", "02_yolov10m_quantized": "YOLOv10m INT8",
         "03_yolov10l_quantized": "YOLOv10l INT8", "04_yolov10m_kd": "YOLOv10m KD",
         "05_yolov10l_kd": "YOLOv10l KD"}
MSIZE = {320: 3.5, 416: 5, 512: 6.5, 640: 8}


def db_name(entry: dict, imgsz: int) -> str:
    arch, variant, stem = entry["name"].split("__")
    return entry["name"] if imgsz == 640 else f"{arch}__{variant}_imgsz{imgsz}__{stem}"


def load() -> list[dict]:
    manifest = json.loads((TOP / "manifest.json").read_text())
    maps = defaultdict(list)
    for r in csv.DictReader((TOP / "resolution_sweep_map.csv").open()):
        maps[(r["dir"], r["dataset"], int(r["imgsz"]))].append(r)
    db = fetch_db_results(DEFAULT_PG)
    rows = []
    for e in manifest["models"]:
        for ds in ("caviar", "oxford"):
            for s in SIZES:
                m = maps[(e["dir"], ds, s)]
                res = db.get((db_name(e, s), DB_DATASET[ds]))
                if not m or res is None:
                    raise SystemExit(f"faltam dados: {e['dir']} {ds} {s} (map={len(m)}, energia={res is not None})")
                n = res["total_imagens"] or res["num_images"]
                rows.append({
                    "rank": e["rank"], "dir": e["dir"], "model": SHORT[e["dir"]], "dataset": ds, "imgsz": s,
                    "mAP50": sum(float(x["mAP50"]) for x in m) / len(m),
                    "mAP50-95": sum(float(x["mAP50-95"]) for x in m) / len(m),
                    "J_per_image": res["joules_per_image"] or res["joules_por_imagem"],
                    "ms_per_image": res["duracao_total"] * 1000 / n,
                    "avg_watt": res["avg_watt"], "images": n,
                    "measured": res["data_execucao"].date().isoformat(),
                })
    return rows


def pareto_front(points: list[dict]) -> list[dict]:
    """Nao dominados: nenhum outro gasta <= energia com >= mAP (um estritamente)."""
    front = []
    for p in points:
        if not any(q is not p and q["J_per_image"] <= p["J_per_image"] and q["mAP50-95"] >= p["mAP50-95"]
                   and (q["J_per_image"] < p["J_per_image"] or q["mAP50-95"] > p["mAP50-95"]) for q in points):
            front.append(p)
    return sorted(front, key=lambda p: p["J_per_image"])


def figure(rows: list[dict], out: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=False)
    dirs = sorted({r["dir"] for r in rows})
    color = {d: SERIES[i] for i, d in enumerate(dirs)}
    fronts = {}
    for ax, ds in zip(axes, ("caviar", "oxford")):
        pts = [r for r in rows if r["dataset"] == ds]
        front = pareto_front(pts)
        fronts[ds] = front
        # fronteira em degraus (recessiva, atras das series)
        xs, ys = [], []
        for i, p in enumerate(front):
            if i:
                xs.append(p["J_per_image"]); ys.append(front[i - 1]["mAP50-95"])
            xs.append(p["J_per_image"]); ys.append(p["mAP50-95"])
        ax.plot(xs, ys, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        for d in dirs:
            ser = sorted((r for r in pts if r["dir"] == d), key=lambda r: r["imgsz"])
            ax.plot([r["J_per_image"] for r in ser], [r["mAP50-95"] for r in ser], color=color[d], lw=1.5, zorder=2)
            for r in ser:
                ax.plot(r["J_per_image"], r["mAP50-95"], marker="o", ms=MSIZE[r["imgsz"]], color=color[d],
                        mec=SURFACE, mew=1.2, zorder=3)
        # rotulos de resolucao so' no modelo mais leve (o padrao se repete nos outros)
        for r in (r for r in pts if r["dir"] == dirs[0]):
            ax.annotate(str(r["imgsz"]), (r["J_per_image"], r["mAP50-95"]), xytext=(6, -2) if r["imgsz"] == 320 else (-5, 3),
                        textcoords="offset points", ha="left" if r["imgsz"] == 320 else "right",
                        va="top" if r["imgsz"] == 320 else "bottom", fontsize=6.3, color=INK_2)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(log_formatter())
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:.0f}" if round(v) in (2, 3, 5, 20, 30) else ""))
        ax.tick_params(axis="x", which="minor", labelsize=6)
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(AXIS)
        ax.set_title(DS_LABEL[ds], fontsize=8.5, color=INK, loc="left", fontweight="bold")
        ax.set_xlabel("Energy per image on the Raspberry Pi (J, log scale)", fontsize=7.5, color=INK_2)
        lo = min(r["mAP50-95"] for r in pts)
        ax.set_ylim(lo - 0.04, max(r["mAP50-95"] for r in pts) + 0.02)
    axes[0].set_ylabel("mAP50-95", fontsize=7.5, color=INK_2)
    handles = [Line2D([], [], color=color[d], marker="o", ms=5, mec=SURFACE, lw=1.5, label=SHORT[d]) for d in dirs]
    handles.append(Line2D([], [], color=MUTED, lw=1.0, ls=(0, (4, 3)), label="Pareto front"))
    header(fig, "Inference resolution: energy vs. detection",
           "Each line is one selected model at input size 320 → 416 → 512 → 640 (larger marker = larger size).",
           handles=handles, ncol=6)
    save(fig, out, "fig7_resolution_energy_map")
    return fronts


def main():
    setup_style()
    rows = load()
    out_csv = TOP / "resolution_sweep_energy.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  {out_csv.relative_to(ROOT)}")
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fronts = figure(rows, out)

    print("\nLinhas LaTeX (J/imagem; 640, 512, 416, 320):")
    for d in sorted({r["dir"] for r in rows}):
        for ds in ("caviar", "oxford"):
            v = {r["imgsz"]: r["J_per_image"] for r in rows if r["dir"] == d and r["dataset"] == ds}
            print(f"  {SHORT[d]:14s} {ds:6s} & " + " & ".join(f"{v[s]:.2f}".replace(".", ",") for s in (640, 512, 416, 320)))
    for ds, front in fronts.items():
        print(f"\nFronteira de Pareto — {ds}:")
        for p in front:
            print(f"  {p['model']:14s} {p['imgsz']}: {p['J_per_image']:.2f} J/img, mAP50-95 {p['mAP50-95']:.3f}")


if __name__ == "__main__":
    main()
