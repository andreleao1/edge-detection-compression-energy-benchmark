#!/usr/bin/env python3
"""
Gera as figuras da dissertacao sobre o comportamento das tecnicas de
compressao na Raspberry (energia x deteccao) e sobre a selecao dos modelos
de melhor custo/beneficio.

Figuras (PDF vetorial + PNG 300 dpi em --out, padrao figures/):

  fig1_tecnicas_por_taxa_<base>.pdf
      Uma figura por base. Colunas = tecnicas com taxa de poda (Poda
      estruturada, Poda + Quantizacao, Poda + KD, Poda + KD + Quantizacao);
      linhas = J/imagem (escala log) e mAP50-95; uma curva por arquitetura,
      da taxa 0% ate 80%. O ponto 0% e' o modelo sem poda da mesma tecnica:
      baseline FP32 para Poda estruturada e Poda + KD; o modelo so'
      quantizado para as tecnicas com quantizacao.

  fig2_tradeoff_tecnicas.pdf
      Um painel por tecnica: variacao de energia (J/imagem, %) contra a
      variacao de mAP50-95 (pontos percentuais), ambas em relacao ao
      baseline da mesma arquitetura na mesma base. Cada ponto e' um modelo;
      cor/forma = base. Quadrante superior esquerdo = economiza energia sem
      perder deteccao. A mediana de cada base aparece no painel.

  fig3_selecao_pareto.pdf
      Explica a selecao do top 5 (select_top_models.py): cada modelo no
      plano J/imagem na pior base x mAP50-95 na pior base; modelos que nao
      passam no piso de mAP50, candidatos dominados, a fronteira de Pareto
      e os 5 selecionados (rotulados).

  fig4_caviar_vs_oxford.pdf
      mAP50-95 e J/imagem da Caviar contra a Oxford, com a diagonal y = x:
      justifica usar a pior base (a Oxford e' sempre a mais dificil).

Tambem grava dados_figuras.csv com todos os valores usados.

Os textos das figuras (titulos, eixos, legendas) sao em ingles, com ponto decimal.

Uso:
    venv/bin/python plot_dissertation_figures.py
    venv/bin/python plot_dissertation_figures.py --out figures --min-map50 0.8
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from rasp_sheet_map import DEFAULT_PG, ROOT, SHEET_XLSX  # noqa: E402
from select_top_models import DATASETS, load_candidates, pareto  # noqa: E402

# ---------------------------------------------------------------------------
# Estilo (paleta categorica validada: ordem fixa, nunca reciclada)
# ---------------------------------------------------------------------------
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]

ARCHS = [  # ordem fixa -> cor e marcador fixos por arquitetura em todas as figuras
    ("yolov10n", "YOLOv10n", "o"),
    ("yolov10m", "YOLOv10m", "s"),
    ("yolov10l", "YOLOv10l", "^"),
    ("yolov10x", "YOLOv10x", "D"),
    ("faster_rcnn", "Faster R-CNN", "v"),
    ("retinanet", "RetinaNet", "P"),
    ("mobilenet", "MobileNet", "X"),
]
ARCH_STYLE = {a: (label, SERIES[i], mk) for i, (a, label, mk) in enumerate(ARCHS)}

DATASET_LABEL = {"caviar": "Caviar", "oxford": "Oxford Town Centre"}
DATASET_STYLE = {"caviar": (SERIES[0], "o"), "oxford": (SERIES[1], "s")}

TECH_LABEL = {
    "baseline": "Baseline",
    "pruned_structured": "Structured pruning",
    "quantized": "Quantization (INT8)",
    "pruned_structured_quantized": "Pruning + Quantization",
    "pruned_unstructured_quantized": "Pruning + Quantization",
    "pruned_structured_kd": "Pruning + KD",
    "pruned_structured_kd_quantized": "Pruning + KD + Quantization",
    "kd": "KD",
}
RATE_TECHS = ["Structured pruning", "Pruning + Quantization", "Pruning + KD", "Pruning + KD + Quantization"]
# tecnica -> variante usada no ponto 0% da curva
RATE_ZERO = {"Structured pruning": "baseline", "Pruning + KD": "baseline",
             "Pruning + Quantization": "quantized", "Pruning + KD + Quantization": "quantized"}
TRADEOFF_TECHS = ["Structured pruning", "Quantization (INT8)", "Pruning + Quantization",
                  "Pruning + KD", "Pruning + KD + Quantization", "KD"]


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
    })
    for side in ("top", "right"):
        plt.rcParams[f"axes.spines.{side}"] = False


def br(x: float, decimals: int = 2) -> str:
    """Numero formatado para as figuras (ponto decimal: textos em ingles)."""
    return f"{x:.{decimals}f}"


def br_formatter(decimals: int):
    return FuncFormatter(lambda v, _: br(v, decimals))


def log_formatter():
    def fmt(v, _):
        if v >= 1:
            return f"{v:.0f}"
        return br(v, 1)
    return FuncFormatter(fmt)


def header(fig, title: str, subtitle: str | None = None, handles: list | None = None,
           ncol: int = 1, **tight) -> None:
    """Titulo, subtitulo e legenda no topo, em faixas fixas (em polegadas),
    e os eixos ajustados abaixo delas — nada se sobrepoe."""
    h = fig.get_figheight()
    y = 1 - 0.04 / h
    fig.text(0.01, y, title, ha="left", va="top", fontsize=9.5, fontweight="bold", color=INK)
    y -= 0.22 / h
    if subtitle:
        fig.text(0.01, y, subtitle, ha="left", va="top", fontsize=7, color=INK_2, linespacing=1.3)
        y -= 0.17 * (subtitle.count("\n") + 1) / h
    if handles:
        fig.legend(handles=handles, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y),
                   handlelength=1.6, columnspacing=1.2, borderaxespad=0)
        y -= 0.24 / h
    fig.tight_layout(rect=(0, 0, 1, y - 0.04 / h), **tight)


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

def build_rows(models: dict) -> list[dict]:
    """Uma linha por (modelo, base) com tecnica, taxa e variacoes vs baseline."""
    rows = []
    for m in models.values():
        if not all(d in m for d in DATASETS):
            continue
        variant = m["variant"]
        rate_match = re.search(r"_p(\d+)$", variant)
        family = re.sub(r"_p\d+$", "", variant)
        for d in DATASETS:
            rows.append({
                "name": m["name"], "arch": m["arch"], "dataset": d, "variant": variant,
                "technique": TECH_LABEL[family], "rate": int(rate_match.group(1)) if rate_match else 0,
                **{k: m[d][k] for k in ("energy", "latency", "watt", "cpu", "mem_raw", "map50", "map5095", "measured")},
            })
    base = {(r["arch"], r["dataset"]): r for r in rows if r["variant"] == "baseline"}
    for r in rows:
        b = base.get((r["arch"], r["dataset"]))
        r["d_energy_pct"] = 100 * (r["energy"] - b["energy"]) / b["energy"] if b else None
        r["d_map5095_pp"] = 100 * (r["map5095"] - b["map5095"]) if b and r["map5095"] is not None else None
    return rows


# ---------------------------------------------------------------------------
# Figura 1 — cada tecnica ao longo da taxa de poda
# ---------------------------------------------------------------------------

def fig_rates(rows: list[dict], dataset: str, out: Path) -> None:
    sub = [r for r in rows if r["dataset"] == dataset]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.8), sharex=True, sharey="row")
    for col, tech in enumerate(RATE_TECHS):
        ax_e, ax_q = axes[0, col], axes[1, col]
        # titulos longos em duas linhas: com 4 paineis lado a lado eles se encostariam
        ax_e.set_title(tech.replace(" + Quantization", " +\nQuantization"), loc="left")
        for arch, (label, color, mk) in ARCH_STYLE.items():
            pts = [r for r in sub if r["arch"] == arch and r["technique"] == tech and r["rate"] > 0]
            zero = [r for r in sub if r["arch"] == arch and r["variant"] == RATE_ZERO[tech]]
            if not pts:
                continue
            pts = sorted(zero + pts, key=lambda r: r["rate"])
            x = [r["rate"] for r in pts]
            for ax, key in ((ax_e, "energy"), (ax_q, "map5095")):
                ax.plot(x, [r[key] for r in pts], color=color, marker=mk, markersize=4.2,
                        markeredgecolor=SURFACE, markeredgewidth=0.6, label=label)
        ax_e.set_yscale("log")
        ax_q.set_ylim(0, 1)
        ax_q.set_xticks([0, 20, 40, 60, 80])
        ax_q.set_xlabel("Pruning rate (%)")
    axes[0, 0].set_ylabel("Energy (J/image, log)")
    axes[1, 0].set_ylabel("mAP50-95")
    axes[0, 0].yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axes[0, 0].yaxis.set_major_formatter(log_formatter())
    axes[0, 0].yaxis.set_minor_formatter(NullFormatter())
    axes[1, 0].yaxis.set_major_formatter(br_formatter(1))

    handles = [Line2D([], [], color=c, marker=mk, markersize=5, markeredgecolor=SURFACE, label=lb)
               for lb, c, mk in ARCH_STYLE.values()]
    header(fig, f"{DATASET_LABEL[dataset]} — energy and detection by pruning rate",
           "Rate 0%: FP32 baseline (Structured pruning, Pruning + KD) or the quantized-only model "
           "(techniques with quantization).", handles, ncol=7, h_pad=0.8, w_pad=0.6)
    save(fig, out, f"fig1_tecnicas_por_taxa_{dataset}")


# ---------------------------------------------------------------------------
# Figura 2 — trade-off energia x deteccao por tecnica
# ---------------------------------------------------------------------------

def fig_tradeoff(rows: list[dict], out: Path) -> None:
    pts = [r for r in rows if r["technique"] in TRADEOFF_TECHS and r["d_energy_pct"] is not None
           and r["d_map5095_pp"] is not None]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.0), sharex=True, sharey=True)
    for ax, tech in zip(axes.flat, TRADEOFF_TECHS):
        ax.set_title(tech, loc="left")
        ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
        ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
        lines = []
        for d in DATASETS:
            color, mk = DATASET_STYLE[d]
            sel = [r for r in pts if r["technique"] == tech and r["dataset"] == d]
            ax.scatter([r["d_energy_pct"] for r in sel], [r["d_map5095_pp"] for r in sel], s=22, marker=mk,
                       color=color, edgecolors=SURFACE, linewidths=0.6, zorder=3, alpha=0.95)
            if sel:
                xs = sorted(r["d_energy_pct"] for r in sel)
                ys = sorted(r["d_map5095_pp"] for r in sel)
                med = lambda v: (v[len(v) // 2] + v[(len(v) - 1) // 2]) / 2  # noqa: E731
                lines.append(f"{DATASET_LABEL[d].split()[0]}: {br(med(xs), 0)}% · {br(med(ys), 1)} pp")
        ax.text(0.97, 0.04, "median\n" + "\n".join(lines), transform=ax.transAxes, fontsize=6.5,
                color=INK_2, va="bottom", ha="right", linespacing=1.3, zorder=5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE, edgecolor=GRID, linewidth=0.6))
    for ax in axes[1]:
        ax.set_xlabel("Δ energy vs. baseline (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ mAP50-95 (pp)")
    axes[0, 0].xaxis.set_major_formatter(br_formatter(0))
    axes[0, 0].yaxis.set_major_formatter(br_formatter(0))
    handles = [Line2D([], [], linestyle="", marker=DATASET_STYLE[d][1], color=DATASET_STYLE[d][0],
                      markersize=5.5, markeredgecolor=SURFACE, label=DATASET_LABEL[d]) for d in DATASETS]
    header(fig, "Energy and detection change of each technique relative to the baseline",
           "Each point is a model (architecture × rate), compared with the baseline of the same architecture "
           "and dataset. Up and left: saves energy without losing detection.", handles, ncol=2, h_pad=0.8, w_pad=0.6)
    save(fig, out, "fig2_tradeoff_tecnicas")


# ---------------------------------------------------------------------------
# Figura 3 — selecao por fronteira de Pareto
# ---------------------------------------------------------------------------

def _short(m: dict) -> str:
    tech = TECH_LABEL[re.sub(r"_p\d+$", "", m["variant"])]
    tech = {"Quantization (INT8)": "quantized"}.get(tech, tech)
    return f"{ARCH_STYLE[m['arch']][0]} {tech}"


def fig_pareto(models_sel: dict, min_map50: float, out: Path) -> list:
    reps, _ = pareto(models_sel, min_map50, "map5095")
    for m in models_sel.values():  # metricas de pior base para TODOS (inclusive os filtrados)
        if all(d in m for d in DATASETS) and all(m[d]["map5095"] is not None for d in DATASETS):
            m["q"] = min(m[d]["map5095"] for d in DATASETS)
            m["e"] = max(m[d]["energy"] for d in DATASETS)
            m["passes"] = min(m[d]["map50"] for d in DATASETS) >= min_map50
    allm = [m for m in models_sel.values() if "q" in m]
    cands = [m for m in allm if m["passes"]]
    front = sorted([x for x in cands if not any(
        y["q"] >= x["q"] and y["e"] <= x["e"] and (y["q"] > x["q"] or y["e"] < x["e"]) for y in cands)],
        key=lambda m: m["e"])
    selected = sorted(reps, key=lambda m: m["e"])[:5]
    sel_ids = {id(m) for m in selected}
    front_ids = {id(m) for m in front}

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    blue = SERIES[0]
    filt = [m for m in allm if not m["passes"]]
    dom = [m for m in cands if id(m) not in front_ids]
    eq = [m for m in front if id(m) not in sel_ids]
    ax.scatter([m["e"] for m in filt], [m["q"] for m in filt], s=18, marker="x", color=AXIS,
               linewidths=1.0, zorder=2, label=f"Below floor (mAP50 < {br(min_map50, 1)} on either dataset)")
    ax.scatter([m["e"] for m in dom], [m["q"] for m in dom], s=22, facecolors="none", edgecolors=MUTED,
               linewidths=1.0, zorder=3, label="Dominated candidate")
    # fronteira em escada: para cada orcamento de energia, a melhor deteccao possivel
    xs, ys = [front[0]["e"]], [front[0]["q"]]
    for m in front[1:]:
        xs += [m["e"], m["e"]]
        ys += [ys[-1], m["q"]]
    xmax = max(m["e"] for m in allm) * 1.15
    xs.append(xmax)
    ys.append(ys[-1])
    ax.fill_between(xs, ys, 0, color=blue, alpha=0.07, zorder=1, linewidth=0)
    ax.plot(xs, ys, color=blue, linewidth=1.5, zorder=4, label="Pareto front")
    ax.scatter([m["e"] for m in eq], [m["q"] for m in eq], s=26, facecolors=SURFACE, edgecolors=blue,
               linewidths=1.2, zorder=5, label="On the front, grouped equivalent variant")
    ax.scatter([m["e"] for m in selected], [m["q"] for m in selected], s=58, color=blue,
               edgecolors=SURFACE, linewidths=1.4, zorder=6, label="Selected (top 5)")

    # posicao do texto (J/imagem, mAP50-95) de cada rotulo, nas areas vazias acima da fronteira
    label_pos = [(2.3, 0.53), (5.6, 0.585), (8.2, 0.648), (26.0, 0.672), (80.0, 0.665)]
    for i, m in enumerate(selected):
        tx, ty = label_pos[i] if i < len(label_pos) else (m["e"] * 1.3, m["q"] + 0.04)
        ax.annotate(f"{i + 1}. {_short(m)}\n{br(m['e'], 1)} J · {br(m['q'], 3)}",
                    (m["e"], m["q"]), xytext=(tx, ty), textcoords="data", ha="center",
                    va="center", fontsize=7, color=INK, linespacing=1.15, zorder=7,
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6, shrinkA=7, shrinkB=5))

    ax.set_xscale("log")
    ax.set_xlim(min(m["e"] for m in allm) * 0.8, xmax)
    ax.set_ylim(0, 0.71)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(log_formatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(br_formatter(1))
    ax.set_xlabel("Energy on the worst dataset (J/image, log scale)  →  more savings to the left")
    ax.set_ylabel("mAP50-95 on the worst dataset  →  better detection up")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, handletextpad=0.5,
              columnspacing=1.4)
    header(fig, "Selection of the best cost/benefit models on both datasets",
           f"Each point is a model; worst dataset = lowest mAP50-95 and highest J/image between Caviar and Oxford. "
           f"{len(cands)} of {len(allm)} models pass the mAP50 floor.")
    save(fig, out, "fig3_selecao_pareto")
    return selected


# ---------------------------------------------------------------------------
# Figura 4 — Caviar x Oxford
# ---------------------------------------------------------------------------

def cross_video_map5095() -> dict:
    """{nome do modelo: mAP50-95 medio nas sequencias extras do CAVIAR} a partir
    do CSV de evaluate_cross_video.py (vazio se ainda nao foi gerado)."""
    import json
    top = ROOT / "experiments" / "top_models"
    csv_path, man_path = top / "cross_video_caviar.csv", top / "manifest.json"
    if not (csv_path.exists() and man_path.exists()):
        return {}
    name_by_rank = {e["rank"]: e["name"] for e in json.loads(man_path.read_text())["models"]}
    vals: dict = {}
    for r in csv.DictReader(csv_path.open()):
        if r["set"] != "teste atual (vazamento)":
            vals.setdefault(name_by_rank[int(r["rank"])], []).append(float(r["mAP50-95"]))
    return {k: sum(v) / len(v) for k, v in vals.items()}


def fig_bases(models_sel: dict, selected: list, out: Path) -> None:
    allm = [m for m in models_sel.values() if "q" in m]
    sel_ids = {id(m) for m in selected}
    cross = cross_video_map5095()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.9))
    for ax, key, title in ((a1, "map5095", "mAP50-95"), (a2, "energy", "Energy (J/image)")):
        xs = [m["caviar"][key] for m in allm]
        ys = [m["oxford"][key] for m in allm]
        lo, hi = min(xs + ys), max(xs + ys)
        if key == "energy":
            ax.set_xscale("log")
            ax.set_yscale("log")
            lo, hi = lo * 0.8, hi * 1.25
            for a in (ax.xaxis, ax.yaxis):
                a.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
                a.set_major_formatter(log_formatter())
                a.set_minor_formatter(NullFormatter())
        else:
            lo, hi = 0, 1
            ax.xaxis.set_major_formatter(br_formatter(1))
            ax.yaxis.set_major_formatter(br_formatter(1))
        ax.plot([lo, hi], [lo, hi], color=AXIS, linewidth=0.9, zorder=1)
        ax.text(0.97, 0.97 if key == "map5095" else 0.03, "y = x",
                transform=ax.transAxes, ha="right", va="top" if key == "map5095" else "bottom",
                fontsize=6.5, color=MUTED)
        rest = [m for m in allm if id(m) not in sel_ids]
        ax.scatter([m["caviar"][key] for m in rest], [m["oxford"][key] for m in rest], s=18,
                   facecolors="none", edgecolors=SERIES[0], linewidths=0.9, zorder=2, alpha=0.8)
        ax.scatter([m["caviar"][key] for m in selected], [m["oxford"][key] for m in selected], s=34,
                   color=SERIES[0], edgecolors=SURFACE, linewidths=1.0, zorder=3)
        if key == "map5095" and cross:
            # top 5: posicao com a Caviar avaliada em sequencias nao vistas (sem vazamento)
            pts = [(m["caviar"][key], cross[m["name"]], m["oxford"][key]) for m in selected if m["name"] in cross]
            for x_old, x_new, y in pts:
                ax.annotate("", (x_new, y), (x_old, y), zorder=3,
                            arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=0.7, shrinkA=4, shrinkB=4,
                                            mutation_scale=7))
            ax.scatter([p[1] for p in pts], [p[2] for p in pts], s=34, marker="s", color=SERIES[1],
                       edgecolors=SURFACE, linewidths=1.0, zorder=4)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Caviar")
        ax.set_ylabel("Oxford Town Centre")
    below = sum(1 for m in allm if m["oxford"]["map5095"] < m["caviar"]["map5095"])
    handles = [Line2D([], [], linestyle="", marker="o", markerfacecolor="none", markeredgecolor=SERIES[0],
                      markersize=5, label="Model"),
               Line2D([], [], linestyle="", marker="o", color=SERIES[0], markeredgecolor=SURFACE,
                      markersize=6, label="Selected (top 5)")]
    subtitle = (f"{below} of {len(allm)} models have a lower mAP50-95 on Oxford (below the diagonal); "
                "this is why the selection uses the worst dataset.")
    if cross:
        handles.append(Line2D([], [], linestyle="", marker="s", color=SERIES[1], markeredgecolor=SURFACE,
                              markersize=6, label="Top 5, Caviar on unseen sequences"))
        subtitle += ("\nArrows: the top 5 moved to Caviar measured on other CAVIAR sequences (no leakage) "
                     "— still below the diagonal.")
    header(fig, "Oxford Town Centre is the harder dataset", subtitle, handles, ncol=len(handles), w_pad=2.0)
    save(fig, out, "fig4_caviar_vs_oxford")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--min-map50", type=float, default=0.8, help="piso de mAP50 da selecao (padrao 0.8)")
    ap.add_argument("--xlsx", type=Path, default=SHEET_XLSX)
    args = ap.parse_args()

    setup_style()
    args.out.mkdir(parents=True, exist_ok=True)
    # tecnicas: inclui YOLO "Poda + Quantizacao" (= pruned_unstructured_quantized na planilha)
    models_all = load_candidates(args.xlsx, DEFAULT_PG, yolo_pq_variant="pruned_unstructured_quantized")
    # selecao: exatamente os mesmos dados de select_top_models.py
    models_sel = load_candidates(args.xlsx, DEFAULT_PG)
    rows = build_rows(models_all)

    print(f"Figuras em {args.out}/ ({len({r['name'] for r in rows})} modelos medidos nas duas bases):")
    for d in DATASETS:
        fig_rates(rows, d, args.out)
    fig_tradeoff(rows, args.out)
    selected = fig_pareto(models_sel, args.min_map50, args.out)
    fig_bases(models_sel, selected, args.out)

    keys = ["name", "arch", "dataset", "technique", "rate", "variant", "energy", "latency", "watt", "cpu",
            "mem_raw", "map50", "map5095", "d_energy_pct", "d_map5095_pp", "measured"]
    with (args.out / "dados_figuras.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(sorted(rows, key=lambda r: (r["dataset"], r["arch"], r["technique"], r["rate"])))
    print("  dados_figuras.csv")


if __name__ == "__main__":
    main()
