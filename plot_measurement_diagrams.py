#!/usr/bin/env python3
"""
Diagramas do ambiente de medicao para a dissertacao (textos em ingles,
PDF vetorial + PNG em figures/):

  fig_measurement_wiring.pdf   montagem eletrica: fonte -> INA219 (em serie
                               na linha de +5 V) -> Raspberry Pi; INA219 e
                               DS18B20 lidos pelo Arduino Uno; Arduino -> PC.
  fig_measurement_dataflow.pdf fluxo de dados: exportadores -> Prometheus ->
                               main_models.py -> PostgreSQL.

Valores conferidos no codigo: shunt de 0,1 ohm e leitura a cada 5 s
(arduino/ina219_without_ethernet), DS18B20 no pino digital 4, exportador do
Arduino na porta 8080 e rasp_monitor na 9100, coleta do Prometheus a cada 5 s.

Uso:
    venv/bin/python plot_measurement_diagrams.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path("figures")
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURF, FRAME = "#ffffff", "#c3c2b7"
RED, BLUE, ORANGE, GREEN = "#e34948", "#2a78d6", "#eb6834", "#008300"
BOX_FILL = "#f6f6f3"


def setup():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42,
                         "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.04})


def box(ax, x, y, w, h, title, sub=None, fill=BOX_FILL, edge=INK_2, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2))
    cy = y + h / 2 + (1.6 if sub else 0)
    ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK, zorder=3)
    if sub:
        ax.text(x + w / 2, cy - 2.2, sub, ha="center", va="top", fontsize=6.8, color=INK_2, zorder=3,
                linespacing=1.25)


def frame(ax, x, y, w, h, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                facecolor="none", edgecolor=FRAME, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1))
    ax.text(x + 1.2, y + h - 1.2, label, ha="left", va="top", fontsize=7.5, fontweight="bold", color=MUTED)


def wire(ax, pts, color, lw=1.8, label=None, lpos=None, ha="center", va="bottom", dashed=False, arrow=False):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle="round", solid_joinstyle="round",
            linestyle=(0, (4, 2.5)) if dashed else "-", zorder=1.5)
    if arrow:
        ax.annotate("", pts[-1], pts[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                                          mutation_scale=9, shrinkA=0, shrinkB=0), zorder=1.6)
    if label:
        ax.text(*lpos, label, ha=ha, va=va, fontsize=6.8, color=INK_2, zorder=3)


def dot(ax, x, y, color):
    ax.plot([x], [y], marker="o", markersize=3.2, color=color, zorder=3)


def pin(ax, x, y, name, va="bottom"):
    ax.text(x, y, name, ha="center", va=va, fontsize=6, color=INK_2, zorder=3)


# ---------------------------------------------------------------------------
# 1. Montagem eletrica
# ---------------------------------------------------------------------------

def wiring():
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 57)
    ax.axis("off")

    box(ax, 1, 33, 17, 13, "Power supply", "USB-C, 5 V / 3 A")
    box(ax, 31, 36, 23, 14, "INA219", "current / voltage sensor\n0.1 Ω shunt on the board")
    box(ax, 70, 31, 22, 17, "Raspberry Pi 4", "Model B, 4 GB\ndevice under test\n(inference on CPU)")
    box(ax, 33, 4, 19, 13, "Arduino Uno", "reads the sensors\nevery 5 s")
    box(ax, 1, 5, 15, 11, "DS18B20", "ambient\ntemperature")
    box(ax, 70, 3, 22, 14, "Desktop PC", "Arduino exporter,\nPrometheus, PostgreSQL")

    # linha de +5 V: fonte -> VIN+ ; VIN- -> Raspberry Pi
    wire(ax, [(18, 42), (31, 42)], RED, label="+5 V", lpos=(24.5, 42.6))
    pin(ax, 30.4, 41.3, "VIN+", va="top")
    ax.texts[-1].set_ha("right")
    wire(ax, [(54, 42), (70, 42)], RED, label="+5 V (measured)", lpos=(62, 42.6))
    pin(ax, 54.6, 41.3, "VIN−", va="top")
    ax.texts[-1].set_ha("left")
    # GND comum
    wire(ax, [(18, 36), (25, 36), (25, 31.5), (62, 31.5), (62, 35), (70, 35)], INK, lw=1.6,
         label="GND (common)", lpos=(56.5, 30.6), va="top")

    # barramento I2C + alimentacao do INA219 (para baixo, ate o Arduino)
    labels = [("VCC", RED, "5V"), ("GND", INK, "GND"), ("SDA", BLUE, "A4"), ("SCL", ORANGE, "A5")]
    for i, (name, color, apin) in enumerate(labels):
        x = 36 + i * 4.3
        wire(ax, [(x, 36), (x, 17)], color, lw=1.5)
        pin(ax, x, 35.2, name, va="top")
        pin(ax, x, 17.6, apin, va="bottom")
    ax.text(56, 23.5, "I²C bus\n+ sensor power", ha="left", va="center", fontsize=6.8, color=INK_2)
    dot(ax, 40.3, 31.5, INK)   # GND do sensor no mesmo terra

    # DS18B20 -> Arduino (1-Wire no pino digital 4)
    wire(ax, [(16, 10.5), (33, 10.5)], GREEN, lw=1.5, label="DQ → D4 (1-Wire)", lpos=(24.5, 11.1))
    ax.text(24.5, 9.8, "with pull-up", ha="center", va="top", fontsize=6.3, color=MUTED)

    # Arduino -> PC (USB serial)
    wire(ax, [(52, 10), (70, 10)], INK_2, lw=1.5, dashed=True, arrow=True,
         label="USB (serial data)", lpos=(61, 10.6))

    ax.text(1, 55.5, "Energy measurement wiring", ha="left", va="top", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(1, 52.8, "The INA219 sits in series with the +5 V line, so it measures the power of the whole "
            "Raspberry Pi board (idle + inference).", ha="left", va="top", fontsize=7, color=INK_2)
    save(fig, "fig_measurement_wiring")


# ---------------------------------------------------------------------------
# 2. Fluxo de dados
# ---------------------------------------------------------------------------

def dataflow():
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")
    PI, PC, HW = "#e6effa", BOX_FILL, SURF          # cor de fundo = onde roda

    # fontes de metricas (esquerda)
    box(ax, 1, 25, 29, 13, "Energy exporter :8080", "INA219 → Arduino (USB serial)\nV, I, P, ambient temperature", fill=PC)
    box(ax, 1, 4, 29, 13, "rasp_monitor :9100", "CPU %, RAM %,\nCPU temperature", fill=PI)
    # armazenamento de series temporais (centro)
    box(ax, 38, 14.5, 24, 13, "Prometheus", "time series,\nscrapes every 5 s", fill=PC)
    # consumidores (direita)
    box(ax, 70, 25, 29, 13, "main_models.py", "runs the inference window\nand keeps its start/end time", fill=PI)
    box(ax, 70, 4, 29, 13, "PostgreSQL", "one row per run: W, J/image,\nms/image, CPU, RAM, temp.", fill=PC)

    def arrow(p, q, color):
        ax.annotate("", q, p, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, mutation_scale=9,
                                              shrinkA=0, shrinkB=0), zorder=1.6)

    arrow((30, 31.5), (38, 24), BLUE)
    arrow((30, 10.5), (38, 18), BLUE)
    ax.text(34, 21, "scrape", ha="center", va="center", fontsize=6.5, color=INK_2,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=SURF, edgecolor="none"))
    arrow((70, 31.5), (62, 24), ORANGE)
    ax.text(61.5, 33, "query avg / max\nin [start, end]", ha="center", va="bottom", fontsize=6.5, color=INK_2)
    arrow((84.5, 25), (84.5, 17), GREEN)
    ax.text(85.5, 21, "save results", ha="left", va="center", fontsize=6.5, color=INK_2)

    # legenda de onde cada parte roda
    for i, (fill, label) in enumerate(((PI, "runs on the Raspberry Pi"), (PC, "runs on the desktop PC"))):
        x = 38 + i * 31
        ax.add_patch(FancyBboxPatch((x, 0.2), 3, 2.2, boxstyle="round,pad=0,rounding_size=0.5",
                                    facecolor=fill, edgecolor=INK_2, linewidth=0.8))
        ax.text(x + 4, 1.3, label, ha="left", va="center", fontsize=6.8, color=INK_2)

    ax.text(1, 49.5, "Measurement data flow", ha="left", va="top", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(1, 46.8, "Energy is measured outside the device (INA219); system metrics come from the Raspberry Pi; "
            "Prometheus aligns both in time.", ha="left", va="top", fontsize=7, color=INK_2)
    save(fig, "fig_measurement_dataflow")


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  {OUT / name}.pdf / .png")


if __name__ == "__main__":
    setup()
    wiring()
    dataflow()
