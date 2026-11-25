#!/usr/bin/env python3
"""
Modelo de custo de energia da inferencia guiada por movimento
(roi_inference.py), a partir das medicoes reais na Raspberry.

1. Curva de energia por inferencia. Para cada modelo de
   experiments/top_models e cada base, as medicoes da varredura de resolucao
   (imgsz 640/512/416/320, tabela experiment_results do Postgres) dao
   pontos (A, J/imagem), onde A e' a area de entrada do modelo em pixels:
     - ONNX: entrada quadrada fixa, A = imgsz^2;
     - .pt: letterbox minimo do ultralytics (lados arredondados para
       multiplos de 32), ex. Caviar 384x288 em 640 -> 640x480.
   Ajusta-se uma reta E(A) = a + b*A: "a" e' o custo fixo de cada chamada
   ao detector (pre/pos-processamento, Python), "b" o custo por pixel. O
   mesmo vale para o tempo (ms/imagem).

2. Energia da inferencia por regiao. Os arquivos roi/costs/*.json gravam,
   para cada quadro, os recortes inferidos ([imgsz, area]). A energia de
   inferencia do quadro e' a soma de E(area) dos recortes; quadro sem
   recorte custa zero de inferencia. O modo "full" (quadro inteiro) passa
   pelo mesmo calculo e serve de referencia.

3. Custo da subtracao de fundo (o "porteiro"), que roda em todo quadro.
   Medido na Raspberry com --bench-bg (tempo por quadro). Em joules,
   estima-se como tempo x potencia media do modelo na Raspberry — um limite
   superior, pois a subtracao de fundo usa menos CPU que a inferencia.

Uso (no PC, depois das medicoes):
    venv/bin/python roi_energy_model.py
    venv/bin/python roi_energy_model.py --bg roi_bg_bench.json --tag _teste

Medir o porteiro (na Raspberry, com os videos copiados para la'):
    venv/bin/python roi_energy_model.py --bench-bg caviar=video1.mpg,video2.mpg \\
        oxford=oxfordtower.mp4 --frames 1500 --out-bench roi_bg_bench.json

Saidas (experiments/top_models/roi/):
    energy_curve.csv   pontos medidos, coeficientes a e b, R^2 e o erro de
                       ajuste em cada ponto
    roi_energy.csv     por (modelo, sequencia, modo): mAP, J/quadro
                       (inferencia, porteiro e total), ms/quadro e economia
                       em relacao ao quadro inteiro
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TOP_DIR = ROOT / "experiments" / "top_models"
ROI_DIR = TOP_DIR / "roi"
SIZES = (640, 512, 416, 320)
FRAME = {"caviar": (384, 288), "oxford": (1920, 1080)}      # quadros medidos na Raspberry
DB_DATASET = {"caviar": "CAVIAR", "oxford": "OXFORD_TOWER"}


# ---------------------------------------------------------------------------
# Curva J x area a partir do banco
# ---------------------------------------------------------------------------

def letterbox_area(w: int, h: int, imgsz: int) -> int:
    r = imgsz / max(w, h)
    return math.ceil(round(w * r) / 32) * 32 * math.ceil(round(h * r) / 32) * 32


def db_name(entry: dict, imgsz: int) -> str:
    if imgsz == 640:
        return entry["name"]
    f = entry["files"]["caviar"]
    stem = Path(f.get("reference") or f["source"]).stem
    return f"{entry['arch']}__{entry['variant']}_imgsz{imgsz}__{stem}"


def fit_line(x: list, y: list) -> tuple[float, float, float]:
    """(a, b, R^2) de y = a + b*x."""
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    b, a = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    pred = a + b * np.asarray(x, float)
    ss_res = float(np.sum((np.asarray(y) - pred) ** 2))
    ss_tot = float(np.sum((np.asarray(y) - np.mean(y)) ** 2))
    return float(a), float(b), (1 - ss_res / ss_tot) if ss_tot else float("nan")


def build_curves(manifest: dict) -> dict:
    from rasp_sheet_map import DEFAULT_PG, fetch_db_results

    db = fetch_db_results(DEFAULT_PG)
    curves, rows = {}, []
    for e in manifest["models"]:
        onnx = e["requantize_after_modifying"]
        for ds, (w, h) in FRAME.items():
            pts = []
            for s in SIZES:
                r = db.get((db_name(e, s), DB_DATASET[ds]))
                if not r or not r.get("total_imagens") or r.get("joules_por_imagem") is None:
                    continue
                area = s * s if onnx else letterbox_area(w, h, s)
                pts.append({"imgsz": s, "area": area, "J": r["joules_por_imagem"],
                            "ms": r["duracao_total"] * 1000 / r["total_imagens"], "W": r["avg_watt"]})
            a, b, r2 = fit_line([p["area"] for p in pts], [p["J"] for p in pts])
            ta, tb, tr2 = fit_line([p["area"] for p in pts], [p["ms"] for p in pts])
            watt = next((p["W"] for p in pts if p["imgsz"] == 640), None)
            curves[(e["dir"], ds)] = {"a": a, "b": b, "r2": r2, "ta": ta, "tb": tb, "W": watt,
                                      "amin": min((p["area"] for p in pts), default=None), "n": len(pts)}
            for p in pts:
                rows.append({"dir": e["dir"], "dataset": ds, **p, "J_fit": a + b * p["area"],
                             "a": a, "b": b, "R2": r2, "ms_a": ta, "ms_b": tb, "ms_R2": tr2})
            if len(pts) < len(SIZES):
                missing = sorted(set(SIZES) - {p["imgsz"] for p in pts})
                print(f"  [aviso] {e['dir']} {ds}: sem medicao em imgsz {missing}")
    return curves, rows


# ---------------------------------------------------------------------------
# Energia por quadro da inferencia por regiao
# ---------------------------------------------------------------------------

def frame_costs(path: Path, curve: dict) -> tuple[float, float, float]:
    """(J de inferencia/quadro, ms de inferencia/quadro, fracao de recortes
    menores que o menor ponto medido — extrapolacao da reta)."""
    frames = json.loads(path.read_text())
    J = ms = 0.0
    crops = below = 0
    for crops_in_frame in frames:
        for c in crops_in_frame:
            area = c[1] if isinstance(c, list) else c * c       # arquivos antigos: so' o lado
            J += curve["a"] + curve["b"] * area
            ms += curve["ta"] + curve["tb"] * area
            crops += 1
            below += area < (curve["amin"] or 0)
    n = max(1, len(frames))
    return J / n, ms / n, below / crops if crops else 0.0


def build_roi_energy(curves: dict, bg: dict, tag: str) -> list[dict]:
    res_csv = ROI_DIR / f"roi_results{tag}.csv"
    if not res_csv.exists():
        raise SystemExit(f"nao encontrei {res_csv.relative_to(ROOT)} — rode roi_inference.py antes")
    results = list(csv.DictReader(res_csv.open()))
    full = {(r["dir"], r["sequence"]): r for r in results if r["mode"] == "full"}
    out = []
    for r in results:
        curve = curves.get((r["dir"], r["dataset"]))
        cpath = ROI_DIR / "costs" / f"{r['dir']}_{r['sequence']}_{r['mode']}{tag}.json"
        if not curve or curve["n"] < 2 or not cpath.exists():
            print(f"  [aviso] sem curva/custos para {r['dir']} {r['sequence']} {r['mode']}")
            continue
        j_inf, ms_inf, extrap = frame_costs(cpath, curve)
        bg_ms = bg.get(r["dataset"], 0.0) if r["mode"] == "roi" else 0.0
        bg_j = bg_ms / 1000 * (curve["W"] or 0.0)
        row = {"dir": r["dir"], "label": r["label"], "dataset": r["dataset"], "sequence": r["sequence"],
               "mode": r["mode"], "mAP50": float(r["mAP50"]), "mAP50-95": float(r["mAP50-95"]),
               "infer_frac": float(r["infer_frac"]), "pixel_frac": float(r["pixel_frac"]),
               "J_infer_per_frame": j_inf, "J_bg_per_frame": bg_j, "J_per_frame": j_inf + bg_j,
               "ms_per_frame": ms_inf + bg_ms, "crops_below_measured_area": extrap}
        out.append(row)
    for row in out:
        ref = next((x for x in out if x["dir"] == row["dir"] and x["sequence"] == row["sequence"]
                    and x["mode"] == "full"), None)
        if ref and (row["dir"], row["sequence"]) in full:
            row["energy_saving"] = 1 - row["J_per_frame"] / ref["J_per_frame"]
            row["d_mAP50-95"] = row["mAP50-95"] - ref["mAP50-95"]
            row["speedup"] = ref["ms_per_frame"] / row["ms_per_frame"] if row["ms_per_frame"] else None
    return out


# ---------------------------------------------------------------------------
# Medicao do porteiro (subtracao de fundo) — roda na Raspberry
# ---------------------------------------------------------------------------

def bench_bg(specs: list[str], frames: int, out: Path) -> None:
    import cv2

    from roi_inference import MotionRegions

    result = {}
    for spec in specs:
        ds, _, videos = spec.partition("=")
        times = []
        for v in videos.split(","):
            cap = cv2.VideoCapture(v)
            w, h = int(cap.get(3)), int(cap.get(4))
            motion = MotionRegions(w, h, min_area_frac=0.0005, margin=0.15)
            n = 0
            while n < frames:
                ok, frame = cap.read()
                if not ok:
                    break
                t0 = time.perf_counter()
                motion(frame)                      # mascara + regioes, sem decodificacao do video
                times.append(time.perf_counter() - t0)
                n += 1
        warm = times[50:] or times
        result[ds] = 1000 * float(np.median(warm))
        print(f"  {ds}: {result[ds]:.2f} ms/quadro (mediana de {len(warm)} quadros)")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Gravado: {out}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="", help="mesmo --tag usado no roi_inference.py")
    ap.add_argument("--bg", type=Path, default=ROI_DIR / "roi_bg_bench.json",
                    help="JSON do --bench-bg (ms/quadro por base); sem ele, o porteiro conta como 0")
    ap.add_argument("--bench-bg", nargs="+", metavar="BASE=VIDEO[,VIDEO...]",
                    help="mede o tempo da subtracao de fundo (rodar na Raspberry)")
    ap.add_argument("--frames", type=int, default=1500, help="--bench-bg: quadros por video")
    ap.add_argument("--out-bench", type=Path, default=Path("roi_bg_bench.json"))
    args = ap.parse_args()

    if args.bench_bg:
        bench_bg(args.bench_bg, args.frames, args.out_bench)
        return

    manifest = json.loads((TOP_DIR / "manifest.json").read_text())
    bg = json.loads(args.bg.read_text()) if args.bg.exists() else {}
    if not bg:
        print("[aviso] sem medicao do porteiro (--bench-bg): J_bg_per_frame = 0")
    curves, curve_rows = build_curves(manifest)
    ROI_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(ROI_DIR / "energy_curve.csv", curve_rows)

    print("\nCurva J/inferencia = a + b x area (pontos medidos na Raspberry):")
    for (d, ds), c in sorted(curves.items()):
        print(f"  {d:22s} {ds:7s} n={c['n']}  a={c['a']:.3f} J  b={c['b'] * 1e6:.3f} J/Mpx  R2={c['r2']:.3f}")

    rows = build_roi_energy(curves, bg, args.tag)
    write_csv(ROI_DIR / f"roi_energy{args.tag}.csv", rows)
    print("\nInferencia por regiao vs quadro inteiro:")
    for r in rows:
        if r["mode"] != "roi":
            continue
        print(f"  {r['dir']:22s} {r['sequence']:22s} economia {r.get('energy_saving', float('nan')):6.1%}  "
              f"J/quadro {r['J_per_frame']:.3f}  Δ mAP50-95 {r.get('d_mAP50-95', float('nan')):+.3f}  "
              f"recortes abaixo da menor area medida {r['crops_below_measured_area']:.0%}")
    print(f"\nCSV: {(ROI_DIR / f'roi_energy{args.tag}.csv').relative_to(ROOT)}, "
          f"{(ROI_DIR / 'energy_curve.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
