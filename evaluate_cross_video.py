#!/usr/bin/env python3
"""
Avalia os modelos de experiments/top_models (manifest.json) nas sequencias
extras do CAVIAR preparadas por prepare_caviar_cross_video.py e, lado a lado,
no teste atual da Caviar (quadros sorteados do proprio video de treino —
com vazamento). Usa os modelos da Caviar de cada pasta: o ONNX medido na
Raspberry quando existe (reference/), senao o checkpoint PyTorch (source/).

A diferenca entre as duas colunas mede o quanto o mAP atual da Caviar esta'
inflado pelo vazamento.

Saida: tabela no terminal e experiments/top_models/cross_video_caviar.csv

Uso:
    venv/bin/python evaluate_cross_video.py
    venv/bin/python evaluate_cross_video.py --device 0
"""
import argparse
import csv
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOP_DIR = ROOT / "experiments" / "top_models"
CROSS_DIR = ROOT / "caviar" / "datasets" / "cross_video"
LEAKY_YAML = ROOT / "runs" / "yaml" / "caviar.yaml"
METRICS = ("mAP50", "mAP50-95", "Precision", "Recall", "F1")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=0, help="GPU (padrao 0)")
    ap.add_argument("--out", type=Path, default=TOP_DIR / "cross_video_caviar.csv")
    args = ap.parse_args()

    logging.disable(logging.WARNING)
    import evaluate_models as em

    manifest = json.loads((TOP_DIR / "manifest.json").read_text())
    seqs = sorted(d for d in CROSS_DIR.iterdir() if (d / "data.yaml").exists())
    if not seqs:
        raise SystemExit("nenhuma sequencia preparada — rode prepare_caviar_cross_video.py antes")
    sets = [("teste atual (vazamento)", LEAKY_YAML)] + [(d.name, d / "data.yaml") for d in seqs]

    rows = []
    for m in manifest["models"]:
        files = m["files"]["caviar"]
        model_path = TOP_DIR / (files.get("reference") or files["source"])
        for set_name, yaml_path in sets:
            r = em.evaluate_yolo(model_path, str(yaml_path), f"{m['dir']}@{set_name}", args.device)
            rows.append({"rank": m["rank"], "model": m["label"], "file": str(model_path.relative_to(ROOT)),
                         "set": set_name, **{k: r.get(k) for k in METRICS}})
            print(f"  {m['label'][:38]:38s} {set_name:24s} mAP50 {r['mAP50']:.3f}  mAP50-95 {r['mAP50-95']:.3f}",
                  flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["rank", "model", "file", "set", *METRICS])
        wr.writeheader()
        wr.writerows(rows)

    print("\nmAP50-95 por conjunto (a primeira coluna tem vazamento):")
    names = [s for s, _ in sets]
    print(f"{'modelo':38s} " + " ".join(f"{n[:22]:>22s}" for n in names) + f" {'queda media':>12s}")
    for m in manifest["models"]:
        vals = {r["set"]: r["mAP50-95"] for r in rows if r["rank"] == m["rank"]}
        cross = [vals[n] for n in names[1:]]
        drop = vals[names[0]] - sum(cross) / len(cross)
        print(f"{m['label'][:38]:38s} " + " ".join(f"{vals[n]:>22.3f}" for n in names) + f" {drop:>+12.3f}")
    print(f"\nCSV: {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
