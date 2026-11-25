#!/usr/bin/env python3
"""
Avalia os modelos YOLOv10 treinados so' com Knowledge Distillation
(models_to_deploy/<ds>/<student>/kd/<student>_kd.pt, gerados por run_kd.py) e
preenche a parte "Desktop (GPU)" (colunas C..L) da planilha
"Experimentos Metrado Andre - atualizada.xlsx".

Acrescenta, ao final de cada aba, uma linha "Knowledge Distillation" por
student (YOLO10n/m/l); se a linha ja existir, so' atualiza. Colunas M..R
(Raspberry) nao sao tocadas. Metodologia identica ao baseline
(evaluate_models.evaluate_yolo: split test, batch 1, thop p/ FLOPs).

Uso:
    venv/bin/python evaluate_kd_to_sheet.py --dry-run
    venv/bin/python evaluate_kd_to_sheet.py            # faz backup .bak e grava in-place
"""
import argparse
import copy
import shutil
import time
from copy import copy as cp
from pathlib import Path

import openpyxl
import torch
from ultralytics import YOLO

import evaluate_models as em
from evaluate_models import DEFAULTS
from evaluate_pruned_to_sheet import COLUMNS, SHEETS, find_baseline_row

ROOT = Path(__file__).resolve().parent
XLSX = ROOT / "Experimentos Metrado Andre - atualizada.xlsx"
STUDENTS = {"YOLO10n": "yolov10n", "YOLO10m": "yolov10m", "YOLO10l": "yolov10l"}
LABEL = "Knowledge Distillation"


def evaluate(arch, dataset, device, device_int):
    ckpt = ROOT / "models_to_deploy" / dataset / arch / "kd" / f"{arch}_kd.pt"
    data_yaml = str(ROOT / "runs" / "yaml" / f"{dataset}.yaml")
    # Params/FLOPs antes do val() (que funde Conv+BN e reduziria a contagem)
    fresh = YOLO(str(ckpt)).model.to(device).float().eval()
    params_m = em.count_params(fresh)
    flops_g = em.estimate_flops(copy.deepcopy(fresh), device, DEFAULTS["yolo"]["imgsz"])
    del fresh
    row = em.evaluate_yolo(ckpt, data_yaml, f"{arch}_kd_{dataset}", device_int, compressao=LABEL)
    row["Params (M)"] = round(params_m, 2)
    row["FLOPS (G)"] = round(flops_g, 2) if flops_g else None
    return row


def target_row(ws, name):
    last = 0
    for r in range(3, ws.max_row + 1):
        if ws[f"A{r}"].value is None:
            continue
        last = r
        if ws[f"A{r}"].value == name and ws[f"B{r}"].value == LABEL:
            return r, False
    return last + 1, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default=XLSX)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = [(ds, name, arch) for ds in SHEETS for name, arch in STUDENTS.items()]
    for ds, name, arch in jobs:
        ck = ROOT / "models_to_deploy" / ds / arch / "kd" / f"{arch}_kd.pt"
        print(f"  {ds:7s} {name:8s} {'OK' if ck.exists() else 'SEM CHECKPOINT'}  {ck.relative_to(ROOT)}")
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise SystemExit("GPU nao encontrada.")
    device, device_int = torch.device("cuda:0"), 0

    bak = args.xlsx.with_name(f"{args.xlsx.stem}.{time.strftime('%Y%m%d-%H%M%S')}.bak.xlsx")
    shutil.copy2(args.xlsx, bak)
    print(f"backup: {bak}")
    wb = openpyxl.load_workbook(args.xlsx)   # sem data_only: preserva formulas

    for ds, name, arch in jobs:
        ws = wb[SHEETS[ds]]
        print(f"\n[{ds}] {name}")
        res = evaluate(arch, ds, device, device_int)
        r, is_new = target_row(ws, name)
        if is_new:  # copia o estilo da ultima linha de dados
            for c in range(1, 19):
                src, dst = ws.cell(r - 1, c), ws.cell(r, c)
                dst.font, dst.border, dst.fill = cp(src.font), cp(src.border), cp(src.fill)
                dst.number_format, dst.alignment = src.number_format, cp(src.alignment)
            ws[f"A{r}"], ws[f"B{r}"] = name, LABEL
        for key, col in COLUMNS.items():
            ws[f"{col}{r}"] = res.get(key)
        b = find_baseline_row(ws, name)
        ws[f"D{r}"] = f'=IF(OR(C{r}="",$C${b}=""),"",100*($C${b}-C{r})/$C${b})'
        wb.save(args.xlsx)
        print(f"  linha {r}: " + " | ".join(f"{k}={res.get(k)}" for k in COLUMNS))
    print(f"\nPlanilha salva em: {args.xlsx}")


if __name__ == "__main__":
    main()
