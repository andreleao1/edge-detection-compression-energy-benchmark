#!/usr/bin/env python3
"""
Avalia os modelos podados (poda ESTRUTURAL, gerados por run_all_pruning.sh) e
preenche a parte "Desktop (GPU)" da planilha "Experimentos Metrado Andre.xlsx":

    C Params (M) | D Taxa compressao (%) | E mAP50 | F mAP50-95 | G Precision |
    H Recall | I F1 | J Inferencia (ms) | K FPS | L FLOPS (G)

so' nas linhas pedidas (padrao 10-37) das abas "Base Oxford Town Centre" e
"Base Caviar". A linha e' identificada pelas colunas A (modelo) e B
("Pruning Estruturado (NN%)") — nao por posicao fixa.

- Reaproveita evaluate_yolo / evaluate_torchvision de evaluate_models.py
  (mesma metodologia do baseline: split test, batch 1, thop p/ FLOPs).
- Nao usa Postgres.
- Coluna D e' formula (100*(base-C)/base): preservada se existir, criada se faltar.
- Cada resultado e' guardado em runs/pruned_eval_cache.json: se interromper,
  rode de novo e so' o que falta e' avaliado.
- Por seguranca grava numa COPIA da planilha (--output); use --in-place para
  sobrescrever o original (um backup .bak com data e' feito antes).

Uso:
    python evaluate_pruned_to_sheet.py --dry-run          # so' lista o que faria
    python evaluate_pruned_to_sheet.py                    # avalia tudo -> copia "_atualizada"
    python evaluate_pruned_to_sheet.py --only caviar --rows 10-13
    python evaluate_pruned_to_sheet.py --in-place
"""

import argparse
import copy
import json
import re
import shutil
import time
from pathlib import Path

import torch
import openpyxl

import evaluate_models as em          # evaluate_yolo, evaluate_torchvision, count_params, estimate_flops
from evaluate_models import DEFAULTS, NUM_CLASSES, load_torchvision_model
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
XLSX = ROOT / "Experimentos Metrado Andre.xlsx"
CACHE = ROOT / "runs" / "pruned_eval_cache.json"

SHEETS = {"oxford": "Base Oxford Town Centre", "caviar": "Base Caviar"}
# nome na coluna A da planilha -> arquitetura
MODEL_MAP = {
    "YOLO10n": "yolov10n", "YOLO10s": "yolov10s", "YOLO10m": "yolov10m",
    "YOLO10b": "yolov10b", "YOLO10l": "yolov10l", "YOLO10x": "yolov10x",
    "FasterRCNN": "faster_rcnn", "RetinaNet": "retinanet", "MobileNet": "mobilenet",
}
# chave do dict de resultado -> coluna da planilha (D e' formula)
COLUMNS = {"Params (M)": "C", "mAP50": "E", "mAP50-95": "F", "Precision": "G",
           "Recall": "H", "F1": "I", "Inferencia (ms)": "J", "FPS": "K", "FLOPS (G)": "L"}


def checkpoint_path(arch: str, dataset: str, pct: int) -> Path:
    if arch.startswith("yolov10"):
        return ROOT / "runs" / "train" / "pruned_structured" / str(pct) / dataset / f"{arch}_pruned.pt"
    return ROOT / "runs" / arch / "pruned_structured" / str(pct) / dataset / "best_pruned.pth"


def parse_rows(spec: str) -> tuple[int, int]:
    a, _, b = spec.partition("-")
    return int(a), int(b or a)


def find_baseline_row(ws, model_name: str) -> int | None:
    for r in range(3, ws.max_row + 1):
        if ws[f"A{r}"].value == model_name and "baseline" in str(ws[f"B{r}"].value or "").lower():
            return r
    return None


def evaluate_one(arch: str, dataset: str, pct: int, device, device_int: int) -> dict:
    ckpt = checkpoint_path(arch, dataset, pct)
    dataset_dir = ROOT / dataset
    label = f"{arch}_pruned_structured_p{pct}_{dataset}"

    if arch.startswith("yolov10"):
        data_yaml = str(ROOT / "runs" / "yaml" / f"{dataset}.yaml")
        # Params/FLOPs medidos ANTES do val(): o val() do ultralytics funde
        # Conv+BN in-place e a contagem pos-val seria menor que a do baseline.
        fresh = YOLO(str(ckpt)).model.to(device).float().eval()
        params_m = em.count_params(fresh)
        flops_g = em.estimate_flops(copy.deepcopy(fresh), device, DEFAULTS["yolo"]["imgsz"])
        del fresh
        row = em.evaluate_yolo(ckpt, data_yaml, label, device_int,
                               compressao=f"Pruning Estruturado ({pct}%)", pruning_rate=float(pct))
        row["Params (M)"] = round(params_m, 2)
        row["FLOPS (G)"] = round(flops_g, 2) if flops_g else None
        return row

    cfg = DEFAULTS[arch]
    model = load_torchvision_model(arch, ckpt, NUM_CLASSES, device, cfg["imgsz"]).eval()
    try:
        return em.evaluate_torchvision(
            model, label, dataset_dir, device, cfg["imgsz"], cfg["square"], ckpt,
            compressao=f"Pruning Estruturado ({pct}%)", pruning_rate=float(pct))
    finally:
        del model
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", type=Path, default=XLSX)
    ap.add_argument("--output", type=Path, default=None,
                    help="planilha de saida (padrao: '<nome> - atualizada.xlsx' ao lado da original)")
    ap.add_argument("--in-place", action="store_true", help="sobrescreve a planilha original (faz backup .bak)")
    ap.add_argument("--rows", default="10-37", help="intervalo de linhas (padrao 10-37)")
    ap.add_argument("--only", choices=list(SHEETS), help="so' uma base")
    ap.add_argument("--dry-run", action="store_true", help="nao avalia nem grava; so' lista os jobs")
    ap.add_argument("--force", action="store_true", help="ignora o cache e reavalia tudo")
    args = ap.parse_args()

    r0, r1 = parse_rows(args.rows)
    out_path = args.xlsx if args.in_place else (
        args.output or args.xlsx.with_name(args.xlsx.stem + " - atualizada.xlsx"))

    wb = openpyxl.load_workbook(args.xlsx)       # sem data_only: preserva formulas
    cache = json.loads(CACHE.read_text()) if CACHE.exists() and not args.force else {}

    # ---- monta a lista de jobs a partir da propria planilha -------------------
    jobs = []
    for dataset, sheet in SHEETS.items():
        if args.only and dataset != args.only:
            continue
        ws = wb[sheet]
        for r in range(r0, r1 + 1):
            name, comp = ws[f"A{r}"].value, str(ws[f"B{r}"].value or "")
            m = re.search(r"\((\d+)%\)", comp)
            if name not in MODEL_MAP or "Pruning Estruturado" not in comp or not m:
                print(f"[aviso] {sheet} linha {r}: '{name}' / '{comp}' ignorada (nao e' poda estrutural conhecida)")
                continue
            arch, pct = MODEL_MAP[name], int(m.group(1))
            ckpt = checkpoint_path(arch, dataset, pct)
            jobs.append(dict(dataset=dataset, sheet=sheet, row=r, name=name, arch=arch, pct=pct, ckpt=ckpt,
                             key=f"{dataset}|{arch}|{pct}"))

    print(f"{len(jobs)} job(s); saida: {out_path}")
    for j in jobs:
        status = "cache" if j["key"] in cache else ("OK" if j["ckpt"].exists() else "SEM CHECKPOINT")
        print(f"  {j['sheet'][:12]:12s} L{j['row']:<3d} {j['name']:10s} {j['pct']:>2d}%  [{status}]  {j['ckpt'].relative_to(ROOT)}")
    if args.dry_run:
        return

    if not torch.cuda.is_available():
        raise SystemExit("GPU nao encontrada (os dados da planilha sao 'Desktop (GPU)').")
    device, device_int = torch.device("cuda:0"), 0
    if not em.HAS_THOP:
        print("[aviso] thop nao instalado — FLOPS ficara vazio (pip install thop)")

    if args.in_place:
        bak = args.xlsx.with_name(f"{args.xlsx.stem}.{time.strftime('%Y%m%d-%H%M%S')}.bak.xlsx")
        shutil.copy2(args.xlsx, bak)
        print(f"backup: {bak}")

    CACHE.parent.mkdir(parents=True, exist_ok=True)

    def write_row(j, res):
        ws = wb[j["sheet"]]
        r = j["row"]
        for key, col in COLUMNS.items():
            ws[f"{col}{r}"] = res.get(key)
        # D: taxa de compressao (formula da propria planilha, mantida)
        d = ws[f"D{r}"]
        if not (isinstance(d.value, str) and d.value.startswith("=")):
            b = find_baseline_row(ws, j["name"])
            if b:
                d.value = f'=IF(OR(C{r}="",$C${b}=""),"",100*($C${b}-C{r})/$C${b})'

    for i, j in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] {j['dataset']} | {j['arch']} | {j['pct']}%  (linha {j['row']})")
        if j["key"] not in cache:
            if not j["ckpt"].exists():
                print(f"  checkpoint nao encontrado: {j['ckpt']} — pulando")
                continue
            try:
                cache[j["key"]] = evaluate_one(j["arch"], j["dataset"], j["pct"], device, device_int)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  ERRO ({type(e).__name__}) — pulando")
                continue
            CACHE.write_text(json.dumps(cache, indent=1))
        res = cache[j["key"]]
        write_row(j, res)
        wb.save(out_path)                         # salva a cada linha (resiste a interrupcao)
        print("  " + " | ".join(f"{k}={res.get(k)}" for k in COLUMNS))

    wb.save(out_path)
    print(f"\nPlanilha salva em: {out_path}")


if __name__ == "__main__":
    main()
