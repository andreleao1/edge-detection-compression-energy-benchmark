#!/usr/bin/env python3
"""
Preenche a parte "Raspberry (CPU)" (colunas M..R) da planilha de experimentos
a partir dos resultados ja' gravados no Postgres pela Raspberry
(rasp_evaluate_models/main_models.py, que roda na Raspberry -> tabela
experiment_results).

So' escreve em celulas VAZIAS — valores ja' presentes na planilha nunca sao
sobrescritos. Usa o mesmo formato de formula que as linhas preenchidas a mao:

    M RAM_pico (MB)      =Sheet1!B2*<avg_mem/100>
    N Inferencia (ms)    =((<duracao_total>*1000)/<total_imagens>)
    O FPS                =<total_imagens>/<duracao_total>
    P Joules/imagem      joules_por_imagem
    Q Reduction (%)      formula relativa ao baseline (so' se vazia)
    R Potencia (W)       avg_watt

Uso:
    venv/bin/python fill_rasp_to_sheet.py --dry-run
    venv/bin/python fill_rasp_to_sheet.py              # backup .bak + grava in-place
"""
import argparse
import shutil
import time
from pathlib import Path

import openpyxl

from rasp_sheet_map import DEFAULT_PG, SHEET_XLSX, fetch_db_results, find_baseline_row, iter_sheet_rows


def cell_values(res: dict) -> dict:
    """Valores para as colunas M..R a partir de uma linha do banco."""
    out = {}
    dur, n = res.get("duracao_total"), res.get("total_imagens")
    if res.get("avg_mem") is not None:
        out["M"] = f"=Sheet1!B2*{res['avg_mem'] / 100:.4f}"
    if dur and n:
        out["N"] = f"=(({dur}*1000)/{n})"
        out["O"] = f"={n}/{dur}"
    if res.get("joules_por_imagem") is not None:
        out["P"] = round(res["joules_por_imagem"], 4)
    if res.get("avg_watt") is not None:
        out["R"] = round(res["avg_watt"], 4)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", type=Path, default=SHEET_XLSX)
    ap.add_argument("--dry-run", action="store_true", help="so' mostra o que seria preenchido")
    ap.add_argument("--yolo-pq-variant", default=None,
                    help="variante usada nas linhas YOLO 'Pruning + Quantização' "
                         "(ex. pruned_unstructured_quantized); padrao: ignora essas linhas")
    ap.add_argument("--pg-host", default=DEFAULT_PG["host"])
    ap.add_argument("--pg-port", type=int, default=DEFAULT_PG["port"])
    args = ap.parse_args()

    pg = {**DEFAULT_PG, "host": args.pg_host, "port": args.pg_port}
    results = fetch_db_results(pg)
    print(f"{len(results)} resultado(s) com sucesso no banco.")

    wb = openpyxl.load_workbook(args.xlsx)   # sem data_only: preserva formulas
    filled_rows, still_missing, stuck = [], [], []

    for row in iter_sheet_rows(wb, args.yolo_pq_variant):
        if not row["empty_cols"]:
            continue
        res = results.get((row["db_name"], row["db_dataset"])) if row["db_name"] else None
        label = f"{row['sheet'][:12]:12s} L{row['row']:<3d} {row['name']:10s} {row['compression']}"
        if res is None:
            still_missing.append(label)
            continue

        ws = wb[row["sheet"]]
        r = row["row"]
        values = cell_values(res)
        written = []
        for col in row["empty_cols"]:
            if col in values:
                ws[f"{col}{r}"] = values[col]
                written.append(col)

        # Q: reducao de energia vs baseline do mesmo modelo, se ainda vazia
        base = find_baseline_row(ws, row["name"])
        if ws[f"Q{r}"].value in (None, "") and base and base != r and "P" in values:
            ws[f"Q{r}"] = f'=IF(OR(P{r}="",$P${base}=""),"",100*($P${base}-P{r})/$P${base})'
            written.append("Q")

        if written:
            filled_rows.append(f"{label}  <- {','.join(written)}  ({row['db_name']})")
        if res.get("avg_watt") is not None and res["avg_watt"] == res.get("max_watt"):
            stuck.append(label)

    print(f"\nPreenchidas a partir do banco ({len(filled_rows)}):")
    for line in filled_rows:
        print("  " + line)
    print(f"\nAinda sem resultado no banco ({len(still_missing)}) — rodar na Raspberry "
          f"(main_models.py) ou sem modelo correspondente:")
    for line in still_missing:
        print("  " + line)
    if stuck:
        print(f"\n[aviso] {len(stuck)} linha(s) com leitura de energia possivelmente travada "
              f"(avg_watt == max_watt) — considere medir de novo:")
        for line in stuck:
            print("  " + line)

    if args.dry_run or not filled_rows:
        print("\n(dry-run: nada gravado)" if args.dry_run else "\nNada a gravar.")
        return

    bak = args.xlsx.with_name(f"{args.xlsx.stem}.{time.strftime('%Y%m%d-%H%M%S')}.bak.xlsx")
    shutil.copy2(args.xlsx, bak)
    wb.save(args.xlsx)
    print(f"\nBackup: {bak.name}\nGravado: {args.xlsx.name}")


if __name__ == "__main__":
    main()
