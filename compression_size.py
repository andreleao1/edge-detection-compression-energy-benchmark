#!/usr/bin/env python3
"""
Taxa de compressao pelo TAMANHO EM BITS dos pesos, para todos os modelos de
models_to_deploy/<base>/<arch>/<variante>/ (exceto as copias *_imgsz<N>).

Por que: a coluna D da planilha comparava so' o numero de parametros com o
baseline. Assim a quantizacao (mesmos parametros, menos bits) e a poda nao
estrutural (pesos zerados, mesmo tensor) apareciam com 0%.

Definicao:
    bits(modelo)    = pesos nao nulos x bits medios por peso
    compressao (%)  = 100 x (1 - bits(modelo) / bits(baseline FP32))

- .pt / .pth (PyTorch): pesos em FP32 (32 bits); conta os tensores do
  state_dict sem os buffers de BatchNorm (running_mean/var, num_batches).
  Checkpoints Ultralytics guardam os pesos em FP16, mas a inferencia e' FP32:
  conta-se 32 bits.
- .onnx quantizado: bits medios por peso = soma dos bits dos inicializadores
  (INT8/UINT8 = 8, FP32/INT32 = 32, FP16 = 16; int64 de forma/indices ignorados)
  / numero de pesos. A cabeca de deteccao e os bias ficam em FP32/INT32, por
  isso a media fica acima de 8 bits. Os pesos nao nulos vem do modelo PyTorch
  de origem (a variante sem "_quantized"), pois o grafo exportado funde o
  BatchNorm e descarta partes so' de treino (ex. cabeca one-to-many do YOLOv10),
  o que distorceria a comparacao com o baseline PyTorch.
- Poda nao estrutural: o ganho em bits supoe armazenamento esparso (so' os
  pesos nao nulos); e' uma compressao teorica (sem o custo dos indices).

Saidas:
    experiments/compression_size.csv   uma linha por (base, arch, variante)
    planilha (com --sheet): coluna D = compressao em bits (%), e duas colunas
        novas ao fim da aba: "Tamanho efetivo (MB)" e "Reducao de params (%)".
        Faz backup .bak com data antes de gravar.

Uso:
    venv/bin/python compression_size.py              # so' o CSV + resumo
    venv/bin/python compression_size.py --sheet      # tambem grava a planilha
"""
import argparse
import csv
import re
import shutil
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

from rasp_sheet_map import DEPLOY_DIR, ROOT, SHEET_XLSX, iter_sheet_rows  # noqa: E402

OUT_CSV = ROOT / "experiments" / "compression_size.csv"
BN_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")
ONNX_BITS = {1: 32, 2: 8, 3: 8, 6: 32, 10: 16, 11: 64}  # FLOAT, UINT8, INT8, INT32, FLOAT16, DOUBLE


# ---------------------------------------------------------------------------
# Contagem
# ---------------------------------------------------------------------------

def torch_counts(path: Path) -> tuple[int, int]:
    """(pesos, pesos nao nulos) de um checkpoint PyTorch."""
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and ("model" in ck or "ema" in ck):  # Ultralytics
        mod = ck.get("ema") or ck.get("model")
        sd = mod.state_dict()
    elif hasattr(ck, "state_dict"):  # nn.Module inteiro (torchvision podado)
        sd = ck.state_dict()
    else:  # state_dict puro
        sd = ck
    total = nz = 0
    for k, t in sd.items():
        if not hasattr(t, "is_floating_point") or not t.is_floating_point() or k.endswith(BN_BUFFERS):
            continue
        total += t.numel()
        nz += int((t != 0).sum())
    return total, nz


def onnx_bits_per_weight(path: Path) -> tuple[float, int]:
    """(bits medios por peso, numero de pesos) dos inicializadores de um ONNX."""
    import onnx

    m = onnx.load(str(path), load_external_data=False)
    bits = n = 0
    for t in m.graph.initializer:
        b = ONNX_BITS.get(t.data_type)
        if b is None or b == 64:
            continue
        size = 1
        for d in t.dims:
            size *= d
        bits += size * b
        n += size
    return bits / n, n


def onnx_nonzero_fraction(path: Path) -> float:
    """Fracao de pesos nao nulos (tensores com mais de 1 dimensao: Conv/Gemm)."""
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(path), load_external_data=False)
    n = nz = 0
    for t in m.graph.initializer:
        if len(t.dims) < 2 or ONNX_BITS.get(t.data_type) in (None, 64):
            continue
        a = numpy_helper.to_array(t)
        n += a.size
        nz += int((a != 0).sum())
    return nz / n if n else 1.0


# ---------------------------------------------------------------------------
# Varredura
# ---------------------------------------------------------------------------

def model_file(d: Path) -> Path | None:
    files = [p for p in d.iterdir() if p.suffix in (".pt", ".pth", ".onnx")]
    return files[0] if len(files) == 1 else None


def source_variant(variant: str) -> str:
    """Variante PyTorch de onde saiu um ONNX quantizado."""
    if variant == "quantized":
        return "baseline"
    return re.sub(r"_quantized(_p\d+)$", r"\1", variant)


def scan() -> list[dict]:
    rows, cache = [], {}

    def counts(p: Path):
        if p not in cache:
            cache[p] = torch_counts(p)
        return cache[p]

    for ds_dir in sorted(DEPLOY_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        for arch_dir in sorted(ds_dir.iterdir()):
            base_dir = arch_dir / "baseline"
            if not base_dir.is_dir() or not (base_file := model_file(base_dir)):
                continue
            p_base, _ = counts(base_file)
            for var_dir in sorted(arch_dir.iterdir()):
                if not var_dir.is_dir() or "imgsz" in var_dir.name or not (f := model_file(var_dir)):
                    continue
                note = ""
                if f.suffix == ".onnx":
                    bpw, n_onnx = onnx_bits_per_weight(f)
                    src = model_file(arch_dir / source_variant(var_dir.name)) \
                        if (arch_dir / source_variant(var_dir.name)).is_dir() else None
                    if src and src.suffix != ".onnx":
                        total, nz = counts(src)
                        note = f"pesos nao nulos de {source_variant(var_dir.name)}"
                    else:  # sem origem PyTorch: fracao de nao nulos do proprio ONNX
                        total, nz = p_base, round(p_base * onnx_nonzero_fraction(f))
                        note = "sem modelo de origem: nao nulos estimados pelo ONNX"
                else:
                    total, nz = counts(f)
                    bpw = 32.0
                bits = nz * bpw
                rows.append({
                    "dataset": ds_dir.name, "arch": arch_dir.name, "variant": var_dir.name,
                    "file": str(f.relative_to(ROOT)),
                    "params_M": round(total / 1e6, 4), "nonzero_M": round(nz / 1e6, 4),
                    "bits_per_weight": round(bpw, 2),
                    "size_MB": round(bits / 8 / 1e6, 3),
                    "baseline_size_MB": round(p_base * 4 / 1e6, 3),
                    "param_reduction_pct": round(100 * (1 - total / p_base), 2),
                    "compression_bits_pct": round(100 * (1 - bits / (p_base * 32)), 2),
                    "note": note,
                })
    return rows


# ---------------------------------------------------------------------------
# Planilha
# ---------------------------------------------------------------------------

def write_sheet(rows: list[dict], xlsx: Path, yolo_pq_variant: str) -> None:
    import openpyxl
    from openpyxl.utils import get_column_letter

    by_key = {(r["dataset"], r["arch"], r["variant"]): r for r in rows}
    bak = xlsx.with_name(f"{xlsx.stem}.{datetime.now():%Y%m%d-%H%M%S}.bak{xlsx.suffix}")
    shutil.copy2(xlsx, bak)
    print(f"backup: {bak.name}")

    wb = openpyxl.load_workbook(xlsx)
    cols = {}
    for sheet in ("Base Oxford Town Centre", "Base Caviar"):
        ws = wb[sheet]
        hdr = {ws.cell(2, c).value: c for c in range(1, ws.max_column + 1)}
        for title in ("Tamanho efetivo (MB)", "Redução de params (%)"):
            if title not in hdr:
                c = ws.max_column + 1
                ws.cell(2, c, title)
                ws.cell(2, c)._style = ws.cell(2, 4)._style
                hdr[title] = c
        cols[sheet] = (hdr["Tamanho efetivo (MB)"], hdr["Redução de params (%)"])
        ws.cell(2, 4, "Compressão em bits (%)")

    n = 0
    for row in iter_sheet_rows(wb, yolo_pq_variant=yolo_pq_variant):
        ws = wb[row["sheet"]]
        c_size, c_par = cols[row["sheet"]]
        r = by_key.get((row["dataset"], row["arch"], row["variant"])) if row["variant"] else None
        is_base = "baseline" in str(row["compression"]).lower()
        if r is None and not is_base:
            continue
        if is_base:
            r = by_key.get((row["dataset"], row["arch"], "baseline"))
            if r is None:
                continue
        ws.cell(row["row"], 4, None if is_base else r["compression_bits_pct"])
        ws.cell(row["row"], c_size, r["size_MB"])
        ws.cell(row["row"], c_par, None if is_base else r["param_reduction_pct"])
        for c in (4, c_size, c_par):
            ws.cell(row["row"], c).number_format = "0.00"
        n += 1
    for sheet, (c_size, c_par) in cols.items():
        for c in (c_size, c_par):
            wb[sheet].column_dimensions[get_column_letter(c)].width = 14
    wb.save(xlsx)
    print(f"planilha: {n} linha(s) atualizadas em {xlsx.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", action="store_true", help="grava a planilha (com backup)")
    ap.add_argument("--xlsx", type=Path, default=SHEET_XLSX)
    ap.add_argument("--yolo-pq-variant", default="pruned_unstructured_quantized",
                    help="variante usada nas linhas YOLO 'Pruning + Quantizacao'")
    args = ap.parse_args()

    rows = scan()
    OUT_CSV.parent.mkdir(exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} modelos -> {OUT_CSV.relative_to(ROOT)}\n")
    print(f"{'base':7s} {'arch':12s} {'variante':38s} {'params':>7s} {'nz':>7s} {'bits':>5s} {'MB':>7s} {'par%':>6s} {'bits%':>6s}")
    for r in rows:
        if r["dataset"] != "oxford":
            continue
        print(f"{r['dataset']:7s} {r['arch']:12s} {r['variant']:38s} {r['params_M']:7.2f} {r['nonzero_M']:7.2f} "
              f"{r['bits_per_weight']:5.1f} {r['size_MB']:7.2f} {r['param_reduction_pct']:6.1f} {r['compression_bits_pct']:6.1f}"
              + (f"  ({r['note']})" if r["note"].startswith("sem") else ""))
    if args.sheet:
        write_sheet(rows, args.xlsx, args.yolo_pq_variant)


if __name__ == "__main__":
    main()
