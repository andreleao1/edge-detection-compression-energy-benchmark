#!/usr/bin/env python3
"""
Seleciona os modelos com melhor custo/beneficio entre deteccao (mAP) e
consumo na Raspberry, considerando as DUAS bases (Caviar e Oxford) ao mesmo
tempo. A ideia e' usa-los como ponto de partida para modificacoes de
arquitetura (ex.: trocar a funcao de ativacao) e medir o impacto em mAP,
recursos e energia.

Fontes:
  - energia/recursos: Postgres (experiment_results, gravado por
    main_models.py na Raspberry) — J/imagem, latencia, CPU e RAM medios;
  - qualidade: planilha de experimentos (mAP50 / mAP50-95 da avaliacao em GPU).

Modo padrao (--mode pareto): custo/beneficio
  - qualidade = mAP50-95 na PIOR base (o mAP50 satura em ~0.995 na Caviar e
    nao diferencia os modelos; a Oxford e' quem separa);
  - custo     = J/imagem na PIOR base;
  - fica na fronteira de Pareto o modelo que nenhum outro supera nas duas
    coisas ao mesmo tempo (mais preciso E mais economico);
  - variantes praticamente iguais da mesma arquitetura (ex.: YOLOv10m
    quantizado com e sem poda) sao agrupadas e fica uma so', a mais simples;
  - mostra o ganho marginal (mAP50-95 ganho por joule a mais) entre degraus.

Modo --mode score: ranking por custo ponderado de recursos (abaixo).

Criterio do --mode score:
  1. Filtro de qualidade: mAP50 >= --min-map50 nas DUAS bases (sem ele, os
     "mais economicos" sao modelos que nao detectam nada, ex. MobileNet
     quantizado com mAP50 = 0).
  2. Custo por base: soma ponderada das metricas normalizadas (min-max entre
     os candidatos daquela base) — J/imagem, latencia, tempo de CPU por
     imagem (CPU% x latencia; CPU% sozinho engana: o ONNX quantizado usa 100%
     da CPU por menos tempo e gasta menos energia) e RAM. Normalizar por base
     evita que a base com imagens maiores domine.
  3. "Simultaneamente" = pior caso: custo final = max(custo_caviar,
     custo_oxford). Um modelo so' fica bem colocado se for bom nas duas.
  4. Com --one-per-arch (padrao), no maximo um modelo por arquitetura, para
     o estudo nao virar 5 variantes quase iguais do mesmo YOLOv10n.

Para cada modelo escolhido, indica tambem o checkpoint PyTorch editavel:
ONNX quantizado nao da' para trocar ativacao e re-treinar, entao a
modificacao parte da versao .pt/.pth anterior a quantizacao (e depois
re-quantiza).

Uso:
    venv/bin/python select_top_models.py                     # custo/beneficio (Pareto)
    venv/bin/python select_top_models.py --quality map50
    venv/bin/python select_top_models.py --mode score        # ranking de recursos
    venv/bin/python select_top_models.py --export experiments/top_models
    venv/bin/python select_top_models.py --top 5 --min-map50 0.85 --csv top5.csv
    venv/bin/python select_top_models.py --all-variants      # sem limite por arquitetura
    venv/bin/python select_top_models.py --weights energy=1  # so' energia
"""
import argparse
import csv
import re
from pathlib import Path

import openpyxl

from rasp_sheet_map import DEFAULT_PG, DEPLOY_DIR, ROOT, SHEET_XLSX, fetch_db_results, iter_sheet_rows

DATASETS = ("caviar", "oxford")
DEFAULT_WEIGHTS = {"energy": 0.5, "latency": 0.2, "cpu_time": 0.15, "mem": 0.15}

# Execucoes antigas cuja RAM media saiu muito acima das demais do mesmo
# modelo (ex.: YOLOv10n Oxford de 18-19/10 com 63-81%, contra ~20% nas
# execucoes de novembro) — provavel processo concorrente na Raspberry. Essas
# leituras ficam de fora do custo (o peso de RAM e' redistribuido) e geram aviso.
MEM_SUSPECT_PCT = 60.0


def parse_weights(spec: str | None) -> dict:
    if not spec:
        return dict(DEFAULT_WEIGHTS)
    w = {k: 0.0 for k in DEFAULT_WEIGHTS}
    for part in spec.split(","):
        k, _, v = part.partition("=")
        if k not in w:
            raise SystemExit(f"peso desconhecido '{k}' (use {', '.join(w)})")
        w[k] = float(v)
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def editable_variant(variant: str) -> str:
    """Variante PyTorch de onde partir a modificacao (antes da quantizacao)."""
    if variant == "quantized":
        return "baseline"
    return re.sub(r"_quantized(_p\d+)$", r"\1", variant)


def load_candidates(xlsx: Path, pg: dict) -> dict:
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    db = fetch_db_results(pg)
    models: dict = {}
    for row in iter_sheet_rows(wb, yolo_pq_variant=None):
        if not row["db_name"]:
            continue
        res = db.get((row["db_name"], row["db_dataset"]))
        ws = wb[row["sheet"]]
        r = row["row"]
        entry = models.setdefault(row["db_name"], {
            "name": row["db_name"], "arch": row["arch"], "variant": row["variant"],
            "label": f"{row['name']} — {row['compression']}",
        })
        if res is None or not res.get("total_imagens") or res.get("joules_por_imagem") is None:
            continue
        entry[row["dataset"]] = {
            "energy": res["joules_por_imagem"],
            "latency": res["duracao_total"] * 1000 / res["total_imagens"],
            "cpu": res["avg_cpu"],
            "cpu_time": res["avg_cpu"] / 100 * res["duracao_total"] * 1000 / res["total_imagens"],
            "mem": res["avg_mem"] if (res["avg_mem"] or 0) <= MEM_SUSPECT_PCT else None,
            "mem_raw": res["avg_mem"],
            "watt": res["avg_watt"],
            "map50": ws[f"E{r}"].value,
            "map5095": ws[f"F{r}"].value,
            "measured": str(res["data_execucao"])[:10],
        }
    return models


def score(models: dict, weights: dict, min_map50: float) -> tuple[list, list]:
    complete, dropped = [], []
    for m in models.values():
        if not all(d in m for d in DATASETS):
            dropped.append((m["name"], "sem medicao nas duas bases"))
        elif any(m[d]["map50"] is None for d in DATASETS):
            dropped.append((m["name"], "sem mAP na planilha"))
        elif min(m[d]["map50"] for d in DATASETS) < min_map50:
            dropped.append((m["name"], f"mAP50 < {min_map50} em alguma base"))
        else:
            complete.append(m)

    for d in DATASETS:
        for metric in weights:
            vals = [m[d][metric] for m in complete if m[d][metric] is not None]
            lo, hi = min(vals), max(vals)
            for m in complete:
                v = m[d][metric]
                m[d][f"n_{metric}"] = None if v is None else (0.0 if hi == lo else (v - lo) / (hi - lo))
        for m in complete:
            used = {k: w for k, w in weights.items() if w and m[d][f"n_{k}"] is not None}
            m[d]["cost"] = sum(w * m[d][f"n_{k}"] for k, w in used.items()) / sum(used.values())

    for m in complete:
        costs = [m[d]["cost"] for d in DATASETS]
        m["cost"] = max(costs)
        m["cost_mean"] = sum(costs) / len(costs)
        m["warnings"] = [f"RAM {m[d]['mem_raw']:.0f}% em {d} (medicao suspeita, fora do custo — "
                         f"medido em {m[d]['measured']}; vale medir de novo)"
                         for d in DATASETS if m[d]["mem"] is None]
    complete.sort(key=lambda m: (m["cost"], m["cost_mean"]))
    return complete, dropped


def pick(ranked: list, top: int, one_per_arch: bool) -> list:
    chosen, seen = [], set()
    for m in ranked:
        if one_per_arch and m["arch"] in seen:
            continue
        chosen.append(m)
        seen.add(m["arch"])
        if len(chosen) == top:
            break
    return chosen


def _simplicity(variant: str) -> int:
    """0 = variante 'limpa' (baseline/kd/quantized), 1 = com poda. Usado para
    desempatar variantes equivalentes: na YOLO, as podas antigas
    (pruned_structured_kd_*) nao reduziram o modelo de fato."""
    return 0 if variant in ("baseline", "kd", "quantized") else 1


def pareto(models: dict, min_map50: float, quality: str) -> tuple[list, int]:
    qkey = "map5095" if quality == "map5095" else "map50"
    cands = []
    for m in models.values():
        if not all(d in m for d in DATASETS):
            continue
        if any(m[d][qkey] is None or m[d]["map50"] is None for d in DATASETS):
            continue
        if min(m[d]["map50"] for d in DATASETS) < min_map50:
            continue
        m["q"] = min(m[d][qkey] for d in DATASETS)
        m["e"] = max(m[d]["energy"] for d in DATASETS)
        cands.append(m)

    def front_of(items):
        return [x for x in items if not any(
            y["q"] >= x["q"] and y["e"] <= x["e"] and (y["q"] > x["q"] or y["e"] < x["e"])
            for y in items)]

    groups: dict = {}
    for m in front_of(cands):
        groups.setdefault((m["arch"], "quantized" in m["variant"]), []).append(m)
    reps = []
    for members in groups.values():
        members.sort(key=lambda m: (-round(m["q"], 2), _simplicity(m["variant"]), m["e"]))
        rep = members[0]
        rep["grouped"] = [m["name"] for m in members[1:]]
        reps.append(rep)
    reps = sorted(front_of(reps), key=lambda m: m["e"])
    for prev, cur in zip([None] + reps[:-1], reps):
        cur["gain"] = None if prev is None else (cur["q"] - prev["q"]) / (cur["e"] - prev["e"])
    return reps, len(cands)


def pick_front(front: list, top: int) -> list:
    """Se a fronteira tiver mais que 'top' degraus: mantem o mais economico e
    o mais preciso e completa com os maiores ganhos marginais."""
    if len(front) <= top:
        return front
    keep = {id(front[0]), id(front[-1])}
    for m in sorted(front[1:-1], key=lambda m: -(m["gain"] or 0)):
        if len(keep) >= top:
            break
        keep.add(id(m))
    return [m for m in front if id(m) in keep]


def print_sources(chosen: list) -> None:
    print("\nCheckpoint PyTorch de partida para as modificacoes:")
    for i, m in enumerate(chosen, 1):
        src = source_paths(m)
        note = "" if src["variant"] == m["variant"] else f"  (re-quantizar depois; origem: {src['variant']})"
        print(f"{i:>2} {m['name']}{note}")
        for d in DATASETS:
            print(f"     {d:7s} {src[d]}")
        for w in m.get("warnings", []):
            print(f"     [aviso] {w}")


def run_pareto(args, models: dict) -> None:
    front, n_cands = pareto(models, args.min_map50, args.quality)
    chosen = pick_front(front, args.top)
    qname = "mAP50-95" if args.quality == "map5095" else "mAP50"
    print(f"{len(models)} modelos na planilha; {n_cands} medidos nas duas bases com mAP50 >= "
          f"{args.min_map50}; {len(front)} degraus na fronteira de Pareto "
          f"(qualidade = {qname} na pior base, custo = J/imagem na pior base).\n")
    hdr = (f"{'#':>2} {'modelo':40s} {'J/img cav/oxf':>15} {'mAP50-95 cav/oxf':>16} "
           f"{'mAP50 cav/oxf':>13} {'ganho/J':>8} {'ms/img cav/oxf':>15} {'RAM%':>7}")
    print(hdr)
    print("-" * len(hdr))
    for i, m in enumerate(chosen, 1):
        c, o = m["caviar"], m["oxford"]
        gain = "-" if m.get("gain") is None else f"{m['gain'] * 1000:+.1f}"
        print(f"{i:>2} {m['label'][:40]:40s} {c['energy']:7.2f}/{o['energy']:<7.2f} "
              f"{c['map5095']:7.3f}/{o['map5095']:<8.3f} {c['map50']:.3f}/{o['map50']:.3f} {gain:>8} "
              f"{c['latency']:7.0f}/{o['latency']:<7.0f} {c['mem_raw']:3.0f}/{o['mem_raw']:<3.0f}")
    print(f"\nganho/J = milesimos de {qname} ganhos por joule a mais, em relacao ao degrau anterior.")
    grouped = [(m["label"], m["grouped"]) for m in chosen if m.get("grouped")]
    if grouped:
        print("Variantes equivalentes agrupadas (mesma arquitetura, mAP e energia praticamente iguais):")
        for label, names in grouped:
            print(f"  {label}: tambem {', '.join(names)}")
    for m in chosen:
        m["warnings"] = [f"RAM {m[d]['mem_raw']:.0f}% em {d} (medicao suspeita de {m[d]['measured']})"
                         for d in DATASETS if (m[d]["mem_raw"] or 0) > MEM_SUSPECT_PCT]
    print_sources(chosen)
    if args.export:
        export_models(chosen, args.export, "pareto")

    if args.csv:
        keys = ("energy", "latency", "cpu", "mem_raw", "watt", "map50", "map5095", "measured")
        with args.csv.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["step", "name", "arch", "variant", "quality", "energy_worst",
                                                "gain_per_J", "grouped"] + [f"{d}_{k}" for d in DATASETS for k in keys])
            wr.writeheader()
            for i, m in enumerate(front, 1):
                wr.writerow({"step": i, "name": m["name"], "arch": m["arch"], "variant": m["variant"],
                             "quality": m["q"], "energy_worst": m["e"], "gain_per_J": m.get("gain"),
                             "grouped": ";".join(m.get("grouped", [])),
                             **{f"{d}_{k}": m[d][k] for d in DATASETS for k in keys}})
        print(f"\nFronteira completa ({len(front)} degraus): {args.csv}")


def source_paths(m: dict) -> dict:
    variant = editable_variant(m["variant"])
    out = {}
    for d in DATASETS:
        folder = DEPLOY_DIR / d / m["arch"] / variant
        files = sorted(p for p in folder.glob("*") if p.suffix in (".pt", ".pth")) if folder.is_dir() else []
        out[d] = str(files[0].relative_to(ROOT)) if files else f"(nao encontrado: {folder.relative_to(ROOT)})"
    return {"variant": variant, **out}


def _deployed_file(dataset: str, arch: str, variant: str) -> Path | None:
    folder = DEPLOY_DIR / dataset / arch / variant
    files = sorted(p for p in folder.glob("*") if p.suffix in (".pt", ".pth", ".onnx")) if folder.is_dir() else []
    return files[0] if len(files) == 1 else None


def export_models(chosen: list, out_dir: Path, mode: str) -> None:
    """Copia os modelos escolhidos para out_dir, um diretorio por modelo:

        <NN>_<arch>_<variante>/<dataset>/source/     checkpoint PyTorch de partida
        <NN>_<arch>_<variante>/<dataset>/reference/  modelo exatamente medido na
                                                     Raspberry (so' quando difere
                                                     do source, ex. ONNX quantizado)
        <NN>_<arch>_<variante>/variants/             onde vao as modificacoes

    e grava manifest.json + README.md com as metricas de referencia."""
    import json
    import shutil
    from datetime import date

    def copy(src: Path, dst_dir: Path) -> str:
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if not (dst.exists() and dst.stat().st_size == src.stat().st_size):
            shutil.copy2(src, dst)
        return str(dst.relative_to(out_dir))

    keys = ("energy", "latency", "cpu", "mem_raw", "watt", "map50", "map5095", "measured")
    manifest = {"generated": date.today().isoformat(), "mode": mode,
                "criterion": "fronteira de Pareto: mAP50-95 x J/imagem, ambos na pior base (Caviar/Oxford)"
                if mode == "pareto" else "custo ponderado de recursos, pior caso entre as bases",
                "models": []}
    for i, m in enumerate(chosen, 1):
        src_variant = editable_variant(m["variant"])
        mdir = out_dir / f"{i:02d}_{m['arch']}_{m['variant']}"
        (mdir / "variants").mkdir(parents=True, exist_ok=True)
        entry = {"rank": i, "dir": mdir.name, "name": m["name"], "label": m["label"], "arch": m["arch"],
                 "variant": m["variant"], "source_variant": src_variant,
                 "requantize_after_modifying": src_variant != m["variant"],
                 "gain_per_J": m.get("gain"), "files": {}, "metrics": {}}
        for d in DATASETS:
            src = _deployed_file(d, m["arch"], src_variant)
            ref = _deployed_file(d, m["arch"], m["variant"])
            if src is None or ref is None:
                raise SystemExit(f"arquivo nao encontrado para {m['name']} em {d} "
                                 f"(source={src_variant}, reference={m['variant']})")
            files = {"source": copy(src, mdir / d / "source"),
                     "source_origin": str(src.relative_to(ROOT))}
            if ref != src:
                files["reference"] = copy(ref, mdir / d / "reference")
                files["reference_origin"] = str(ref.relative_to(ROOT))
            entry["files"][d] = files
            entry["metrics"][d] = {k: m[d][k] for k in keys}
        manifest["models"].append(entry)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    rows = "\n".join(
        f"| {e['rank']} | `{e['dir']}` | {e['label']} | "
        f"{e['metrics']['caviar']['energy']:.2f} / {e['metrics']['oxford']['energy']:.2f} | "
        f"{e['metrics']['caviar']['map5095']:.3f} / {e['metrics']['oxford']['map5095']:.3f} | "
        f"{e['metrics']['caviar']['map50']:.3f} / {e['metrics']['oxford']['map50']:.3f} | "
        f"{'sim (origem: ' + e['source_variant'] + ')' if e['requantize_after_modifying'] else 'nao'} |"
        for e in manifest["models"])
    (out_dir / "README.md").write_text(f"""# Modelos de melhor custo/beneficio

Gerado por `select_top_models.py --export` em {manifest['generated']}.
Criterio: {manifest['criterion']}.

| # | Diretorio | Modelo | J/imagem cav / oxf | mAP50-95 cav / oxf | mAP50 cav / oxf | Re-quantizar? |
|---|---|---|---|---|---|---|
{rows}

Estes valores sao a **referencia** de cada modelo: toda modificacao deve ser
comparada com a linha do proprio modelo (mesmo protocolo de medicao na
Raspberry: `main_models.py --duration-minutes 10 --cooldown-minutes 3`).

## Estrutura

```
<NN>_<arch>_<variante>/
  caviar/source/      checkpoint PyTorch de partida para modificar (Caviar)
  caviar/reference/   modelo exatamente medido na Raspberry, quando difere do
                      source (ex.: ONNX quantizado) — so' para comparacao
  oxford/...          idem para a Oxford
  variants/           uma pasta por experimento, ex. variants/relu/
```

Os modelos sao treinados por base: cada modificacao precisa ser feita e
medida separadamente para Caviar e Oxford.

Nos modelos marcados com "Re-quantizar", a modificacao parte do checkpoint
PyTorch anterior a quantizacao; depois de modificar e ajustar, gere o ONNX
quantizado de novo (`quantize_models.py`) antes de medir na Raspberry.

`manifest.json` tem os caminhos de origem e todas as metricas de referencia
(energia, latencia, CPU, RAM, potencia, mAP, data da medicao).
Os pesos (`.pt`, `.onnx`) nao vao para o git.
""")
    print(f"\nExportado para {out_dir}: {len(chosen)} modelo(s), manifest.json e README.md.")


def run_score(args, models: dict) -> None:
    weights = parse_weights(args.weights)
    ranked, dropped = score(models, weights, args.min_map50)
    chosen = pick(ranked, args.top, one_per_arch=not args.all_variants)

    print(f"{len(models)} modelos na planilha; {len(ranked)} passam no filtro "
          f"(medidos nas duas bases e mAP50 >= {args.min_map50}); {len(dropped)} descartados.")
    print("Pesos: " + ", ".join(f"{k}={v:.2f}" for k, v in weights.items())
          + " | custo final = pior caso entre Caviar e Oxford (0 = melhor)\n")

    hdr = f"{'#':>2} {'modelo':48s} {'custo':>5}  {'J/img cav/oxf':>15} {'ms/img cav/oxf':>17} {'CPU%':>7} {'RAM%':>7} {'mAP50 cav/oxf':>13} {'mAP50-95':>11}"
    print(hdr)
    print("-" * len(hdr))
    for i, m in enumerate(chosen, 1):
        c, o = m["caviar"], m["oxford"]
        print(f"{i:>2} {m['label'][:48]:48s} {m['cost']:5.3f}  "
              f"{c['energy']:7.2f}/{o['energy']:<7.2f} {c['latency']:8.0f}/{o['latency']:<8.0f} "
              f"{c['cpu']:3.0f}/{o['cpu']:<3.0f} {c['mem_raw']:3.0f}/{o['mem_raw']:<3.0f} "
              f"{c['map50']:.3f}/{o['map50']:.3f} {c['map5095']:.3f}/{o['map5095']:.3f}")
    print_sources(chosen)
    if args.export:
        export_models(chosen, args.export, "score")

    if args.csv:
        keys = ("energy", "latency", "cpu", "cpu_time", "mem_raw", "watt", "map50", "map5095", "measured")
        fields = ["rank", "name", "arch", "variant", "cost", "cost_mean"] + [f"{d}_{k}" for d in DATASETS for k in keys]
        with args.csv.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            wr.writeheader()
            for i, m in enumerate(ranked, 1):
                wr.writerow({"rank": i, "name": m["name"], "arch": m["arch"], "variant": m["variant"],
                             "cost": round(m["cost"], 4), "cost_mean": round(m["cost_mean"], 4),
                             **{f"{d}_{k}": m[d][k] for d in DATASETS for k in keys}})
        print(f"\nRanking completo ({len(ranked)} modelos): {args.csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("pareto", "score"), default="pareto",
                    help="pareto = custo/beneficio deteccao x energia (padrao); score = ranking de recursos")
    ap.add_argument("--quality", choices=("map5095", "map50"), default="map5095",
                    help="metrica de qualidade no modo pareto (padrao: mAP50-95)")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--min-map50", type=float, default=0.8,
                    help="mAP50 minimo exigido nas duas bases (padrao: 0.8)")
    ap.add_argument("--weights", default=None,
                    help="--mode score: pesos, ex. energy=0.5,latency=0.2,cpu_time=0.15,mem=0.15 (padrao)")
    ap.add_argument("--all-variants", action="store_true",
                    help="--mode score: permite varias variantes da mesma arquitetura no top")
    ap.add_argument("--xlsx", type=Path, default=SHEET_XLSX)
    ap.add_argument("--csv", type=Path, default=None, help="grava o resultado completo em CSV")
    ap.add_argument("--export", type=Path, default=None, metavar="DIR",
                    help="copia os modelos escolhidos para DIR (ex. experiments/top_models), com manifest.json e README.md")
    ap.add_argument("--pg-host", default=DEFAULT_PG["host"])
    ap.add_argument("--pg-port", type=int, default=DEFAULT_PG["port"])
    args = ap.parse_args()

    pg = {**DEFAULT_PG, "host": args.pg_host, "port": args.pg_port}
    models = load_candidates(args.xlsx, pg)
    if args.mode == "pareto":
        run_pareto(args, models)
    else:
        run_score(args, models)


if __name__ == "__main__":
    main()
