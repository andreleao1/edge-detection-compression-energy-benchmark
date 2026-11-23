"""
Mapeamento entre as linhas da planilha de experimentos e os resultados da
Raspberry gravados no Postgres (tabela experiment_results, gravada por
rasp_evaluate_models/main_models.py na Raspberry).

Usado por fill_rasp_to_sheet.py (copia do banco para as colunas M..R).

Chave do banco: nome_modelo = "<arch>__<variante>__<stem do arquivo>", com
<variante> igual ao nome da pasta em models_to_deploy/<ds>/<arch>/<variante>/,
e nome_dataset = CAVIAR | OXFORD_TOWER.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEPLOY_DIR = ROOT / "models_to_deploy"
SHEET_XLSX = ROOT / "Experimentos Metrado Andre - atualizada.xlsx"

# pasta do dataset -> (aba da planilha, nome_dataset no banco)
DATASETS = {
    "oxford": ("Base Oxford Town Centre", "OXFORD_TOWER"),
    "caviar": ("Base Caviar", "CAVIAR"),
}

ARCHS = {
    "YOLO10n": "yolov10n", "YOLO10s": "yolov10s", "YOLO10m": "yolov10m",
    "YOLO10b": "yolov10b", "YOLO10l": "yolov10l", "YOLO10x": "yolov10x",
    "FasterRCNN": "faster_rcnn", "RetinaNet": "retinanet", "MobileNet": "mobilenet",
}

# Colunas "Raspberry (CPU)" da planilha
RASP_COLS = {"ram": "M", "ms": "N", "fps": "O", "joules": "P", "reduction": "Q", "watt": "R"}
DATA_COLS = ("M", "N", "O", "P", "R")   # Q e' formula derivada de P

# (regex sobre a coluna B, variante — "{p}" e' trocado pelo percentual)
_RULES = [
    (r"^nenhum \(baseline\)$", "baseline"),
    (r"^pruning estruturado \((\d+)%\)$", "pruned_structured_p{p}"),
    (r"^quantiza[cç][aã]o \(8 bits\)$", "quantized"),
    (r"^pruning \+ quantiza[cç][aã]o \((\d+)%\)$", "pruned_structured_quantized_p{p}"),
    (r"^pruning \+ kd \((\d+)%\)$", "pruned_structured_kd_p{p}"),
    (r"^pruning \+ kd \+ quantiza[cç][aã]o \((\d+)%\)$", "pruned_structured_kd_quantized_p{p}"),
    (r"^knowledge distillation$", "kd"),
]


def row_variant(arch: str, compression: str, yolo_pq_variant: str | None = None) -> str | None:
    """Variante (nome da pasta em models_to_deploy) de uma linha da planilha,
    ou None se a linha nao tem modelo correspondente (ex.: Energy Aware).

    YOLO "Pruning + Quantização" e' ambiguo: nao existe versao estrutural
    quantizada de YOLO, so' 'pruned_unstructured_quantized_pXX'. Por padrao a
    linha e' ignorada; passe yolo_pq_variant (ex. "pruned_unstructured_quantized")
    para mapea-la explicitamente."""
    text = (compression or "").strip().lower()
    for pattern, variant in _RULES:
        m = re.match(pattern, text)
        if not m:
            continue
        pct = m.group(1) if m.groups() else None
        if variant.startswith("pruned_structured_quantized") and arch.startswith("yolov10"):
            if not yolo_pq_variant:
                return None
            variant = f"{yolo_pq_variant}_p{{p}}"
        return variant.format(p=pct) if pct else variant
    return None


def model_file(dataset: str, arch: str, variant: str) -> Path | None:
    """Unico arquivo de modelo (.pt/.pth/.onnx) em models_to_deploy/<ds>/<arch>/<variante>/."""
    d = DEPLOY_DIR / dataset / arch / variant
    if not d.is_dir():
        return None
    files = sorted(p for p in d.iterdir() if p.suffix in (".pt", ".pth", ".onnx"))
    return files[0] if len(files) == 1 else None


def db_model_name(arch: str, variant: str, stem: str) -> str:
    return f"{arch}__{variant}__{stem}"


def find_baseline_row(ws, model_name: str) -> int | None:
    for r in range(3, ws.max_row + 1):
        if ws[f"A{r}"].value == model_name and "baseline" in str(ws[f"B{r}"].value or "").lower():
            return r
    return None


def iter_sheet_rows(wb, yolo_pq_variant: str | None = None):
    """Gera um dict por linha de modelo de cada aba, com a variante e o
    arquivo correspondente em models_to_deploy (quando existir)."""
    for dataset, (sheet, db_dataset) in DATASETS.items():
        ws = wb[sheet]
        for r in range(3, ws.max_row + 1):
            name, comp = ws[f"A{r}"].value, ws[f"B{r}"].value
            arch = ARCHS.get(name)
            if not arch:
                continue
            variant = row_variant(arch, comp, yolo_pq_variant)
            path = model_file(dataset, arch, variant) if variant else None
            empty = [c for c in DATA_COLS if ws[f"{c}{r}"].value in (None, "")]
            yield dict(
                dataset=dataset, sheet=sheet, db_dataset=db_dataset, row=r,
                name=name, compression=comp, arch=arch, variant=variant, path=path,
                db_name=db_model_name(arch, variant, path.stem) if path else None,
                empty_cols=empty,
            )


def fetch_db_results(pg: dict) -> dict:
    """{(nome_modelo, nome_dataset): linha mais recente com sucesso}."""
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(host=pg["host"], port=pg["port"], dbname=pg["database"],
                            user=pg["user"], password=pg["password"])
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT DISTINCT ON (nome_modelo, nome_dataset) *
                FROM experiment_results
                WHERE erro IS NULL
                ORDER BY nome_modelo, nome_dataset, data_execucao DESC
            """)
            return {(row["nome_modelo"], row["nome_dataset"]): row for row in cur.fetchall()}
    finally:
        conn.close()


DEFAULT_PG = {"host": "localhost", "port": 5432, "database": "benchmark_results",
              "user": "benchmark", "password": "benchmark"}
