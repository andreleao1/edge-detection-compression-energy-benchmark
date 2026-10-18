"""
Exporta os dados de energia/recursos (experiment_results, gravados pelo
benchmark que roda na rasp) e de acuracia (model_accuracy_results, gravados
por evaluate_models.py na GPU) para uma unica planilha Excel, unificando por
(nome_modelo, nome_dataset).

Le a conexao do Postgres da secao 'postgres:' de
rasp_evaluate_models/settings.yaml (mesmo banco usado pelos dois pipelines) —
nao depende de nenhum outro codigo da rasp, so leitura via SQL.

Uso:
    python export_results.py [--output relatorio_completo.xlsx]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from accuracy_db import _load_postgres_cfg

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

ENERGY_QUERY = """
    SELECT nome_modelo, nome_dataset, avg_watt, max_watt, avg_cpu, avg_mem,
           avg_temp, duracao_total, total_imagens, joules_por_imagem,
           data_execucao, erro
    FROM experiment_results
    ORDER BY nome_modelo, nome_dataset
"""

ACCURACY_QUERY = """
    SELECT nome_modelo, nome_dataset, compressao, pruning_rate, model_size_mb,
           params_m, flops_g, map50, map50_95, map70, precision, recall, f1,
           inferencia_ms, fps, data_avaliacao
    FROM model_accuracy_results
    ORDER BY nome_modelo, nome_dataset
"""

# Ordem de colunas da aba "combinado" (so entram as que existirem de fato,
# dataframes vazios/tabelas ainda nao populadas nao quebram o script).
COMBINED_COLUMN_ORDER = [
    "nome_modelo", "nome_dataset", "compressao", "pruning_rate",
    "avg_watt", "max_watt", "joules_por_imagem",
    "avg_cpu", "avg_mem", "avg_temp", "duracao_total", "total_imagens",
    "map50", "map50_95", "map70", "precision", "recall", "f1",
    "params_m", "flops_g", "model_size_mb", "inferencia_ms", "fps",
    "data_execucao", "data_avaliacao", "erro",
]


def _strip_timezones(df: pd.DataFrame) -> pd.DataFrame:
    """Excel (openpyxl) nao aceita datetimes com timezone — as colunas
    timestamptz (data_execucao, data_avaliacao) vem tz-aware do Postgres,
    entao removemos o tz (mantendo o horario em UTC) antes de escrever."""
    for col in df.columns:
        if isinstance(df[col].dtype, pd.DatetimeTZDtype):
            df[col] = df[col].dt.tz_convert(None)
    return df


def load_data(engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    energy_df = _strip_timezones(pd.read_sql_query(ENERGY_QUERY, engine))
    accuracy_df = _strip_timezones(pd.read_sql_query(ACCURACY_QUERY, engine))
    return energy_df, accuracy_df


def build_combined(energy_df: pd.DataFrame, accuracy_df: pd.DataFrame) -> pd.DataFrame:
    """Full outer join por (nome_modelo, nome_dataset) — mantem linhas que so
    existem de um lado (ex.: modelo ja avaliado na GPU mas ainda nao
    benchmarkado na rasp, ou vice-versa) em vez de descarta-las."""
    combined = pd.merge(
        accuracy_df, energy_df,
        on=["nome_modelo", "nome_dataset"], how="outer",
    )
    columns = [c for c in COMBINED_COLUMN_ORDER if c in combined.columns]
    return combined[columns]


def main():
    parser = argparse.ArgumentParser(
        description="Exporta experiment_results + model_accuracy_results para Excel."
    )
    parser.add_argument(
        "--output", default=str(BASE_DIR / "relatorio_completo.xlsx"),
        help="Caminho do arquivo Excel de saida (padrao: relatorio_completo.xlsx)",
    )
    args = parser.parse_args()

    cfg = _load_postgres_cfg()
    dsn = (
        f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    logger.info("Conectando ao Postgres em %s:%s/%s", cfg["host"], cfg["port"], cfg["database"])
    engine = create_engine(dsn)

    energy_df, accuracy_df = load_data(engine)
    combined_df = build_combined(energy_df, accuracy_df)

    output_path = Path(args.output)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        combined_df.to_excel(writer, sheet_name="combinado", index=False)
        energy_df.to_excel(writer, sheet_name="energia", index=False)
        accuracy_df.to_excel(writer, sheet_name="acuracia", index=False)

    logger.info(
        "Planilha salva em %s — %d linha(s) combinada(s) (%d energia, %d acuracia).",
        output_path, len(combined_df), len(energy_df), len(accuracy_df),
    )


if __name__ == "__main__":
    main()
