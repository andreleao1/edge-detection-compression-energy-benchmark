"""
Persistencia dos resultados de avaliacao de acuracia (mAP50/mAP50-95/mAP70,
Precision/Recall/F1, Params/FLOPs) gerados por evaluate_models.py.

Tabela nova e independente (model_accuracy_results) no MESMO Postgres usado
pelo benchmark de energia da rasp — mas sem tocar em
rasp_evaluate_models/database.py nem nas migrations yoyo de la, para nao
correr risco de quebrar o script que roda na Raspberry Pi. As credenciais sao
lidas da secao 'postgres:' de rasp_evaluate_models/settings.yaml (arquivo
gitignored, ja com as credenciais certas) em vez de duplicadas aqui.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml
from sqlalchemy import Column, DateTime, Float, Integer, String, UniqueConstraint, create_engine, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
SETTINGS_PATH = BASE_DIR / "rasp_evaluate_models" / "settings.yaml"

_engine = None
_SessionLocal = None


class Base(DeclarativeBase):
    pass


class ModelAccuracyResult(Base):
    __tablename__ = "model_accuracy_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    nome_modelo = Column(String(255), nullable=False)
    nome_dataset = Column(String(255), nullable=False)
    compressao = Column(String(50), nullable=True)
    pruning_rate = Column(Float, nullable=True)
    model_size_mb = Column(Float, nullable=True)
    params_m = Column(Float, nullable=True)
    flops_g = Column(Float, nullable=True)
    map50 = Column(Float, nullable=True)
    map50_95 = Column(Float, nullable=True)
    map70 = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1 = Column(Float, nullable=True)
    inferencia_ms = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    data_avaliacao = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("nome_modelo", "nome_dataset", name="uq_model_dataset_accuracy"),
    )


def _load_postgres_cfg() -> dict:
    if not SETTINGS_PATH.exists():
        raise FileNotFoundError(
            f"Nao encontrei {SETTINGS_PATH} — preciso da secao 'postgres:' de la "
            f"pra saber onde conectar (mesmo banco usado pela rasp)."
        )
    with SETTINGS_PATH.open("r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg["postgres"]


def init_db() -> None:
    """Conecta no Postgres e cria a tabela model_accuracy_results se ainda nao
    existir. Nao mexe em nenhuma outra tabela — Base e' propria deste modulo."""
    global _engine, _SessionLocal

    cfg = _load_postgres_cfg()
    dsn = (
        f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    logger.info(
        "Conectando ao Postgres em %s:%s/%s (tabela model_accuracy_results)",
        cfg["host"], cfg["port"], cfg["database"],
    )

    _engine = create_engine(dsn, pool_pre_ping=True)
    Base.metadata.create_all(_engine, checkfirst=True)
    _SessionLocal = sessionmaker(bind=_engine)
    logger.info("Tabela model_accuracy_results pronta.")


def get_result(nome_modelo: str, nome_dataset: str) -> dict | None:
    """Retorna a linha ja gravada (no formato de row dict usado pelo Excel/
    evaluate_models.py) para este par, ou None se ainda nao foi avaliado —
    usado pra pular jobs ja concluidos e retomar so os que faltam/falharam."""
    if _SessionLocal is None:
        raise RuntimeError("accuracy_db nao inicializado. Chame init_db() primeiro.")

    with _SessionLocal() as session:
        stmt = select(ModelAccuracyResult).where(
            ModelAccuracyResult.nome_modelo == nome_modelo,
            ModelAccuracyResult.nome_dataset == nome_dataset,
        )
        result = session.execute(stmt).scalar_one_or_none()
        if result is None:
            return None

        return {
            "Modelo": result.nome_modelo,
            "Compressao": result.compressao,
            "Pruning rate (%)": result.pruning_rate,
            "Tamanho (MB)": result.model_size_mb,
            "Params (M)": result.params_m,
            "mAP50": result.map50,
            "mAP50-95": result.map50_95,
            "mAP70": result.map70,
            "Precision": result.precision,
            "Recall": result.recall,
            "F1": result.f1,
            "Inferencia (ms)": result.inferencia_ms,
            "FPS": result.fps,
            "FLOPS (G)": result.flops_g,
        }


def save_accuracy_result(row: dict, dataset_name: str) -> None:
    """Grava (ou atualiza, se ja existir para o mesmo nome_modelo+nome_dataset)
    uma linha de resultado de avaliacao de acuracia."""
    if _SessionLocal is None:
        raise RuntimeError("accuracy_db nao inicializado. Chame init_db() primeiro.")

    values = {
        "nome_modelo": row["Modelo"],
        "nome_dataset": dataset_name,
        "compressao": row.get("Compressao"),
        "pruning_rate": row.get("Pruning rate (%)"),
        "model_size_mb": row.get("Tamanho (MB)"),
        "params_m": row.get("Params (M)"),
        "flops_g": row.get("FLOPS (G)"),
        "map50": row.get("mAP50"),
        "map50_95": row.get("mAP50-95"),
        "map70": row.get("mAP70"),
        "precision": row.get("Precision"),
        "recall": row.get("Recall"),
        "f1": row.get("F1"),
        "inferencia_ms": row.get("Inferencia (ms)"),
        "fps": row.get("FPS"),
        "data_avaliacao": datetime.now(timezone.utc),
    }

    stmt = insert(ModelAccuracyResult).values(**values)
    update_cols = {k: v for k, v in values.items() if k not in ("nome_modelo", "nome_dataset")}
    stmt = stmt.on_conflict_do_update(
        index_elements=["nome_modelo", "nome_dataset"],
        set_=update_cols,
    )

    with _SessionLocal() as session:
        session.execute(stmt)
        session.commit()

    logger.info("  Gravado no banco: %s / %s", row["Modelo"], dataset_name)
