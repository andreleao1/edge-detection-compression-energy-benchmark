import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class ExperimentResult(Base):
    __tablename__ = "experiment_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    nome_modelo = Column(String(255), nullable=False, index=True)
    nome_dataset = Column(String(255), nullable=False, index=True)
    avg_watt = Column(Float, nullable=True)
    max_watt = Column(Float, nullable=True)
    avg_cpu = Column(Float, nullable=True)
    avg_mem = Column(Float, nullable=True)
    avg_temp = Column(Float, nullable=True)
    data_execucao = Column(DateTime(timezone=True), nullable=False)
    duracao_total = Column(Float, nullable=False)
    total_imagens = Column(Integer, nullable=True)
    joules_por_imagem = Column(Float, nullable=True)
    erro = Column(Text, nullable=True)


class BaselineSession(Base):
    """Uma linha por sessao de coleta baseline/idle (ver
    rasp_evaluate_models/baseline_report.py), agregada a partir do
    Prometheus. session_id e unico — reprocessar a mesma sessao sobrescreve
    a linha existente (ver save_baseline_session) em vez de duplicar."""

    __tablename__ = "baseline_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False)
    period = Column(String(20), nullable=True)
    day = Column(String(20), nullable=True)
    mode = Column(String(20), nullable=True)
    window_start = Column(DateTime(timezone=True), nullable=False)
    window_end = Column(DateTime(timezone=True), nullable=False)
    analysis_start = Column(DateTime(timezone=True), nullable=False)
    analysis_end = Column(DateTime(timezone=True), nullable=False)
    avg_cpu_pct = Column(Float, nullable=True)
    max_cpu_pct = Column(Float, nullable=True)
    avg_mem_pct = Column(Float, nullable=True)
    avg_mem_mb = Column(Float, nullable=True)
    max_mem_mb = Column(Float, nullable=True)
    avg_current_a = Column(Float, nullable=True)
    max_current_a = Column(Float, nullable=True)
    avg_power_w = Column(Float, nullable=True)
    max_power_w = Column(Float, nullable=True)
    data_quality_ok = Column(Boolean, nullable=False)
    quality_warnings = Column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("session_id", name="uq_baseline_sessions_session_id"),
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _build_dsn(cfg: dict) -> str:
    return (
        f"postgresql+psycopg2://"
        f"{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}"
        f"/{cfg['database']}"
    )


def init_db(postgres_cfg: dict) -> None:
    """Connect to PostgreSQL and apply pending migrations."""
    global _engine, _SessionLocal

    dsn = _build_dsn(postgres_cfg)
    logger.info(
        "Connecting to PostgreSQL at %s:%s/%s",
        postgres_cfg["host"],
        postgres_cfg["port"],
        postgres_cfg["database"],
    )

    _engine = create_engine(dsn, echo=False, pool_pre_ping=True)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)

    _run_migrations(dsn)
    logger.info("Database ready.")


def _run_migrations(dsn: str) -> None:
    from yoyo import get_backend, read_migrations

    migrations_dir = Path(__file__).parent / "migrations"
    logger.info("Applying migrations from: %s", migrations_dir)

    # yoyo-migrations >=9 nao reconhece o driver "+psycopg2" (esquema usado pelo
    # SQLAlchemy) — so aceita "postgresql://" puro. Usa uma DSN separada aqui.
    yoyo_dsn = dsn.replace("postgresql+psycopg2://", "postgresql://")
    backend = get_backend(yoyo_dsn)
    migrations = read_migrations(str(migrations_dir))

    with backend.lock():
        # Nao converter para list() — apply_migrations() espera o tipo
        # MigrationList do proprio yoyo (com atributo .post_apply), que se
        # perde ao envolver em list().
        pending = backend.to_apply(migrations)
        if pending:
            logger.info("Applying %d pending migration(s)...", len(pending))
            backend.apply_migrations(pending)
            logger.info("Migrations applied successfully.")
        else:
            logger.info("Schema is up to date — no pending migrations.")


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

@contextmanager
def get_session():
    """Provide a transactional scope around a series of operations."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def has_successful_result(model_name: str, dataset_name: str) -> bool:
    """True if a successful (erro IS NULL) run already exists for this
    model x dataset pair — used to resume the pipeline after a crash without
    re-running tests that already completed."""
    with get_session() as session:
        stmt = (
            select(ExperimentResult.id)
            .where(
                ExperimentResult.nome_modelo == model_name,
                ExperimentResult.nome_dataset == dataset_name,
                ExperimentResult.erro.is_(None),
            )
            .limit(1)
        )
        return session.execute(stmt).first() is not None


def get_stuck_energy_results() -> list[dict]:
    """Rows whose energy reading looks frozen — avg_watt exactly equals
    max_watt, which in practice only happens when every Prometheus sample in
    the window carries the same value (e.g. the INA219/Arduino got stuck and
    kept reporting a fixed reading instead of a real, noisy measurement).

    Returns plain dicts (not ORM instances) since callers use these outside
    of any session scope.
    """
    with get_session() as session:
        stmt = select(ExperimentResult).where(
            ExperimentResult.erro.is_(None),
            ExperimentResult.avg_watt.is_not(None),
            ExperimentResult.avg_watt == ExperimentResult.max_watt,
        )
        rows = session.execute(stmt).scalars().all()
        return [
            {
                "id": row.id,
                "nome_modelo": row.nome_modelo,
                "nome_dataset": row.nome_dataset,
                "avg_watt": row.avg_watt,
                "max_watt": row.max_watt,
                "total_imagens": row.total_imagens,
            }
            for row in rows
        ]


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def save_result(
    model_name: str,
    dataset_name: str,
    metrics: dict,
    start_time: float,
    end_time: float,
    image_count: int | None = None,
) -> None:
    """Persist a successful experiment result.

    joules_por_imagem = avg_watt * duracao_total / image_count — energia
    total da janela de inferencia (Watts medios x segundos = Joules),
    dividida pelo numero de imagens processadas.
    """
    duration = round(end_time - start_time, 2)

    avg_watt = metrics.get("avg_watt")
    joules_por_imagem = None
    if avg_watt is not None and image_count:
        joules_por_imagem = round((avg_watt * duration) / image_count, 4)

    record = ExperimentResult(
        nome_modelo=model_name,
        nome_dataset=dataset_name,
        avg_watt=avg_watt,
        max_watt=metrics.get("max_watt"),
        avg_cpu=metrics.get("avg_cpu"),
        avg_mem=metrics.get("avg_mem"),
        avg_temp=metrics.get("avg_temp"),
        data_execucao=datetime.fromtimestamp(start_time, tz=timezone.utc),
        duracao_total=duration,
        total_imagens=image_count,
        joules_por_imagem=joules_por_imagem,
    )

    with get_session() as session:
        session.add(record)
        session.flush()
        record_id = record.id

    logger.info(
        "Persisted result #%d — model=%s dataset=%s duration=%.2fs",
        record_id,
        model_name,
        dataset_name,
        duration,
    )


def update_energy_metrics(
    record_id: int,
    metrics: dict,
    start_time: float,
    end_time: float,
    image_count: int | None = None,
) -> None:
    """Overwrite an existing row's metrics after reprocessing a frozen energy
    reading — updates in place instead of inserting a new row, since the
    original run's Prometheus window is stuck in the past and re-querying it
    would just return the same frozen value again (see
    get_stuck_energy_results()). Only a fresh inference run produces a new,
    hopefully-unstuck measurement window.
    """
    duration = round(end_time - start_time, 2)

    avg_watt = metrics.get("avg_watt")
    joules_por_imagem = None
    if avg_watt is not None and image_count:
        joules_por_imagem = round((avg_watt * duration) / image_count, 4)

    with get_session() as session:
        record = session.get(ExperimentResult, record_id)
        if record is None:
            raise ValueError(f"No experiment_results row with id={record_id}")

        record.avg_watt = avg_watt
        record.max_watt = metrics.get("max_watt")
        record.avg_cpu = metrics.get("avg_cpu")
        record.avg_mem = metrics.get("avg_mem")
        record.avg_temp = metrics.get("avg_temp")
        record.data_execucao = datetime.fromtimestamp(start_time, tz=timezone.utc)
        record.duracao_total = duration
        record.total_imagens = image_count
        record.joules_por_imagem = joules_por_imagem

    logger.info("Updated result #%d with reprocessed energy metrics", record_id)


def save_baseline_session(row: dict) -> None:
    """Persist (or overwrite, on a session_id conflict) one aggregated
    baseline session row produced by baseline_report.py.

    Upsert instead of plain insert because re-running baseline_report.py for
    the same session_id (e.g. after fixing something) should replace the
    previous aggregation, not accumulate duplicate rows for the same
    session — mirrors the ON CONFLICT pattern used in accuracy_db.py for the
    same reason.
    """
    stmt = insert(BaselineSession).values(**row)
    update_cols = {
        col: stmt.excluded[col]
        for col in row
        if col != "session_id"
    }
    stmt = stmt.on_conflict_do_update(
        index_elements=["session_id"], set_=update_cols
    )

    with get_session() as session:
        session.execute(stmt)

    logger.info("Persisted baseline session — session_id=%s", row["session_id"])


def save_error(model_name: str, dataset_name: str, error_message: str) -> None:
    """Persist a failed experiment run so the pipeline remains auditable."""
    record = ExperimentResult(
        nome_modelo=model_name,
        nome_dataset=dataset_name,
        data_execucao=datetime.now(tz=timezone.utc),
        duracao_total=0.0,
        erro=error_message,
    )

    with get_session() as session:
        session.add(record)

    logger.info("Persisted error for model=%s dataset=%s", model_name, dataset_name)
