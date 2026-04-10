import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
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
    erro = Column(Text, nullable=True)


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

    backend = get_backend(dsn)
    migrations = read_migrations(str(migrations_dir))

    with backend.lock():
        pending = list(backend.to_apply(migrations))
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
# Write helpers
# ---------------------------------------------------------------------------

def save_result(
    model_name: str,
    dataset_name: str,
    metrics: dict,
    start_time: float,
    end_time: float,
) -> None:
    """Persist a successful experiment result."""
    duration = round(end_time - start_time, 2)

    record = ExperimentResult(
        nome_modelo=model_name,
        nome_dataset=dataset_name,
        avg_watt=metrics.get("avg_watt"),
        max_watt=metrics.get("max_watt"),
        avg_cpu=metrics.get("avg_cpu"),
        avg_mem=metrics.get("avg_mem"),
        avg_temp=metrics.get("avg_temp"),
        data_execucao=datetime.fromtimestamp(start_time, tz=timezone.utc),
        duracao_total=duration,
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
