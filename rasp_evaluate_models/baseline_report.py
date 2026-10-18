"""
Agrega uma sessao de coleta baseline/idle a partir do Prometheus e persiste
uma linha resumo (tabela baseline_sessions no Postgres) com medias/maximos e
uma checagem de qualidade dos dados.

Uma sessao = periodo em que rasp_monitor (Go) e/ou arduino/main.py (Python)
rodaram com as mesmas flags --session-id/--period/--day/--mode. Como esses
labels sao constLabels fixos para a vida do processo, o primeiro e o ultimo
timestamp de qualquer serie com aquele session_id SAO o inicio/fim reais da
sessao — nao e preciso anotar horarios manualmente.

Fluxo:
  1. Descobre a janela real da sessao (min/max timestamp das series com o
     session_id informado, dentro de --lookback-hours).
  2. Descarta os primeiros --warmup-minutes (rampa de aquecimento termico) —
     agrega so sobre o restante da janela.
  3. Roda avg_over_time/max_over_time (escopados por session_id) para CPU%,
     memoria% / memoria MB, corrente e potencia.
  4. Verifica a metrica 'up' dos jobs raspberry_pi/arduino_energia na mesma
     janela — sinaliza gaps de scrape ou o alvo caido em algum ponto.
  5. Grava (ou sobrescreve, se o session_id ja existir) uma linha na tabela
     baseline_sessions do mesmo Postgres usado por experiment_results.

Uso:
    python baseline_report.py --session-id 2026-07-20-tarde
    python baseline_report.py --session-id 2026-07-20-tarde --settings settings.yaml
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from database import init_db, save_baseline_session
from prometheus_client import PrometheusClient

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

# job_name dos dois exporters em arduino/prometheus/prometheus.yml
JOB_RASPBERRY = "raspberry_pi"
JOB_ARDUINO = "arduino_energia"

# Metricas usadas para descobrir a janela real da sessao (min/max timestamp
# de qualquer amostra com o session_id informado) e para ler de volta os
# labels period/day/mode. Cobre os dois exporters — um deles ja basta.
SESSION_DISCOVERY_METRICS = [
    'rasp_cpu_usage_percent{{session_id="{sid}", cpu="total"}}',
    'energia_potencia{{session_id="{sid}"}}',
]

WARMUP_MINUTES_DEFAULT = 5
LOOKBACK_HOURS_DEFAULT = 6
SCRAPE_INTERVAL_S_DEFAULT = 5
GAP_TOLERANCE = 0.10  # 10% de amostras faltando ainda passa


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def load_settings(settings_path: Path) -> dict:
    with settings_path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    missing = {"prometheus", "postgres"} - cfg.keys()
    if missing:
        raise ValueError(f"{settings_path} is missing required keys: {missing}")
    return cfg


def find_session_window(
    client: PrometheusClient, session_id: str, lookback_hours: float
) -> tuple[float, float] | None:
    """Retorna (window_start, window_end) como o menor/maior timestamp entre
    todas as amostras com esse session_id nas ultimas lookback_hours, ou None
    se nada foi encontrado (session_id errado, sessao fora do lookback, ou
    nenhum exporter rodou com esses labels)."""
    now = time.time()
    start = now - lookback_hours * 3600

    timestamps: list[float] = []
    for metric_template in SESSION_DISCOVERY_METRICS:
        query = metric_template.format(sid=session_id)
        samples = client.query_range(query, start, now, step="5s")
        timestamps.extend(ts for ts, _ in samples)

    if not timestamps:
        return None
    return min(timestamps), max(timestamps)


def get_session_labels(client: PrometheusClient, session_id: str, at_ts: float) -> dict:
    """Le period/day/mode de volta de uma amostra real, para conferencia —
    esses valores vieram das flags passadas aos exporters no inicio da sessao."""
    for metric_template in SESSION_DISCOVERY_METRICS:
        query = metric_template.format(sid=session_id)
        labels = client.get_series_labels(query, at_ts)
        if labels:
            return {
                "period": labels.get("period", ""),
                "day": labels.get("day", ""),
                "mode": labels.get("mode", ""),
            }
    return {"period": "", "day": "", "mode": ""}


def aggregate_metrics(
    client: PrometheusClient, session_id: str, analysis_start: float, analysis_end: float
) -> dict:
    duration_s = max(1, int(analysis_end - analysis_start))
    window = f"{duration_s}s"
    sid = session_id

    queries = {
        "avg_cpu_pct": f'avg(avg_over_time(rasp_cpu_usage_percent{{session_id="{sid}", cpu="total"}}[{window}]))',
        "max_cpu_pct": f'max(max_over_time(rasp_cpu_usage_percent{{session_id="{sid}", cpu="total"}}[{window}]))',
        "avg_mem_pct": f'avg_over_time(rasp_memory_used_percent{{session_id="{sid}"}}[{window}])',
        "avg_mem_mb": f'avg_over_time(rasp_memory_used_mb{{session_id="{sid}"}}[{window}])',
        "max_mem_mb": f'max_over_time(rasp_memory_used_mb{{session_id="{sid}"}}[{window}])',
        "avg_current_a": f'avg_over_time(energia_corrente{{session_id="{sid}"}}[{window}])',
        "max_current_a": f'max_over_time(energia_corrente{{session_id="{sid}"}}[{window}])',
        "avg_power_w": f'avg_over_time(energia_potencia{{session_id="{sid}"}}[{window}])',
        "max_power_w": f'max_over_time(energia_potencia{{session_id="{sid}"}}[{window}])',
    }

    metrics: dict[str, float | None] = {}
    for key, query in queries.items():
        metrics[key] = client.instant_query(query, analysis_end)
        if metrics[key] is None:
            logger.warning("  Sem dado para %s (query: %s)", key, query)

    return metrics


def check_data_quality(
    client: PrometheusClient, window_start: float, window_end: float, scrape_interval_s: int
) -> tuple[bool, list[str]]:
    """Verifica a serie 'up' dos dois jobs na janela da sessao. 'up' nao
    carrega o label session_id (e uma serie sintetica por target do proprio
    Prometheus) — por isso usamos a janela de tempo descoberta em
    find_session_window() em vez de um filtro por label."""
    warnings: list[str] = []
    ok = True

    for job in (JOB_RASPBERRY, JOB_ARDUINO):
        samples = client.query_range(
            f'up{{job="{job}"}}', window_start, window_end, step=f"{scrape_interval_s}s"
        )
        if not samples:
            ok = False
            warnings.append(f"sem dado de 'up' para job={job} na janela da sessao")
            continue

        if any(v == 0 for _, v in samples):
            ok = False
            warnings.append(f"job={job} ficou down (up=0) em algum ponto da janela")

        expected = (window_end - window_start) / scrape_interval_s
        actual = len(samples)
        if expected > 0 and actual < expected * (1 - GAP_TOLERANCE):
            ok = False
            warnings.append(
                f"gap de scrape em job={job}: esperado ~{expected:.0f} amostras, "
                f"obtido {actual}"
            )

    return ok, warnings


def run(args: argparse.Namespace) -> int:
    settings_path = Path(args.settings)
    if not settings_path.exists():
        logger.critical("Settings file not found: %s", settings_path.resolve())
        return 1

    settings = load_settings(settings_path)
    client = PrometheusClient(
        host=settings["prometheus"]["host"], port=settings["prometheus"]["port"]
    )

    logger.info("Procurando sessao session_id=%s (lookback=%dh)...", args.session_id, args.lookback_hours)
    window = find_session_window(client, args.session_id, args.lookback_hours)
    if window is None:
        logger.critical(
            "Nenhuma amostra encontrada para session_id=%s dentro de %dh. "
            "Confira o session_id ou aumente --lookback-hours.",
            args.session_id, args.lookback_hours,
        )
        return 1

    window_start, window_end = window
    logger.info("Janela da sessao: %s -> %s", _iso(window_start), _iso(window_end))

    analysis_start = window_start + args.warmup_minutes * 60
    analysis_end = window_end
    if analysis_end - analysis_start < 60:
        logger.critical(
            "Sessao tem menos de %d min uteis apos descartar o warmup de %d min "
            "— sessao curta demais para agregar com confianca.",
            int((analysis_end - analysis_start) / 60) if analysis_end > analysis_start else 0,
            args.warmup_minutes,
        )
        return 1

    logger.info(
        "Janela de analise (pos-warmup de %d min): %s -> %s",
        args.warmup_minutes, _iso(analysis_start), _iso(analysis_end),
    )

    labels = get_session_labels(client, args.session_id, analysis_end)
    metrics = aggregate_metrics(client, args.session_id, analysis_start, analysis_end)

    logger.info("Checando qualidade dos dados (metrica 'up')...")
    quality_ok, quality_warnings = check_data_quality(
        client, window_start, window_end, args.scrape_interval_s
    )
    if quality_ok:
        logger.info("  OK — sem gaps de scrape nem targets down na janela.")
    else:
        for w in quality_warnings:
            logger.warning("  %s", w)

    row = {
        "session_id": args.session_id,
        "period": labels["period"],
        "day": labels["day"],
        "mode": labels["mode"],
        "window_start": _dt(window_start),
        "window_end": _dt(window_end),
        "analysis_start": _dt(analysis_start),
        "analysis_end": _dt(analysis_end),
        **metrics,
        "data_quality_ok": quality_ok,
        "quality_warnings": "; ".join(quality_warnings),
    }

    init_db(settings["postgres"])
    save_baseline_session(row)

    return 0 if quality_ok else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agrega uma sessao baseline do Prometheus e grava na tabela baseline_sessions."
    )
    parser.add_argument("--session-id", required=True, help="session_id da sessao a agregar.")
    parser.add_argument(
        "--settings", default=str(BASE_DIR / "settings.yaml"),
        help="Caminho do settings.yaml (usa as secoes 'prometheus' e 'postgres'). Padrao: settings.yaml local.",
    )
    parser.add_argument(
        "--warmup-minutes", type=int, default=WARMUP_MINUTES_DEFAULT,
        help=f"Minutos iniciais descartados da agregacao (padrao: {WARMUP_MINUTES_DEFAULT}).",
    )
    parser.add_argument(
        "--lookback-hours", type=float, default=LOOKBACK_HOURS_DEFAULT,
        help=f"Ate quantas horas atras procurar a sessao (padrao: {LOOKBACK_HOURS_DEFAULT}).",
    )
    parser.add_argument(
        "--scrape-interval-s", type=int, default=SCRAPE_INTERVAL_S_DEFAULT,
        help=(
            "Scrape interval configurado nos jobs raspberry_pi/arduino_energia "
            f"em prometheus.yml (padrao: {SCRAPE_INTERVAL_S_DEFAULT}s) — usado para "
            "calcular quantas amostras 'up' eram esperadas."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit(run(_parse_args()))
