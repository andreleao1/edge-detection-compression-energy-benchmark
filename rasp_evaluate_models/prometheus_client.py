"""
Thin wrapper around the Prometheus HTTP API.

All queries use avg_over_time / max_over_time evaluated at `end_time` with a
lookback window equal to the execution duration so only the data produced
during the inference run is considered.

Metrics assumed to exist in Prometheus:
  - energia_potencia          (watts)  — arduino/main.py exporter
  - energia_corrente          (amps)   — arduino/main.py exporter
  - rasp_cpu_usage_percent    (%)      — rasp_monitor/main.go exporter
  - rasp_memory_used_percent  (%)      — rasp_monitor/main.go exporter
  - rasp_memory_used_mb       (MB)     — rasp_monitor/main.go exporter
  - rasp_cpu_temperature_celsius (°C)  — rasp_monitor/main.go exporter
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class PrometheusClient:
    def __init__(self, host: str, port: int):
        self.base_url = f"http://{host}:{port}"

    # ------------------------------------------------------------------
    # Low-level
    # ------------------------------------------------------------------

    def _instant_query(self, query: str, timestamp: float) -> Optional[float]:
        """
        Execute an instant PromQL query at a specific Unix timestamp.

        Returns the first scalar result value, or None if the query returned
        no data or an error occurred.
        """
        url = f"{self.base_url}/api/v1/query"
        params = {"query": query, "time": str(timestamp)}

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.error("Prometheus HTTP error for query '%s': %s", query, exc)
            return None

        if payload.get("status") != "success":
            logger.warning(
                "Prometheus returned non-success for query '%s': %s",
                query,
                payload.get("error", "<no error field>"),
            )
            return None

        results = payload["data"]["result"]
        if not results:
            logger.warning("No data returned by Prometheus for query: %s", query)
            return None

        try:
            value = float(results[0]["value"][1])
        except (KeyError, IndexError, ValueError) as exc:
            logger.error(
                "Could not parse Prometheus result for query '%s': %s", query, exc
            )
            return None

        return value

    def instant_query(self, query: str, timestamp: float) -> Optional[float]:
        """Public entry point for an arbitrary instant PromQL query — used by
        callers (e.g. baseline_report.py) that need session-scoped queries
        get_metrics() doesn't build (it hardcodes unscoped metric names)."""
        return self._instant_query(query, timestamp)

    def get_series_labels(self, query: str, timestamp: float) -> Optional[dict]:
        """Instant query that returns the label set of the first matched
        series (instead of its value) — used to read constLabels such as
        session_id/period/day/mode straight off a real sample."""
        url = f"{self.base_url}/api/v1/query"
        params = {"query": query, "time": str(timestamp)}

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.error("Prometheus HTTP error for query '%s': %s", query, exc)
            return None

        if payload.get("status") != "success":
            return None

        results = payload["data"]["result"]
        if not results:
            return None

        return results[0].get("metric", {})

    def query_range(
        self, query: str, start_time: float, end_time: float, step: str = "5s"
    ) -> list[tuple[float, float]]:
        """
        Execute a PromQL range query over [start_time, end_time].

        Returns a list of (timestamp, value) tuples from the first series in
        the result, or an empty list if the query returned no data, multiple
        series (ambiguous — callers should scope the query with label
        selectors to a single series), or an error occurred.
        """
        url = f"{self.base_url}/api/v1/query_range"
        params = {
            "query": query,
            "start": str(start_time),
            "end": str(end_time),
            "step": step,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.error("Prometheus HTTP error for range query '%s': %s", query, exc)
            return []

        if payload.get("status") != "success":
            logger.warning(
                "Prometheus returned non-success for range query '%s': %s",
                query,
                payload.get("error", "<no error field>"),
            )
            return []

        results = payload["data"]["result"]
        if not results:
            return []
        if len(results) > 1:
            logger.warning(
                "Range query '%s' matched %d series — using the first one. "
                "Scope the query with label selectors to avoid ambiguity.",
                query,
                len(results),
            )

        try:
            return [(float(ts), float(val)) for ts, val in results[0]["values"]]
        except (KeyError, IndexError, ValueError) as exc:
            logger.error(
                "Could not parse Prometheus range result for query '%s': %s", query, exc
            )
            return []

    # ------------------------------------------------------------------
    # High-level
    # ------------------------------------------------------------------

    def get_metrics(self, start_time: float, end_time: float) -> dict:
        """
        Query consolidated metrics for the inference window [start_time, end_time].

        Strategy: instant queries evaluated at `end_time` using a lookback window
        of (end_time - start_time) seconds. This avoids a range-query + manual
        aggregation step.

        Returns a dict with keys:
            avg_watt, max_watt, avg_current, max_current, avg_cpu, avg_mem, avg_temp
        Values are floats or None when data is unavailable.
        """
        duration_s = max(1, int(end_time - start_time))
        window = f"{duration_s}s"

        # avg() wrapper handles multi-series metrics (e.g. per-core CPU) gracefully
        # by collapsing them into a single scalar before the over_time aggregation.
        queries: dict[str, str] = {
            "avg_watt": f"avg_over_time(energia_potencia[{window}])",
            "max_watt": f"max_over_time(energia_potencia[{window}])",
            "avg_current": f"avg_over_time(energia_corrente[{window}])",
            "max_current": f"max_over_time(energia_corrente[{window}])",
            "avg_cpu": (
                f"avg(avg_over_time("
                f"rasp_cpu_usage_percent{{cpu='total'}}[{window}]))"
            ),
            "avg_mem": f"avg_over_time(rasp_memory_used_percent[{window}])",
            "avg_temp": f"avg_over_time(rasp_cpu_temperature_celsius[{window}])",
        }

        metrics: dict[str, Optional[float]] = {}
        for key, query in queries.items():
            value = self._instant_query(query, end_time)
            metrics[key] = value
            logger.debug("  %-12s = %s", key, value)

        return metrics
