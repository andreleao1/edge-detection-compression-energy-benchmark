"""
Thin wrapper around the Prometheus HTTP API.

All queries use avg_over_time / max_over_time evaluated at `end_time` with a
lookback window equal to the execution duration so only the data produced
during the inference run is considered.

Metrics assumed to exist in Prometheus:
  - energia_potencia          (watts)  — arduino/main.py exporter
  - rasp_cpu_usage_percent    (%)      — rasp_monitor/main.go exporter
  - rasp_memory_used_percent  (%)      — rasp_monitor/main.go exporter
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
            avg_watt, max_watt, avg_cpu, avg_mem, avg_temp
        Values are floats or None when data is unavailable.
        """
        duration_s = max(1, int(end_time - start_time))
        window = f"{duration_s}s"

        # avg() wrapper handles multi-series metrics (e.g. per-core CPU) gracefully
        # by collapsing them into a single scalar before the over_time aggregation.
        queries: dict[str, str] = {
            "avg_watt": f"avg_over_time(energia_potencia[{window}])",
            "max_watt": f"max_over_time(energia_potencia[{window}])",
            "avg_cpu": (
                f"avg(avg_over_time("
                f"rasp_cpu_usage_percent{{mode='aggregate'}}[{window}]))"
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
