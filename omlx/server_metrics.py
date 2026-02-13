# SPDX-License-Identifier: Apache-2.0
"""
Server-level metrics for the oMLX admin dashboard.

Provides a thread-safe singleton that aggregates serving metrics
across all engines/models. Metrics reset on server start and
persist until shutdown.
"""

import threading
import time
from typing import Any, Dict, Optional


class ServerMetrics:
    """
    Global server-level metrics for the Status dashboard.

    Thread-safe: uses threading.Lock since scheduler runs in ThreadPoolExecutor.
    Tracks cumulative totals and average speeds across all requests,
    with optional per-model breakdown.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Cumulative totals (reset only on server restart or clear)
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cached_tokens: int = 0
        self.total_requests: int = 0

        # Cumulative durations for average speed
        self.total_prefill_duration: float = 0.0
        self.total_generation_duration: float = 0.0

        # Per-model counters
        self._per_model: Dict[str, Dict[str, Any]] = {}

        self._start_time = time.time()

    @staticmethod
    def _new_model_counters() -> Dict[str, Any]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "requests": 0,
            "prefill_duration": 0.0,
            "generation_duration": 0.0,
        }

    def record_request_complete(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        prefill_duration: float = 0.0,
        generation_duration: float = 0.0,
        model_id: str = "",
    ) -> None:
        """Record a completed request. Thread-safe."""
        with self._lock:
            # Global counters
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_cached_tokens += cached_tokens
            self.total_requests += 1
            self.total_prefill_duration += prefill_duration
            self.total_generation_duration += generation_duration

            # Per-model counters
            if model_id:
                if model_id not in self._per_model:
                    self._per_model[model_id] = self._new_model_counters()
                m = self._per_model[model_id]
                m["prompt_tokens"] += prompt_tokens
                m["completion_tokens"] += completion_tokens
                m["cached_tokens"] += cached_tokens
                m["requests"] += 1
                m["prefill_duration"] += prefill_duration
                m["generation_duration"] += generation_duration

    def get_snapshot(self, model_id: str = "") -> Dict[str, Any]:
        """Get current metrics snapshot. Thread-safe.

        Args:
            model_id: If provided and tracked, return per-model metrics.
                      Otherwise return global aggregate.
        """
        with self._lock:
            now = time.time()

            if model_id:
                # Specific model: return its data or zeros
                m = self._per_model.get(model_id)
                if m:
                    prompt = m["prompt_tokens"]
                    completion = m["completion_tokens"]
                    cached = m["cached_tokens"]
                    requests = m["requests"]
                    prefill_dur = m["prefill_duration"]
                    gen_dur = m["generation_duration"]
                else:
                    prompt = completion = cached = requests = 0
                    prefill_dur = gen_dur = 0.0
            else:
                # All models: global aggregate
                prompt = self.total_prompt_tokens
                completion = self.total_completion_tokens
                cached = self.total_cached_tokens
                requests = self.total_requests
                prefill_dur = self.total_prefill_duration
                gen_dur = self.total_generation_duration

            # Average speed (all time) - prefill excludes cached tokens
            actual_processed = prompt - cached
            avg_prefill_tps = (
                actual_processed / prefill_dur if prefill_dur > 0 else 0.0
            )
            avg_generation_tps = (
                completion / gen_dur if gen_dur > 0 else 0.0
            )

            # Cache efficiency (cached / prompt tokens)
            cache_efficiency = (
                (cached / prompt * 100) if prompt > 0 else 0.0
            )

            return {
                "total_tokens_served": prompt + completion,
                "total_cached_tokens": cached,
                "cache_efficiency": round(cache_efficiency, 1),
                "total_prompt_tokens": prompt,
                "total_completion_tokens": completion,
                "total_requests": requests,
                "avg_prefill_tps": round(avg_prefill_tps, 1),
                "avg_generation_tps": round(avg_generation_tps, 1),
                "uptime_seconds": round(now - self._start_time, 1),
            }

    def clear_metrics(self) -> None:
        """Clear all metrics. Thread-safe."""
        with self._lock:
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_cached_tokens = 0
            self.total_requests = 0
            self.total_prefill_duration = 0.0
            self.total_generation_duration = 0.0
            self._per_model.clear()


# Global singleton
_server_metrics: Optional[ServerMetrics] = None


def get_server_metrics() -> ServerMetrics:
    """Get the global ServerMetrics singleton."""
    global _server_metrics
    if _server_metrics is None:
        _server_metrics = ServerMetrics()
    return _server_metrics


def reset_server_metrics() -> None:
    """Reset metrics (called on server start)."""
    global _server_metrics
    _server_metrics = ServerMetrics()
