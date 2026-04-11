"""
Optional evolution/simulation timing (EVOL_TRACE=1).

Aggregates wall time per span name; prints one JSON line per evaluate() flush.
Zero overhead when EVOL_TRACE is unset (single env read, branch only).

Optional OpenTelemetry: set EVOL_OTEL=1 and install optional [trace] deps;
configures ConsoleSpanExporter for local inspection.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

_NS: dict[str, int] = {}
_COUNTS: dict[str, int] = {}
_ENABLED: bool | None = None
_OTEL_READY: bool | None = None


def is_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        v = os.environ.get("EVOL_TRACE", "").strip().lower()
        _ENABLED = v in ("1", "true", "yes")
    return _ENABLED


def reset_accumulators() -> None:
    _NS.clear()
    _COUNTS.clear()


def _add_ns(name: str, dt_ns: int) -> None:
    _NS[name] = _NS.get(name, 0) + dt_ns
    _COUNTS[name] = _COUNTS.get(name, 0) + 1


@contextmanager
def span(name: str) -> Iterator[None]:
    if not is_enabled():
        yield
        return
    otel_cm = _otel_span_cm(name)
    t0 = time.perf_counter_ns()
    with otel_cm:
        try:
            yield
        finally:
            _add_ns(name, time.perf_counter_ns() - t0)


def _otel_span_cm(name: str):
    if os.environ.get("EVOL_OTEL", "").strip().lower() not in ("1", "true", "yes"):
        return nullcontext()
    global _OTEL_READY
    if _OTEL_READY is False:
        return nullcontext()
    if _OTEL_READY is None:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                ConsoleSpanExporter,
                SimpleSpanProcessor,
            )

            provider = TracerProvider()
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
            trace.set_tracer_provider(provider)
            _OTEL_READY = True
        except Exception:
            _OTEL_READY = False
            return nullcontext()
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("active_inference.evol_trace")
        return tracer.start_as_current_span(name)
    except Exception:
        return nullcontext()


def flush_json_line(extra: dict[str, Any] | None = None) -> None:
    """Emit one JSON object to stderr (works with multiprocessing workers)."""
    if not is_enabled():
        return
    payload: dict[str, Any] = {
        "evol_trace": True,
        "pid": os.getpid(),
        "spans_ns": dict(_NS),
        "spans_count": dict(_COUNTS),
    }
    if extra:
        payload.update(extra)
    # Convert to ms for readability
    payload["spans_ms"] = {k: v / 1e6 for k, v in _NS.items()}
    total = sum(_NS.values())
    payload["total_ms"] = total / 1e6 if total else 0.0
    if total > 0:
        payload["pct"] = {k: round(100.0 * v / total, 2) for k, v in _NS.items()}
    print(json.dumps(payload), file=sys.stderr, flush=True)


def summarize_for_file() -> str:
    """Plain-text table for benchmark_results_evol_trace.txt."""
    if not _NS:
        return "(no spans accumulated)\n"
    total = sum(_NS.values())
    lines = ["span\tms\tcount\tpct"]
    for name in sorted(_NS.keys(), key=lambda k: -_NS[k]):
        ns = _NS[name]
        ms = ns / 1e6
        c = _COUNTS.get(name, 0)
        pct = 100.0 * ns / total if total else 0.0
        lines.append(f"{name}\t{ms:.2f}\t{c}\t{pct:.1f}%")
    lines.append(f"total\t{total / 1e6:.2f}\t\t")
    return "\n".join(lines) + "\n"
