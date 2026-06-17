"""ZAPBench-derived tail-action decoder utilities.

The trained artifacts are produced by
``zebrafish_live_demo_v2/analysis/zapbench_action_decoder.py`` and
``zebrafish_live_demo_v2/analysis/zapbench_ephys_action_decoder.py``. They
decode ZAPBench calcium trace windows into a compact motor command representation:
whether a tail kick should occur, the lateral side, and the force magnitude.

The decoder is intentionally separate from the PAULA circuit.  ZAPBench traces
are a 71k-neuron experimental imaging space, while the current embodied
zebrafish simulation uses a reduced PAULA circuit.  This module gives us a
reproducible bridge artifact without pretending those two state spaces are
already anatomically registered.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TailAction:
    """Compact action command decoded from a brain-activity window."""

    kick: bool
    side: str
    force: float
    kick_score: float
    side_score: float


class LinearTailActionDecoder:
    """Linear ridge decoder over random-projected ZAPBench trace windows."""

    def __init__(
        self,
        *,
        projection: np.ndarray,
        weights: np.ndarray,
        trace_mean: np.ndarray,
        trace_std: np.ndarray,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        context: int,
        thresholds: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.projection = np.asarray(projection, dtype=np.float32)
        self.weights = np.asarray(weights, dtype=np.float32)
        self.trace_mean = np.asarray(trace_mean, dtype=np.float32)
        self.trace_std = np.asarray(trace_std, dtype=np.float32)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)
        self.context = int(context)
        self.thresholds = dict(thresholds)
        self.metadata = dict(metadata or {})
        if self.projection.ndim != 2:
            raise ValueError("projection must be 2D")
        if self.weights.ndim != 2 or self.weights.shape[1] != 3:
            raise ValueError("weights must have shape [features + bias, 3]")
        if self.trace_mean.shape != self.trace_std.shape:
            raise ValueError("trace_mean and trace_std shapes differ")
        if self.trace_mean.shape[0] != self.projection.shape[0]:
            raise ValueError("trace statistics and projection neuron counts differ")
        if self.feature_mean.shape != self.feature_std.shape:
            raise ValueError("feature_mean and feature_std shapes differ")
        if self.feature_mean.shape[0] + 1 != self.weights.shape[0]:
            raise ValueError("feature statistics and weight shapes differ")

    @classmethod
    def from_npz(cls, path: str | Path) -> "LinearTailActionDecoder":
        """Load a decoder artifact saved by the ZAPBench training script."""

        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            thresholds = json.loads(str(data["thresholds_json"]))
            return cls(
                projection=data["projection"],
                weights=data["weights"],
                trace_mean=data["trace_mean"],
                trace_std=data["trace_std"],
                feature_mean=data["feature_mean"],
                feature_std=data["feature_std"],
                context=int(data["context"]),
                thresholds=thresholds,
                metadata=metadata,
            )

    def _features_from_trace_window(self, trace_window: np.ndarray) -> np.ndarray:
        window = np.asarray(trace_window, dtype=np.float32)
        if window.ndim != 2:
            raise ValueError("trace_window must have shape [time, selected_neurons]")
        if window.shape[1] != self.projection.shape[0]:
            raise ValueError(
                "trace_window neuron dimension does not match trained decoder: "
                f"{window.shape[1]} != {self.projection.shape[0]}"
            )
        if window.shape[0] < self.context:
            raise ValueError(
                f"trace_window needs at least {self.context} timesteps, "
                f"got {window.shape[0]}"
            )
        window = window[-self.context :]
        normalized = (window - self.trace_mean) / self.trace_std
        projected = normalized @ self.projection
        current = projected[-1]
        mean = projected.mean(axis=0)
        slope = projected[-1] - projected[0]
        features = np.concatenate([current, mean, slope])
        features = (features - self.feature_mean) / self.feature_std
        return np.concatenate([features, np.ones(1, dtype=np.float32)])

    def raw_scores(self, trace_window: np.ndarray) -> np.ndarray:
        """Return raw `[kick_score, side_score, force]` predictions."""

        features = self._features_from_trace_window(trace_window)
        scores = features @ self.weights
        scores = np.asarray(scores, dtype=np.float32)
        scores[0] = float(np.clip(scores[0], 0.0, 1.0))
        scores[2] = float(np.clip(scores[2], 0.0, 1.0))
        return scores

    def decode(self, trace_window: np.ndarray) -> TailAction:
        """Decode a trace window into a tail action."""

        kick_score, side_score, force = [float(v) for v in self.raw_scores(trace_window)]
        kick = kick_score >= float(self.thresholds.get("kick", 0.5))
        if not kick or force < float(self.thresholds.get("force_none", 0.15)):
            side = "none"
        elif abs(side_score) < float(self.thresholds.get("side_abs", 0.20)):
            side = "none"
        else:
            side = "left" if side_score < 0.0 else "right"
        return TailAction(
            kick=bool(kick),
            side=side,
            force=float(np.clip(force, 0.0, 1.0)),
            kick_score=kick_score,
            side_score=side_score,
        )
