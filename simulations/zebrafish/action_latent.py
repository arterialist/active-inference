"""Shared zebrafish motor-action latent used by replay branches.

Both current lab replay paths eventually need the same low-dimensional motor
contract: kick/bout probability, force, turn side, confidence, and optional
segment-level tail targets.  The provenance lives beside the values so the UI
and analysis scripts can distinguish a ZAPBench ephys-derived action from an
estimated natural-video OMR action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simulations.zebrafish import config as zfc


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _clip11(value: float) -> float:
    return float(np.clip(float(value), -1.0, 1.0))


def tail_targets_from_scores(
    *,
    force: float,
    side_score: float,
    phase: float,
    n_segments: int = zfc.N_BODY_SEGMENTS,
    amplitude: float | None = None,
) -> tuple[float, ...]:
    """Generate normalized segment bend targets from a bout latent.

    Convention: positive target means stronger right-side motor activation,
    matching ``ZebrafishNervousSystem`` turn-bias convention.  ZAPBench
    ``side_score`` convention is kept separately: negative means left fictive
    action, so ``-side_score`` contributes a positive right-muscle target.
    """

    force = _clip01(force)
    side = _clip11(side_score)
    amp = _clip01(force if amplitude is None else amplitude)
    if n_segments <= 0:
        return ()
    targets: list[float] = []
    for i in range(n_segments):
        frac = i / max(1, n_segments - 1)
        envelope = 0.22 + 0.92 * frac
        wave = np.sin(float(phase) - i * zfc.TAIL_BEAT_PHASE_LAG_RAD)
        turn = -side * (1.0 - 0.68 * frac)
        target = amp * (0.74 * envelope * wave + 0.42 * turn)
        targets.append(_clip11(target))
    return tuple(targets)


def coerce_tail_targets(
    values: Any,
    *,
    n_segments: int = zfc.N_BODY_SEGMENTS,
) -> tuple[float, ...]:
    if values is None:
        return tuple(0.0 for _ in range(n_segments))
    try:
        raw = list(values)
    except TypeError:
        return tuple(0.0 for _ in range(n_segments))
    out = [_clip11(v) for v in raw[:n_segments]]
    if len(out) < n_segments:
        out.extend(0.0 for _ in range(n_segments - len(out)))
    return tuple(out)


@dataclass(frozen=True)
class ZebrafishActionLatent:
    """One frame of common video/calcium-to-motor command state."""

    source: str
    branch: str
    frame_index: int = 0
    source_time_s: float = 0.0
    kick: float = 0.0
    kick_score: float = 0.0
    force: float = 0.0
    side_score: float = 0.0
    confidence: float = 0.0
    bout_type: str = "none"
    tail_frequency_hz: float = 0.0
    tail_amplitude: float = 0.0
    tail_targets: tuple[float, ...] = field(default_factory=tuple)
    zapbench_row: int | None = None
    ephys_row: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def clipped(self) -> "ZebrafishActionLatent":
        targets = coerce_tail_targets(self.tail_targets)
        return ZebrafishActionLatent(
            source=str(self.source),
            branch=str(self.branch),
            frame_index=int(self.frame_index),
            source_time_s=float(self.source_time_s),
            kick=_clip01(self.kick),
            kick_score=_clip01(self.kick_score),
            force=_clip01(self.force),
            side_score=_clip11(self.side_score),
            confidence=_clip01(self.confidence),
            bout_type=str(self.bout_type or "none"),
            tail_frequency_hz=float(max(0.0, self.tail_frequency_hz)),
            tail_amplitude=_clip01(self.tail_amplitude),
            tail_targets=targets,
            zapbench_row=None if self.zapbench_row is None else int(self.zapbench_row),
            ephys_row=None if self.ephys_row is None else int(self.ephys_row),
            metadata=dict(self.metadata),
        )

    @property
    def side_label(self) -> str:
        side = _clip11(self.side_score)
        if self.kick < 0.5 or self.force <= 0.02:
            return "none"
        if side < -0.05:
            return "left"
        if side > 0.05:
            return "right"
        return "forward"

    def to_legacy_action_fields(self) -> dict[str, Any]:
        clipped = self.clipped()
        return {
            "action_source": clipped.source,
            "action_branch": clipped.branch,
            "action_kick": clipped.kick,
            "action_force": clipped.force,
            "action_side_score": clipped.side_score,
            "action_kick_score": clipped.kick_score,
            "action_confidence": clipped.confidence,
            "action_bout_type": clipped.bout_type,
            "tail_frequency_hz": clipped.tail_frequency_hz,
            "tail_amplitude": clipped.tail_amplitude,
            "tail_targets": list(clipped.tail_targets),
            "zapbench_row": int(clipped.zapbench_row or 0),
        }

    def to_state_dict(self) -> dict[str, Any]:
        clipped = self.clipped()
        return {
            "source": clipped.source,
            "branch": clipped.branch,
            "frame_index": clipped.frame_index,
            "source_time_s": clipped.source_time_s,
            "kick": clipped.kick,
            "kick_score": clipped.kick_score,
            "force": clipped.force,
            "side_score": clipped.side_score,
            "side": clipped.side_label,
            "confidence": clipped.confidence,
            "bout_type": clipped.bout_type,
            "tail_frequency_hz": clipped.tail_frequency_hz,
            "tail_amplitude": clipped.tail_amplitude,
            "tail_targets": list(clipped.tail_targets),
            "zapbench_row": clipped.zapbench_row,
            "ephys_row": clipped.ephys_row,
            "metadata": dict(clipped.metadata),
        }

    @classmethod
    def from_zapbench_ephys(
        cls,
        *,
        source: str,
        row: int,
        frame_index: int,
        calcium_time_s: float,
        kick: float,
        side_score: float,
        force: float,
        kick_score: float,
        confidence: float,
        phase: float = 0.0,
    ) -> "ZebrafishActionLatent":
        force = _clip01(force)
        side_score = _clip11(side_score)
        kick_score = _clip01(kick_score)
        confidence = _clip01(confidence)
        kick_value = 1.0 if float(kick) >= 0.5 else 0.0
        targets = tail_targets_from_scores(
            force=force,
            side_score=side_score,
            phase=phase,
            amplitude=force,
        )
        return cls(
            source=source,
            branch="zapbench_calcium_ephys",
            frame_index=frame_index,
            source_time_s=calcium_time_s,
            kick=kick_value,
            kick_score=kick_score,
            force=force,
            side_score=side_score,
            confidence=confidence,
            bout_type="fictive_tail_bout" if kick_value >= 0.5 else "none",
            tail_frequency_hz=zfc.TAIL_BEAT_FREQ_MIN_HZ
            + force * (zfc.TAIL_BEAT_FREQ_MAX_HZ - zfc.TAIL_BEAT_FREQ_MIN_HZ),
            tail_amplitude=force,
            tail_targets=targets,
            zapbench_row=row,
            ephys_row=row,
            metadata={"label_source": "zapbench_raw_tail_ephys_decoder"},
        ).clipped()

