"""simZFish-inspired retinal OMR adapter for natural-video replay.

This is a Python runtime bridge from video frames to the shared zebrafish action
latent.  It mirrors the public Z-Robot/simZFish chain at the level needed by
the lab: lower-field OFF retinal motion counters, an OMR-like low-pass
hindbrain controller, and segment-level tail targets.  It intentionally emits
diagnostics for every intermediate stage because arbitrary POV video is not the
same data distribution as the ZAPBench VR covariates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from simulations.zebrafish import config as zfc
from simulations.zebrafish.action_latent import (
    ZebrafishActionLatent,
    tail_targets_from_scores,
)


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _clip11(value: float) -> float:
    return float(np.clip(float(value), -1.0, 1.0))


def _sigmoid(x: np.ndarray | float, omega: float = 1.0, bias: float = 0.0):
    z = np.clip(omega * (x - bias), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class SimZFishRetinalMotion:
    """Lower-field OFF direction-selective counters for the two eyes."""

    left_superior: float
    left_anterior: float
    left_inferior: float
    left_posterior: float
    right_superior: float
    right_anterior: float
    right_inferior: float
    right_posterior: float
    off_left: float
    off_right: float
    lower_field_fraction: float

    def as_dict(self) -> dict[str, float]:
        return {
            "left_superior": float(self.left_superior),
            "left_anterior": float(self.left_anterior),
            "left_inferior": float(self.left_inferior),
            "left_posterior": float(self.left_posterior),
            "right_superior": float(self.right_superior),
            "right_anterior": float(self.right_anterior),
            "right_inferior": float(self.right_inferior),
            "right_posterior": float(self.right_posterior),
            "off_left": float(self.off_left),
            "off_right": float(self.off_right),
            "lower_field_fraction": float(self.lower_field_fraction),
        }


class SimZFishRetina:
    """Vectorized approximation of the public simZFish ``Image.c`` retina."""

    def __init__(self) -> None:
        self.gap = 3
        self.off_bias_uint8 = 3.0

    def extract(
        self,
        *,
        current_gray: np.ndarray,
        previous_gray: np.ndarray,
        residual_flow: np.ndarray,
    ) -> SimZFishRetinalMotion:
        current = np.asarray(current_gray, dtype=np.float32)
        previous = np.asarray(previous_gray, dtype=np.float32)
        flow = np.asarray(residual_flow, dtype=np.float32)
        if current.shape != previous.shape:
            raise ValueError("current and previous retina frames must have the same shape")
        if flow.shape[:2] != current.shape or flow.shape[2] != 2:
            raise ValueError("residual_flow must have shape HxWx2")

        h, w = current.shape
        scaled_bias = self.off_bias_uint8 / 255.0
        off = _sigmoid(previous - current, omega=255.0, bias=scaled_bias).astype(np.float32)

        # Image.c samples the lower/posterior visual field.  Scale the original
        # 25-pixel row offset from 240px source frames down to the lab frame.
        lower_offset = max(2, int(round(25.0 * h / 240.0)))
        row0 = min(h - 2, h // 2 + lower_offset)
        rows = np.arange(row0, max(row0 + 1, h - 2), self.gap)
        left_cols = np.arange(2, max(3, w // 2), self.gap)
        right_cols = np.arange(max(2, w // 2), max(w // 2 + 1, w - 2), self.gap)
        lower_fraction = float(len(rows) / max(1, h))

        left = self._eye_counters(off, flow, rows, left_cols, left_eye=True, width=w)
        right = self._eye_counters(off, flow, rows, right_cols, left_eye=False, width=w)
        return SimZFishRetinalMotion(
            left_superior=left["superior"],
            left_anterior=left["anterior"],
            left_inferior=left["inferior"],
            left_posterior=left["posterior"],
            right_superior=right["superior"],
            right_anterior=right["anterior"],
            right_inferior=right["inferior"],
            right_posterior=right["posterior"],
            off_left=left["off"],
            off_right=right["off"],
            lower_field_fraction=lower_fraction,
        )

    def _eye_counters(
        self,
        off: np.ndarray,
        flow: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        *,
        left_eye: bool,
        width: int,
    ) -> dict[str, float]:
        if rows.size == 0 or cols.size == 0:
            return {"superior": 0.0, "anterior": 0.0, "inferior": 0.0, "posterior": 0.0, "off": 0.0}
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        off_patch = off[rr, cc]
        flow_patch = flow[rr, cc]
        u = flow_patch[:, :, 0] / max(1.0, width * 0.035)
        v = flow_patch[:, :, 1] / max(1.0, off.shape[0] * 0.035)

        # Convert image coordinates into eye-centered anterior/posterior axes.
        # For the left eye, rightward image motion is treated as anterior; for
        # the right eye, the horizontal axis is mirrored.
        anterior_axis = u if left_eye else -u
        posterior_axis = -anterior_axis
        superior_axis = -v
        inferior_axis = v
        off_gate = np.clip(off_patch, 0.0, 1.0)

        def _counter(axis: np.ndarray) -> float:
            return _clip01(float(np.mean(np.maximum(0.0, axis) * off_gate)) * 2.4)

        return {
            "superior": _counter(superior_axis),
            "anterior": _counter(anterior_axis),
            "inferior": _counter(inferior_axis),
            "posterior": _counter(posterior_axis),
            "off": _clip01(float(np.mean(off_gate))),
        }


class SimZFishOMRActionAdapter:
    """Stateful OMR/bout controller based on Z-Robot/simZFish motifs."""

    def __init__(self) -> None:
        self.retina = SimZFishRetina()
        self._left_pt = 0.0
        self._right_pt = 0.0
        self._turn_state = 0.0
        self._bout_state = 0.0
        self._ss_mlf = 0.0
        self._bout_ticks_left = 0
        self._bout_duration = 1
        self._active_force = 0.0
        self._active_side_score = 0.0
        self._active_bout_type = "coast"
        self._phase = 0.0

    def action_from_frame(
        self,
        *,
        current_gray: np.ndarray,
        previous_gray: np.ndarray,
        residual_flow: np.ndarray,
        features: dict[str, float],
        diagnostics: dict[str, Any],
        file_name: str,
        frame_index: int,
        video_time_s: float,
        sample_hz: float,
    ) -> tuple[ZebrafishActionLatent, dict[str, Any]]:
        retinal = self.retina.extract(
            current_gray=current_gray,
            previous_gray=previous_gray,
            residual_flow=residual_flow,
        )
        retinal_dict = retinal.as_dict()
        reliability = _clip01(float(diagnostics.get("flow_reliability", 0.0)))
        coherence = _clip01(float(diagnostics.get("residual_flow_coherence", diagnostics.get("flow_coherence", 0.0))))
        motion = _clip01(float(features.get("motion_energy", 0.0)))
        startle = _clip01(float(features.get("startle", 0.0)))

        left_progressive = retinal.left_posterior - retinal.left_anterior
        right_progressive = retinal.right_posterior - retinal.right_anterior
        bilateral_omr = _clip01(0.5 * (left_progressive + right_progressive) + 0.35 * motion)
        raw_turn = _clip11((right_progressive - left_progressive) * 1.35 + 0.35 * float(features.get("asymmetry", 0.0)))

        # These low-pass rates mirror the slow PT/MLF accumulation in the C
        # controller without copying its hidden parameter fit verbatim.
        alpha_pt = 0.10 if self._bout_state < 0.1 else 0.035
        self._left_pt = (1.0 - alpha_pt) * self._left_pt + alpha_pt * left_progressive
        self._right_pt = (1.0 - alpha_pt) * self._right_pt + alpha_pt * right_progressive
        self._turn_state = 0.78 * self._turn_state + 0.22 * raw_turn

        bout_evidence = _clip01(0.55 * bilateral_omr + 0.25 * motion + 0.20 * startle)
        kick_score = _clip01(float(_sigmoid(bout_evidence, omega=8.0, bias=0.30)))
        candidate_force = _clip01(0.72 * bout_evidence + 0.18 * startle)
        if reliability < 0.18 and startle < 0.4:
            candidate_force *= 0.45
            kick_score *= 0.55

        side_score = _clip11(-self._turn_state)
        if startle >= 0.55 and abs(side_score) < 0.08:
            side_score = _clip11(-float(features.get("asymmetry", 0.0)))
        confidence = _clip01(0.52 * reliability + 0.24 * coherence + 0.24 * min(1.0, motion * 1.5))
        sample_hz = max(1.0, float(sample_hz))

        if self._bout_ticks_left <= 0:
            # Normalized analogue of LeakyIntegrator.c: visual/MLF evidence
            # accumulates during glide, crosses a threshold, then emits a
            # finite bout. During glide there must be no tail-target injection.
            leak = 0.985
            increment = (0.030 + 0.155 * candidate_force + 0.245 * startle) * max(0.18, reliability)
            self._ss_mlf = float(np.clip(self._ss_mlf * leak + increment, 0.0, 1.25))
            threshold = 0.42 if startle < 0.55 else 0.16
            if self._ss_mlf >= threshold:
                duration_s = 0.16 + 0.18 * candidate_force + 0.10 * startle
                self._bout_duration = max(4, int(round(duration_s * sample_hz)))
                self._bout_ticks_left = self._bout_duration
                self._active_force = _clip01(0.22 + 0.78 * candidate_force + 0.22 * startle)
                self._active_side_score = side_score
                if startle >= 0.55:
                    self._active_bout_type = "startle_c_bend"
                elif abs(side_score) >= 0.18:
                    self._active_bout_type = "omr_turn_left" if side_score < 0.0 else "omr_turn_right"
                else:
                    self._active_bout_type = "omr_forward_bout"
                self._ss_mlf = 0.0

        if self._bout_ticks_left > 0:
            progress = self._bout_ticks_left / max(1, self._bout_duration)
            force = _clip01(self._active_force * (0.18 + 0.82 * progress))
            kick = 1.0
            bout_type = self._active_bout_type
            tail_frequency = zfc.TAIL_BEAT_FREQ_MIN_HZ + force * (
                zfc.TAIL_BEAT_FREQ_MAX_HZ - zfc.TAIL_BEAT_FREQ_MIN_HZ
            )
            self._phase = (self._phase + zfc.TWO_PI * tail_frequency / sample_hz) % zfc.TWO_PI
            tail_targets = tail_targets_from_scores(
                force=force,
                side_score=self._active_side_score,
                phase=self._phase,
                amplitude=force,
            )
            side_score = self._active_side_score
            self._bout_ticks_left -= 1
        else:
            force = 0.0
            kick = 0.0
            tail_frequency = 0.0
            tail_targets = tuple(0.0 for _ in range(zfc.N_BODY_SEGMENTS))
            bout_type = "coast"
        self._bout_state = (
            self._bout_ticks_left / max(1, self._bout_duration)
            if self._bout_ticks_left > 0
            else self._bout_state * 0.72
        )
        latent = ZebrafishActionLatent(
            source=file_name,
            branch="video_simzfish_omr",
            frame_index=int(frame_index),
            source_time_s=float(video_time_s),
            kick=kick,
            kick_score=kick_score,
            force=force,
            side_score=side_score,
            confidence=confidence,
            bout_type=bout_type,
            tail_frequency_hz=tail_frequency,
            tail_amplitude=force,
            tail_targets=tail_targets,
            metadata={
                "retina_model": "simzfish_image_c_off_dsc_python",
                "controller_model": "simzfish_omr_leaky_integrator_python",
                "distribution": "natural_video_estimated_retina_not_projector_raw",
            },
        ).clipped()
        diagnostics_out = {
            "simzfish_action": latent.to_state_dict(),
            "simzfish_retinal_counters": retinal_dict,
            "simzfish_left_pt": float(self._left_pt),
            "simzfish_right_pt": float(self._right_pt),
            "simzfish_turn_state": float(self._turn_state),
            "simzfish_bout_state": float(self._bout_state),
            "simzfish_ss_mlf": float(self._ss_mlf),
            "simzfish_bout_ticks_left": float(self._bout_ticks_left),
            "simzfish_candidate_force": float(candidate_force),
            "simzfish_primary_action_model": "retina_off_dsc_to_omr_bout",
        }
        return latent, diagnostics_out
