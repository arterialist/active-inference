"""
Larval zebrafish aquatic arena.

The environment exposes stimulus channels that are common in public zebrafish
behavior and physiology datasets: visual targets / optic flow, odor gradients,
lateral-line water flow, thermal preference, wall proximity, and startle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from simulations.base_environment import BaseEnvironment, EnvironmentObservation
from simulations.zebrafish import config as zfc
from simulations.zebrafish.action_latent import (
    ZebrafishActionLatent,
    coerce_tail_targets,
)


def _angle_wrap(x: float) -> float:
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class FoodParticle:
    position: np.ndarray
    intensity: float = 1.0


@dataclass
class VideoStimulusFrame:
    """Retinal / optic-flow features extracted from a user-selected video."""

    file_name: str = ""
    frame_index: int = 0
    video_time_s: float = 0.0
    visual_left: float = 0.0
    visual_right: float = 0.0
    optic_flow_left: float = 0.0
    optic_flow_right: float = 0.0
    lateral_line_left: float = 0.0
    lateral_line_right: float = 0.0
    visual_up: float = 0.0
    visual_down: float = 0.0
    light_level: float = 0.0
    startle: float = 0.0
    motion_energy: float = 0.0
    asymmetry: float = 0.0
    action_kick: float = 0.0
    action_force: float = 0.0
    action_side_score: float = 0.0
    action_kick_score: float = 0.0
    action_confidence: float = 0.0
    action_source: str = ""
    action_branch: str = ""
    action_bout_type: str = "none"
    tail_frequency_hz: float = 0.0
    tail_amplitude: float = 0.0
    tail_targets: tuple[float, ...] = ()
    zapbench_row: int = 0
    backend_extracted: bool = False
    received_tick: int = 0


@dataclass
class CalciumActionFrame:
    """Tail-action command decoded from a ZAPBench calcium-imaging window."""

    source: str = ""
    row: int = 0
    frame_index: int = 0
    calcium_time_s: float = 0.0
    kick: float = 0.0
    side_score: float = 0.0
    force: float = 0.0
    kick_score: float = 0.0
    confidence: float = 0.0
    action_branch: str = "zapbench_calcium_ephys"
    action_bout_type: str = "none"
    tail_frequency_hz: float = 0.0
    tail_amplitude: float = 0.0
    tail_targets: tuple[float, ...] = ()
    received_tick: int = 0


class AquaticArenaEnvironment(BaseEnvironment):
    """Circular shallow-water arena for larval zebrafish."""

    def __init__(
        self,
        food_positions: list[tuple[float, float, float]] | None = None,
        arena_radius_m: float = zfc.ARENA_RADIUS_M,
        water_flow_m_s: float = zfc.WATER_FLOW_M_S,
        temperature_c: float = zfc.TEMPERATURE_C_DEFAULT,
        light_level: float = zfc.LIGHT_LEVEL_DEFAULT,
    ):
        self._arena_radius_m = float(arena_radius_m)
        self._water_flow = np.array([float(water_flow_m_s), 0.0, 0.0])
        self._temperature_c = float(temperature_c)
        self._light_level = float(light_level)
        self._food: list[FoodParticle] = []
        self._head_position = np.zeros(3)
        self._heading = 0.0
        self._tick = 0
        self._startle_until_tick = -1
        self._video_enabled = False
        self._video_gain = 1.0
        self._video_frame: VideoStimulusFrame | None = None
        self._calcium_enabled = False
        self._calcium_gain = 1.0
        self._calcium_frame: CalciumActionFrame | None = None
        self.replace_food_sources(food_positions or zfc.DEFAULT_FOOD_POSITIONS)

    @property
    def arena_radius_m(self) -> float:
        return self._arena_radius_m

    def reset(self) -> EnvironmentObservation:
        self._head_position = np.zeros(3)
        self._heading = 0.0
        self._tick = 0
        self._startle_until_tick = -1
        if self._video_frame is not None:
            self._video_frame.received_tick = 0
        if self._calcium_frame is not None:
            self._calcium_frame.received_tick = 0
        return self._build_observation()

    def step(self, body_state: dict[str, Any]) -> EnvironmentObservation:
        self._tick += 1
        head = body_state.get("head_position", np.zeros(3))
        self._head_position = np.asarray(head, dtype=float).copy()
        extra = body_state.get("extra", {}) or {}
        self._heading = float(extra.get("heading", self._heading))
        return self._build_observation()

    def post_body_step(self, body_state: dict[str, Any]) -> None:
        head = np.asarray(body_state.get("head_position", np.zeros(3)), dtype=float)
        self._food = [
            fp
            for fp in self._food
            if float(np.linalg.norm(fp.position[:2] - head[:2]))
            > zfc.FOOD_CONSUMPTION_RADIUS_M
        ]

    def render(self) -> np.ndarray | None:
        size = 256
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:, :, :] = [18, 35, 48]
        half = self._arena_radius_m
        xs = np.linspace(-half, half, size)
        ys = np.linspace(-half, half, size)
        xx, yy = np.meshgrid(xs, ys)
        rr = np.sqrt(xx * xx + yy * yy)
        img[rr <= half] = [23, 70, 86]
        for fp in self._food:
            px, py = self._world_to_px(fp.position[:2], size)
            img[max(0, py - 2) : min(size, py + 3), max(0, px - 2) : min(size, px + 3)] = [
                255,
                170,
                64,
            ]
        hx, hy = self._world_to_px(self._head_position[:2], size)
        img[max(0, hy - 2) : min(size, hy + 3), max(0, hx - 2) : min(size, hx + 3)] = [
            120,
            220,
            255,
        ]
        return img

    def get_chemical_concentration(
        self, molecule: str, position: np.ndarray
    ) -> float:
        if molecule not in {"food_odor", "amino_acid"}:
            return 0.0
        return self._odor_at(np.asarray(position, dtype=float))

    def get_active_food_positions(self) -> list[tuple[float, float, float]]:
        return [tuple(fp.position.tolist()) for fp in self._food]

    def replace_food_sources(self, positions: list[tuple[float, float, float]]) -> None:
        self._food = [
            FoodParticle(position=np.asarray(pos, dtype=float).copy())
            for pos in positions
        ]

    def add_food(self, position: tuple[float, float, float]) -> bool:
        pos = np.asarray(position, dtype=float)
        if np.linalg.norm(pos[:2]) > self._arena_radius_m:
            return False
        self._food.append(FoodParticle(position=pos))
        return True

    def clear_food(self) -> None:
        self._food.clear()

    def set_water_flow(self, x_m_s: float, y_m_s: float = 0.0) -> None:
        self._water_flow = np.array([float(x_m_s), float(y_m_s), 0.0])

    def set_temperature(self, temperature_c: float) -> None:
        self._temperature_c = float(temperature_c)

    def set_light_level(self, light_level: float) -> None:
        self._light_level = float(np.clip(light_level, 0.0, 1.0))

    def set_video_stimulus(
        self,
        *,
        enabled: bool | None = None,
        gain: float | None = None,
        file_name: str | None = None,
    ) -> None:
        if enabled is not None:
            self._video_enabled = bool(enabled)
        if gain is not None:
            self._video_gain = float(np.clip(gain, 0.0, 3.0))
        if file_name is not None:
            if self._video_frame is None:
                self._video_frame = VideoStimulusFrame()
            self._video_frame.file_name = str(file_name)
            self._video_frame.received_tick = int(self._tick)

    def push_video_stimulus_frame(self, features: dict[str, Any]) -> dict[str, Any]:
        def _f(name: str, default: float = 0.0) -> float:
            return float(np.clip(float(features.get(name, default)), 0.0, 1.0))

        tail_targets = coerce_tail_targets(features.get("tail_targets"))

        frame = VideoStimulusFrame(
            file_name=str(features.get("file_name", "")),
            frame_index=int(features.get("frame_index", 0)),
            video_time_s=float(features.get("video_time_s", 0.0)),
            visual_left=_f("visual_left"),
            visual_right=_f("visual_right"),
            optic_flow_left=_f("optic_flow_left"),
            optic_flow_right=_f("optic_flow_right"),
            lateral_line_left=_f("lateral_line_left"),
            lateral_line_right=_f("lateral_line_right"),
            visual_up=_f("visual_up"),
            visual_down=_f("visual_down"),
            light_level=_f("light_level"),
            startle=_f("startle"),
            motion_energy=_f("motion_energy"),
            asymmetry=float(np.clip(float(features.get("asymmetry", 0.0)), -1.0, 1.0)),
            action_kick=_f("action_kick"),
            action_force=_f("action_force"),
            action_side_score=float(np.clip(float(features.get("action_side_score", 0.0)), -1.0, 1.0)),
            action_kick_score=_f("action_kick_score"),
            action_confidence=_f("action_confidence"),
            action_source=str(features.get("action_source", features.get("file_name", ""))),
            action_branch=str(features.get("action_branch", "")),
            action_bout_type=str(features.get("action_bout_type", "none")),
            tail_frequency_hz=float(max(0.0, float(features.get("tail_frequency_hz", 0.0)))),
            tail_amplitude=_f("tail_amplitude"),
            tail_targets=tail_targets,
            zapbench_row=int(features.get("zapbench_row", 0)),
            backend_extracted=bool(features.get("backend_extracted", False)),
            received_tick=int(self._tick),
        )
        self._video_frame = frame
        if "enabled" in features:
            self._video_enabled = bool(features["enabled"])
        return self.video_stimulus_state()

    def clear_video_stimulus(self) -> None:
        self._video_enabled = False
        self._video_frame = None

    def video_stimulus_state(self) -> dict[str, Any]:
        frame = self._video_frame
        return {
            "enabled": bool(self._video_enabled),
            "gain": float(self._video_gain),
            "has_frame": frame is not None,
            "file_name": frame.file_name if frame else "",
            "frame_index": int(frame.frame_index) if frame else 0,
            "video_time_s": float(frame.video_time_s) if frame else 0.0,
            "received_tick": int(frame.received_tick) if frame else None,
            "age_ticks": int(self._tick - frame.received_tick) if frame else None,
            "visual_left": float(frame.visual_left) if frame else 0.0,
            "visual_right": float(frame.visual_right) if frame else 0.0,
            "optic_flow_left": float(frame.optic_flow_left) if frame else 0.0,
            "optic_flow_right": float(frame.optic_flow_right) if frame else 0.0,
            "lateral_line_left": float(frame.lateral_line_left) if frame else 0.0,
            "lateral_line_right": float(frame.lateral_line_right) if frame else 0.0,
            "visual_up": float(frame.visual_up) if frame else 0.0,
            "visual_down": float(frame.visual_down) if frame else 0.0,
            "light_level": float(frame.light_level) if frame else 0.0,
            "startle": float(frame.startle) if frame else 0.0,
            "motion_energy": float(frame.motion_energy) if frame else 0.0,
            "asymmetry": float(frame.asymmetry) if frame else 0.0,
            "action_kick": float(frame.action_kick) if frame else 0.0,
            "action_force": float(frame.action_force) if frame else 0.0,
            "action_side_score": float(frame.action_side_score) if frame else 0.0,
            "action_kick_score": float(frame.action_kick_score) if frame else 0.0,
            "action_confidence": float(frame.action_confidence) if frame else 0.0,
            "action_source": frame.action_source if frame else "",
            "action_branch": frame.action_branch if frame else "",
            "action_bout_type": frame.action_bout_type if frame else "none",
            "tail_frequency_hz": float(frame.tail_frequency_hz) if frame else 0.0,
            "tail_amplitude": float(frame.tail_amplitude) if frame else 0.0,
            "tail_targets": list(frame.tail_targets) if frame else [],
            "zapbench_row": int(frame.zapbench_row) if frame else 0,
            "backend_extracted": bool(frame.backend_extracted) if frame else False,
        }

    def set_calcium_stimulus(
        self,
        *,
        enabled: bool | None = None,
        gain: float | None = None,
        source: str | None = None,
    ) -> None:
        if enabled is not None:
            self._calcium_enabled = bool(enabled)
        if gain is not None:
            self._calcium_gain = float(np.clip(gain, 0.0, 3.0))
        if source is not None:
            if self._calcium_frame is None:
                self._calcium_frame = CalciumActionFrame()
            self._calcium_frame.source = str(source)
            self._calcium_frame.received_tick = int(self._tick)

    def push_calcium_action_frame(self, action: dict[str, Any]) -> dict[str, Any]:
        def _f(name: str, default: float = 0.0, lo: float = 0.0, hi: float = 1.0) -> float:
            return float(np.clip(float(action.get(name, default)), lo, hi))

        phase = float(action.get("tail_phase", 0.0))
        latent = ZebrafishActionLatent.from_zapbench_ephys(
            source=str(action.get("source", "")),
            row=int(action.get("row", 0)),
            frame_index=int(action.get("frame_index", 0)),
            calcium_time_s=float(action.get("calcium_time_s", 0.0)),
            kick=_f("kick"),
            side_score=_f("side_score", lo=-1.0, hi=1.0),
            force=_f("force"),
            kick_score=_f("kick_score"),
            confidence=_f("confidence"),
            phase=phase,
        )
        frame = CalciumActionFrame(
            source=str(action.get("source", "")),
            row=int(action.get("row", 0)),
            frame_index=int(action.get("frame_index", 0)),
            calcium_time_s=float(action.get("calcium_time_s", 0.0)),
            kick=latent.kick,
            side_score=latent.side_score,
            force=latent.force,
            kick_score=latent.kick_score,
            confidence=latent.confidence,
            action_branch=latent.branch,
            action_bout_type=latent.bout_type,
            tail_frequency_hz=latent.tail_frequency_hz,
            tail_amplitude=latent.tail_amplitude,
            tail_targets=latent.tail_targets,
            received_tick=int(self._tick),
        )
        self._calcium_frame = frame
        if "enabled" in action:
            self._calcium_enabled = bool(action["enabled"])
        return self.calcium_stimulus_state()

    def clear_calcium_stimulus(self) -> None:
        self._calcium_enabled = False
        self._calcium_frame = None

    def calcium_stimulus_state(self) -> dict[str, Any]:
        frame = self._calcium_frame
        side = "none"
        if frame and frame.kick >= 0.5 and frame.force > 0.02:
            if frame.side_score < -0.05:
                side = "left"
            elif frame.side_score > 0.05:
                side = "right"
        return {
            "enabled": bool(self._calcium_enabled),
            "gain": float(self._calcium_gain),
            "has_frame": frame is not None,
            "source": frame.source if frame else "",
            "row": int(frame.row) if frame else 0,
            "frame_index": int(frame.frame_index) if frame else 0,
            "calcium_time_s": float(frame.calcium_time_s) if frame else 0.0,
            "received_tick": int(frame.received_tick) if frame else None,
            "age_ticks": int(self._tick - frame.received_tick) if frame else None,
            "kick": float(frame.kick) if frame else 0.0,
            "side": side,
            "side_score": float(frame.side_score) if frame else 0.0,
            "force": float(frame.force) if frame else 0.0,
            "kick_score": float(frame.kick_score) if frame else 0.0,
            "confidence": float(frame.confidence) if frame else 0.0,
            "action_branch": frame.action_branch if frame else "zapbench_calcium_ephys",
            "action_bout_type": frame.action_bout_type if frame else "none",
            "tail_frequency_hz": float(frame.tail_frequency_hz) if frame else 0.0,
            "tail_amplitude": float(frame.tail_amplitude) if frame else 0.0,
            "tail_targets": list(frame.tail_targets) if frame else [],
        }

    def environment_state(self) -> dict[str, Any]:
        return {
            "arena_radius_m": self._arena_radius_m,
            "arena_depth_m": zfc.ARENA_DEPTH_M,
            "water_flow_m_s": self._water_flow.tolist(),
            "temperature_c": self._temperature_c,
            "light_level": self._light_level,
            "food_count": len(self._food),
            "video_stimulus": self.video_stimulus_state(),
            "calcium_stimulus": self.calcium_stimulus_state(),
        }

    def remove_food_near(
        self,
        position: tuple[float, float, float],
        radius_m: float = zfc.FOOD_CONSUMPTION_RADIUS_M * 2.0,
    ) -> bool:
        if not self._food:
            return False
        pos = np.asarray(position, dtype=float)
        distances = [float(np.linalg.norm(fp.position[:2] - pos[:2])) for fp in self._food]
        idx = int(np.argmin(distances))
        if distances[idx] <= radius_m:
            del self._food[idx]
            return True
        return False

    def trigger_startle(self, duration_ticks: int = 18) -> None:
        self._startle_until_tick = max(self._startle_until_tick, self._tick + duration_ticks)

    def _build_observation(self) -> EnvironmentObservation:
        target = self._nearest_food()
        visual_left = 0.0
        visual_right = 0.0
        target_distance = float("inf")
        target_bearing = 0.0
        if target is not None:
            vec = target.position[:2] - self._head_position[:2]
            target_distance = float(np.linalg.norm(vec))
            if target_distance > 1e-9:
                target_bearing = _angle_wrap(float(np.arctan2(vec[1], vec[0])) - self._heading)
                visibility = float(np.exp(-target_distance / zfc.VISUAL_RANGE_M))
                if abs(target_bearing) < np.deg2rad(115.0):
                    side = np.sin(target_bearing)
                    midline = max(0.0, np.cos(target_bearing))
                    visual_left = visibility * max(0.0, side + 0.35 * midline)
                    visual_right = visibility * max(0.0, -side + 0.35 * midline)

        heading_vec = np.array([np.cos(self._heading), np.sin(self._heading), 0.0])
        lateral_vec = np.array([-np.sin(self._heading), np.cos(self._heading), 0.0])
        optic_flow = -float(np.dot(self._water_flow, heading_vec))
        lateral_flow = float(np.dot(self._water_flow, lateral_vec))

        radial = float(np.linalg.norm(self._head_position[:2]))
        wall_margin = self._arena_radius_m - radial
        wall_signal = max(0.0, (zfc.WALL_AVOIDANCE_MARGIN_M - wall_margin) / zfc.WALL_AVOIDANCE_MARGIN_M)
        wall_left = 0.0
        wall_right = 0.0
        if wall_signal > 0.0 and radial > 1e-9:
            outward = float(np.arctan2(self._head_position[1], self._head_position[0]))
            bearing = _angle_wrap(outward - self._heading)
            if np.sin(bearing) > 0:
                wall_left = wall_signal
            else:
                wall_right = wall_signal

        temp_delta = self._temperature_c - zfc.TEMPERATURE_PREFERRED_C
        hot = max(0.0, temp_delta / 5.0)
        cold = max(0.0, -temp_delta / 5.0)
        startle = 1.0 if self._tick <= self._startle_until_tick else 0.0
        depth_m = float(np.clip(-self._head_position[2], 0.0, zfc.ARENA_DEPTH_M))
        preferred_depth_m = float(np.clip(-zfc.DEPTH_HOME_M, 1e-6, zfc.ARENA_DEPTH_M))
        shallow = max(0.0, (preferred_depth_m - depth_m) / preferred_depth_m)
        deep_room = max(1e-6, zfc.ARENA_DEPTH_M - preferred_depth_m)
        deep = max(0.0, (depth_m - preferred_depth_m) / deep_room)

        chemicals = {
            "food_odor": self._odor_at(self._head_position),
            "amino_acid": self._odor_at(self._head_position) * 0.85,
        }
        extra = {
            "visual_left": float(np.clip(visual_left, 0.0, 1.0)),
            "visual_right": float(np.clip(visual_right, 0.0, 1.0)),
            "target_distance_m": target_distance,
            "target_bearing_rad": target_bearing,
            "optic_flow_left": float(np.clip(max(0.0, optic_flow + lateral_flow) * 18.0, 0.0, 1.0)),
            "optic_flow_right": float(np.clip(max(0.0, optic_flow - lateral_flow) * 18.0, 0.0, 1.0)),
            "lateral_line_left": float(np.clip(max(0.0, lateral_flow) * 120.0, 0.0, 1.0)),
            "lateral_line_right": float(np.clip(max(0.0, -lateral_flow) * 120.0, 0.0, 1.0)),
            "thermo_hot": float(np.clip(hot, 0.0, 1.0)),
            "thermo_cold": float(np.clip(cold, 0.0, 1.0)),
            "wall_left": float(np.clip(wall_left, 0.0, 1.0)),
            "wall_right": float(np.clip(wall_right, 0.0, 1.0)),
            "depth_m": depth_m,
            "depth_shallow": float(np.clip(shallow, 0.0, 1.0)),
            "depth_deep": float(np.clip(deep, 0.0, 1.0)),
            "startle": startle,
            "light_level": self._light_level,
            "water_flow_m_s": self._water_flow.copy(),
        }
        self._apply_video_stimulus(extra)
        self._apply_calcium_stimulus(extra)
        return EnvironmentObservation(chemicals=chemicals, extra=extra)

    def _apply_video_stimulus(self, extra: dict[str, Any]) -> None:
        frame = self._video_frame
        if not self._video_enabled or frame is None:
            extra["stimulus_mode"] = "arena"
            extra["video_active"] = 0.0
            return
        gain = float(self._video_gain)
        extra["stimulus_mode"] = "video"
        extra["video_active"] = 1.0
        extra["video_file_name"] = frame.file_name
        extra["video_frame_index"] = int(frame.frame_index)
        extra["video_time_s"] = float(frame.video_time_s)
        extra["video_motion_energy"] = float(frame.motion_energy)
        extra["video_asymmetry"] = float(frame.asymmetry)
        extra["video_action_kick"] = float(frame.action_kick)
        extra["video_action_force"] = float(frame.action_force)
        extra["video_action_side_score"] = float(frame.action_side_score)
        extra["video_action_kick_score"] = float(frame.action_kick_score)
        extra["video_action_confidence"] = float(frame.action_confidence)
        extra["video_action_source"] = frame.action_source
        extra["video_action_branch"] = frame.action_branch
        extra["video_action_bout_type"] = frame.action_bout_type
        extra["video_tail_frequency_hz"] = float(frame.tail_frequency_hz)
        extra["video_tail_amplitude"] = float(frame.tail_amplitude)
        extra["video_zapbench_row"] = int(frame.zapbench_row)
        extra["video_backend_extracted"] = 1.0 if frame.backend_extracted else 0.0
        self._apply_common_action_extra(
            extra,
            active=1.0,
            branch=frame.action_branch or "video",
            kick=frame.action_kick,
            force=frame.action_force,
            side_score=frame.action_side_score,
            kick_score=frame.action_kick_score,
            confidence=frame.action_confidence,
            bout_type=frame.action_bout_type,
            tail_frequency_hz=frame.tail_frequency_hz,
            tail_amplitude=frame.tail_amplitude,
            tail_targets=frame.tail_targets,
        )
        extra["visual_left"] = float(np.clip(frame.visual_left * gain, 0.0, 1.0))
        extra["visual_right"] = float(np.clip(frame.visual_right * gain, 0.0, 1.0))
        extra["optic_flow_left"] = float(np.clip(frame.optic_flow_left * gain, 0.0, 1.0))
        extra["optic_flow_right"] = float(np.clip(frame.optic_flow_right * gain, 0.0, 1.0))
        extra["lateral_line_left"] = float(np.clip(frame.lateral_line_left * gain, 0.0, 1.0))
        extra["lateral_line_right"] = float(np.clip(frame.lateral_line_right * gain, 0.0, 1.0))
        extra["visual_up"] = float(np.clip(frame.visual_up * gain, 0.0, 1.0))
        extra["visual_down"] = float(np.clip(frame.visual_down * gain, 0.0, 1.0))
        extra["light_level"] = float(np.clip(frame.light_level, 0.0, 1.0))
        extra["startle"] = max(float(extra.get("startle", 0.0)), float(frame.startle))
        vertical_delta = extra["visual_up"] - extra["visual_down"]
        deadband = float(np.clip(zfc.VIDEO_DEPTH_CUE_DEADBAND, 0.0, 0.95))
        if abs(vertical_delta) > deadband:
            cue = (abs(vertical_delta) - deadband) / max(1e-6, 1.0 - deadband)
            cue *= float(np.clip(zfc.VIDEO_DEPTH_CUE_GAIN, 0.0, 1.0))
            if vertical_delta > 0.0:
                extra["depth_deep"] = max(float(extra.get("depth_deep", 0.0)), cue)
            else:
                extra["depth_shallow"] = max(float(extra.get("depth_shallow", 0.0)), cue)

    def _apply_calcium_stimulus(self, extra: dict[str, Any]) -> None:
        frame = self._calcium_frame
        if not self._calcium_enabled or frame is None:
            extra["calcium_active"] = 0.0
            return
        gain = float(self._calcium_gain)
        age_ticks = max(0, int(self._tick) - int(frame.received_tick))
        if age_ticks <= int(zfc.CALCIUM_ACTION_PULSE_TICKS):
            pulse = float(np.exp(-age_ticks / float(zfc.CALCIUM_ACTION_DECAY_TICKS)))
        else:
            pulse = 0.0
        kick = float(np.clip(frame.kick, 0.0, 1.0))
        force = float(np.clip(frame.force * gain * zfc.CALCIUM_FORCE_GAIN * pulse, 0.0, 1.0))
        extra["calcium_active"] = 1.0
        extra["calcium_source"] = frame.source
        extra["calcium_row"] = int(frame.row)
        extra["calcium_frame_index"] = int(frame.frame_index)
        extra["calcium_time_s"] = float(frame.calcium_time_s)
        extra["calcium_age_ticks"] = float(age_ticks)
        extra["calcium_pulse"] = pulse
        extra["calcium_kick"] = kick if kick >= 0.5 and pulse > 0.05 else 0.0
        extra["calcium_side_score"] = float(np.clip(frame.side_score * pulse, -1.0, 1.0))
        extra["calcium_force"] = force
        extra["calcium_kick_score"] = float(np.clip(frame.kick_score, 0.0, 1.0))
        extra["calcium_confidence"] = float(np.clip(frame.confidence, 0.0, 1.0))
        extra["calcium_action_branch"] = frame.action_branch
        extra["calcium_action_bout_type"] = frame.action_bout_type
        extra["calcium_tail_frequency_hz"] = float(frame.tail_frequency_hz)
        extra["calcium_tail_amplitude"] = float(frame.tail_amplitude)
        pulse_targets = tuple(float(np.clip(v * gain * pulse, -1.0, 1.0)) for v in frame.tail_targets)
        self._apply_common_action_extra(
            extra,
            active=1.0,
            branch=frame.action_branch,
            kick=extra["calcium_kick"],
            force=force,
            side_score=extra["calcium_side_score"],
            kick_score=frame.kick_score,
            confidence=frame.confidence,
            bout_type=frame.action_bout_type,
            tail_frequency_hz=frame.tail_frequency_hz,
            tail_amplitude=frame.tail_amplitude * pulse,
            tail_targets=pulse_targets,
        )

    def _apply_common_action_extra(
        self,
        extra: dict[str, Any],
        *,
        active: float,
        branch: str,
        kick: float,
        force: float,
        side_score: float,
        kick_score: float,
        confidence: float,
        bout_type: str,
        tail_frequency_hz: float,
        tail_amplitude: float,
        tail_targets: tuple[float, ...],
    ) -> None:
        targets = coerce_tail_targets(tail_targets)
        extra["action_active"] = float(np.clip(active, 0.0, 1.0))
        extra["action_branch"] = str(branch)
        extra["action_kick"] = float(np.clip(kick, 0.0, 1.0))
        extra["action_force"] = float(np.clip(force, 0.0, 1.0))
        extra["action_side_score"] = float(np.clip(side_score, -1.0, 1.0))
        extra["action_kick_score"] = float(np.clip(kick_score, 0.0, 1.0))
        extra["action_confidence"] = float(np.clip(confidence, 0.0, 1.0))
        extra["action_bout_type"] = str(bout_type)
        extra["action_tail_frequency_hz"] = float(max(0.0, tail_frequency_hz))
        extra["action_tail_amplitude"] = float(np.clip(tail_amplitude, 0.0, 1.0))
        extra["action_tail_target_confidence"] = float(
            np.clip(confidence * max(force, tail_amplitude), 0.0, 1.0)
        )
        for idx, value in enumerate(targets):
            extra[f"action_tail_target_{idx:02d}"] = float(np.clip(value, -1.0, 1.0))

    def _nearest_food(self) -> FoodParticle | None:
        if not self._food:
            return None
        distances = [
            float(np.linalg.norm(fp.position[:2] - self._head_position[:2]))
            for fp in self._food
        ]
        return self._food[int(np.argmin(distances))]

    def _odor_at(self, position: np.ndarray) -> float:
        best = 0.0
        for fp in self._food:
            dist = float(np.linalg.norm(fp.position[:2] - position[:2]))
            best = max(best, fp.intensity * np.exp(-zfc.FOOD_GRADIENT_DECAY * dist))
        return float(np.clip(best, 0.0, 1.0))

    def _world_to_px(self, xy: np.ndarray, size: int) -> tuple[int, int]:
        half = self._arena_radius_m
        px = int((float(xy[0]) + half) / (2 * half) * (size - 1))
        py = int((half - float(xy[1])) / (2 * half) * (size - 1))
        return int(np.clip(px, 0, size - 1)), int(np.clip(py, 0, size - 1))
