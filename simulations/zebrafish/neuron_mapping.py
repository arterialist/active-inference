"""
PAULA-backed reduced nervous system for larval zebrafish.

The wrapper uses PAULA neurons for sensory integration and circuit state, then
adds explicit zebrafish motor motifs outside ``neuron.py``: bout timing,
reticulospinal turn bias, and a spinal tail-wave CPG. This keeps PAULA as the
local steerable dynamical substrate while avoiding changes to the core neuron
model.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from simulations.base_nervous_system import BaseNervousSystem
from simulations.connectome_loader import ConnectomeData, build_paula_network
from simulations.paula_loader import ensure_paula_available
from simulations.zebrafish import config as zfc

ensure_paula_available()
from neuron.neuron import NeuronParameters  # noqa: E402
from neuron.network import NeuronNetwork  # noqa: E402


class ZebrafishNervousSystem(BaseNervousSystem):
    """Reduced zebrafish nervous system backed by a PAULA network."""

    def __init__(
        self,
        connectome: ConnectomeData,
        log_level: str = "WARNING",
        seed: int | None = 7,
    ):
        self._connectome = connectome
        self._log_level = log_level
        self._rng = np.random.default_rng(seed)
        self._network: NeuronNetwork | None = None
        self._name_to_id: dict[str, int] = {}
        self._motor_outputs = {name: 0.0 for name in zfc.MUSCLE_NAMES}
        self._last_sensory: dict[str, float] = {}
        self._bout_ticks_left = 0
        self._coast_ticks_left = 20
        self._bout_drive = 0.0
        self._turn_bias = 0.0
        self._pitch_bias = 0.0
        self._tail_phase = 0.0
        self._external_tail_targets = np.zeros(zfc.N_BODY_SEGMENTS, dtype=float)
        self._external_tail_confidence = 0.0
        self._startle_ticks_left = 0
        self._enable_m0 = True
        self._enable_m1 = True
        self._plasticity_disabled = True
        self._plasticity_eta_post = 0.001
        self._plasticity_eta_retro = 0.001
        self.tail_cpg_enabled = True
        self.spontaneous_bout_prob = float(zfc.SPONTANEOUS_BOUT_PROB)
        self.baseline_drive = 0.10
        self.visual_approach_gain = 0.45
        self.odor_approach_gain = 0.55
        self.optic_flow_drive_gain = 0.28
        self.wall_avoidance_drive_gain = 0.65
        self.visual_turn_gain = 0.42
        self.odor_turn_gain = 0.28
        self.optic_flow_turn_gain = 0.20
        self.lateral_line_turn_gain = 0.18
        self.wall_turn_gain = 0.70
        self.paula_activity_gain = 3.0
        self.sensory_input_gain = 1.0
        self._build()

    def reset(self, *, rebuild_network: bool = True) -> None:
        if rebuild_network or self._network is None:
            self._build()
        else:
            self._network.reset_simulation()
        self._motor_outputs = {name: 0.0 for name in zfc.MUSCLE_NAMES}
        self._last_sensory.clear()
        self._bout_ticks_left = 0
        self._coast_ticks_left = 20
        self._bout_drive = 0.0
        self._turn_bias = 0.0
        self._pitch_bias = 0.0
        self._tail_phase = 0.0
        self._external_tail_targets = np.zeros(zfc.N_BODY_SEGMENTS, dtype=float)
        self._external_tail_confidence = 0.0
        self._startle_ticks_left = 0

    def tick(
        self,
        sensory_inputs: dict[str, float],
        current_tick: int,
    ) -> dict[str, float]:
        if self._network is None:
            raise RuntimeError("Network not initialised")

        self._inject_sensory(sensory_inputs)
        self._network.run_tick()
        self._update_behavior_state(sensory_inputs, current_tick)
        self._motor_outputs = self._decode_tail_motors(current_tick)
        self._last_sensory = dict(sensory_inputs)
        return dict(self._motor_outputs)

    def get_neuron_states(self) -> dict[str, Any]:
        if self._network is None:
            return {}
        states: dict[str, Any] = {}
        for nid, neuron in self._network.network.neurons.items():
            name = str(neuron.metadata.get("name", str(nid)))
            states[f"{name}_S"] = float(neuron.S)
            states[f"{name}_fired"] = float(neuron.O > 0)
        states["behavior_swim_drive"] = float(self._bout_drive)
        states["behavior_turn_bias"] = float(self._turn_bias)
        states["behavior_pitch_bias"] = float(self._pitch_bias)
        return states

    def get_neuron_names_paula_order(self) -> list[str]:
        if self._network is None:
            return []
        neurons = self._network.network.neurons
        return [
            str(neurons[i].metadata.get("name", str(i)))
            for i in sorted(neurons.keys(), key=int)
        ]

    def get_compact_neural_snapshot(self) -> tuple[list[float], list[int], list[float]]:
        if self._network is None:
            return [], [], []
        neurons = self._network.network.neurons
        ids = sorted(neurons.keys(), key=int)
        return (
            [float(neurons[i].S) for i in ids],
            [1 if float(neurons[i].O) > 0 else 0 for i in ids],
            [float(neurons[i].r) for i in ids],
        )

    def get_compact_neural_detail_snapshot(self) -> dict[str, list[Any]]:
        """Return the live PAULA state vectors used by the lab inspector."""
        if self._network is None:
            return {
                "names": [],
                "s": [],
                "o": [],
                "fired": [],
                "r": [],
                "b": [],
                "t_ref": [],
                "f_avg": [],
                "t_last_fire": [],
                "m0": [],
                "m1": [],
                "pq_len": [],
            }
        neurons = self._network.network.neurons
        ids = sorted(neurons.keys(), key=int)
        names: list[str] = []
        s_vals: list[float] = []
        o_vals: list[float] = []
        fired: list[int] = []
        r_vals: list[float] = []
        b_vals: list[float] = []
        t_ref_vals: list[float] = []
        f_avg_vals: list[float] = []
        t_last_fire_vals: list[float | None] = []
        m0_vals: list[float] = []
        m1_vals: list[float] = []
        pq_len_vals: list[int] = []
        for nid in ids:
            neuron = neurons[nid]
            m_vec = np.asarray(getattr(neuron, "M_vector", []), dtype=float)
            t_last_fire = float(getattr(neuron, "t_last_fire", 0.0))
            names.append(str(neuron.metadata.get("name", str(nid))))
            s_vals.append(float(neuron.S))
            o_vals.append(float(neuron.O))
            fired.append(1 if float(neuron.O) > 0 else 0)
            r_vals.append(float(neuron.r))
            b_vals.append(float(neuron.b))
            t_ref_vals.append(float(neuron.t_ref))
            f_avg_vals.append(float(neuron.F_avg))
            t_last_fire_vals.append(t_last_fire if np.isfinite(t_last_fire) else None)
            m0_vals.append(float(m_vec[0]) if m_vec.size > 0 else 0.0)
            m1_vals.append(float(m_vec[1]) if m_vec.size > 1 else 0.0)
            pq_len_vals.append(int(len(neuron.propagation_queue)))
        return {
            "names": names,
            "s": s_vals,
            "o": o_vals,
            "fired": fired,
            "r": r_vals,
            "b": b_vals,
            "t_ref": t_ref_vals,
            "f_avg": f_avg_vals,
            "t_last_fire": t_last_fire_vals,
            "m0": m0_vals,
            "m1": m1_vals,
            "pq_len": pq_len_vals,
        }

    @property
    def n_neurons(self) -> int:
        return len(self._name_to_id)

    @property
    def behavior_state(self) -> dict[str, float | int]:
        return {
            "swim_drive": float(self._bout_drive),
            "turn_bias": float(self._turn_bias),
            "pitch_bias": float(self._pitch_bias),
            "external_tail_confidence": float(self._external_tail_confidence),
            "bout_ticks_left": int(self._bout_ticks_left),
            "coast_ticks_left": int(self._coast_ticks_left),
            "startle_ticks_left": int(self._startle_ticks_left),
        }

    @property
    def neuromod_levels(self) -> tuple[float, float]:
        """Return lab-level M0/M1 proxies, honoring neuromodulation toggles."""
        m0 = abs(float(self._turn_bias)) + abs(float(self._pitch_bias))
        m1 = float(self._bout_drive)
        return (
            m0 if self._enable_m0 else 0.0,
            m1 if self._enable_m1 else 0.0,
        )

    @property
    def name_to_id(self) -> dict[str, int]:
        return dict(self._name_to_id)

    @property
    def connectome(self) -> ConnectomeData:
        return self._connectome

    def get_neuron_by_name(self, name: str):
        if self._network is None:
            return None
        nid = self._name_to_id.get(name)
        if nid is None:
            return None
        return self._network.network.neurons.get(nid)

    def _build(self) -> None:
        network, name_to_id = build_paula_network(
            self._connectome,
            base_params=_base_params(),
            sensory_params=_sensory_params(),
            motor_params=_motor_params(),
            interneuron_params=_interneuron_params(),
            weight_max=3.0,
            log_level=self._log_level,
        )
        self._network = network
        self._name_to_id = name_to_id
        self._apply_plasticity_setting()

    def set_neuromodulator_enabled(self, index: int, enabled: bool) -> None:
        if index == 0:
            self._enable_m0 = bool(enabled)
        elif index == 1:
            self._enable_m1 = bool(enabled)
        else:
            raise ValueError(f"unsupported neuromodulator index: {index}")
        if not enabled:
            self._zero_neuromodulator(index)

    def set_plasticity_disabled(self, disabled: bool) -> None:
        self._plasticity_disabled = bool(disabled)
        self._apply_plasticity_setting()

    def _apply_plasticity_setting(self) -> None:
        if self._network is None:
            return
        eta_post = 0.0 if self._plasticity_disabled else self._plasticity_eta_post
        eta_retro = 0.0 if self._plasticity_disabled else self._plasticity_eta_retro
        for neuron in self._network.network.neurons.values():
            neuron.params.eta_post = float(eta_post)
            neuron.params.eta_retro = float(eta_retro)

    def _zero_neuromodulator(self, index: int) -> None:
        if self._network is None:
            return
        for neuron in self._network.network.neurons.values():
            if index < len(neuron.M_vector):
                neuron.M_vector[index] = 0.0

    def _inject_sensory(self, sensory_inputs: dict[str, float]) -> None:
        assert self._network is not None
        for name, raw in sensory_inputs.items():
            nid = self._name_to_id.get(name)
            if nid is None:
                continue
            neuron = self._network.network.neurons.get(nid)
            if neuron is None or neuron.params.num_inputs <= 0:
                continue
            value = float(np.clip(raw * self.sensory_input_gain, 0.0, 1.5))
            prev = self._last_sensory.get(name, value)
            delta = value - prev
            # M0 carries sudden adverse/surprise signals; M1 carries approach
            # and improving sensory evidence.
            mod = np.array([max(0.0, -delta) * 2.0, max(0.0, delta) * 2.0])
            if name in {
                "WALL_L",
                "WALL_R",
                "STARTLE",
                "THERMO_HOT",
                "THERMO_COLD",
                "DEPTH_SHALLOW",
                "DEPTH_DEEP",
                "VESTIBULAR_UP",
                "VESTIBULAR_DOWN",
            }:
                mod[0] += value
            if name in {"RETINA_L", "RETINA_R", "OLFACTORY_L", "OLFACTORY_R"}:
                mod[1] += value * 0.8
            if not self._enable_m0:
                mod[0] = 0.0
            if not self._enable_m1:
                mod[1] = 0.0
            self._network.set_external_input(nid, 0, value, mod=mod)

    def _update_behavior_state(
        self,
        sensory_inputs: dict[str, float],
        current_tick: int,
    ) -> None:
        visual_l = sensory_inputs.get("RETINA_L", 0.0)
        visual_r = sensory_inputs.get("RETINA_R", 0.0)
        odor_l = sensory_inputs.get("OLFACTORY_L", 0.0)
        odor_r = sensory_inputs.get("OLFACTORY_R", 0.0)
        flow_l = sensory_inputs.get("OPTIC_FLOW_L", 0.0)
        flow_r = sensory_inputs.get("OPTIC_FLOW_R", 0.0)
        wall_l = sensory_inputs.get("WALL_L", 0.0)
        wall_r = sensory_inputs.get("WALL_R", 0.0)
        line_l = sensory_inputs.get("LATERAL_LINE_L", 0.0)
        line_r = sensory_inputs.get("LATERAL_LINE_R", 0.0)
        depth_shallow = sensory_inputs.get("DEPTH_SHALLOW", 0.0)
        depth_deep = sensory_inputs.get("DEPTH_DEEP", 0.0)
        vertical_up = sensory_inputs.get("VESTIBULAR_UP", 0.0)
        vertical_down = sensory_inputs.get("VESTIBULAR_DOWN", 0.0)
        startle = sensory_inputs.get("STARTLE", 0.0)
        video_active = sensory_inputs.get("VIDEO_ACTIVE", 0.0) > 0.5
        video_motion = sensory_inputs.get("VIDEO_MOTION", 0.0)
        video_action_kick = sensory_inputs.get("VIDEO_ACTION_KICK", 0.0) > 0.5
        video_action_force = sensory_inputs.get("VIDEO_ACTION_FORCE", 0.0)
        video_action_side_score = sensory_inputs.get("VIDEO_ACTION_SIDE_SCORE", 0.0)
        video_action_confidence = sensory_inputs.get("VIDEO_ACTION_CONFIDENCE", 0.0)
        calcium_active = sensory_inputs.get("CALCIUM_ACTIVE", 0.0) > 0.5
        calcium_kick = sensory_inputs.get("CALCIUM_KICK", 0.0) > 0.5
        calcium_force = sensory_inputs.get("CALCIUM_FORCE", 0.0)
        calcium_side_score = sensory_inputs.get("CALCIUM_SIDE_SCORE", 0.0)
        calcium_confidence = sensory_inputs.get("CALCIUM_CONFIDENCE", 0.0)
        externally_driven = video_active or calcium_active
        action_tail_confidence = sensory_inputs.get("ACTION_TAIL_TARGET_CONFIDENCE", 0.0)
        tail_confidence_min = (
            zfc.VIDEO_TAIL_TARGET_CONFIDENCE_MIN
            if video_active
            else zfc.EXTERNAL_TAIL_TARGET_CONFIDENCE_MIN
        )
        if externally_driven and action_tail_confidence > tail_confidence_min:
            raw_targets = np.array(
                [
                    float(np.clip(sensory_inputs.get(f"ACTION_TAIL_TARGET_{i:02d}", 0.0), -1.0, 1.0))
                    for i in range(zfc.N_BODY_SEGMENTS)
                ],
                dtype=float,
            )
            alpha = 0.44 if calcium_active else 0.34
            self._external_tail_targets = (
                (1.0 - alpha) * self._external_tail_targets + alpha * raw_targets
            )
            self._external_tail_confidence = float(
                np.clip(action_tail_confidence, 0.0, 1.0)
            )
        else:
            self._external_tail_targets *= 0.72
            self._external_tail_confidence *= 0.72

        if video_active:
            # Natural underwater footage carries background luminance and reef /
            # substrate salience that should not be interpreted as prey.  The
            # current ZAPBench direct-ephys labels have a kick-positive rate of
            # ~0.24, so video-mode drive is gated by calibrated motion/startle
            # energy rather than raw brightness.  Left/right visual channels are
            # still used below for steering.
            visual_appetitive = 0.0
            motion_drive = max(0.0, video_motion - 0.12) * 4.0
            flow_drive = max(0.0, max(flow_l, flow_r) - 0.55) * 0.12
        else:
            visual_appetitive = max(visual_l, visual_r) * self.visual_approach_gain
            motion_drive = 0.0
            flow_drive = 0.0

        appetitive = visual_appetitive + max(odor_l, odor_r) * self.odor_approach_gain
        stabilizing = max(flow_l, flow_r) * self.optic_flow_drive_gain
        avoidance = max(wall_l, wall_r) * self.wall_avoidance_drive_gain
        if calcium_active:
            calcium_drive = calcium_force if calcium_kick else 0.0
            drive_target = float(np.clip(calcium_drive + 0.25 * avoidance, 0.0, 1.0))
        elif video_active:
            video_calibrated_drive = (
                video_action_force * max(0.25, video_action_confidence)
                if video_action_kick
                else 0.0
            )
            startle_sensory_drive = max(0.0, motion_drive + flow_drive) if startle > 0.45 else 0.0
            drive_target = float(
                np.clip(
                    max(video_calibrated_drive, 0.35 * startle_sensory_drive)
                    + 0.25 * avoidance
                    + 0.75 * startle,
                    0.0,
                    1.0,
                )
            )
        else:
            drive_target = float(np.clip(self.baseline_drive + appetitive + stabilizing + avoidance, 0.0, 1.0))

        if startle > 0.4 and self._startle_ticks_left <= 0:
            self._startle_ticks_left = 22
            self._bout_ticks_left = max(self._bout_ticks_left, 26)
            self._bout_drive = 1.0

        if externally_driven:
            self._coast_ticks_left = 0
            if drive_target > 0.03:
                self._bout_ticks_left = max(self._bout_ticks_left - 1, 2)
                self._bout_drive = 0.84 * self._bout_drive + 0.16 * drive_target
            else:
                self._bout_ticks_left = 0
                self._bout_drive *= 0.88
        elif self._bout_ticks_left > 0:
            self._bout_ticks_left -= 1
            self._bout_drive = 0.82 * self._bout_drive + 0.18 * max(0.35, drive_target)
        else:
            self._bout_drive *= 0.84
            self._coast_ticks_left -= 1
            p = self.spontaneous_bout_prob + 0.040 * drive_target
            if self._coast_ticks_left <= 0 or self._rng.random() < p:
                self._bout_ticks_left = int(
                    self._rng.integers(zfc.BOUT_MIN_TICKS, zfc.BOUT_MAX_TICKS + 1)
                )
                self._coast_ticks_left = int(
                    self._rng.integers(zfc.COAST_MIN_TICKS, zfc.COAST_MAX_TICKS + 1)
                )
                self._bout_drive = max(0.35, drive_target)

        if self._startle_ticks_left > 0:
            self._startle_ticks_left -= 1
            cstart = 0.85 if (current_tick // 7) % 2 == 0 else -0.85
        else:
            cstart = 0.0

        # Positive turn bias means stronger right-side motor activation, which
        # bends the model left. Visual/odor terms steer toward the stronger side.
        sensory_turn = (
            self.visual_turn_gain * (visual_l - visual_r)
            + self.odor_turn_gain * (odor_l - odor_r)
            + self.optic_flow_turn_gain * (flow_r - flow_l)
            + self.lateral_line_turn_gain * (line_r - line_l)
            + self.wall_turn_gain * (wall_r - wall_l)
            + cstart
        )
        if calcium_active:
            # Decoder convention: negative side_score means left motor nerve /
            # leftward fictive action. Positive PAULA turn bias activates the
            # right-side body motors and bends the body left, so invert sign.
            calcium_turn = -float(
                np.clip(
                    calcium_side_score,
                    -zfc.CALCIUM_TURN_SCORE_LIMIT,
                    zfc.CALCIUM_TURN_SCORE_LIMIT,
                )
            )
            turn_force = max(0.18, float(calcium_force))
            turn_confidence = max(0.35, float(calcium_confidence))
            sensory_turn += zfc.CALCIUM_TURN_GAIN * calcium_turn * turn_force * turn_confidence
        elif video_active:
            video_turn = -float(
                np.clip(
                    video_action_side_score,
                    -zfc.CALCIUM_TURN_SCORE_LIMIT,
                    zfc.CALCIUM_TURN_SCORE_LIMIT,
                )
            )
            video_turn_force = float(video_action_force) if video_action_kick else 0.0
            video_turn_confidence = max(0.25, float(video_action_confidence))
            sensory_turn += (
                0.80
                * zfc.CALCIUM_TURN_GAIN
                * video_turn
                * video_turn_force
                * video_turn_confidence
            )
        noise = 0.0 if externally_driven else float(self._rng.normal(0.0, 0.035)) if self._bout_ticks_left > 0 else 0.0
        target_turn = float(np.clip(sensory_turn + noise, -1.0, 1.0))
        turn_alpha = 0.44 if calcium_active else 0.22 if video_active else 0.16 if self._bout_ticks_left > 0 else 0.05
        self._turn_bias = (1.0 - turn_alpha) * self._turn_bias + turn_alpha * target_turn
        if self._bout_ticks_left <= 0 and not externally_driven:
            self._turn_bias *= 0.82

        depth_error = float(np.clip(depth_deep - depth_shallow, -1.0, 1.0))
        vertical_rate = float(np.clip(vertical_down - vertical_up, -1.0, 1.0))
        if video_active:
            startle_pitch = (
                zfc.VIDEO_STARTLE_PITCH_GAIN
                if self._startle_ticks_left > 0 and (current_tick // 11) % 2 == 0
                else 0.0
            )
            pitch_target = float(
                np.clip(
                    zfc.VIDEO_DEPTH_PITCH_GAIN * depth_error
                    + zfc.VIDEO_VERTICAL_RATE_PITCH_GAIN * vertical_rate
                    + startle_pitch,
                    -zfc.VIDEO_PITCH_TARGET_LIMIT,
                    zfc.VIDEO_PITCH_TARGET_LIMIT,
                )
            )
            pitch_alpha = float(np.clip(zfc.VIDEO_PITCH_ALPHA, 0.0, 1.0))
        else:
            startle_pitch = 0.35 if self._startle_ticks_left > 0 and (current_tick // 11) % 2 == 0 else 0.0
            pitch_target = float(np.clip(0.72 * depth_error + 0.26 * vertical_rate + startle_pitch, -1.0, 1.0))
            pitch_alpha = 0.16 if calcium_active else 0.10 if self._bout_ticks_left > 0 else 0.035
        self._pitch_bias = (1.0 - pitch_alpha) * self._pitch_bias + pitch_alpha * pitch_target
        if self._bout_ticks_left <= 0 and not externally_driven:
            self._pitch_bias *= 0.90

        # Light touch into PAULA state variables for visible neural dynamics.
        self._nudge("SWIM_GATE", self._bout_drive * 0.025 * self.paula_activity_gain)
        self._nudge("RETICULOSPINAL_L", max(0.0, self._turn_bias) * 0.025 * self.paula_activity_gain)
        self._nudge("RETICULOSPINAL_R", max(0.0, -self._turn_bias) * 0.025 * self.paula_activity_gain)
        self._nudge("DEPTH_HOMEOSTAT", abs(depth_error) * 0.018 * self.paula_activity_gain)
        self._nudge("ASCEND_GATE", max(0.0, self._pitch_bias) * 0.020 * self.paula_activity_gain)
        self._nudge("DIVE_GATE", max(0.0, -self._pitch_bias) * 0.020 * self.paula_activity_gain)

    def _decode_tail_motors(self, current_tick: int) -> dict[str, float]:
        drive = float(np.clip(self._bout_drive, 0.0, 1.0))
        freq = zfc.TAIL_BEAT_FREQ_MIN_HZ + drive * (
            zfc.TAIL_BEAT_FREQ_MAX_HZ - zfc.TAIL_BEAT_FREQ_MIN_HZ
        )
        if self.tail_cpg_enabled:
            self._tail_phase = (self._tail_phase + zfc.TWO_PI * freq * zfc.PHYSICS_TIMESTEP_S) % zfc.TWO_PI
        out: dict[str, float] = {}
        maneuver_drive = max(0.18, drive) if drive >= zfc.TAIL_MOTOR_DRIVE_DEADBAND else 0.0
        for i in range(zfc.N_BODY_SEGMENTS):
            frac = i / max(1, zfc.N_BODY_SEGMENTS - 1)
            envelope = 0.30 + 0.95 * frac
            phase = self._tail_phase - i * zfc.TAIL_BEAT_PHASE_LAG_RAD
            wave = np.sin(phase) * drive * envelope if self.tail_cpg_enabled else 0.0
            turn = self._turn_bias * (1.0 - 0.72 * frac) * maneuver_drive
            pitch = self._pitch_bias * (1.0 - 0.56 * frac) * maneuver_drive
            left = np.clip(0.5 * (wave - turn), 0.0, 1.0)
            right = np.clip(0.5 * (-wave + turn), 0.0, 1.0)
            ext_blend = float(np.clip(self._external_tail_confidence, 0.0, 0.50))
            if ext_blend > 0.01:
                ext = float(np.clip(self._external_tail_targets[i], -1.0, 1.0))
                direct_left = max(0.0, -ext) * maneuver_drive
                direct_right = max(0.0, ext) * maneuver_drive
                left = np.clip((1.0 - ext_blend) * left + ext_blend * direct_left, 0.0, 1.0)
                right = np.clip((1.0 - ext_blend) * right + ext_blend * direct_right, 0.0, 1.0)
            dorsal = np.clip(0.34 * max(0.0, pitch) + 0.08 * abs(wave) * max(0.0, pitch), 0.0, 1.0)
            ventral = np.clip(0.34 * max(0.0, -pitch) + 0.08 * abs(wave) * max(0.0, -pitch), 0.0, 1.0)
            out[f"tail_{i:02d}_left"] = float(left)
            out[f"tail_{i:02d}_right"] = float(right)
            out[f"tail_{i:02d}_dorsal"] = float(dorsal)
            out[f"tail_{i:02d}_ventral"] = float(ventral)
            self._nudge(f"MOTOR_L_{i:02d}", left * 0.006 * self.paula_activity_gain)
            self._nudge(f"MOTOR_R_{i:02d}", right * 0.006 * self.paula_activity_gain)
            self._nudge(f"MOTOR_D_{i:02d}", dorsal * 0.006 * self.paula_activity_gain)
            self._nudge(f"MOTOR_V_{i:02d}", ventral * 0.006 * self.paula_activity_gain)
        return out

    def _nudge(self, name: str, amount: float) -> None:
        if self._network is None or amount == 0.0:
            return
        nid = self._name_to_id.get(name)
        if nid is None:
            return
        neuron = self._network.network.neurons.get(nid)
        if neuron is not None:
            neuron.S += float(amount)


def _params(**overrides: Any) -> NeuronParameters:
    base = NeuronParameters(
        r_base=0.75,
        b_base=1.05,
        c=8,
        lambda_param=18.0,
        p=1.0,
        gamma=np.array([0.985, 0.992]),
        w_r=np.array([-0.08, 0.04]),
        w_b=np.array([-0.04, 0.03]),
        w_tref=np.array([-8.0, 4.0]),
        eta_post=0.0,
        eta_retro=0.0,
        num_neuromodulators=2,
        num_inputs=1,
    )
    return replace(base, **overrides)


def _base_params() -> NeuronParameters:
    return _params()


def _sensory_params() -> NeuronParameters:
    return _params(r_base=0.55, b_base=0.85, lambda_param=12.0)


def _interneuron_params() -> NeuronParameters:
    return _params(r_base=0.68, b_base=1.0, lambda_param=20.0)


def _motor_params() -> NeuronParameters:
    return _params(r_base=0.62, b_base=0.92, lambda_param=15.0)
