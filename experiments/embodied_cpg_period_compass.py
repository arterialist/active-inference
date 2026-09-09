"""Separate CPG-period sweep for embodied PAULA compass transfer.

This runner changes *only* the construction-time period of the existing
four-phase PAULA CPG in its own process.  Each CPG phase is an ordinary PAULA
neuron with refractory period ``p`` and receives the preceding phase through a
synapse of delay ``p``; the nominal full motor cycle is therefore ``4p``
neural ticks.  The default agent and live demo are never changed.

For each period, the standard 1,400-tick TR course first runs through normal
PAULA motor relays, graded muscle membranes, and MuJoCo.  It retains the full
body/CPG trace and measures actual (not assumed) phase intervals, yaw jitter,
net turn, and locomotor viability.  Only viable periods are then tested using
the existing opt-in sensorimotor V2 compass.  A no-physics V2 replay receives
the exact raw yaw and speed samples that were presented before each embodied
PAULA tick; all replay comparisons are observational.

This tests an engineering timing hypothesis.  It is not a claim that changing
the simulator tick-level CPG period supplies a biologically calibrated speed
controller, nor a navigation/homing result.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from simulations.active_inference import aif_agent3d as base
from simulations.active_inference.sensorimotor_agent3d_v2 import SensorimotorEstimatorV2Agent3D


BURN_IN = 200
TURN_START = 500
TURN_STOP = 900
TOTAL_TICKS = 1400
HEADING_WINDOW = 24
MUJOCO_DT_SECONDS = 0.004
BASE_PERIOD = 10
DEFAULT_PERIODS = (BASE_PERIOD, 6, 4)
QUARANTINED_BUILD = {
    "turn_probe_ports": True,
    "w_opp": 0.0,
    "w_cpu1": 0.0,
    "w_musf_mode": 0.0,
}


@contextmanager
def _temporary_cpg_period(period: int):
    """Change the construction constant only while an opt-in case is built/run.

    ``CPG_PERIOD`` is currently a module-level build constant: it sets both
    the recurrent CPG synaptic delay and the CPG cells' refractory period.
    Keeping this mutation process-local and restoring it after every case
    prevents any effect on defaults, later callers, or the live demo.
    """

    if period < 2:
        raise ValueError("CPG period must be at least two neural ticks")
    original = base.CPG_PERIOD
    try:
        base.CPG_PERIOD = int(period)
        yield
    finally:
        base.CPG_PERIOD = original


def _heading(window: deque[np.ndarray]) -> float | None:
    vector = np.sum(np.sum(window, axis=0) * np.exp(1j * base.cc.PHI))
    return None if abs(vector) < 1e-12 else float(np.angle(vector))


def _turn_delta(rows: list[dict], key: str) -> float | None:
    selected = [row[key] for row in rows if row["turn_stimulus_active"]]
    if len(selected) < 2 or any(value is None for value in selected):
        return None
    values = np.unwrap(np.asarray(selected, dtype=float))
    return float(np.degrees(values[-1] - values[0]))


def _phase_metrics(rows: list[dict], period: int) -> dict:
    """Measure the actual recurrent CPG timing, including its phase order."""

    settled = [row for row in rows if row["neural_tick_input"] >= BURN_IN + 4 * period]
    times = [
        [int(row["neural_tick_input"]) for row in settled if row["cpg_spikes"][phase]]
        for phase in range(4)
    ]
    intervals = [np.diff(series) for series in times]
    phase_lags: list[list[int]] = []
    for phase in range(4):
        target = times[(phase + 1) % 4]
        lags = []
        for tick in times[phase]:
            later = next((candidate for candidate in target if candidate > tick), None)
            if later is not None:
                lags.append(later - tick)
        phase_lags.append(lags)

    def stats(values: np.ndarray | list[int]) -> dict:
        data = np.asarray(values, dtype=float)
        return {
            "count": int(len(data)),
            "mean_ticks": None if not len(data) else float(np.mean(data)),
            "std_ticks": None if not len(data) else float(np.std(data)),
            "unique_ticks": [] if not len(data) else sorted({int(value) for value in data}),
        }

    recurrence = [stats(values) for values in intervals]
    lag_stats = [stats(values) for values in phase_lags]
    # The PAULA CPG's refractory/recurrent construction produces a
    # deterministic two-interval rhythm per phase (for example 13,31 at
    # p=10), not one uniform IPI.  Treating its non-zero IPI standard
    # deviation as instability would reject the unmodified standard CPG.
    # Stability here means a repeatable one- or two-value alternating pattern
    # plus a fixed downstream phase lag.
    def periodic_intervals(values: np.ndarray) -> bool:
        unique = sorted({int(value) for value in values})
        if len(values) < 5 or len(unique) > 2:
            return False
        if len(unique) == 1:
            return True
        return bool(all(int(values[index]) == int(values[index % 2]) for index in range(len(values))))

    stable = bool(
        all(len(series) >= 5 for series in times)
        and all(periodic_intervals(values) for values in intervals)
        and all(item["std_ticks"] is not None and item["std_ticks"] <= 0.05 for item in lag_stats)
    )
    return {
        "construction": {
            "phase_count": 4,
            "recurrent_synaptic_delay_ticks": period,
            "per_phase_refractory_ticks": period,
            "nominal_full_cycle_ticks": 4 * period,
            "nominal_full_cycle_hz_at_mujoco_dt": 1.0 / (4 * period * MUJOCO_DT_SECONDS),
        },
        "observed": {
            "spike_ticks_by_phase": times,
            "same_phase_intervals": recurrence,
            "following_phase_lags": lag_stats,
            "phase_stable": stable,
        },
    }


def _body_metrics(rows: list[dict], period: int) -> dict:
    after_birth = [row for row in rows if row["neural_tick_input"] >= BURN_IN]
    turn = [row for row in rows if row["turn_stimulus_active"]]
    raw = np.asarray([row["raw_yaw_rate_pre_tick"] for row in turn], dtype=float)
    start = after_birth[0]["physical_pose_before"]
    end = after_birth[-1]["physical_pose_after"]
    displacement = float(np.hypot(end["x"] - start["x"], end["y"] - start["y"]))
    mean_speed = float(np.mean([row["physical_speed_after"] for row in after_birth]))
    turn_delta = _turn_delta(rows, "physical_yaw_radians_after")
    phases = _phase_metrics(rows, period)
    viable = bool(
        phases["observed"]["phase_stable"]
        and mean_speed > 0.01
        and displacement > 0.05
        and turn_delta is not None
        and abs(turn_delta) >= 20.0
    )
    return {
        "physical_turn_delta_degrees": turn_delta,
        "turn_raw_yaw": {
            "mean_signed_rad_per_sec": float(np.mean(raw)),
            "mean_abs_rad_per_sec": float(np.mean(np.abs(raw))),
            "std_rad_per_sec": float(np.std(raw)),
            "p95_abs_rad_per_sec": float(np.quantile(np.abs(raw), 0.95)),
            "sign_changes": int(np.sum(np.signbit(raw[1:]) != np.signbit(raw[:-1]))),
        },
        "locomotion": {
            "mean_speed_after_birth": mean_speed,
            "displacement_after_birth": displacement,
        },
        "cpg": phases,
        "viable_for_v2_test": viable,
        "viability_rule": {
            "phase_stable": True,
            "mean_speed_after_birth_gt": 0.01,
            "displacement_after_birth_gt": 0.05,
            "absolute_turn_degrees_at_least": 20.0,
        },
    }


def _capture_row(current, *, input_tick: int, raw_yaw: float, speed: float,
                 turn_active: bool, window: deque[np.ndarray], include_v2: bool,
                 physical_reference: dict | None = None) -> dict:
    active = np.asarray([current.nb[nid].O > 0.0 for nid in base.cc.RING], dtype=float)
    window.append(active)
    if physical_reference is None:
        x, y, yaw = current.world.pose()
        yaw_rate_after = float(current.world.yaw_rate())
        speed_after = float(current.world.speed())
    else:
        pose = physical_reference["physical_pose_after"]
        x, y, yaw = float(pose["x"]), float(pose["y"]), float(pose["yaw"])
        yaw_rate_after = float(physical_reference["physical_yaw_rate_after"])
        speed_after = float(physical_reference["physical_speed_after"])
    row = {
        "neural_tick_input": input_tick,
        "neural_tick_after_physics": int(current.t),
        "turn_stimulus_active": turn_active,
        "raw_yaw_rate_pre_tick": raw_yaw,
        "physical_speed_pre_tick": speed,
        "physical_pose_before": {},  # filled by the input hook before PAULA and physics
        "physical_pose_after": {"x": float(x), "y": float(y), "yaw": float(yaw)},
        "physical_yaw_radians_after": float(yaw),
        "physical_yaw_rate_after": yaw_rate_after,
        "physical_speed_after": speed_after,
        "cpg_spikes": [int(current.nb[nid].O > 0.0) for nid in base.CPGP],
        "turn_spike": int(current.nb[base.TR].O > 0.0),
        "motor_relay_spikes": {
            label: [int(current.nb[nid].O > 0.0) for nid in ids]
            for label, ids in base.RLY.items()
        },
        "muscle_membranes": {
            "MLp": float(current.nb[base.MLp].S), "MLr": float(current.nb[base.MLr].S),
            "MRp": float(current.nb[base.MRp].S), "MRr": float(current.nb[base.MRr].S),
        },
        "ring_spikes_by_column": active.astype(int).tolist(),
        "ring_heading_radians": _heading(window),
        "ring_live": bool(np.any(active)),
    }
    if include_v2:
        row.update({
            "sensory_residual": {
                "CCW": [float(current.nb[nid].O) for nid in base.SM2_SENS_CCW],
                "CW": [float(current.nb[nid].O) for nid in base.SM2_SENS_CW],
            },
            "motor_prediction": {
                "CCW": [float(current.nb[nid].O) for nid in base.SM2_PRED_CCW],
                "CW": [float(current.nb[nid].O) for nid in base.SM2_PRED_CW],
            },
            "prediction_error": {
                "CCW": float(current.nb[base.SM2_PE_CCW].O),
                "CW": float(current.nb[base.SM2_PE_CW].O),
            },
            "fused_update": {
                "CCW": float(current.nb[base.SM2_UPDATE_CCW].O),
                "CW": float(current.nb[base.SM2_UPDATE_CW].O),
            },
            "p_en_spikes": {
                "CL": int(sum(current.nb[nid].O > 0.0 for column in base.cc.CL for nid in column)),
                "CR": int(sum(current.nb[nid].O > 0.0 for column in base.cc.CR for nid in column)),
            },
            "p_en_release_by_column": {
                "CL": [float(sum(current.nb[nid].O for nid in column)) for column in base.cc.CL],
                "CR": [float(sum(current.nb[nid].O for nid in column)) for column in base.cc.CR],
            },
        })
    return row


def _run_embodied(*, period: int, seed: int, v2: bool) -> dict:
    """Run one physical course; CPG period is set only inside this function."""

    with _temporary_cpg_period(period):
        np.random.seed(seed)
        agent_class = SensorimotorEstimatorV2Agent3D if v2 else base.AIFAgent3D
        agent = agent_class(seed=seed, **QUARANTINED_BUILD)
        agent.world = base.w3.World3D(seed=seed, n_food=0, n_tox=0, arena=8.0)
        agent.img = agent.world.retina()
        agent.birth()
        agent._kicked = True
        window: deque[np.ndarray] = deque(maxlen=HEADING_WINDOW)
        trace: list[dict] = []
        pending: dict[int, dict] = {}

        def input_hook(current) -> None:
            tick = int(current.t)
            raw = float(current.world.yaw_rate())
            speed = float(current.world.speed())
            turn_active = TURN_START <= tick < TURN_STOP
            x, y, yaw = current.world.pose()
            pending[tick] = {
                "raw": raw,
                "speed": speed,
                "turn_active": turn_active,
                "pose": {"x": float(x), "y": float(y), "yaw": float(yaw)},
            }
            if tick == BURN_IN:
                current.net.set_external_input(base.CPGP[0], 0, 5.0)
            if turn_active:
                current.net.set_external_input(base.TR, base.TURN_PROBE_SYN[base.TR], 3.0)

        def capture(current) -> None:
            input_tick = int(current.t) - 1
            meta = pending.pop(input_tick)
            row = _capture_row(
                current, input_tick=input_tick, raw_yaw=meta["raw"], speed=meta["speed"],
                turn_active=meta["turn_active"], window=window, include_v2=v2,
            )
            row["physical_pose_before"] = meta["pose"]
            trace.append(row)

        base.run_episode(
            agent, steps=TOTAL_TICKS, sub=1, vision=False, render_every=10**9,
            render_ticks=0, log_every=10**9, neural_input_hook=input_hook, tick_hook=capture,
        )
        if pending:
            raise RuntimeError(f"missing physical capture for ticks {sorted(pending)}")
        return {
            "condition": "sensorimotor_v2" if v2 else "cpg_body_mapping",
            "period": period,
            "seed": seed,
            "build": dict(QUARANTINED_BUILD),
            "cpg_construction": {
                "recurrent_delay_ticks": period,
                "refractory_ticks": period,
                "phase_count": 4,
            },
            "tick_trace": trace,
        }


class _ReplayWorld:
    """No-physics sensor source for exact V2 replay of a recorded course."""

    def __init__(self, source: list[dict]) -> None:
        self.source = source
        self.index = 0
        self.eaten = 0
        self.tox_hits = 0
        self.wz_tau = 60.0
        self.wz_comp = 0.0

    def odour(self):
        return ((0.0, 0.0), (0.0, 0.0))

    def odour_identity(self):
        return (0.0, 0.0)

    def take_event(self):
        return None

    def speed(self):
        return float(self.source[self.index]["physical_speed_pre_tick"])

    def yaw_rate(self):
        return float(self.source[self.index]["raw_yaw_rate_pre_tick"])

    def pose(self):
        pose = self.source[self.index]["physical_pose_before"]
        return float(pose["x"]), float(pose["y"]), float(pose["yaw"])

    def act_muscles(self, *_args, **_kwargs):
        # Preserve run_episode's order while explicitly omitting MuJoCo.
        self.index += 1

    def dist_home(self):
        return 0.0


def _replay_v2(*, period: int, seed: int, embodied_rows: list[dict]) -> dict:
    """Replay the full V2 circuit from recorded pre-physics yaw and speed."""

    with _temporary_cpg_period(period):
        np.random.seed(seed)
        agent = SensorimotorEstimatorV2Agent3D(seed=seed, **QUARANTINED_BUILD)
        agent.world = _ReplayWorld(embodied_rows)
        agent.img = None
        agent.birth()
        agent._kicked = True
        window: deque[np.ndarray] = deque(maxlen=HEADING_WINDOW)
        trace: list[dict] = []
        input_ticks: list[int] = []

        def input_hook(current) -> None:
            tick = int(current.t)
            input_ticks.append(tick)
            if tick == BURN_IN:
                current.net.set_external_input(base.CPGP[0], 0, 5.0)
            if TURN_START <= tick < TURN_STOP:
                current.net.set_external_input(base.TR, base.TURN_PROBE_SYN[base.TR], 3.0)

        def capture(current) -> None:
            source = embodied_rows[len(trace)]
            row = _capture_row(
                current, input_tick=input_ticks[-1], raw_yaw=source["raw_yaw_rate_pre_tick"],
                speed=source["physical_speed_pre_tick"], turn_active=source["turn_stimulus_active"],
                window=window, include_v2=True, physical_reference=source,
            )
            # No physical pose or step occurs in replay. Preserve the input
            # pose only as a display reference, never as neural input.
            row["physical_pose_before"] = source["physical_pose_before"]
            trace.append(row)

        base.run_episode(
            agent, steps=TOTAL_TICKS, sub=1, vision=False, render_every=10**9,
            render_ticks=0, log_every=10**9, neural_input_hook=input_hook, tick_hook=capture,
        )
        return {
            "condition": "sensorimotor_v2_exact_raw_yaw_speed_replay",
            "no_physics": True,
            "input": "recorded pre-PAULA raw yaw and speed; no pose/heading feedback",
            "tick_trace": trace,
        }


def _replay_view(row: dict) -> dict:
    return {
        key: row[key] for key in (
            "turn_stimulus_active", "raw_yaw_rate_pre_tick", "physical_speed_pre_tick",
            "cpg_spikes", "turn_spike", "motor_relay_spikes", "muscle_membranes",
            "ring_spikes_by_column", "ring_heading_radians", "ring_live", "sensory_residual",
            "motor_prediction", "prediction_error", "fused_update", "p_en_spikes",
            "p_en_release_by_column",
        )
    }


def _compare_replay(embodied: list[dict], replay: list[dict]) -> dict:
    mismatches: dict[str, int] = {}
    mismatch_ticks: list[int] = []
    for original, regenerated in zip(embodied, replay, strict=True):
        left, right = _replay_view(original), _replay_view(regenerated)
        if left != right:
            mismatch_ticks.append(original["neural_tick_input"])
            for key in left:
                if left[key] != right[key]:
                    mismatches[key] = mismatches.get(key, 0) + 1
    return {
        "exact_neural_replay": not mismatch_ticks,
        "mismatch_ticks": len(mismatch_ticks),
        "first_mismatch_input_ticks": mismatch_ticks[:20],
        "mismatching_fields": mismatches,
    }


def _v2_metrics(rows: list[dict], replay: list[dict]) -> dict:
    body_turn = _turn_delta(rows, "physical_yaw_radians_after")
    ring_turn = _turn_delta(rows, "ring_heading_radians")
    gain = None if body_turn is None or ring_turn is None or abs(body_turn) < 1e-9 else ring_turn / body_turn
    turn = [row for row in rows if row["turn_stimulus_active"]]
    return {
        "body_turn_delta_degrees": body_turn,
        "ring_turn_delta_degrees": ring_turn,
        "tracking_error_degrees": None if body_turn is None or ring_turn is None else ring_turn - body_turn,
        "signed_tracking_gain": gain,
        "leading_route_undergain_fraction": None if gain is None else 1.0 - gain,
        "ring_live_fraction": float(np.mean([row["ring_live"] for row in rows])),
        "mean_fused_update": {
            direction: float(np.mean([row["fused_update"][direction] for row in turn]))
            for direction in ("CCW", "CW")
        },
        "mean_p_en_spikes": {
            direction: float(np.mean([row["p_en_spikes"][direction] for row in turn]))
            for direction in ("CL", "CR")
        },
        "raw_replay": _compare_replay(rows, replay),
    }


def run_sweep(*, seed: int, periods: tuple[int, ...]) -> dict:
    if BASE_PERIOD not in periods:
        raise ValueError("include period 10 as the standard-course control")
    if len([period for period in periods if period < BASE_PERIOD]) < 2:
        raise ValueError("provide at least two faster periods below the standard period of 10")
    if len(set(periods)) != len(periods):
        raise ValueError("periods must be unique")

    cases: dict[str, dict] = {}
    for period in periods:
        body = _run_embodied(period=period, seed=seed, v2=False)
        body_metrics = _body_metrics(body["tick_trace"], period)
        item: dict[str, Any] = {"body_mapping": body, "body_metrics": body_metrics}
        if body_metrics["viable_for_v2_test"]:
            embodied_v2 = _run_embodied(period=period, seed=seed, v2=True)
            replay = _replay_v2(period=period, seed=seed, embodied_rows=embodied_v2["tick_trace"])
            item["v2_compass"] = {
                "embodied": embodied_v2,
                "raw_replay": replay,
                "metrics": _v2_metrics(embodied_v2["tick_trace"], replay["tick_trace"]),
            }
        else:
            item["v2_compass"] = {
                "skipped": True,
                "reason": "body/CPG did not meet predeclared viability rule",
            }
        cases[str(period)] = item
    return {
        "experiment": "embodied_paula_cpg_period_compass_sweep",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "scope": {
            "separate_opt_in_experiment": True,
            "default_or_demo_modified": False,
            "host_timing_controller": False,
            "physical_course": "PAULA TR -> PAULA relay/muscle -> MuJoCo body -> raw yaw",
            "v2_test_only_after_body_viability": True,
            "no_decoded_heading_or_pose_feedback": True,
        },
        "periods": list(periods),
        "standard_period": BASE_PERIOD,
        "course": {
            "burn_in_tick": BURN_IN, "turn_start_tick": TURN_START,
            "turn_stop_tick": TURN_STOP, "total_ticks": TOTAL_TICKS,
            "turn_neuron": "TR", "mujoco_dt_seconds": MUJOCO_DT_SECONDS,
        },
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--periods", nargs="+", type=int, default=DEFAULT_PERIODS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    record = run_sweep(seed=args.seed, periods=tuple(args.periods))
    for period, case in record["cases"].items():
        body = case["body_metrics"]
        v2 = case["v2_compass"]
        v2_text = "skipped" if v2.get("skipped") else (
            f"ring={v2['metrics']['ring_turn_delta_degrees']:+.2f}° "
            f"gain={v2['metrics']['signed_tracking_gain']:.3f} "
            f"replay={v2['metrics']['raw_replay']['exact_neural_replay']}"
        )
        print(
            f"period={period}: body={body['physical_turn_delta_degrees']:+.2f}° "
            f"speed={body['locomotion']['mean_speed_after_birth']:.3f} "
            f"raw_std={body['turn_raw_yaw']['std_rad_per_sec']:.3f} "
            f"phase_stable={body['cpg']['observed']['phase_stable']} v2={v2_text}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
