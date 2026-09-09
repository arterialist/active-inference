"""Build a seekable, per-neural-tick embodied-versus-replay compass record.

The two sides begin from the same PAULA seed and run the same strict
stroke-reset compass configuration.  The right side drives the MuJoCo body.
The left side has *no physics*: it receives the raw angular-velocity current
recorded just before each right-side neural tick.  It is therefore a causal
boundary experiment, not a second controller or a smoothed reconstruction.

The JSON bundle deliberately retains every ring and P-EN column at every
tick.  ``compass_replay.html`` reads it directly and its only decoder is a
display measurement; no result is fed back into either network.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from simulations.active_inference import aif_agent3d as ag
from simulations.active_inference import central_complex as cc
from simulations.active_inference.experiments import embodied_compass_longturn_diagnostic as longturn


HERE = Path(__file__).resolve().parent
WINDOW = 24
DT_SECONDS = 0.004  # MuJoCo timestep in this diagnostic's body XML.
DEFAULT_CASE = "ordinary_stroke_reset44_conjunctive_rate_ring"


def _heading(window: deque[np.ndarray]) -> float | None:
    vector = np.sum(np.sum(window, axis=0) * np.exp(1j * cc.PHI))
    return None if abs(vector) < 1e-12 else float(np.angle(vector))


def _shift_columns(agent: ag.AIFAgent3D) -> dict[str, list[int]]:
    return {
        "CL": [sum(int(agent.nb[nid].O > 0) for nid in column) for column in cc.CL],
        "CR": [sum(int(agent.nb[nid].O > 0) for nid in column) for column in cc.CR],
    }


def _replay_raw_gyro(seed: int, build: dict, embodied: dict) -> dict:
    """Run the PAULA-only counterpart from the recorded pre-tick gyro input."""
    np.random.seed(seed)
    agent = ag.AIFAgent3D(seed=seed, **build)
    agent.birth()
    # Match longturn.run_case(): birth seeded the CPG, and no later automatic
    # kick is allowed.  The replay does not need a world or a motor plant.
    agent._kicked = True
    window: deque[np.ndarray] = deque(maxlen=WINDOW)
    trace: list[dict] = []

    source_rows = embodied["tick_trace"]
    first_pose = source_rows[0]["physical_pose"]
    virtual_x, virtual_y = 0.0, 0.0
    virtual_yaw = float(first_pose["yaw"])
    for index, source in enumerate(source_rows):
        raw_rate = float(source["applied_raw_yaw_rate"])
        # ``tick`` makes the same PAULA sensory-port writes and ring tonic
        # writes as a closed-loop agent.  It has no body, no host filter and
        # no access to the physical pose.
        agent.tick(
            ccw=max(0.0, raw_rate),
            cw=max(0.0, -raw_rate),
            speed=0.0,
            vision=False,
        )
        active = np.asarray([agent.nb[nid].O > 0 for nid in cc.RING], dtype=float)
        window.append(active)
        shifts = _shift_columns(agent)
        if index:
            virtual_yaw += raw_rate * DT_SECONDS
            speed = float(source["physical_speed"])
            virtual_x += speed * DT_SECONDS * float(np.cos(virtual_yaw))
            virtual_y += speed * DT_SECONDS * float(np.sin(virtual_yaw))
        trace.append({
            "neural_tick": int(source["neural_tick"]),
            "input_raw_yaw_rate": raw_rate,
            "ring_heading_radians": _heading(window),
            "ring_live": bool(np.any(active)),
            "ring_spikes_by_column": active.astype(int).tolist(),
            "shift_spikes_by_column": shifts,
            "shift_spikes": {side: int(sum(values)) for side, values in shifts.items()},
            "pb_shift_spikes": (
                {
                    "CL": sum(int(agent.nb[nid].O > 0) for nid in (
                        cc.PB_PHASE_CL if build.get("pb_phase_update", False) else cc.PB_CL)),
                    "CR": sum(int(agent.nb[nid].O > 0) for nid in (
                        cc.PB_PHASE_CR if build.get("pb_phase_update", False) else cc.PB_CR)),
                }
                if build.get("pb_eb_bridge", False) else None
            ),
            "pb_shift_release": (
                {
                    "CL": float(sum(agent.nb[nid].O for nid in (
                        cc.PB_PHASE_CL if build.get("pb_phase_update", False) else cc.PB_CL))),
                    "CR": float(sum(agent.nb[nid].O for nid in (
                        cc.PB_PHASE_CR if build.get("pb_phase_update", False) else cc.PB_CR))),
                }
                if build.get("pb_eb_bridge", False) else None
            ),
            "pb_phase_clock": (
                [index for index, nid in enumerate(cc.PB_PHASE_CLOCK) if agent.nb[nid].O > 0]
                if build.get("pb_phase_update", False) else None
            ),
            "pb_phase_gate_membrane": (
                {"CL": [float(agent.nb[nid].S) for nid in cc.PB_PHASE_CL],
                 "CR": [float(agent.nb[nid].S) for nid in cc.PB_PHASE_CR]}
                if build.get("pb_phase_update", False) else None
            ),
            "pb_phase_fast_vestibular": (
                {"CCW": float(agent.nb[ag.PHASE_VEST_CCW].O),
                 "CW": float(agent.nb[ag.PHASE_VEST_CW].O)}
                if build.get("pb_phase_update", False) else None
            ),
            "stroke_reset_release": {
                "CCW": float(agent.nb[ag.STROKE_RESET_CCW].O),
                "CW": float(agent.nb[ag.STROKE_RESET_CW].O),
                "CCW_membrane": float(agent.nb[ag.STROKE_RESET_CCW].S),
                "CW_membrane": float(agent.nb[ag.STROKE_RESET_CW].S),
                "clock": int(agent.nb[ag.STROKE_RESET_CLOCK].O > 0),
            },
            # A visual-only kinematic reference.  It shares the raw input,
            # not the body state, and makes the left pane's no-physics status
            # inspectable rather than suggesting it is another MuJoCo run.
            "virtual_pose": {"x": virtual_x, "y": virtual_y, "yaw": virtual_yaw},
        })
    return {
        "condition": "paula_only_raw_gyro_replay",
        "seed": seed,
        "build": build,
        "input_source": "embodied.applied_raw_yaw_rate (recorded before the matching neural tick)",
        "no_physics": True,
        "tick_trace": trace,
    }


def _unwrap(rows: list[dict], key: str) -> np.ndarray:
    values = [row[key] for row in rows]
    if any(value is None for value in values):
        raise RuntimeError(f"{key} contains silence; this bundle requires a live ring to compare phase")
    return np.unwrap(np.asarray(values, dtype=float))


def _blocks(embodied: list[dict], replay: list[dict], period: int) -> list[dict]:
    ticks = np.asarray([row["neural_tick"] for row in embodied])
    body = _unwrap(embodied, "physical_yaw_radians")
    live_ring = _unwrap(embodied, "ring_heading_radians")
    replay_ring = _unwrap(replay, "ring_heading_radians")
    result: list[dict] = []
    starts = range(longturn.BURN_IN, int(ticks[-1]) + 1, period)
    for start in starts:
        selection = np.flatnonzero((ticks >= start) & (ticks <= start + period))
        if len(selection) < 2:
            continue
        a, b = int(selection[0]), int(selection[-1])
        body_delta = float(np.degrees(body[b] - body[a]))
        ring_delta = float(np.degrees(live_ring[b] - live_ring[a]))
        replay_delta = float(np.degrees(replay_ring[b] - replay_ring[a]))
        cl = int(sum(embodied[i]["shift_spikes"]["CL"] for i in range(a, b + 1)))
        cr = int(sum(embodied[i]["shift_spikes"]["CR"] for i in range(a, b + 1)))
        result.append({
            "start_tick": int(ticks[a]),
            "end_tick": int(ticks[b]),
            "body_yaw_delta_degrees": body_delta,
            "ring_yaw_delta_degrees": ring_delta,
            "replay_ring_yaw_delta_degrees": replay_delta,
            "tracking_error_degrees": ring_delta - body_delta,
            "CL_spikes": cl,
            "CR_spikes": cr,
            "p_en_net_spikes_CL_minus_CR": cl - cr,
            "ring_opposes_body": bool(abs(body_delta) >= 4.0 and abs(ring_delta) >= 4.0 and body_delta * ring_delta < 0.0),
        })
    return result


def _events(blocks: list[dict]) -> list[dict]:
    events = [
        {"tick": longturn.BURN_IN, "kind": "cpg_seed", "label": "CPG/reset-clock seed"},
        {"tick": longturn.TURN_START, "kind": "turn_start", "label": "PAULA turn stimulus begins"},
        {"tick": longturn.TURN_STOP, "kind": "turn_stop", "label": "PAULA turn stimulus ends"},
    ]
    for block in blocks:
        if block["ring_opposes_body"]:
            events.append({
                "tick": block["start_tick"],
                "end_tick": block["end_tick"],
                "kind": "ring_reversal",
                "label": "Ring moves opposite to body over one 44-tick stroke",
                "detail": (
                    f"body {block['body_yaw_delta_degrees']:+.2f}°, "
                    f"ring {block['ring_yaw_delta_degrees']:+.2f}°, "
                    f"P-EN CL−CR {block['p_en_net_spikes_CL_minus_CR']:+d}"
                ),
            })
    return events


def _summary(embodied: dict, replay: dict, blocks: list[dict]) -> dict:
    rows, replay_rows = embodied["tick_trace"], replay["tick_trace"]
    body = _unwrap(rows, "physical_yaw_radians")
    ring = _unwrap(rows, "ring_heading_radians")
    replay_ring = _unwrap(replay_rows, "ring_heading_radians")
    phase_difference = np.angle(np.exp(1j * (ring - replay_ring)))
    turn = np.flatnonzero(np.asarray([longturn.TURN_START <= r["neural_tick"] < longturn.TURN_STOP for r in rows]))
    a, b = int(turn[0]), int(turn[-1])
    return {
        "body_turn_delta_degrees": float(np.degrees(body[b] - body[a])),
        "ring_turn_delta_degrees": float(np.degrees(ring[b] - ring[a])),
        "tracking_error_turn_degrees": float(np.degrees((ring[b] - ring[a]) - (body[b] - body[a]))),
        "mean_abs_embodied_replay_phase_difference_degrees": float(np.degrees(np.mean(np.abs(phase_difference)))),
        "max_abs_embodied_replay_phase_difference_degrees": float(np.degrees(np.max(np.abs(phase_difference)))),
        "reversal_blocks": int(sum(block["ring_opposes_body"] for block in blocks)),
        "stroke_period_ticks": 44,
        "interpretation": (
            "The replay receives the exact recorded raw gyro before each matching neural tick. "
            "Near-zero phase difference therefore rules out unrecorded body pose or physics state as the "
            "immediate cause of ring divergence; it does not make the transducer or compass correct."
        ),
    }


def build_bundle(seed: int, case: str) -> dict:
    if case not in longturn.CASES:
        raise ValueError(f"unknown case {case}")
    turn_neuron = ag.TR
    embodied = longturn.run_case(seed, case, longturn.CASES[case], turn_neuron)
    replay = _replay_raw_gyro(seed, longturn.CASES[case], embodied)
    blocks = _blocks(embodied["tick_trace"], replay["tick_trace"], period=44)
    return {
        "format": "compass_replay_bundle.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case": case,
        "seed": seed,
        "course": {
            "cpg_seed_tick": longturn.BURN_IN,
            "turn_start_tick": longturn.TURN_START,
            "turn_stop_tick": longturn.TURN_STOP,
            "stroke_period_ticks": 44,
            "mujoco_timestep_seconds": DT_SECONDS,
        },
        "embodied": embodied,
        "replay": replay,
        "analysis_blocks": blocks,
        "events": _events(blocks),
        "summary": _summary(embodied, replay, blocks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--case", choices=tuple(longturn.CASES), default=DEFAULT_CASE)
    parser.add_argument("--output", type=Path, required=True,
                        help="output bundle JSON; parent must already exist")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing bundle: {args.output}")
    bundle = build_bundle(args.seed, args.case)
    args.output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    summary = bundle["summary"]
    print(
        f"wrote {args.output}: body={summary['body_turn_delta_degrees']:+.2f}° "
        f"ring={summary['ring_turn_delta_degrees']:+.2f}° "
        f"replay phase MAE={summary['mean_abs_embodied_replay_phase_difference_degrees']:.3f}° "
        f"reversals={summary['reversal_blocks']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
