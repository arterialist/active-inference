"""Isolated acceptance harness for the explicit PAULA PB/EB bridge.

This is deliberately a *whole compass* test with no MuJoCo body: normal PAULA
gyro ports, reset clock, P-ENs, PB relays and E-PG ring all tick together.  It
proves only the prerequisites for an embodied trial:

* the bridge relay actually carries P-EN activity;
* the ring remains alive during a hold and two opposite signed drives; and
* the two signed drives move the decoded E-PG phase in opposite directions.

It neither supplies a heading estimate nor writes to the circuit after
measurement.  It is therefore a necessary gate, not a navigation claim.
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


HERE = Path(__file__).resolve().parent
WINDOW = 24
SETTLE = 200
DRIVE = 400
HOLD = 200

COMMON = {
    "vestibular_stroke_reset": True,
    "vop_reset_period": 44,
    "vop_reset_input_gain": 1.0,
    "vop_reset_gain": 40.0,
    "vop_reset_lam": 44.0,
    "vop_reset_output_gain": 1.0,
    "vop_reset_output_S0": -1.5,
    "vop_reset_pen_gain": 1.0,
    "d_push": 4,
    "graded_shift": True,
    "graded_shift_gain": 0.5,
    "graded_shift_S0": 0.0,
    "conjunctive_graded_shift": True,
    "conjunctive_graded_S0": 0.0,
    "graded_ring": True,
    "graded_ring_gain": 0.08,
    "graded_ring_S0": 0.2,
    "w_push_ccw": 0.34,
    "w_push_cw": 0.67,
}


def _heading(window: deque[np.ndarray]) -> float | None:
    vector = np.sum(np.sum(window, axis=0) * np.exp(1j * cc.PHI))
    return None if abs(vector) < 1e-12 else float(np.angle(vector))


def _phase_delta(rows: list[dict], start: int, end: int) -> float | None:
    segment = [row["ring_heading_radians"] for row in rows if start <= row["tick"] < end]
    if any(value is None for value in segment) or len(segment) < 2:
        return None
    phase = np.unwrap(np.asarray(segment, dtype=float))
    return float(np.degrees(phase[-1] - phase[0]))


def run_case(seed: int, bridge: bool, maintenance: bool, w_pb_peg: float = 0.8,
             w_pb_cross_inhib: float = 0.0, w_pb_opponent: float = 0.0,
             pb_shift_lam: float = 1.0, pb_reset_gain: float = 0.0,
             phase_update: bool = False, phase_r: float = 2.1,
             phase_push: float = 2.1, phase_ring: float = 0.7,
             phase_clock: float = 0.7) -> dict:
    build = dict(COMMON)
    if bridge:
        build.update({"pb_eb_bridge": True, "w_pb_cross_inhib": float(w_pb_cross_inhib),
                      "w_pb_opponent": float(w_pb_opponent),
                      "pb_shift_lam": float(pb_shift_lam)})
        if phase_update:
            # A selected phase has about nine local PB gates rather than the
            # ~56 concurrently active direct P-ENs. Hold the aggregate
            # update conductance in the same order before assessing the new
            # temporal wiring; do not mistake sparse population size for an
            # architectural phase-gating failure.
            build.update({"pb_phase_update": True, "r_pb_phase": float(phase_r),
                          "w_push_ccw": float(phase_push), "w_push_cw": float(phase_push),
                          "w_pb_phase_ring": float(phase_ring),
                          "w_pb_phase_clock": float(phase_clock)})
        if pb_reset_gain > 0.0:
            build.update({"pb_stroke_reset": True, "pb_reset_gain": float(pb_reset_gain)})
    if maintenance:
        if not bridge:
            raise ValueError("maintenance bridge requires --bridge")
        build.update({"w_peg": 1.6, "w_pb_peg": float(w_pb_peg), "w_peg_ring": 0.8})
    np.random.seed(seed)
    agent = ag.AIFAgent3D(seed=seed, **build)
    agent.birth()
    window: deque[np.ndarray] = deque(maxlen=WINDOW)
    trace: list[dict] = []
    pb_cl = cc.PB_PHASE_CL if phase_update else cc.PB_CL
    pb_cr = cc.PB_PHASE_CR if phase_update else cc.PB_CR
    # A fixed signed current is intentionally simple.  The later replay test
    # asks whether the same bridge remains stable under the physical waveform.
    course = [("settle", 0.0, SETTLE), ("CCW", 4.0, DRIVE),
              ("CW", -4.0, DRIVE), ("hold", 0.0, HOLD)]
    for label, raw, duration in course:
        for _ in range(duration):
            agent.tick(ccw=max(raw, 0.0), cw=max(-raw, 0.0), speed=0.0, vision=False)
            active = np.asarray([agent.nb[nid].O > 0 for nid in cc.RING], dtype=float)
            window.append(active)
            trace.append({
                "tick": agent.t,
                "phase": label,
                "input_raw_yaw_rate": raw,
                "ring_heading_radians": _heading(window),
                "ring_live": bool(np.any(active)),
                "ring_spikes_by_column": active.astype(int).tolist(),
                "pb_shift_spikes": (
                    {"CL": sum(int(agent.nb[nid].O > 0) for nid in pb_cl),
                     "CR": sum(int(agent.nb[nid].O > 0) for nid in pb_cr)}
                    if bridge else None
                ),
                "pb_maintenance_spikes": (
                    sum(int(agent.nb[nid].O > 0) for nid in cc.PB_ML + cc.PB_MR)
                    if maintenance else None
                ),
                "peg_spikes": (
                    sum(int(agent.nb[nid].O > 0) for nid in cc.PEG)
                    if maintenance else None
                ),
            })
    # Agent tick starts from t=0 after birth; use phase labels rather than
    # constructor timing to make the metrics robust to future birth changes.
    phases = {name: [row for row in trace if row["phase"] == name] for name, _, _ in course}
    deltas = {}
    for name in ("CCW", "CW"):
        vals = [row["ring_heading_radians"] for row in phases[name]]
        deltas[name] = None if any(v is None for v in vals) else float(np.degrees(np.unwrap(vals)[-1] - np.unwrap(vals)[0]))
    bridge_spikes = sum(sum(row["pb_shift_spikes"].values()) for row in trace if row["pb_shift_spikes"] is not None)
    maintenance_spikes = sum(row["pb_maintenance_spikes"] or 0 for row in trace)
    peg_spikes = sum(row["peg_spikes"] or 0 for row in trace)
    return {
        "seed": seed,
        "build": build,
        "course": {"settle": SETTLE, "drive": DRIVE, "hold": HOLD, "raw_drive": 4.0},
        "tick_trace": trace,
        "summary": {
            "ring_live_fraction": float(np.mean([row["ring_live"] for row in trace])),
            "ccw_phase_delta_degrees": deltas["CCW"],
            "cw_phase_delta_degrees": deltas["CW"],
            "signed_response": bool(deltas["CCW"] is not None and deltas["CW"] is not None and deltas["CCW"] * deltas["CW"] < 0),
            "pb_shift_spikes_total": int(bridge_spikes),
            "pb_maintenance_spikes_total": int(maintenance_spikes),
            "peg_spikes_total": int(peg_spikes),
            "hold_live_fraction": float(np.mean([row["ring_live"] for row in phases["hold"]])),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--bridge", action="store_true", help="enable PB update bridge")
    parser.add_argument("--maintenance", action="store_true", help="also enable PB/P-EG maintenance")
    parser.add_argument("--w-pb-peg", type=float, default=0.8,
                        help="PB-to-P-EG synaptic weight for the maintenance-only check")
    parser.add_argument("--w-pb-cross-inhib", type=float, default=0.0,
                        help="reciprocal inhibition between ordinary PB update relays")
    parser.add_argument("--w-pb-opponent", type=float, default=0.0,
                        help="opponent P-EN dendritic weight on each PB update relay")
    parser.add_argument("--pb-shift-lam", type=float, default=1.0,
                        help="ordinary PAULA PB relay membrane time constant")
    parser.add_argument("--pb-reset-gain", type=float, default=0.0,
                        help="ordinary stroke-clock inhibitory reset on PB relays (0 disables)")
    parser.add_argument("--phase-update", action="store_true",
                        help="use the explicit CPG-phase-gated PB update population")
    parser.add_argument("--phase-r", type=float, default=2.1,
                        help="threshold of each ordinary PB phase-gate neuron")
    parser.add_argument("--phase-push", type=float, default=2.1,
                        help="per-gate PB-to-E-PG pulse weight (aggregate-matched to direct P-EN activity)")
    parser.add_argument("--phase-ring", type=float, default=0.7,
                        help="local E-PG dendritic weight on an ordinary PB phase gate")
    parser.add_argument("--phase-clock", type=float, default=0.7,
                        help="CPG-phase dendritic weight on an ordinary PB phase gate")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = run_case(args.seed, args.bridge, args.maintenance, args.w_pb_peg,
                      args.w_pb_cross_inhib, args.w_pb_opponent, args.pb_shift_lam,
                      args.pb_reset_gain, args.phase_update, args.phase_r, args.phase_push,
                      args.phase_ring, args.phase_clock)
    record["created_utc"] = datetime.now(timezone.utc).isoformat()
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    summary = record["summary"]
    print(
        f"bridge={args.bridge} maintenance={args.maintenance} live={summary['ring_live_fraction']:.3f} "
        f"CCW={summary['ccw_phase_delta_degrees']:+.1f} CW={summary['cw_phase_delta_degrees']:+.1f} "
        f"signed={summary['signed_response']} pb_spikes={summary['pb_shift_spikes_total']} "
        f"peg_spikes={summary['peg_spikes_total']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
