"""Full-tick probe for the separate PAULA sensorimotor-compass agent.

This is intentionally a component diagnostic, not a claim of path integration
or homing.  It drives the existing PAULA turn neuron, lets the normal relay /
graded-muscle / MuJoCo path make the body turn, and records the alternative
agent's motor prediction, raw-yaw residual, PAULA prediction error, fused
update, and E-PG raster on every neural tick.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from simulations.active_inference import aif_agent3d as base
from simulations.active_inference.sensorimotor_agent3d import SensorimotorEstimatorAgent3D
from simulations.active_inference.sensorimotor_agent3d_v2 import SensorimotorEstimatorV2Agent3D
from simulations.active_inference.sensorimotor_agent3d_v3 import SensorimotorEstimatorV3Agent3D
from simulations.active_inference.sensorimotor_agent3d_v4 import SensorimotorEstimatorV4Agent3D
from simulations.active_inference.sensorimotor_agent3d_v5 import SensorimotorEstimatorV5Agent3D
from simulations.active_inference.sensorimotor_agent3d_v6 import SensorimotorEstimatorV6Agent3D
from simulations.active_inference.sensorimotor_agent3d_v7 import SensorimotorEstimatorV7Agent3D


BURN_IN = 200
TURN_START = 500
TURN_STOP = 900
TOTAL_TICKS = 1400
WINDOW = 24


def _heading(window: deque[np.ndarray]) -> float | None:
    vector = np.sum(np.sum(window, axis=0) * np.exp(1j * base.cc.PHI))
    return None if abs(vector) < 1e-12 else float(np.angle(vector))


def _delta_degrees(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if TURN_START <= row["neural_tick"] < TURN_STOP]
    if len(values) < 2 or any(value is None for value in values):
        return None
    return float(np.degrees(np.unwrap(np.asarray(values, dtype=float))[-1]
                            - np.unwrap(np.asarray(values, dtype=float))[0]))


def run_case(seed: int, ablation: str = "none", version: str = "v2",
             build_overrides: dict | None = None) -> dict:
    if ablation not in {"none", "prediction", "sensory"}:
        raise ValueError("ablation must be none, prediction, or sensory")
    if version not in {"v1", "v2", "v3", "v4", "v5", "v6", "v7"}:
        raise ValueError("version must be v1, v2, v3, v4, v5, v6, or v7")
    build = {
        "turn_probe_ports": True,
        "w_opp": 0.0,
        "w_cpu1": 0.0,
        "w_musf_mode": 0.0,
    }
    prefix = "sensorimotor_" if version == "v1" else "sensorimotor_v2_"
    if ablation == "prediction":
        build[f"{prefix}prediction_weight"] = 0.0
    elif ablation == "sensory":
        build[f"{prefix}sensory_weight"] = 0.0
    if build_overrides:
        build.update(build_overrides)

    np.random.seed(seed)
    agent_cls = {
        "v1": SensorimotorEstimatorAgent3D,
        "v2": SensorimotorEstimatorV2Agent3D,
        "v3": SensorimotorEstimatorV3Agent3D,
        "v4": SensorimotorEstimatorV4Agent3D,
        "v5": SensorimotorEstimatorV5Agent3D,
        "v6": SensorimotorEstimatorV6Agent3D,
        "v7": SensorimotorEstimatorV7Agent3D,
    }[version]
    agent = agent_cls(seed=seed, **build)
    if version == "v1":
        sensory_ccw, sensory_cw = base.SM_SENS_CCW, base.SM_SENS_CW
        prediction_ccw, prediction_cw = base.SM_PRED_CCW, base.SM_PRED_CW
        pe_ccw, pe_cw = base.SM_PE_CCW, base.SM_PE_CW
        update_ccw, update_cw = base.SM_UPDATE_CCW, base.SM_UPDATE_CW
    else:
        sensory_ccw, sensory_cw = base.SM2_SENS_CCW, base.SM2_SENS_CW
        prediction_ccw, prediction_cw = base.SM2_PRED_CCW, base.SM2_PRED_CW
        pe_ccw, pe_cw = base.SM2_PE_CCW, base.SM2_PE_CW
        update_ccw, update_cw = base.SM2_UPDATE_CCW, base.SM2_UPDATE_CW
    if version in {"v3", "v4"}:
        copies = 6 if version == "v3" else 4
        lead_ccw = tuple(nid for phase in range(4) for nid in base.SM3_LEAD_CCW[
            phase * base.SM3_PHASE_COPIES:phase * base.SM3_PHASE_COPIES + copies
        ])
        lead_cw = tuple(nid for phase in range(4) for nid in base.SM3_LEAD_CW[
            phase * base.SM3_PHASE_COPIES:phase * base.SM3_PHASE_COPIES + copies
        ])
    else:
        lead_ccw, lead_cw = (), ()
    maintenance_pb = tuple(base.cc.PB_ML + base.cc.PB_MR) if version in {"v5", "v6"} else ()
    peg = tuple(base.cc.PEG) if version in {"v5", "v6"} else ()
    penb = tuple(base.cc.PENB_CL + base.cc.PENB_CR) if version == "v6" else ()
    pena = tuple(nid for columns in (base.cc.PENA_CL, base.cc.PENA_CR) for column in columns for nid in column) if version == "v7" else ()
    agent.world = base.w3.World3D(seed=seed, n_food=0, n_tox=0, arena=8.0)
    agent.img = agent.world.retina()
    agent.birth()
    # The normal birth kick is deferred until after a declared ring-settling
    # period, exactly as in the existing long-turn diagnostic.
    agent._kicked = True
    window: deque[np.ndarray] = deque(maxlen=WINDOW)
    trace: list[dict] = []

    def input_hook(current) -> None:
        if current.t == BURN_IN:
            current.net.set_external_input(base.CPGP[0], 0, 5.0)
        if TURN_START <= current.t < TURN_STOP:
            current.net.set_external_input(base.TR, base.TURN_PROBE_SYN[base.TR], 3.0)

    def capture(current) -> None:
        active = np.asarray([current.nb[nid].O > 0 for nid in base.cc.RING], dtype=float)
        window.append(active)
        x, y, yaw = current.world.pose()
        trace.append({
            "neural_tick": int(current.t),
            "physical_pose": {"x": float(x), "y": float(y), "yaw": float(yaw)},
            "physical_yaw_radians": float(yaw),
            "physical_yaw_rate": float(current.world.yaw_rate()),
            "applied_raw_yaw_rate": float(current.last_gyro_yaw_rate),
            "ring_heading_radians": _heading(window),
            "ring_live": bool(np.any(active)),
            "ring_spikes_by_column": active.astype(int).tolist(),
            "turn_stimulus_active": bool(TURN_START <= current.t < TURN_STOP),
            "turn_spike": int(current.nb[base.TR].O > 0),
            "sensory_residual": {
                "CCW": [float(current.nb[nid].O) for nid in sensory_ccw],
                "CW": [float(current.nb[nid].O) for nid in sensory_cw],
            },
            "motor_prediction": {
                "CCW": [float(current.nb[nid].O) for nid in prediction_ccw],
                "CW": [float(current.nb[nid].O) for nid in prediction_cw],
            },
            "prediction_error": {
                "CCW": float(current.nb[pe_ccw].O),
                "CW": float(current.nb[pe_cw].O),
            },
            "fused_update": {
                "CCW": float(current.nb[update_ccw].O),
                "CW": float(current.nb[update_cw].O),
            },
            "phase_leading_update": {
                "CCW": [float(current.nb[nid].O) for nid in lead_ccw],
                "CW": [float(current.nb[nid].O) for nid in lead_cw],
            },
            "trailing_stabilizer": {
                "pb_maintenance": [float(current.nb[nid].O) for nid in maintenance_pb],
                "peg": [float(current.nb[nid].O) for nid in peg],
                "penb": [float(current.nb[nid].O) for nid in penb],
                "pena": [float(current.nb[nid].O) for nid in pena],
            },
            "p_en_spikes": {
                "CL": int(sum(current.nb[nid].O > 0 for column in base.cc.CL for nid in column)),
                "CR": int(sum(current.nb[nid].O > 0 for column in base.cc.CR for nid in column)),
            },
            # Population totals hide whether a direction-selective P-EN
            # signal is confined to the E-PG bump or has become a spatially
            # uniform push.  Keep the full per-column release raster so the
            # recurrent wave can be diagnosed from raw ticks rather than a
            # period summary.
            "p_en_release_by_column": {
                "CL": [float(sum(current.nb[nid].O for nid in column)) for column in base.cc.CL],
                "CR": [float(sum(current.nb[nid].O for nid in column)) for column in base.cc.CR],
            },
            # The P-EN populations are opt-in graded coincidence cells in
            # this candidate.  Counting their nonzero releases alone hides a
            # transfer deadband, so retain their full-population mean as well
            # as the binary raster count.
            "p_en_mean_release": {
                "CL": float(np.mean([current.nb[nid].O for column in base.cc.CL for nid in column])),
                "CR": float(np.mean([current.nb[nid].O for column in base.cc.CR for nid in column])),
            },
            "ring_mean_release": float(np.mean([current.nb[nid].O for nid in base.cc.RING])),
        })

    base.run_episode(
        agent,
        steps=TOTAL_TICKS,
        sub=1,
        render_every=10**9,
        render_ticks=0,
        log_every=10**9,
        vision=False,
        neural_input_hook=input_hook,
        tick_hook=capture,
    )
    body_turn = _delta_degrees(trace, "physical_yaw_radians")
    ring_turn = _delta_degrees(trace, "ring_heading_radians")
    return {
        "format": f"sensorimotor_compass_probe.{version}",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "version": version,
        "ablation": ablation,
        "build_overrides": dict(build_overrides or {}),
        "agent_manifest": agent.sensorimotor_manifest(),
        "course": {"burn_in": BURN_IN, "turn_start": TURN_START, "turn_stop": TURN_STOP,
                   "total_ticks": TOTAL_TICKS, "turn_neuron": "TR"},
        "summary": {
            "body_turn_delta_degrees": body_turn,
            "ring_turn_delta_degrees": ring_turn,
            "tracking_error_turn_degrees": None if body_turn is None or ring_turn is None else ring_turn - body_turn,
            "ring_live_fraction": float(np.mean([row["ring_live"] for row in trace])),
            "mean_fused_update_ccw": float(np.mean([
                row["fused_update"]["CCW"] for row in trace if TURN_START <= row["neural_tick"] < TURN_STOP
            ])),
            "mean_fused_update_cw": float(np.mean([
                row["fused_update"]["CW"] for row in trace if TURN_START <= row["neural_tick"] < TURN_STOP
            ])),
            "mean_prediction_error_ccw": float(np.mean([
                row["prediction_error"]["CCW"] for row in trace if TURN_START <= row["neural_tick"] < TURN_STOP
            ])),
            "mean_prediction_error_cw": float(np.mean([
                row["prediction_error"]["CW"] for row in trace if TURN_START <= row["neural_tick"] < TURN_STOP
            ])),
        },
        "tick_trace": trace,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--version", choices=("v1", "v2", "v3", "v4", "v5", "v6", "v7"), default="v2")
    parser.add_argument("--ablation", choices=("none", "prediction", "sensory"), default="none")
    parser.add_argument(
        "--update-lam", type=float,
        help="override the versioned fused-update membrane time constant for a declared phase-matching test",
    )
    parser.add_argument(
        "--pen-gain", type=float,
        help="override the versioned PAULA fused-update-to-P-EN synaptic gain for a transfer test",
    )
    parser.add_argument(
        "--d-push", type=int,
        help="override P-EN-to-ring dendritic delay for a declared pathway-latency matching test",
    )
    parser.add_argument(
        "--w-pb-peg", type=float,
        help="override the explicit PB-to-P-EG synaptic weight for a silent-versus-active stabilizer test",
    )
    parser.add_argument(
        "--w-pena-ring", type=float,
        help="override the P-ENa micro-bank-to-ring conductance for a declared scale-response test",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    overrides = {}
    if args.update_lam is not None:
        key = "sensorimotor_update_lam" if args.version == "v1" else "sensorimotor_v2_update_lam"
        overrides[key] = args.update_lam
    if args.pen_gain is not None:
        key = "sensorimotor_pen_gain" if args.version == "v1" else "sensorimotor_v2_pen_gain"
        overrides[key] = args.pen_gain
    if args.d_push is not None:
        overrides["d_push"] = args.d_push
    if args.w_pb_peg is not None:
        overrides["w_pb_peg"] = args.w_pb_peg
    if args.w_pena_ring is not None:
        overrides["w_pena_ring"] = args.w_pena_ring
    record = run_case(args.seed, args.ablation, args.version, overrides)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    summary = record["summary"]
    print(
        f"version={args.version} ablation={args.ablation} body={summary['body_turn_delta_degrees']:+.2f} "
        f"ring={summary['ring_turn_delta_degrees']:+.2f} live={summary['ring_live_fraction']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
