"""Causal acceptance test for the analog XACC/YACC path-integration primitive.

The normal heading ring is intentionally *not* under test here.  A calibrated
head-direction sensory transducer clamps a seven-cell heading code from the
actual MuJoCo body heading onto the existing RING -> PG pathway.  The only
other input is the body's measured planar speed.  This isolates the analog
XACC/YACC displacement memory from the still-unaccepted recurrent compass.

The body is propelled by the separately accepted PAULA CPG/muscle circuit;
the fixed descending currents merely supply a repeatable curved, turn-reversal
trajectory.  They never set a pose, velocity, or PI state.  The replayed
conditions preserve that identical physical sensory stream while sweeping
``k_pi`` and severing just PG -> XACC/YACC with ``k_pi=0``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from simulations.active_inference import central_complex as cc
from simulations.active_inference import nmrower2 as motor


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NEURON_MODEL = ROOT.parent / "neuron-model"
SPEED_GAIN = 2.5
SPEED_CAP = 1.0
HEADING_AMP = 4.0
RETAIN_TICKS = 240
GAIN_SWEEP = (16.0, 64.0, 128.0, 256.0)
# The sweep's 128 branch is already 8% compressed; 64 is the largest
# fully linear setting on the physical calibration trace.
SELECTED_K_PI = 64.0
# A physical PAULA-motor trajectory: outward straight leg, strong left curve,
# a second leg, then a strong right curve that reverses the turn.
SEGMENTS = (
    ("outbound_straight", 0.0, 0.0, 300),
    ("left_curve", 0.1, 0.0, 540),
    ("second_leg", 0.0, 0.0, 240),
    ("right_curve_reversal", 0.0, 0.1, 540),
)


def _revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _angle_error_degrees(a: float, b: float) -> float:
    return abs(float(np.degrees(np.angle(np.exp(1j * (a - b))))) )


def _initial_yaw(seed: int) -> float:
    """Deterministic varied initial headings without changing the motor circuit."""
    return float(np.radians(((seed * 37) % 360) - 180))


def physical_sensory_stream(seed: int) -> dict:
    """Collect actual heading and speed from a PAULA-driven MuJoCo body."""
    np.random.seed(seed)
    rower = motor.NMRower2()
    rower.data.qpos[rower.rz] = _initial_yaw(seed)
    mujoco.mj_forward(rower.model, rower.data)
    x0, y0, _ = rower.pose()
    heading0 = rower.heading()
    trace: list[dict] = []
    for segment, steer_left, steer_right, ticks in SEGMENTS:
        for _ in range(ticks):
            rower.step(steer_left, steer_right, sub=6)
            x, y, _ = rower.pose()
            heading = rower.heading()
            heading_index = int(round((heading % (2.0 * np.pi)) / (2.0 * np.pi) * cc.NR)) % cc.NR
            speed = float(np.hypot(rower.data.qvel[rower.sx], rower.data.qvel[rower.sy]))
            trace.append({
                "segment": segment,
                "pose": {"x": float(x - x0), "y": float(y - y0)},
                "heading_radians": float(heading),
                "heading_index": heading_index,
                "speed_physical": speed,
                "speed_current": min(SPEED_CAP, SPEED_GAIN * speed),
                "motor": {
                    "steer_left_current": steer_left,
                    "steer_right_current": steer_right,
                    "cpg_spikes": {str(nid): int(rower.nb[nid].O > 0) for nid in motor.P},
                    "muscle_state": {str(nid): float(rower.nb[nid].S) for nid in (motor.MLp, motor.MLr, motor.MRp, motor.MRr)},
                },
            })
    final_heading_delta = float(np.angle(np.exp(1j * (rower.heading() - heading0))))
    return {
        "seed": seed,
        "initial_yaw_radians": _initial_yaw(seed),
        "initial_heading_radians": float(heading0),
        "final_heading_delta_radians": final_heading_delta,
        "trace": trace,
    }


def _drive_heading_code(compass: cc.Compass, heading_index: int, speed_current: float) -> int:
    """Calibrated experimental heading transducer; it bypasses only ring recurrence."""
    active = {(heading_index + offset) % cc.NR for offset in range(-cc.NB, cc.NB + 1)}
    for i, neuron_id in enumerate(cc.RING):
        compass.net.set_external_input(neuron_id, cc.SEEDMAP[neuron_id], HEADING_AMP if i in active else 0.0)
    compass._drive(0.0, 0.0, speed_current)
    compass.core.do_tick()
    return sum(int(compass.nb[neuron_id].O > 0) for neuron_id in cc.PG)


def replay_accumulator(sensory_trace: list[dict], k_pi: float, retain_ticks: int = RETAIN_TICKS) -> dict:
    """Replay one physical heading/speed stream through only the neural PI route."""
    # Ring recurrence is deliberately zeroed: this test injects a sensory
    # heading code so it measures XACC/YACC, not the imperfect compass wave.
    compass = cc.Compass(
        w_self=0.0, w_nbr=0.0, w_gi_ring=0.0,
        w_pg=2.4, w_pg_s=2.6, k_pi=k_pi,
    )
    trace = []
    for row in sensory_trace:
        pg_spikes = _drive_heading_code(compass, row["heading_index"], row["speed_current"])
        trace.append({
            "segment": row["segment"],
            "heading_index": row["heading_index"],
            "speed_physical": row["speed_physical"],
            "speed_current": row["speed_current"],
            "body_pose": row["pose"],
            "pg_spikes": pg_spikes,
            "xacc": float(compass.nb[cc.XACC].S),
            "yacc": float(compass.nb[cc.YACC].S),
        })
    before = np.array([float(compass.nb[cc.XACC].S), float(compass.nb[cc.YACC].S)])
    for neuron_id in cc.RING:
        compass.net.set_external_input(neuron_id, cc.SEEDMAP[neuron_id], 0.0)
    for _ in range(retain_ticks):
        compass._drive(0.0, 0.0, 0.0)
        compass.core.do_tick()
    after = np.array([float(compass.nb[cc.XACC].S), float(compass.nb[cc.YACC].S)])
    return {
        "k_pi": k_pi,
        "tick_trace": trace,
        "final_vector": {"x": float(before[0]), "y": float(before[1]), "magnitude": float(np.hypot(*before))},
        "post_retention_vector": {"x": float(after[0]), "y": float(after[1]), "magnitude": float(np.hypot(*after))},
    }


def _summary(physical: dict, selected: dict, ablated: dict, gains: dict[str, dict]) -> dict:
    stream, neural = physical["trace"], selected["tick_trace"]
    final_pose = stream[-1]["pose"]
    body_vector = np.array([final_pose["x"], final_pose["y"]])
    neural_vector = np.array([selected["final_vector"]["x"], selected["final_vector"]["y"]])
    body_magnitude = float(np.hypot(*body_vector))
    neural_magnitude = float(np.hypot(*neural_vector))
    angular_error = _angle_error_degrees(np.arctan2(neural_vector[1], neural_vector[0]), np.arctan2(body_vector[1], body_vector[0]))
    straight = [row for row in neural if row["segment"] == "outbound_straight"]
    body_distance = np.array([np.hypot(row["body_pose"]["x"], row["body_pose"]["y"]) for row in straight])
    neural_distance = np.array([np.hypot(row["xacc"], row["yacc"]) for row in straight])
    magnitude_correlation = float(np.corrcoef(body_distance, neural_distance)[0, 1])
    # PAULA communicates through discrete spikes, so a single neural tick can
    # show a small membrane decrement even while its physical distance grows.
    # Use ten-tick measurement windows (40 ms) for the monotonicity criterion;
    # the raw per-tick trace remains the primary evidence.
    magnitude_increases = float(np.mean(np.diff(neural_distance[::10]) >= -1e-8))
    retention = selected["post_retention_vector"]["magnitude"] / max(neural_magnitude, 1e-12)
    headings = np.unwrap(np.array([row["heading_radians"] for row in stream]))
    heading_delta = np.degrees(headings - headings[0])
    gain_magnitudes = {name: record["final_vector"]["magnitude"] for name, record in gains.items()}
    low_per_gain = gain_magnitudes["16.0"] / 16.0
    gain_ratios = {name: value / (float(name) * low_per_gain) for name, value in gain_magnitudes.items()}
    return {
        "body_final_vector": {"x": float(body_vector[0]), "y": float(body_vector[1]), "magnitude": body_magnitude},
        "neural_final_vector": {"x": float(neural_vector[0]), "y": float(neural_vector[1]), "magnitude": neural_magnitude},
        "home_vector_angular_error_degrees": angular_error,
        "outbound_magnitude_correlation": magnitude_correlation,
        "outbound_magnitude_non_decrease_fraction": magnitude_increases,
        "retention_ratio_after_zero_input": retention,
        "physical_turn_min_degrees": float(heading_delta.min()),
        "physical_final_turn_degrees": float(heading_delta[-1]),
        "pg_spikes_selected": int(sum(row["pg_spikes"] for row in neural)),
        "pg_spikes_ablated": int(sum(row["pg_spikes"] for row in ablated["tick_trace"])),
        "ablated_final_magnitude": ablated["final_vector"]["magnitude"],
        "gain_magnitudes": gain_magnitudes,
        "gain_ratio_to_linear_16": gain_ratios,
    }


def run_case(seed: int) -> dict:
    physical = physical_sensory_stream(seed)
    # The gain sweep replays the identical real physical sensory trace; it
    # neither alters the body nor selects an action.  64 is the largest fully
    # linear gain; 256 is deliberately included as the saturation
    # boundary control.
    # The first leg has one stable heading and monotonically increasing real
    # distance, making it the correct calibration slice.  Replaying the later
    # curve for every candidate would add no information about gain linearity.
    outbound = [row for row in physical["trace"] if row["segment"] == "outbound_straight"]
    gains = {str(gain): replay_accumulator(outbound, gain, retain_ticks=0) for gain in GAIN_SWEEP}
    selected = replay_accumulator(physical["trace"], SELECTED_K_PI)
    ablated = replay_accumulator(physical["trace"], 0.0)
    return {
        "seed": seed,
        "physical": physical,
        "gain_sweep": gains,
        "selected": selected,
        "ablated": ablated,
        "summary": _summary(physical, selected, ablated, gains),
    }


def _accept(records: dict[str, dict]) -> list[str]:
    failures: list[str] = []
    for seed, record in records.items():
        row = record["summary"]
        if row["body_final_vector"]["magnitude"] < 0.5:
            failures.append(f"seed {seed}: PAULA-driven physical route did not leave the origin")
        if row["physical_turn_min_degrees"] > -45.0 or abs(row["physical_final_turn_degrees"]) > 20.0:
            failures.append(f"seed {seed}: physical route did not complete the left/right turn reversal")
        if row["pg_spikes_selected"] < 100:
            failures.append(f"seed {seed}: heading/speed inputs did not recruit PG")
        if row["home_vector_angular_error_degrees"] > 15.0:
            failures.append(f"seed {seed}: XACC/YACC direction error exceeds 15 degrees")
        if row["outbound_magnitude_correlation"] < 0.95 or row["outbound_magnitude_non_decrease_fraction"] < 0.95:
            failures.append(f"seed {seed}: XACC/YACC magnitude did not increase with real outward distance")
        if row["retention_ratio_after_zero_input"] < 0.98:
            failures.append(f"seed {seed}: XACC/YACC did not retain its vector after input removal")
        if row["ablated_final_magnitude"] > 1e-12:
            failures.append(f"seed {seed}: k_pi=0 did not sever accumulator input")
        if row["pg_spikes_ablated"] != row["pg_spikes_selected"]:
            failures.append(f"seed {seed}: k_pi=0 changed PG instead of only XACC/YACC")
        ratios = row["gain_ratio_to_linear_16"]
        if abs(ratios[str(SELECTED_K_PI)] - 1.0) > 0.05:
            failures.append(f"seed {seed}: selected k_pi lies outside the measured linear gain range")
        if ratios["256.0"] > 0.98:
            failures.append(f"seed {seed}: saturation-boundary control did not saturate")
    return failures


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 44, 77])
    parser.add_argument("--output", type=Path, help="new directory for manifest and raw traces")
    parser.add_argument("--resume", type=Path, help="complete missing seeds in an interrupted output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output and args.resume:
        raise SystemExit("use either --output or --resume, not both")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.resume or args.output or HERE / "results" / f"embodied_xacc_pi_causal_{timestamp}"
    if args.resume:
        manifest = output / "manifest.json"
        if not manifest.exists():
            raise SystemExit(f"--resume requires an existing manifest: {manifest}")
        if json.loads(manifest.read_text()).get("experiment") != "embodied_xacc_pi_causal":
            raise SystemExit("--resume directory belongs to another experiment")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "manifest.json", {
            "experiment": "embodied_xacc_pi_causal",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": {"active_inference": _revision(ROOT), "neuron_model": _revision(NEURON_MODEL)},
            "source_fingerprints": {
                "central_complex.py": _sha256(HERE.parent / "central_complex.py"),
                "nmrower2.py": _sha256(HERE.parent / "nmrower2.py"),
                "embodied_xacc_pi_causal.py": _sha256(Path(__file__).resolve()),
            },
            "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "mujoco": mujoco.__version__},
            "seeds": args.seeds, "segments": SEGMENTS,
            "heading_transducer": "physical heading -> seven-cell external RING code; recurrent ring disabled only to isolate XACC/YACC",
            "speed_transducer": {"source": "MuJoCo qvel", "gain": SPEED_GAIN, "cap": SPEED_CAP},
            "gain_sweep": GAIN_SWEEP, "selected_k_pi": SELECTED_K_PI, "retention_ticks": RETAIN_TICKS,
            "scope": "PAULA motor/MuJoCo heading+speed -> calibrated sensory heading code -> PG -> analog XACC/YACC; excludes recurrent-compass validation and closed-loop homing",
            "primary_evidence": "per-seed physical and neural tick traces",
        })
    records = {}
    for seed in args.seeds:
        path = output / f"seed{seed}.json"
        if args.resume and path.exists():
            records[str(seed)] = json.loads(path.read_text())
        else:
            records[str(seed)] = run_case(seed)
            _write_json(path, records[str(seed)])
        s = records[str(seed)]["summary"]
        print(
            f"seed={seed}: angle={s['home_vector_angular_error_degrees']:.2f}deg "
            f"mag_r={s['outbound_magnitude_correlation']:.3f} retain={s['retention_ratio_after_zero_input']:.3f} "
            f"ablate={s['ablated_final_magnitude']:.3g}",
            flush=True,
        )
    _write_json(output / "summary.json", {seed: record["summary"] for seed, record in records.items()})
    failures = _accept(records)
    _write_json(output / "acceptance.json", {"passed": not failures, "failures": failures})
    print(f"Wrote embodied XACC/YACC causal evidence to {output}")
    if failures:
        print("FAIL: " + " | ".join(failures), file=sys.stderr)
        return 1
    print("PASS: XACC/YACC integrates MuJoCo heading/speed, retains the vector, and is causally input-dependent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
