"""Locate recurrent-compass failures at the isolated-to-embodied boundary.

The same experimental continuous P-EN parameters are examined in three
increasingly realistic conditions:

1. direct isolated P-EN drive (the recurrent-ring mechanism itself),
2. the full PAULA raw-gyro/notch/opponent transducer with a declared rate,
3. ordinary, unscripted food-and-toxin locomotion in MuJoCo, followed by an
   isolated replay of the *recorded raw gyro stream* through the same PAULA
   transducer.

The final comparison deliberately has no host-side heading estimate.  It
answers whether rough embodied movement is enough to explain a failure:
if the replay fails too, the temporal PAULA transducer is at fault; if replay
works but the live trace fails, another embodied input or timing interaction
must be traced.  This is a diagnostic, not a navigation acceptance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from simulations.active_inference import aif_agent3d as ag
from simulations.active_inference import central_complex as cc


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NEURON_MODEL = ROOT.parent / "neuron-model"
WINDOW = 24
ISOLATED_TICKS = 1000
DRIVE_START = 300
DRIVE_STOP = 700

# Kept opt-in.  This is the smallest construction that exercised the
# continuous-P-EN candidate in the long-turn diagnostic; it does not replace
# the normal organism configuration.
BUILD = {
    "vestibular_notch_graded_opponent": True,
    "vop_notch_graded_opp_gain": 4.0,
    "vop_notch_graded_opp_pen_gain": 3.0,
    "graded_shift": True,
    "graded_shift_gain": 0.5,
    "graded_shift_S0": 0.4,
    "conjunctive_graded_shift": True,
    "conjunctive_graded_S0": 0.0,
    "graded_ring": True,
    "graded_ring_gain": 0.08,
    "graded_ring_S0": 0.2,
    "w_push_ccw": 0.34,
    "w_push_cw": 0.67,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _heading(window: deque[np.ndarray]) -> float | None:
    spikes = np.sum(window, axis=0)
    vector = np.sum(spikes * np.exp(1j * cc.PHI))
    return None if abs(vector) < 1e-12 else float(np.angle(vector))


def _ring_row(agent: ag.AIFAgent3D, window: deque[np.ndarray]) -> tuple[float | None, int]:
    active = np.asarray([agent.nb[nid].O > 0 for nid in cc.RING], dtype=float)
    window.append(active)
    return _heading(window), int(active.sum())


def _trace_summary(trace: list[dict], heading_key: str = "ring_heading_radians") -> dict:
    live = [i for i, row in enumerate(trace) if row[heading_key] is not None]
    first_silent = next((row["tick"] for row in trace if row[heading_key] is None), None)
    if not live:
        return {
            "live_fraction": 0.0,
            "first_silent_tick": first_silent,
            "last_live_tick": None,
            "heading_delta_degrees": None,
        }
    prefix_end = next((i for i, row in enumerate(trace) if row[heading_key] is None), len(trace))
    prefix = np.unwrap(np.asarray([trace[i][heading_key] for i in range(prefix_end)]))
    return {
        "live_fraction": float(len(live) / len(trace)),
        "first_silent_tick": first_silent,
        "last_live_tick": trace[live[-1]]["tick"],
        "heading_delta_degrees": float(np.degrees(prefix[-1] - prefix[0])),
    }


def direct_p_en_case(seed: int, direction: str) -> dict:
    """Isolate only the recurrent P-EN/ring mechanism (no gyro transducer)."""
    np.random.seed(seed)
    # Remove the complete gyro route so ``AIFAgent3D.tick`` supplies the
    # declared current to P-EN synapse 1. This still uses the same inherited
    # PAULA class, ring tonic, and recurrent wiring as the embodied candidate.
    direct_build = {key: value for key, value in BUILD.items() if key != "vestibular_notch_graded_opponent"}
    agent = ag.AIFAgent3D(seed=seed, **direct_build)
    agent.birth()
    window: deque[np.ndarray] = deque(maxlen=WINDOW)
    trace: list[dict] = []
    for tick in range(ISOLATED_TICKS):
        velocity = 0.75 if DRIVE_START <= tick < DRIVE_STOP else 0.0
        agent.tick(
            ccw=velocity if direction == "CCW" else 0.0,
            cw=velocity if direction == "CW" else 0.0,
            speed=0.0,
            vision=False,
        )
        heading, ring_spikes = _ring_row(agent, window)
        trace.append({
            "tick": tick,
            "ring_heading_radians": heading,
            "ring_spikes": ring_spikes,
            "direct_p_en_current": velocity,
        })
    return {"condition": "direct_p_en", "seed": seed, "direction": direction, "tick_trace": trace}


def raw_rate_case(seed: int, direction: str, rate: float = 4.0) -> dict:
    """Exercise the complete PAULA gyro route, without physics or motors."""
    np.random.seed(seed)
    agent = ag.AIFAgent3D(seed=seed, **BUILD)
    agent.birth()
    window: deque[np.ndarray] = deque(maxlen=WINDOW)
    trace: list[dict] = []
    for tick in range(ISOLATED_TICKS):
        raw_rate = rate if DRIVE_START <= tick < DRIVE_STOP else 0.0
        agent.tick(
            ccw=raw_rate if direction == "CCW" else 0.0,
            cw=raw_rate if direction == "CW" else 0.0,
            speed=0.0,
            vision=False,
        )
        heading, ring_spikes = _ring_row(agent, window)
        trace.append({
            "tick": tick,
            "ring_heading_radians": heading,
            "ring_spikes": ring_spikes,
            "raw_yaw_rate": raw_rate if direction == "CCW" else -raw_rate,
            "notch_release": {
                "CCW": float(agent.nb[ag.VEST_NET_CCW].O),
                "CW": float(agent.nb[ag.VEST_NET_CW].O),
            },
            "opponent_release": {
                "CCW": float(agent.nb[ag.VEST_OPP_CCW].O),
                "CW": float(agent.nb[ag.VEST_OPP_CW].O),
            },
        })
    return {"condition": "raw_rate_isolated", "seed": seed, "direction": direction, "tick_trace": trace}


def embodied_food_toxin_case(seed: int, steps: int, substeps: int) -> dict:
    """Capture the normal nonvisual agent's unprescribed physical gyro stream."""
    np.random.seed(seed)
    agent = ag.AIFAgent3D(seed=seed, **BUILD)
    agent.birth()
    window: deque[np.ndarray] = deque(maxlen=WINDOW)
    trace: list[dict] = []

    def capture(current: ag.AIFAgent3D) -> None:
        heading, ring_spikes = _ring_row(current, window)
        x, y, yaw = current.world.pose()
        trace.append({
            "tick": current.t,
            "ring_heading_radians": heading,
            "ring_spikes": ring_spikes,
            "raw_yaw_rate": float(current.world.yaw_rate()),
            "applied_raw_yaw_rate": float(current.last_gyro_yaw_rate),
            "speed": float(current.world.speed()),
            "pose": {"x": float(x), "y": float(y), "yaw": float(yaw)},
            "food_eaten": current.world.eaten,
            "toxin_hits": current.world.tox_hits,
            "sensor_current": dict(current.last_sensor_drives),
            "opponent_release": {
                "CCW": float(current.nb[ag.VEST_OPP_CCW].O),
                "CW": float(current.nb[ag.VEST_OPP_CW].O),
            },
        })

    ag.run_episode(
        agent,
        steps=steps,
        sub=substeps,
        vision=False,
        render_every=10**9,
        render_ticks=0,
        log_every=10**9,
        tick_hook=capture,
    )
    return {
        "condition": "embodied_food_toxin",
        "seed": seed,
        "build": BUILD,
        "vision_enabled": False,
        "agent_steps": steps,
        "substeps": substeps,
        "food_sources": agent.world.foods,
        "toxin_sources": agent.world.toxins,
        "tick_trace": trace,
    }


def replay_raw_gyro(seed: int, embodied: dict) -> dict:
    """Replay raw physical observations through the gyro route alone."""
    np.random.seed(seed)
    agent = ag.AIFAgent3D(seed=seed, **BUILD)
    agent.birth()
    window: deque[np.ndarray] = deque(maxlen=WINDOW)
    trace: list[dict] = []
    prior_observed_rate = 0.0
    for source in embodied["tick_trace"]:
        # Current traces contain the exact pre-physics PAULA input. Retain a
        # one-tick shifted fallback for historical traces that only retained
        # the post-physics observation.
        raw_rate = float(source.get("applied_raw_yaw_rate", prior_observed_rate))
        prior_observed_rate = float(source["raw_yaw_rate"])
        agent.tick(ccw=max(0.0, raw_rate), cw=max(0.0, -raw_rate), speed=float(source["speed"]), vision=False)
        heading, ring_spikes = _ring_row(agent, window)
        trace.append({
            "tick": source["tick"],
            "ring_heading_radians": heading,
            "ring_spikes": ring_spikes,
            "raw_yaw_rate": raw_rate,
            "source_embodied_ring_heading_radians": source["ring_heading_radians"],
            "opponent_release": {
                "CCW": float(agent.nb[ag.VEST_OPP_CCW].O),
                "CW": float(agent.nb[ag.VEST_OPP_CW].O),
            },
        })
    return {"condition": "raw_gyro_replay", "seed": seed, "build": BUILD, "tick_trace": trace}


def _replay_difference(live: dict, replay: dict) -> dict:
    pairs = [
        (a["ring_heading_radians"], b["ring_heading_radians"])
        for a, b in zip(live["tick_trace"], replay["tick_trace"], strict=True)
        if a["ring_heading_radians"] is not None and b["ring_heading_radians"] is not None
    ]
    if not pairs:
        return {"overlap_ticks": 0, "mean_abs_phase_difference_degrees": None}
    phase = np.asarray([np.angle(np.exp(1j * (a - b))) for a, b in pairs])
    return {
        "overlap_ticks": len(pairs),
        "mean_abs_phase_difference_degrees": float(np.degrees(np.mean(np.abs(phase)))),
    }


def _embodied_summary(record: dict) -> dict:
    rows = record["tick_trace"]
    result = _trace_summary(rows)
    yaw = np.asarray([row["raw_yaw_rate"] for row in rows])
    pose = np.unwrap(np.asarray([row["pose"]["yaw"] for row in rows]))
    result.update({
        "body_yaw_delta_degrees": float(np.degrees(pose[-1] - pose[0])),
        "raw_yaw_abs_p95": float(np.percentile(np.abs(yaw), 95)),
        "raw_yaw_sign_changes": int(np.count_nonzero(np.diff(np.signbit(yaw)))),
        "food_eaten": rows[-1]["food_eaten"],
        "toxin_hits": rows[-1]["toxin_hits"],
    })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--agent-steps", type=int, default=150)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.agent_steps <= 0 or args.substeps <= 0:
        raise SystemExit("--agent-steps and --substeps must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or HERE / "results" / f"compass_transfer_diagnostic_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "manifest.json", {
        "experiment": "compass_transfer_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": {"active_inference": _revision(ROOT), "neuron_model": _revision(NEURON_MODEL)},
        "source_fingerprints": {
            "aif_agent3d.py": _sha256(HERE.parent / "aif_agent3d.py"),
            "central_complex.py": _sha256(HERE.parent / "central_complex.py"),
            "runner": _sha256(Path(__file__).resolve()),
        },
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "mujoco": mujoco.__version__},
        "build": BUILD,
        "seeds": args.seeds,
        "agent_steps": args.agent_steps,
        "substeps": args.substeps,
        "scope": "isolated direct P-EN vs isolated raw-gyro PAULA route vs raw gyro from ordinary nonvisual food/toxin locomotion; diagnostic only",
    })
    summary: dict[str, dict] = {}
    for seed in args.seeds:
        rows: dict[str, dict] = {}
        for direction in ("CCW", "CW"):
            direct = direct_p_en_case(seed, direction)
            raw = raw_rate_case(seed, direction)
            _write_json(output / f"direct_p_en_{direction}_seed{seed}.json", direct)
            _write_json(output / f"raw_rate_{direction}_seed{seed}.json", raw)
            rows[f"direct_p_en_{direction}"] = _trace_summary(direct["tick_trace"])
            rows[f"raw_rate_{direction}"] = _trace_summary(raw["tick_trace"])
        embodied = embodied_food_toxin_case(seed, args.agent_steps, args.substeps)
        replay = replay_raw_gyro(seed, embodied)
        _write_json(output / f"embodied_food_toxin_seed{seed}.json", embodied)
        _write_json(output / f"raw_gyro_replay_seed{seed}.json", replay)
        rows["embodied_food_toxin"] = _embodied_summary(embodied)
        rows["raw_gyro_replay"] = _trace_summary(replay["tick_trace"])
        rows["embodied_vs_replay"] = _replay_difference(embodied, replay)
        summary[str(seed)] = rows
        print(
            f"seed={seed}: direct={rows['direct_p_en_CCW']['live_fraction']:.2f}/"
            f"{rows['direct_p_en_CW']['live_fraction']:.2f} raw={rows['raw_rate_CCW']['live_fraction']:.2f}/"
            f"{rows['raw_rate_CW']['live_fraction']:.2f} live={rows['embodied_food_toxin']['live_fraction']:.2f} "
            f"replay={rows['raw_gyro_replay']['live_fraction']:.2f}",
            flush=True,
        )
    _write_json(output / "summary.json", summary)
    print(f"Wrote compass transfer diagnostic to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
