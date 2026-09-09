"""Embodied, compass-only test of the experimental PAULA cylinder compass.

This is deliberately a *separate* agent experiment.  It leaves the normal
demo/evolved organism unchanged.  The organism in this file uses the existing
PAULA CPG, motor neurons, muscles, and MuJoCo body, while the experimental
cylinder receives the actual signed MuJoCo yaw rate on every neural tick:

    PAULA CPG / turn port -> PAULA motor -> muscles -> MuJoCo body
        -> raw yaw sensor -> PAULA cylinder vestibular circuit

No decoded cylinder heading, body pose, turn angle, or host-side event is fed
back to a neural circuit.  The base agent's conventional central-complex
heading/PI/motor route is explicitly quarantined: its PI-to-motor weights are
zero, vision is disabled, and the turn probe enters the existing PAULA turn
neuron directly.  This is therefore a compass-under-real-movement experiment,
not a navigation or homing claim, and it does not yet give the cylinder a
motor projection or a visual re-anchoring pathway.

Two cylinder settings can run against the same live physical trajectory:
``baseline`` and ``tuned_seed11``.  The latter is retained only as a named
trace-calibrated hypothesis from the isolated raw-yaw study; this program
never tunes it from its own run.  Full tick rows include the input sample
before physics, PAULA cylinder activity, and the post-physics body state.  A
fresh PAULA cylinder then replays those exact input samples.  Equality of the
two complete circuit traces distinguishes a circuit issue from an accidental
physics/runner handoff discrepancy.

Run from ``active-inference/``:

    uv run python experiments/embodied_cylinder_compass.py \\
      --turn-neuron TR --output experiments/results/embodied_cylinder_tr.json

The result is intentionally large: it is a per-neural-tick audit artifact,
not a period summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from experiments.paula_cylinder_compass_isolated import (
    BIN_DEGREES,
    RAW_YAW_BIAS,
    RAW_YAW_TUNED_ON_SEED11,
    SEED_TICKS,
    BuildParameters,
    CylinderCompass,
    _signed_circular_delta,
)
from simulations.active_inference import aif_agent3d as ag


HERE = Path(__file__).resolve().parent
ACTIVE_INFERENCE = HERE.parent
REPOSITORY = ACTIVE_INFERENCE.parent
NEURON_MODEL = REPOSITORY / "neuron-model"

# Keep the physical course identical in intent to the existing long-turn
# diagnostic.  It contains a ring/CPG settling epoch, then a deliberately
# sustained PAULA TL/TR drive that must turn the real body through muscles.
BURN_IN = 200
TURN_START = 500
TURN_STOP = 900
STEPS = 175
SUBSTEPS = 8

# These disable every base-agent route by which its own conventional compass
# could alter the turn/motor behavior.  The legacy ring still exists inside
# the convenient body/CPG host, but has no projection to the experimental
# cylinder and no PI-to-motor influence in this course.  The cylinder itself
# is also read-only with respect to the motor in this first compass-only run.
QUARANTINED_BASE_BUILD = {
    "turn_probe_ports": True,
    "w_opp": 0.0,
    "w_cpu1": 0.0,
    "w_musf_mode": 0.0,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parameter_variant(name: str) -> BuildParameters:
    if name == "baseline":
        return BuildParameters()
    if name == "tuned_seed11":
        return RAW_YAW_TUNED_ON_SEED11
    if name == "tuned_seed11_sensor_delay1_ablation":
        # One extra causal PAULA sensory-relay stage. This is a timing
        # ablation, not a re-fit: the inherited parameters were selected on
        # the no-extra-relay training waveform, so it must not be presented
        # as a calibrated candidate if it happens to look better elsewhere.
        return replace(RAW_YAW_TUNED_ON_SEED11, raw_sensor_delay_stages=1)
    raise ValueError(f"unknown cylinder variant {name}")


def _seed_compass(compass: CylinderCompass) -> list[dict]:
    """Use the cylinder's declared PAULA birth ports before body interaction."""

    return [compass.tick(phase="cylinder_birth", seed=True) for _ in range(SEED_TICKS)]


def _cylinder_view(row: dict) -> dict:
    """Fields which must be identical under an exact raw-yaw replay.

    Physical pose and course labels deliberately do not occur here.  This
    makes a mismatch unambiguously a different PAULA cylinder state, rather
    than a difference in post-physics observation metadata.
    """

    return {
        key: row[key]
        for key in (
            "raw_yaw_rate",
            "state_spikes",
            "update_spikes",
            "carry_spikes",
            "raw_yaw_transducer",
            "decoded_digits_offline",
            "decoded_code_offline",
            "decoded_heading_degrees_offline",
            "decode_ambiguous",
        )
    }


def _last_decoded_code(rows: Iterable[dict]) -> int | None:
    for row in reversed(list(rows)):
        if row["decoded_code_offline"] is not None:
            return int(row["decoded_code_offline"])
    return None


def _turn_metrics(rows: list[dict]) -> dict:
    """Offline-only score of cylinder tracking over the declared turn epoch."""

    turn = [row for row in rows if row["turn_stimulus_active"]]
    if len(turn) < 2:
        raise RuntimeError("turn window did not produce enough full ticks")
    body_yaw = np.unwrap(np.asarray([row["physical_yaw_radians_after"] for row in turn], dtype=float))
    body_delta_degrees = float(np.degrees(body_yaw[-1] - body_yaw[0]))
    codes = [int(row["decoded_code_offline"]) for row in turn if row["decoded_code_offline"] is not None]
    cylinder_delta_bins = None if len(codes) < 2 else _signed_circular_delta(codes[0], codes[-1])
    expected_bins = int(np.rint(body_delta_degrees / BIN_DEGREES))
    coverage = float(np.mean([row["decoded_code_offline"] is not None for row in turn]))
    error_bins = None if cylinder_delta_bins is None else cylinder_delta_bins - expected_bins
    return {
        "turn_ticks": len(turn),
        "body_turn_degrees": body_delta_degrees,
        "expected_quantized_bins_from_body": expected_bins,
        "cylinder_turn_bins": cylinder_delta_bins,
        "cylinder_turn_degrees": (
            None if cylinder_delta_bins is None else float(cylinder_delta_bins * BIN_DEGREES)
        ),
        "absolute_quantized_error_bins": None if error_bins is None else abs(error_bins),
        "signed_quantized_error_bins": error_bins,
        "direction_matches_body": bool(
            cylinder_delta_bins is not None and body_delta_degrees * cylinder_delta_bins > 0.0
        ),
        "turn_decode_coverage": coverage,
        "level0_event_spikes": {
            "ccw": int(sum(row["raw_yaw_transducer"]["event_spikes"]["ccw_a"] for row in turn)),
            "cw": int(sum(row["raw_yaw_transducer"]["event_spikes"]["cw_a"] for row in turn)),
        },
    }


def _run_exact_replay(*, seed: int, parameters: BuildParameters, embodied_rows: list[dict]) -> dict:
    """Re-run only the PAULA cylinder on the recorded pre-physics gyro stream."""

    cylinder_seed = 100_000 + seed
    np.random.seed(cylinder_seed)
    replay = CylinderCompass(parameters, raw_yaw_transduction=True)
    birth = _seed_compass(replay)
    replay_rows: list[dict] = []
    mismatch_ticks: list[int] = []
    mismatch_fields: dict[str, int] = {}
    for embodied in embodied_rows:
        replay_row = replay.tick(
            phase="exact_raw_yaw_replay",
            raw_yaw_rate=float(embodied["raw_yaw_rate_pre_tick"]),
        )
        replay_rows.append(replay_row)
        expected = _cylinder_view(embodied["cylinder"])
        observed = _cylinder_view(replay_row)
        if expected != observed:
            mismatch_ticks.append(int(embodied["neural_tick_input"]))
            for key in expected:
                if expected[key] != observed[key]:
                    mismatch_fields[key] = mismatch_fields.get(key, 0) + 1
    return {
        "scope": {
            "body_reexecuted": False,
            "input": "the exact pre-physics raw yaw samples captured in this embodied run",
            "purpose": "diagnose circuit determinism versus physics/runner handoff",
        },
        "birth_tick_trace": birth,
        "tick_trace": replay_rows,
        "metrics": {
            "source_body_ticks": len(embodied_rows),
            "replay_ticks": len(replay_rows),
            "cylinder_state_mismatch_ticks": len(mismatch_ticks),
            "first_mismatch_neural_ticks": mismatch_ticks[:20],
            "mismatching_fields": mismatch_fields,
            "exact_replay": not mismatch_ticks,
        },
    }


def _base_quarantine_audit(agent) -> dict:
    """Make the inherited base brain's non-role explicit and inspectable."""

    return {
        "base_agent_used_only_for": ["PAULA CPG", "PAULA motor", "muscles", "MuJoCo body", "raw yaw sensor"],
        "conventional_heading_ring_present_in_host": True,
        "conventional_heading_ring_to_cylinder_connection": False,
        "cylinder_to_motor_connection": False,
        "vision_called": False,
        "pi_to_motor_weights": {
            # These are exploratory structural build keywords, rather than
            # fields of EmbodiedAgentConfig.  Record their declared values
            # directly so the quarantine audit cannot accidentally imply a
            # non-existent shared-config field.
            "w_opp": float(QUARANTINED_BASE_BUILD["w_opp"]),
            "w_cpu1": float(QUARANTINED_BASE_BUILD["w_cpu1"]),
            "w_musf_mode": float(QUARANTINED_BASE_BUILD["w_musf_mode"]),
        },
        "base_configured_with": QUARANTINED_BASE_BUILD,
        "mechanistic_boundary": (
            "The sustained turn is current to the existing PAULA TL/TR probe port. "
            "It drives the base PAULA CPG/motor/body. The cylinder only receives the physical "
            "gyro sample through its own two opponent PAULA receptors; all cylinder decoding is offline."
        ),
    }


def run_embodied_cylinder(*, seed: int, variants: tuple[str, ...], turn_name: str) -> dict:
    """Run independent cylinders on live MuJoCo yaw without changing default code."""

    if not variants:
        raise ValueError("at least one cylinder variant is required")
    if len(set(variants)) != len(variants):
        raise ValueError("each variant may appear only once")
    turn_neuron = ag.TL if turn_name == "TL" else ag.TR

    # Base and experimental networks have independent seeded construction.
    # Fixing the cylinder seed independently prevents unrelated host-neuron
    # construction order from becoming a hidden experimental input.
    np.random.seed(seed)
    agent = ag.AIFAgent3D(seed=seed, **QUARANTINED_BASE_BUILD)
    agent.world = ag.w3.World3D(seed=seed, n_food=0, n_tox=0, arena=8.0)
    agent.img = agent.world.retina()
    agent.birth()
    agent._kicked = True

    cylinders: dict[str, CylinderCompass] = {}
    births: dict[str, list[dict]] = {}
    parameters: dict[str, BuildParameters] = {}
    for name in variants:
        parameters[name] = _parameter_variant(name)
        np.random.seed(100_000 + seed)
        cylinders[name] = CylinderCompass(parameters[name], raw_yaw_transduction=True)
        births[name] = _seed_compass(cylinders[name])

    # One row per physical neural tick, one nested row per cylinder. The
    # ``stimulus`` hook is deliberately before the host core tick / physics;
    # ``capture`` then appends post-physics state to the same row.
    trace: list[dict] = []
    pending: dict[int, dict] = {}

    def stimulus(current) -> None:
        tick = int(current.t)
        raw_yaw = float(current.world.yaw_rate())
        if abs(raw_yaw) > RAW_YAW_BIAS:
            raise RuntimeError(
                f"live MuJoCo yaw {raw_yaw:.6f} exceeds cylinder receptor bias {RAW_YAW_BIAS}; "
                "do not clip a physical sample -- revise the declared receptor range"
            )
        turn_active = TURN_START <= tick < TURN_STOP
        row = {
            "neural_tick_input": tick,
            "raw_yaw_rate_pre_tick": raw_yaw,
            "turn_stimulus_active": turn_active,
            "turn_neuron": turn_name,
            "physical_pose_before": {
                "x": float(current.world.pose()[0]),
                "y": float(current.world.pose()[1]),
                "yaw": float(current.world.pose()[2]),
            },
            "cylinder": {},
        }
        for name, cylinder in cylinders.items():
            row["cylinder"][name] = cylinder.tick(
                phase="embodied_raw_yaw_prephysics", raw_yaw_rate=raw_yaw
            )
        pending[tick] = row

        if tick == BURN_IN:
            current.net.set_external_input(ag.CPGP[0], 0, 5.0)
        if turn_active:
            current.net.set_external_input(turn_neuron, ag.TURN_PROBE_SYN[turn_neuron], 3.0)

    def capture(current) -> None:
        input_tick = int(current.t) - 1
        row = pending.pop(input_tick)
        x, y, yaw = current.world.pose()
        row.update({
            "neural_tick_after_physics": int(current.t),
            "physical_pose_after": {"x": float(x), "y": float(y), "yaw": float(yaw)},
            "physical_yaw_radians_after": float(yaw),
            "physical_yaw_rate_after": float(current.world.yaw_rate()),
            "physical_speed_after": float(current.world.speed()),
            "base_turn_probe_spike": int(current.nb[turn_neuron].O > 0.0),
            "base_motor_membranes": {
                "left_propulsion": float(current.nb[ag.MLp].S),
                "left_retraction": float(current.nb[ag.MLr].S),
                "right_propulsion": float(current.nb[ag.MRp].S),
                "right_retraction": float(current.nb[ag.MRr].S),
            },
            "base_heading_ring_spikes_observer_only": [
                int(current.nb[nid].O > 0.0) for nid in ag.cc.RING
            ],
        })
        trace.append(row)

    ag.run_episode(
        agent,
        steps=STEPS,
        sub=SUBSTEPS,
        vision=False,
        render_every=10**9,
        render_ticks=0,
        log_every=10**9,
        tick_hook=capture,
        neural_input_hook=stimulus,
    )
    if pending:
        raise RuntimeError(f"missing post-physics capture for ticks {sorted(pending)}")

    result_variants: dict[str, dict] = {}
    for name in variants:
        # Flatten the nested trace for the generic cylinder score, retaining
        # the same body labels for all variants.  These offline fields never
        # enter any neural ports.
        rows = [{**body, **body["cylinder"][name]} for body in trace]
        metrics = _turn_metrics(rows)
        replay = _run_exact_replay(seed=seed, parameters=parameters[name], embodied_rows=trace_for_variant(trace, name))
        # This is intentionally a modest engineering criterion for a
        # compass-only test. It is not a criterion that a biological animal
        # must track every degree, and it does not claim navigation.
        metrics["compass_only_candidate_pass"] = bool(
            replay["metrics"]["exact_replay"]
            and metrics["turn_decode_coverage"] >= 0.99
            and metrics["direction_matches_body"]
            and metrics["absolute_quantized_error_bins"] is not None
            and metrics["absolute_quantized_error_bins"] <= 4
        )
        metrics["criterion"] = {
            "purpose": "detect a usable tracking foundation, not organism-level perfection",
            "required": {
                "exact_raw_replay": True,
                "turn_decode_coverage_at_least": 0.99,
                "direction_match": True,
                "absolute_quantized_error_bins_at_most": 4,
            },
            "not_established_by_pass": [
                "visual anchoring", "motor control by the cylinder", "path integration", "homing", "biological fidelity",
            ],
        }
        result_variants[name] = {
            "parameters": asdict(parameters[name]),
            "calibration_status": (
                "untuned baseline" if name == "baseline" else
                "one-extra-PAULA-sensory-relay timing ablation; inherited no-relay seed-11 settings, not recalibrated"
                if name == "tuned_seed11_sensor_delay1_ablation" else
                "fixed before this run; previously selected on a named seed-11 raw-yaw replay. "
                "This embodied execution did not adjust it and is not an independent seed validation."
            ),
            "birth_tick_trace": births[name],
            "metrics": metrics,
            "exact_raw_yaw_replay": replay,
        }

    return {
        "experiment": "embodied_paula_cylinder_compass_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "course": {
            "cpg_birth_tick": BURN_IN,
            "turn_start": TURN_START,
            "turn_stop": TURN_STOP,
            "turn_neuron": turn_name,
            "steps": STEPS,
            "substeps": SUBSTEPS,
            "neural_ticks": STEPS * SUBSTEPS,
        },
        "scope": {
            "separate_opt_in_experiment": True,
            "default_demo_or_agent_modified": False,
            "body_is_real_mujoco": True,
            "body_motion_is_paula_cpg_and_muscle_driven": True,
            "raw_yaw_is_sampled_from_live_body_before_each_paula_tick": True,
            "host_heading_decode_used_for_control": False,
            "cylinder_motor_projection": False,
            "visual_anchoring": False,
            "claim": "compass-only sensorimotor waveform tolerance and exact raw replay audit",
            "not_a_claim": "navigation, homing, or a faithful biological cylinder topology",
        },
        "architecture": {
            "cylinder": {
                "levels": 4,
                "base_per_level": 4,
                "total_bins": 256,
                "bin_degrees": BIN_DEGREES,
                "transducer": (
                    "two biased opponent graded PAULA receptor currents -> signed graded accumulators -> "
                    "ordinary refractory PAULA event doublets -> state-gated quaternary carry counter"
                ),
                "neuron_count_per_variant": len(next(iter(cylinders.values())).nb),
            },
            "body_loop": "PAULA TL/TR port -> PAULA CPG/motor -> muscle actuation -> MuJoCo -> yaw receptor",
            "base_quarantine_audit": _base_quarantine_audit(agent),
        },
        "variants": result_variants,
        "tick_trace": trace,
    }


def trace_for_variant(trace: list[dict], name: str) -> list[dict]:
    """Adapt nested embodied rows for the exact replay helper without copying large state."""

    return [
        {
            "neural_tick_input": row["neural_tick_input"],
            "raw_yaw_rate_pre_tick": row["raw_yaw_rate_pre_tick"],
            "cylinder": row["cylinder"][name],
        }
        for row in trace
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--variants", nargs="+", choices=("baseline", "tuned_seed11", "tuned_seed11_sensor_delay1_ablation"),
                        default=("baseline", "tuned_seed11"))
    parser.add_argument("--turn-neuron", choices=("TL", "TR"), default="TR")
    parser.add_argument("--output", type=Path,
                        help="write a non-overwriting full-tick JSON audit artifact")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record = run_embodied_cylinder(
        seed=args.seed, variants=tuple(args.variants), turn_name=args.turn_neuron
    )
    for name, variant in record["variants"].items():
        metrics = variant["metrics"]
        print(
            f"{name}: body={metrics['body_turn_degrees']:+.2f}deg "
            f"expected={metrics['expected_quantized_bins_from_body']:+d} bins "
            f"cylinder={metrics['cylinder_turn_bins']} bins "
            f"error={metrics['signed_quantized_error_bins']} bins "
            f"direction={metrics['direction_matches_body']} "
            f"exact_replay={variant['exact_raw_yaw_replay']['metrics']['exact_replay']} "
            f"candidate_pass={metrics['compass_only_candidate_pass']}",
            flush=True,
        )
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.output, record)
        print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
