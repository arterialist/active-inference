#!/usr/bin/env python3
"""
Evolutionary optimization of neuromodulation and neuron params for food-seeking.

Robustness protocol: each genome is tested against 4 distinct food positions
to prevent directional overfitting ("lucky torpedo"). Fitness = average of
min_distance achieved per environment (or worst-case with --robust).
Uses min_distance over the run, not final tick, to avoid "final tick trap".

Usage:
  cd active-inference/
  uv run python scripts/evolve_food_seeking.py [--generations N] [--population M]
  uv run python scripts/evolve_food_seeking.py --low-memory   # Raspberry Pi / 8GB RAM

Checkpointing:
  Checkpoints include simulation-ready best config plus full DE state (population,
  energies, nfev, RNG) when available so runs resume exactly where they stopped.
  Use the checkpoint for simulation at any time:
    uv run python scripts/run_c_elegans.py --evol-config evolved_food_seeking_checkpoint.json --viewer

Resume (requires checkpoint de_state from this script):
    uv run python scripts/evolve_food_seeking.py --resume evolved_food_seeking_checkpoint.json
    uv run python scripts/evolve_food_seeking.py --resume ckpt.json --extra-generations 10
  Omit --extra-generations to run until evol_args.generations is reached. With --resume,
  ticks/population/seed/robust come from evol_args; --generations is not used (use --extra-generations).
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import signal
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Suppress simulation logging during evolution
import loguru

loguru.logger.remove()
loguru.logger.add(sys.stderr, level="ERROR")

N_TICKS = 50_000
EAT_RADIUS_M = 0.0001  # Early exit if worm gets this close (0.1 mm)

# MuJoCo gravity settle steps (full interactive default is 2000 in CElegansBody).
EVOLUTION_BODY_SETTLE_STEPS = 600


def _rss_mb() -> float:
    """Peak resident set size in MB (Unix). macOS: bytes, Linux: KB."""
    r = resource.getrusage(resource.RUSAGE_SELF)
    rss = r.ru_maxrss
    return (rss / 1024 / 1024) if sys.platform == "darwin" else (rss / 1024)


# 4 distinct targets to prevent directional overfitting ("lucky torpedo")
TEST_ENVIRONMENTS = [
    # (0.002, 0.002, 0.0),   # Top-right
    (-0.002, 0.002, 0.0),  # Top-left
    (0.002, -0.002, 0.0),  # Bottom-right
    # (0.0, 0.003, 0.0),     # Straight ahead
]


def x_to_config(x: np.ndarray) -> dict:
    """Map normalized [0,1]^d vector to evol_config."""
    # Bounds for each parameter
    # 0: K_STRESS_SYN [1000, 8000]
    # 1: K_REWARD_SYN [1000, 8000]
    # 2: K_VOL_STRESS [500, 4000]
    # 3: K_VOL_REWARD [500, 4000]
    # 4: STRESS_DEADZONE log [1e-6, 0.01]
    # 5: CHEM_EMA_ALPHA_SLOW [0.01, 0.10] — slow filter (environmental gradient)
    # 6: CHEM_EMA_ALPHA_FAST [0.05, 0.40] — fast filter (head-sweep tracking)
    # 7: TONIC_FWD_CMD [0.1, 0.5]
    # 8: TONIC_FWD_MOTOR [0.05, 0.2]
    # 9: motor w_tref M0 [15, 45]
    # 10: motor w_tref M1 [-25, -5]
    # 11: sensory w_tref M0 [8, 25]
    # 12: sensory w_tref M1 [-12, -3]
    cfg: dict = {
        "K_STRESS_SYN": 1000 + 7000 * x[0],
        "K_REWARD_SYN": 1000 + 7000 * x[1],
        "K_VOL_STRESS": 500 + 3500 * x[2],
        "K_VOL_REWARD": 500 + 3500 * x[3],
        "STRESS_DEADZONE": 1e-6 * (0.01 / 1e-6) ** x[4],
        "CHEM_EMA_ALPHA_SLOW": 0.01 + 0.09 * x[5],
        "CHEM_EMA_ALPHA_FAST": 0.05 + 0.35 * x[6],
        "TONIC_FWD_CMD": 0.1 + 0.4 * x[7],
        "TONIC_FWD_MOTOR": 0.05 + 0.15 * x[8],
    }
    cfg["neuron_params"] = {
        "motor": {"w_tref": [15 + 30 * x[9], -5 - 20 * x[10]]},
        "sensory": {"w_tref": [8 + 17 * x[11], -3 - 9 * x[12]]},
    }
    return cfg


def _resolve_project_path(relative_to_script_parent: Path, p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return relative_to_script_parent / path


_DE_STATE_VERSION = 1
_DE_STRATEGY = "best1bin"
_DE_MUTATION = (0.5, 1.0)
_DE_RECOMBINATION = 0.7
_DE_ATOL = 0.0
_DE_TOL = 0.0
_DE_MACHEPS = float(np.finfo(np.float64).eps)


def _rng_to_json(rng: object) -> dict:
    """Serialize numpy RNG used by SciPy DE (RandomState or Generator)."""
    if isinstance(rng, np.random.RandomState):
        st = rng.get_state()
        return {
            "kind": "RandomState",
            "state": [st[0], st[1].tolist(), int(st[2]), int(st[3]), float(st[4])],
        }
    if isinstance(rng, np.random.Generator):
        return {"kind": "Generator", "state": rng.bit_generator.state}
    raise TypeError(f"Unsupported RNG type: {type(rng)!r}")


def _rng_from_json(data: dict) -> np.random.RandomState | np.random.Generator:
    if data.get("kind") == "RandomState":
        st = data["state"]
        rs = np.random.RandomState()
        rs.set_state((st[0], np.asarray(st[1], dtype=np.uint32), st[2], st[3], st[4]))
        return rs
    if data.get("kind") == "Generator":
        g = np.random.Generator(np.random.PCG64())
        g.bit_generator.state = data["state"]
        return g
    raise ValueError(f"Unknown RNG kind in checkpoint: {data.get('kind')!r}")


def _de_state_from_solver(solver: object) -> dict:
    """Snapshot SciPy DifferentialEvolutionSolver for JSON (population in physical space)."""
    pop_phys = np.asarray(solver._scale_parameters(solver.population), dtype=float)
    return {
        "version": _DE_STATE_VERSION,
        "strategy": _DE_STRATEGY,
        "mutation": list(_DE_MUTATION),
        "recombination": float(_DE_RECOMBINATION),
        "population": pop_phys.tolist(),
        "population_energies": np.asarray(
            solver.population_energies, dtype=float
        ).tolist(),
        "nfev": int(solver._nfev),
        "rng": _rng_to_json(solver.random_number_generator),
    }


def _validate_de_state(state: dict, *, n_dim: int) -> None:
    if int(state.get("version", 0)) != _DE_STATE_VERSION:
        raise ValueError(f"Unsupported de_state version (got {state.get('version')!r})")
    if state.get("strategy") != _DE_STRATEGY:
        raise ValueError(
            f"Unsupported DE strategy in checkpoint: {state.get('strategy')!r}"
        )
    pop = np.asarray(state["population"], dtype=float)
    if pop.ndim != 2 or pop.shape[1] != n_dim:
        raise ValueError("de_state population has wrong shape")


def _apply_de_state_post_init(solver: object, state: dict) -> None:
    """After DifferentialEvolutionSolver(..., init=population_phys), restore energies/nfev/RNG."""
    energies = np.asarray(state["population_energies"], dtype=float)
    if energies.shape != (solver.num_population_members,):
        raise ValueError(
            f"population_energies length {energies.shape[0]} != "
            f"solver population {solver.num_population_members}"
        )
    solver.population_energies[:] = energies
    solver._nfev = int(state["nfev"])
    solver.random_number_generator = _rng_from_json(state["rng"])
    if (
        abs(float(state.get("recombination", _DE_RECOMBINATION)) - _DE_RECOMBINATION)
        > 1e-9
    ):
        raise ValueError("Checkpoint DE recombination does not match this script")
    mut = tuple(float(x) for x in state.get("mutation", list(_DE_MUTATION)))
    if mut != _DE_MUTATION:
        raise ValueError("Checkpoint DE mutation does not match this script")


def _de_solve_continue(
    solver: object,
    *,
    maxiter: int,
    gen_done: int,
    callback: object | None,
    disp: bool,
) -> object:
    """Run up to maxiter more DE generations (solver already has finite energies)."""
    from scipy.optimize._optimize import _status_message

    last_nit = 0
    warning_flag = False
    status_message = "in progress"
    for nit in range(1, maxiter + 1):
        try:
            next(solver)
        except StopIteration:
            warning_flag = True
            if solver._nfev > solver.maxfun:
                status_message = _status_message["maxfev"]
            elif solver._nfev == solver.maxfun:
                status_message = (
                    "Maximum number of function evaluations has been reached."
                )
            last_nit = nit
            break

        last_nit = nit
        if disp:
            print(
                f"differential_evolution step {nit}: f(x)= {solver.population_energies[0]}"
            )

        if callback:
            c = solver.tol / (solver.convergence + _DE_MACHEPS)
            res = solver._result(nit=gen_done + nit, message="in progress")
            res.convergence = c
            try:
                warning_flag = bool(callback(res))
            except StopIteration:
                warning_flag = True

            if warning_flag:
                status_message = "callback function requested stop early"

        if warning_flag or solver.converged():
            break
    else:
        status_message = _status_message["maxiter"]
        warning_flag = True

    return solver._result(
        nit=gen_done + last_nit, message=status_message, warning_flag=warning_flag
    )


def _load_resume_plan(
    resume_path: Path,
    *,
    extra_generations: int | None,
) -> dict:
    """Parse checkpoint and compute resume segment. Returns dict with keys:
    best_x, gen_done, n_evals, elapsed_sec, evol_args, maxiter, initial_best_dist, de_state.
    """
    n_dim = 13
    with open(resume_path) as f:
        ckpt = json.load(f)
    if "best_x" not in ckpt:
        raise ValueError(f"Resume file {resume_path} has no 'best_x'")
    best_x = np.clip(np.asarray(ckpt["best_x"], dtype=float), 0.0, 1.0)
    if best_x.shape != (n_dim,):
        raise ValueError(f"best_x must have length {n_dim}, got {best_x.shape}")
    ea = ckpt.get("evol_args")
    if not isinstance(ea, dict):
        raise ValueError(
            f"Resume file {resume_path} has no 'evol_args' object; "
            "only checkpoints written by this script can be resumed."
        )
    ea = dict(ea)
    for key in ("generations", "population", "ticks", "seed"):
        if key not in ea:
            raise ValueError(f"evol_args missing required key {key!r}")
    gen_done = int(ckpt.get("generation", 0))
    planned_total = int(ea["generations"])
    raw_de = ckpt.get("de_state")
    if not isinstance(raw_de, dict):
        raise ValueError(
            f"Checkpoint {resume_path} has no usable 'de_state' (full DE snapshot). "
            "Only checkpoints written by this version of evolve_food_seeking.py can be resumed."
        )
    if gen_done > planned_total:
        raise ValueError(
            f"Checkpoint generation {gen_done} exceeds evol_args.generations {planned_total}"
        )
    base_remaining = planned_total - gen_done
    if extra_generations is not None:
        if extra_generations <= 0:
            raise ValueError("--extra-generations must be positive")
        maxiter = base_remaining + extra_generations
        ea["generations"] = gen_done + maxiter
    else:
        maxiter = base_remaining
    if maxiter <= 0:
        raise ValueError(
            f"No generations left to run (completed {gen_done}/{planned_total}). "
            "Pass --extra-generations N to run N more generations beyond the original target."
        )
    n_evals = int(ckpt.get("n_evals", 0))
    elapsed_sec = float(ckpt.get("elapsed_min", 0.0)) * 60.0
    bd_mm = ckpt.get("best_dist_mm")
    initial_best_dist = float(bd_mm) / 1000.0 if bd_mm is not None else float("inf")
    return {
        "best_x": best_x,
        "gen_done": gen_done,
        "n_evals": n_evals,
        "elapsed_sec": elapsed_sec,
        "evol_args": ea,
        "maxiter": maxiter,
        "initial_best_dist": initial_best_dist,
        "de_state": raw_de,
    }


def _save_checkpoint(
    best_x: np.ndarray | None,
    best_dist: float,
    gen: int,
    n_evals: int,
    elapsed: float,
    out_path: Path,
    evol_args: dict | None = None,
    de_state: dict | None = None,
) -> None:
    """Atomically save best config + metadata + optional full DE state."""
    if best_x is None:
        return
    best_config = x_to_config(np.clip(best_x, 0, 1))
    payload: dict = {
        "config": best_config,
        "best_config": best_config,
        "best_x": best_x.tolist(),
        "best_dist_mm": float(best_dist * 1000),
        "generation": gen,
        "n_evals": n_evals,
        "elapsed_min": elapsed / 60,
    }
    if evol_args is not None:
        payload["evol_args"] = evol_args
    if de_state is not None:
        payload["de_state"] = de_state
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.rename(out_path)


def evaluate(
    x: np.ndarray,
    n_ticks: int = N_TICKS,
    eval_counter: list[int] | None = None,
    show_sim_progress: bool = False,
    use_worst: bool = False,
    low_memory: bool = False,
    body_settle_steps: int | None = EVOLUTION_BODY_SETTLE_STEPS,
) -> float:
    """
    Run simulation against 4 distinct food positions. Return average (or worst)
    of min_distance achieved in each run. DE minimizes this (lower = better).
    """
    from simulations import evol_trace
    from simulations.c_elegans.simulation import build_c_elegans_simulation

    if evol_trace.is_enabled():
        evol_trace.reset_accumulators()

    evol_config = x_to_config(np.clip(x, 0, 1))
    min_distances: list[float] = []

    for env_idx, food_pos in enumerate(TEST_ENVIRONMENTS):
        try:
            with evol_trace.span("eval_build_sim"):
                engine, loop = build_c_elegans_simulation(
                    use_connectome_cache=True,
                    food_positions=[food_pos],
                    log_level="ERROR",
                    record_neural_states=False,
                    evol_config=evol_config,
                    max_history=1 if low_memory else 200,
                    suppress_connectome_summary=low_memory,
                    body_settle_steps=body_settle_steps,
                )
        except Exception:
            return 1e6  # Penalize failed builds (large distance)

        with evol_trace.span("eval_reset"):
            loop.reset(nervous_rebuild=False)
        food = np.array(food_pos)
        min_dist = float("inf")

        tick_range = range(n_ticks - 1)
        if show_sim_progress:
            tick_range = tqdm(
                tick_range,
                unit="tick",
                desc=f"Sim env{env_idx+1}",
                position=1,
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        for _ in tick_range:
            step = engine.step()
            head = step.body_state.head_position
            d = float(np.linalg.norm(head - food))
            min_dist = min(min_dist, d)
            if min_dist < EAT_RADIUS_M:
                break  # Successfully reached food
        min_distances.append(min_dist)

        if low_memory:
            del engine, loop
            gc.collect()

    if eval_counter is not None:
        eval_counter[0] += 1

    if low_memory:
        gc.collect()

    if evol_trace.is_enabled():
        evol_trace.flush_json_line(
            {
                "n_envs": len(TEST_ENVIRONMENTS),
                "n_ticks_requested": n_ticks,
                "body_settle_steps": body_settle_steps,
            }
        )

    # Average across environments (or worst-case for robustness)
    return float(np.max(min_distances) if use_worst else np.mean(min_distances))


def _objective(
    x: np.ndarray,
    n_ticks: int,
    use_worst: bool,
    low_memory: bool,
    body_settle_steps: int | None,
) -> float:
    """
    Picklable objective for differential_evolution workers.
    Must be module-level for multiprocessing.
    """
    return evaluate(
        x,
        n_ticks=n_ticks,
        use_worst=use_worst,
        low_memory=low_memory,
        body_settle_steps=body_settle_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evolve neuromod params for food-seeking"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Fresh run only: total DE generations (default: 30). Incompatible with --resume.",
    )
    parser.add_argument(
        "--extra-generations",
        type=int,
        default=None,
        metavar="N",
        help="With --resume only: run N generations beyond the remaining budget "
        "(extends evol_args.generations).",
    )
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--ticks", type=int, default=N_TICKS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verbose", action="store_true", help="Scipy DE verbose output"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress logging"
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Score on worst environment (max min_dist) instead of average",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint JSON (must include de_state written by this script).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="evolved_food_seeking_checkpoint.json",
        help="Path for periodic checkpoints (default: evolved_food_seeking_checkpoint.json)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save checkpoint every N evals (default: 1 = every eval)",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Keep RAM flat for Raspberry Pi / 8GB: gc after each eval, min history",
    )
    parser.add_argument(
        "--measure-memory",
        action="store_true",
        help="Print peak RSS (MB) each generation",
    )
    parser.add_argument(
        "--full-settle",
        action="store_true",
        help="Use full MuJoCo gravity settle (2000 steps) like run_c_elegans; default is faster evolution settle",
    )
    args = parser.parse_args()

    if args.extra_generations is not None and not args.resume:
        print("error: --extra-generations requires --resume", file=sys.stderr)
        sys.exit(1)
    if args.resume and args.generations is not None:
        print(
            "error: do not pass --generations with --resume; "
            "use --extra-generations to extend the run",
            file=sys.stderr,
        )
        sys.exit(1)

    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_path = _resolve_project_path(repo_root, args.checkpoint)

    resume_plan: dict | None = None
    de_maxiter: int

    if args.resume:
        resume_path = _resolve_project_path(repo_root, args.resume)
        if not resume_path.is_file():
            print(f"error: --resume file not found: {resume_path}", file=sys.stderr)
            sys.exit(1)
        try:
            resume_plan = _load_resume_plan(
                resume_path, extra_generations=args.extra_generations
            )
        except (OSError, ValueError, json.JSONDecodeError) as e:
            print(f"error: cannot resume from {resume_path}: {e}", file=sys.stderr)
            sys.exit(1)
        evol_args = resume_plan["evol_args"]
        de_maxiter = resume_plan["maxiter"]
        loaded_x0 = resume_plan["best_x"]
        args.ticks = int(evol_args["ticks"])
        args.population = int(evol_args["population"])
        args.robust = bool(evol_args.get("robust", False))
        args.seed = int(evol_args["seed"])
        args.generations = int(evol_args["generations"])
        eval_counter: list[int] = [resume_plan["n_evals"]]
        gen_counter: list[int] = [resume_plan["gen_done"]]
        start_time = time.perf_counter() - resume_plan["elapsed_sec"]
        best_state: dict = {
            "best_dist": resume_plan["initial_best_dist"],
            "best_x": loaded_x0.copy(),
        }
    else:
        if args.generations is None:
            args.generations = 30
        de_maxiter = args.generations
        evol_args = {
            "generations": args.generations,
            "population": args.population,
            "ticks": args.ticks,
            "robust": args.robust,
            "seed": args.seed,
        }
        eval_counter = [0]
        gen_counter = [0]
        start_time = time.perf_counter()
        best_state = {"best_dist": np.inf, "best_x": None}

    np.random.seed(args.seed)
    n_dim = 13

    if not args.quiet:
        print("=" * 60)
        print("Evolution: food-seeking optimization (robustness protocol)")
        print("=" * 60)
        if args.resume:
            print(
                f"  RESUME from gen {gen_counter[0]}/{args.generations} "
                f"({de_maxiter} generation(s) this segment)"
            )
        print(
            f"  generations(target)={args.generations}  population={args.population}  ticks={args.ticks}"
        )
        print(
            f"  {len(TEST_ENVIRONMENTS)} targets: avg min_dist"
            + (" (worst)" if args.robust else "")
        )
        print(f"  checkpoint: {checkpoint_path} (every {args.checkpoint_every} eval)")
        if args.low_memory:
            print("  low-memory: enabled (flat RAM for Pi/8GB)")
        if args.full_settle:
            print("  body settle: full (2000 mj_step)")
        else:
            print(f"  body settle: evolution ({EVOLUTION_BODY_SETTLE_STEPS} mj_step)")
        print("-" * 60)

    # Approximate total evals: DE does ~popsize + generations*popsize, use 2x buffer
    total_evals = args.population * (de_maxiter + 1) * 2
    pbar = tqdm(
        total=total_evals,
        disable=args.quiet,
        unit="eval",
        desc="Evolution",
        position=0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    # Picklable objective for workers (module-level function + partial)
    _body_settle = None if args.full_settle else EVOLUTION_BODY_SETTLE_STEPS
    objective = partial(
        _objective,
        n_ticks=args.ticks,
        use_worst=args.robust,
        low_memory=args.low_memory,
        body_settle_steps=_body_settle,
    )

    # Mutable state for callback (runs in main process)
    _abort_requested = [False]  # Mutable for signal handler
    solver_holder: list[object | None] = [None]

    def _on_sigint(signum, frame):
        _abort_requested[0] = True
        if best_state["best_x"] is not None:
            elapsed = time.perf_counter() - start_time
            de_snap = None
            if solver_holder[0] is not None:
                try:
                    de_snap = _de_state_from_solver(solver_holder[0])
                except Exception:
                    de_snap = None
            _save_checkpoint(
                best_state["best_x"],
                best_state["best_dist"],
                gen_counter[0],
                eval_counter[0],
                elapsed,
                checkpoint_path,
                evol_args=evol_args,
                de_state=de_snap,
            )
            tqdm.write(f"\n[Ctrl+C] Checkpoint saved to {checkpoint_path}")
        sys.exit(130)

    signal.signal(signal.SIGINT, _on_sigint)

    def _progress_callback(intermediate_result):
        gen_counter[0] = int(intermediate_result.nit)
        dist = intermediate_result.fun
        if dist < best_state["best_dist"]:
            best_state["best_dist"] = dist
            best_state["best_x"] = np.array(intermediate_result.x)
        if getattr(intermediate_result, "nfev", None) is not None:
            eval_counter[0] = int(intermediate_result.nfev)
        de_snap = None
        if solver_holder[0] is not None:
            try:
                de_snap = _de_state_from_solver(solver_holder[0])
            except Exception:
                de_snap = None
        if best_state["best_x"] is not None:
            elapsed = time.perf_counter() - start_time
            _save_checkpoint(
                best_state["best_x"],
                best_state["best_dist"],
                gen_counter[0],
                eval_counter[0],
                elapsed,
                checkpoint_path,
                evol_args=evol_args,
                de_state=de_snap,
            )
        if not args.quiet:
            pbar.update(args.population)
            postfix = {"best_dist": f"{best_state['best_dist']*1000:.2f}mm"}
            if args.measure_memory:
                postfix["RSS"] = f"{_rss_mb():.0f}MB"
            pbar.set_postfix(postfix)
            convergence = intermediate_result.convergence
            elapsed = time.perf_counter() - start_time
            n_evals = eval_counter[0]
            best_d = best_state["best_dist"]
            evals_per_sec = n_evals / elapsed if elapsed > 0 else 0
            mem = f" | peak {_rss_mb():.0f}MB" if args.measure_memory else ""
            tqdm.write(
                f"  gen {gen_counter[0]:3d}/{args.generations} | "
                f"evals {n_evals:5d} | best {best_d*1000:.2f}mm | "
                f"conv {convergence:.3f} | {elapsed/60:.1f}m | {evals_per_sec:.1f} ev/s{mem}"
            )

    from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

    bounds = [(0.0, 1.0)] * n_dim

    def _open_solver(*, attach_callback: bool = True, **kw):
        return DifferentialEvolutionSolver(
            objective,
            bounds,
            args=(),
            strategy=_DE_STRATEGY,
            maxiter=de_maxiter,
            popsize=args.population,
            tol=_DE_TOL,
            mutation=_DE_MUTATION,
            recombination=_DE_RECOMBINATION,
            rng=args.seed,
            polish=False,
            callback=_progress_callback if attach_callback else None,
            disp=args.verbose,
            init=kw.get("init", "latinhypercube"),
            atol=_DE_ATOL,
            updating="deferred",
            workers=-1,
            constraints=(),
            x0=kw.get("x0", None),
        )

    if resume_plan is not None:
        st = resume_plan["de_state"]
        _validate_de_state(st, n_dim=n_dim)
        pop_init = np.asarray(st["population"], dtype=np.float64)
        with _open_solver(attach_callback=False, init=pop_init) as solver:
            solver_holder[0] = solver
            _apply_de_state_post_init(solver, st)
            result = _de_solve_continue(
                solver,
                maxiter=de_maxiter,
                gen_done=resume_plan["gen_done"],
                callback=_progress_callback,
                disp=args.verbose,
            )
    else:
        with _open_solver() as solver:
            solver_holder[0] = solver
            result = solver.solve()
    solver_holder[0] = None

    if not args.quiet:
        pbar.close()

    best_x = np.clip(result.x, 0, 1)
    best_config = x_to_config(best_x)
    best_distance = result.fun  # DE minimized distance
    total_time = time.perf_counter() - start_time

    print("\n" + "=" * 60)
    print("Evolution complete")
    print("=" * 60)
    print(f"Best avg min_dist: {best_distance*1000:.2f} mm")
    print(f"Total evals: {eval_counter[0]}, time: {total_time/60:.1f} min")
    if args.measure_memory:
        print(f"Peak RSS: {_rss_mb():.0f} MB")
    print("\nBest neuromod config:")
    for k, v in best_config.items():
        if k != "neuron_params":
            print(f"  {k}: {v}")
    print("\nBest neuron_params:")
    for k, v in best_config.get("neuron_params", {}).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Write best config to file for easy application
    out_path = Path(__file__).parent.parent / "evolved_food_seeking_config.json"
    with open(out_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\nConfig saved to {out_path}")


if __name__ == "__main__":
    main()
