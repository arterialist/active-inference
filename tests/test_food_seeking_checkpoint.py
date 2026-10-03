"""Optimizer continuation and safe generation-boundary checkpointing."""

from __future__ import annotations

import json
import signal
import sys
import numpy as np
import pytest
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

from scripts import evolve_food_seeking as evolution


def solver(objective, *, n_dim=4, rng=11, init="latinhypercube", callback=None):
    return DifferentialEvolutionSolver(
        objective, [(0.0, 1.0)] * n_dim,
        rng=rng, popsize=2, init=init, updating="deferred", workers=1,
        mutation=evolution._DE_MUTATION, recombination=evolution._DE_RECOMBINATION,
        strategy=evolution._DE_STRATEGY, tol=0.0, atol=0.0,
        polish=False, callback=callback,
    )


@pytest.mark.parametrize("use_generator", [False, True])
def test_json_checkpoint_preserves_exact_future_candidates(use_generator):
    calls = []

    def objective(x):
        calls.append(x.copy())
        return float(np.sum((x - 0.25) ** 2))

    rng = np.random.default_rng(11) if use_generator else 11
    with solver(objective, rng=rng) as original:
        next(original)
        next(original)
        state = json.loads(json.dumps(evolution._de_state_from_solver(original)))
        with solver(objective, init=state["population"], rng=99) as restored:
            evolution._apply_de_state_post_init(restored, state)
            assert np.array_equal(original.population, restored.population)
            for _ in range(3):
                calls.clear()
                next(original)
                uninterrupted_candidates = np.array(calls)
                calls.clear()
                next(restored)
                assert np.array_equal(uninterrupted_candidates, np.array(calls))
                assert evolution._de_state_from_solver(original) == evolution._de_state_from_solver(restored)


@pytest.mark.parametrize("bit_generator", [np.random.PCG64, np.random.MT19937, np.random.Philox])
def test_generator_kind_and_array_state_survive_json(bit_generator):
    original = np.random.Generator(bit_generator(19))
    original.normal(size=17)
    data = json.loads(json.dumps(evolution._rng_to_json(original)))
    restored = evolution._rng_from_json(data)
    assert type(restored.bit_generator) is type(original.bit_generator)
    assert np.array_equal(original.normal(size=100), restored.normal(size=100))


def test_older_optimizer_checkpoint_warns_about_nonexact_continuation():
    with solver(lambda x: float(np.sum(x ** 2))) as original:
        next(original)
        state = evolution._de_state_from_solver(original)
    del state["random_population_index"]
    del state["population_normalized"]
    with solver(lambda x: float(np.sum(x ** 2)), init=state["population"]) as restored:
        with pytest.warns(RuntimeWarning, match="may differ from the original run"):
            evolution._apply_de_state_post_init(restored, state)


def test_invalid_population_permutation_is_rejected():
    with solver(lambda x: float(np.sum(x ** 2))) as original:
        next(original)
        state = evolution._de_state_from_solver(original)
    state["random_population_index"] = [0] * len(state["population"])
    with pytest.raises(ValueError, match="population permutation"):
        evolution._validate_de_state(state, n_dim=4)


def prepare_main(monkeypatch, tmp_path, *, solver_type=DifferentialEvolutionSolver):
    script_path = tmp_path / "scripts" / "evolve_food_seeking.py"
    script_path.parent.mkdir(exist_ok=True)
    monkeypatch.setattr(evolution, "__file__", str(script_path))
    settle_choices = []

    def objective(x, n_ticks, use_worst, low_memory, body_settle_steps):
        settle_choices.append(body_settle_steps)
        return float(np.sum((x - 0.25) ** 2))

    def open_solver(*args, **kwargs):
        kwargs["workers"] = 1
        return solver_type(*args, **kwargs)

    monkeypatch.setattr(evolution, "_objective", objective)
    monkeypatch.setattr(evolution, "DifferentialEvolutionSolver", open_solver)
    handlers = {}
    monkeypatch.setattr(evolution.signal, "signal", lambda kind, handler: handlers.__setitem__(kind, handler))
    return settle_choices, handlers


def test_main_honors_eval_threshold_and_always_saves_final_generation(monkeypatch, tmp_path):
    prepare_main(monkeypatch, tmp_path)
    saves = []
    save = evolution._save_checkpoint

    def track_save(best_x, best_dist, gen, *args, **kwargs):
        saves.append(gen)
        return save(best_x, best_dist, gen, *args, **kwargs)

    monkeypatch.setattr(evolution, "_save_checkpoint", track_save)
    monkeypatch.setattr(sys, "argv", [
        "evolve", "--generations", "3", "--population", "1",
        "--checkpoint-every", "27", "--quiet",
    ])
    evolution.main()
    assert saves == [2, 3]
    checkpoint = json.loads((tmp_path / "evolved_food_seeking_checkpoint.json").read_text())
    assert checkpoint["generation"] == 3
    assert checkpoint["n_evals"] == 52


@pytest.mark.parametrize("full_settle", [False, True])
def test_resume_keeps_checkpoint_settling_protocol(monkeypatch, tmp_path, full_settle):
    choices, _ = prepare_main(monkeypatch, tmp_path)
    with solver(lambda x: float(np.sum((x - 0.25) ** 2)), n_dim=13) as original:
        next(original)
        evolution._save_checkpoint(
            original.x, float(original.population_energies[0]), 1, original._nfev,
            0.0, tmp_path / "resume.json",
            evol_args={"generations": 1, "population": 2, "ticks": 3,
                       "seed": 11, "robust": False, "full_settle": full_settle},
            de_state=evolution._de_state_from_solver(original),
        )
    monkeypatch.setattr(sys, "argv", [
        "evolve", "--resume", "resume.json", "--extra-generations", "1",
        "--full-settle", "--quiet",
    ])
    evolution.main()
    assert choices
    assert set(choices) == {None if full_settle else evolution.EVOLUTION_BODY_SETTLE_STEPS}
    checkpoint = json.loads((tmp_path / "evolved_food_seeking_checkpoint.json").read_text())
    assert checkpoint["evol_args"]["full_settle"] is full_settle
    assert checkpoint["generation"] == 2


def test_sigint_saves_completed_state_before_inflight_rng_changes(monkeypatch, tmp_path):
    expected = {}
    handlers = {}

    class InterruptedSolver(DifferentialEvolutionSolver):
        def __next__(self):
            if self._nfev >= 26:
                expected.update(evolution._de_state_from_solver(self))
                self.random_number_generator.uniform(size=10)
                self.random_number_generator.shuffle(self._random_population_index)
                self._nfev += 7
                handlers[signal.SIGINT](signal.SIGINT, None)
            return super().__next__()

    _, handlers = prepare_main(monkeypatch, tmp_path, solver_type=InterruptedSolver)
    monkeypatch.setattr(sys, "argv", [
        "evolve", "--generations", "3", "--population", "1",
        "--checkpoint-every", "1000", "--quiet",
    ])
    with pytest.raises(SystemExit) as interrupted:
        evolution.main()
    assert interrupted.value.code == 130
    checkpoint = json.loads((tmp_path / "evolved_food_seeking_checkpoint.json").read_text())
    assert checkpoint["generation"] == 1
    assert checkpoint["n_evals"] == 26
    assert checkpoint["de_state"] == expected
    assert not (tmp_path / "evolved_food_seeking_config.json").exists()
