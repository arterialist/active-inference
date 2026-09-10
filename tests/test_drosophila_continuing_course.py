"""Continuation must save branch-local RNG, not ambient RNG."""
import json
from types import SimpleNamespace

from simulations.drosophila.memory_feedback import continuing_course


def test_checkpoint_uses_advancing_branch_and_restores_original_tools(monkeypatch, tmp_path):
    branch = SimpleNamespace(network=object(), python_rng="before")
    saved = []
    def load(*args, **kwargs):
        return branch
    def save(value, path, **kwargs):
        saved.append((value, value.python_rng))
    def run(receiver, output, **kwargs):
        restored = continuing_course.second_order.load_checkpoint(receiver, trusted=True)
        restored.python_rng = "advanced"
        continuing_course.second_order.save_checkpoint(restored.network, output/"retention.paula")
        return {"probes": {}}
    monkeypatch.setattr(continuing_course.second_order, "load_checkpoint", load)
    monkeypatch.setattr(continuing_course.second_order, "save_checkpoint", save)
    monkeypatch.setattr(continuing_course.second_order, "run", run)
    continuing_course.run(tmp_path/"input", tmp_path)
    assert saved == [(branch, "advanced")]
    assert continuing_course.second_order.load_checkpoint is load
    assert continuing_course.second_order.save_checkpoint is save
    assert json.loads((tmp_path/"summary.json").read_text())["branch_rng_preserved"]
