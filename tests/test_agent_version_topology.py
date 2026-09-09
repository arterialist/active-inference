"""The selectable agent versions are real PAULA topology profiles, not UI masks."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _agent_module():
    spec = importlib.util.spec_from_file_location(
        "aif_agent3d_version_topology_test",
        ROOT / "simulations/active_inference/aif_agent3d.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMPONENTS = {
    "v1": ("sensory.olfactory_valence", "motor.cpg_muscle"),
    "v2": ("sensory.olfactory_valence", "motor.cpg_muscle", "learning.mushroom_body"),
    "v3": (
        "sensory.olfactory_valence",
        "motor.cpg_muscle",
        "learning.mushroom_body",
        "arbitration.foraging_exploration",
        "body.metabolic_organs",
        "arbitration.metabolic_sleep",
    ),
    "v4": (
        "sensory.olfactory_valence",
        "motor.cpg_muscle",
        "learning.mushroom_body",
        "arbitration.foraging_exploration",
        "body.metabolic_organs",
        "arbitration.metabolic_sleep",
        "body.obstacle_geometry",
        "sensory.obstacle_proximity",
        "motor.obstacle_reflex",
    ),
}


def test_versions_only_emit_declared_populations():
    ag = _agent_module()
    built = {name: ag.AIFAgent3D(seed=1, components=components) for name, components in COMPONENTS.items()}

    assert [len(built[name].nb) for name in ("v1", "v2", "v3", "v4")] == [95, 299, 345, 373]
    assert not (set(ag.cc.RING) & set(built["v1"].nb))
    assert not (set(ag.vc.VIS_IDS()) & set(built["v1"].nb))
    assert not (set(ag.ar.HUNGER) & set(built["v2"].nb))
    assert not (set(ag.ar.MODE[0]) & set(built["v2"].nb))
    assert set(ag.ar.HUNGER) <= set(built["v3"].nb)
    assert set(ag.ar.SLEEP_MODE) <= set(built["v3"].nb)
    assert not (set(ag.ar.MODE[1]) & set(built["v3"].nb))
    assert set(ag.OBL + ag.OBR + ag.OBDL + ag.OBDR) <= set(built["v4"].nb)
    assert set((ag.OBS_LEFT, ag.OBS_RIGHT, ag.OBS_BRAKE, ag.OBS_WALL)) <= set(built["v4"].nb)
    assert not (set(ag.OBL + ag.OBR) & set(built["v3"].nb))


def test_strict_versions_run_without_omitted_transducers():
    ag = _agent_module()
    for components in COMPONENTS.values():
        agent = ag.AIFAgent3D(seed=1, components=components)
        agent.birth(ticks=3)
        ag.run_episode(agent, steps=2, sub=2, vision=False, render_ticks=0, log_every=10**9)
        assert agent.t == 4
