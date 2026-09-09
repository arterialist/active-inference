"""Regression coverage for the opt-in graded PI-memory transducer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_agent_module():
    spec = importlib.util.spec_from_file_location(
        "aif_agent3d_pistone_test", ROOT / "simulations/active_inference/aif_agent3d.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pistone_opt_in_builds_and_drives_memory_from_direct_agent_ticks():
    ag = load_agent_module()
    agent = ag.AIFAgent3D(seed=11, pistone=True, graded=True)

    assert agent._pistone is True
    assert set(ag.pstone.MEM).issubset(agent.nb)

    speeds = []
    original_drive = ag.pstone.drive

    def observe(net, speed):
        speeds.append((net, speed))
        return original_drive(net, speed)

    ag.pstone.drive = observe
    try:
        agent.tick(ccw=0.0, cw=0.0, speed=0.73, vision=False)
    finally:
        ag.pstone.drive = original_drive

    assert speeds == [(agent.net, 0.73)]


def test_pistone_remains_off_in_the_default_agent():
    ag = load_agent_module()
    agent = ag.AIFAgent3D(seed=11)

    assert agent._pistone is False
    assert not set(ag.pstone.MEM).intersection(agent.nb)


def test_stone_output_route_reaches_home_mode_and_heading_gated_turn_neurons():
    ag = load_agent_module()
    np.random.seed(11)
    agent = ag.AIFAgent3D(seed=11, pistone=True, pistone_opp=True, graded=True)
    agent.birth()

    opponent_spikes = home_spikes = home_turn_spikes = 0
    for tick in range(180):
        agent.tick(ccw=0.0, cw=0.0, speed=1.0, hunger_fill=0.0, eat=0.0, vision=False)
        if tick >= 100:
            opponent_spikes += sum(int(agent.nb[nid].O > 0) for nid in ag.nv.OPP)
            home_spikes += sum(int(agent.nb[nid].O > 0) for nid in ag.ar.MODE[1])
            home_turn_spikes += int(agent.nb[ag.HTL].O > 0) + int(agent.nb[ag.HTR].O > 0)

    assert sum(float(agent.nb[nid].O) for nid in ag.pstone.MEM) > 0.0
    assert opponent_spikes > 0
    assert home_spikes > 0
    assert home_turn_spikes > 0
