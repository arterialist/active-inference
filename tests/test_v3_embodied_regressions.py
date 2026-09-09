"""Small, fast guards for the strict V3 failure fixes.

The longer acceptance evidence lives in
``experiments/embodied_v3_trajectory_causal.py``.  These tests only protect
the wiring/runtime contracts so a future refactor cannot silently restore the
old un-driven EXPLORE or zeroed hunger ports.
"""

from __future__ import annotations

from simulations.active_inference import aif_agent3d as ag
from simulations.active_inference.agents.interoceptive_v3 import InteroceptiveV3Agent
from simulations.active_inference.decode import Decoder


def test_strict_v3_uses_its_actual_mode_order_and_body_driven_hunger():
    agent = InteroceptiveV3Agent(seed=11)
    assert agent._mode_names == ["FORAGE", "EXPLORE", "SLEEP"]
    agent.birth()
    hunger = 0
    explore = 0
    for _ in range(80):
        agent.tick(0.0, 0.0, 1.0, vision=False)
        hunger += sum(int(agent.nb[n].O > 0) for n in ag.ar.HUNGER)
        explore += sum(int(agent.nb[n].O > 0) for n in ag.ar.MODE[2])
    assert hunger > 0
    assert explore > 0


def test_metabolic_hunger_transducer_drains_from_gut_load():
    agent = InteroceptiveV3Agent(seed=11)
    agent.world.gut_load = 0.82
    agent.world.energy_store = 0.40
    state = agent.drive_metabolic_afferents()
    fill, drain = agent.metabolic_hunger_inputs(state)
    assert fill > 0.0
    assert drain > fill


def test_standalone_decoder_preserves_strict_v3_mode_labels():
    agent = InteroceptiveV3Agent(seed=11)
    ids = sorted(agent.net.network.neurons)
    decoded = Decoder(ag).bind(agent, ids).read([0.0] * len(ids))
    modes = {row["k"].strip() for row in decoded if row["s"] == "drive"}
    assert {"forage", "explore", "sleep"}.issubset(modes)
    assert "home" not in modes
