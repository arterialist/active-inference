"""Isolated and body-side checks for the V4 obstacle delta."""

from __future__ import annotations

import mujoco

from simulations.active_inference import aif_agent3d as ag
from simulations.active_inference.components.body.world import World3D
from simulations.active_inference.live.versions import get_version


def test_barrier_is_physical_and_default_world_is_unchanged():
    baseline = World3D(seed=11)
    challenge = World3D(n_food=1, n_tox=0, arena=7.0, seed=11, barrier="l_gap")
    assert not baseline.barriers
    assert baseline.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)  # V1–V3 unchanged
    assert len(challenge.barriers) == 2
    signal = challenge.obstacle_proximity()
    assert signal["left"] > signal["right"]
    assert signal["left"] > 0.0


def test_v4_fixture_is_full_width_and_has_no_random_food_target():
    world = World3D(n_food=9, n_tox=6, arena=7.0, seed=11, barrier="obstacle_detour")
    assert world.barrier_name == "head_on_wall"
    assert len(world.barriers) == 1
    assert world.foods == []
    assert world.toxins == []
    # The wall spans the arena laterally; callers cannot route around a gap
    # inside the challenge fixture.
    wall = world.barriers[0]
    assert wall["x"] == -2.4
    assert wall["hy"] > world.arena
    signal = world.obstacle_proximity()
    assert signal["distance_left"] < 2.35
    assert signal["distance_right"] < 2.35
    assert signal["left"] > 0.0 and signal["right"] > 0.0


def test_v4_compound_geometry_catalog_has_no_food_or_toxin_side_channel():
    for name, minimum_segments in (("corner", 2), ("chicane", 3), ("maze", 5)):
        world = World3D(n_food=9, n_tox=6, arena=7.0, seed=11, barrier=name)
        assert world.barrier_name == name
        assert len(world.barriers) == minimum_segments
        assert world.foods == [] and world.toxins == []
        assert all(item["hz"] > 0 for item in world.barriers)


def test_obstacle_route_is_paula_and_crossed_in_isolation():
    agent = ag.AIFAgent3D(seed=11, components=get_version("v4").components)
    agent.birth()
    for _ in range(80):
        agent.tick(0.0, 0.0, 0.0, hunger_fill=0.0, eat=0.0, vision=False,
                   obstacle={"left": 1.2, "right": 0.0})
    assert sum(agent.nb[n].O > 0 for n in ag.OBL) > 0
    assert sum(agent.nb[n].O > 0 for n in ag.OBDL) > 0
    assert agent.nb[ag.OBS_LEFT].O > 0 or agent.nb[ag.OBS_LEFT].S > 0
    assert sum(agent.nb[n].O > 0 for n in (ag.TL, ag.TR)) > 0
    # Left obstacle is crossed to the right-turn command; the opposite
    # presentation exercises the mirrored path.
    right = ag.AIFAgent3D(seed=11, components=get_version("v4").components)
    right.birth()
    for _ in range(80):
        right.tick(0.0, 0.0, 0.0, hunger_fill=0.0, eat=0.0, vision=False,
                   obstacle={"left": 0.0, "right": 1.2})
    assert right.nb[ag.OBS_RIGHT].O > 0 or right.nb[ag.OBS_RIGHT].S > 0
    assert sum(right.nb[n].O > 0 for n in (ag.TL, ag.TR)) > 0
