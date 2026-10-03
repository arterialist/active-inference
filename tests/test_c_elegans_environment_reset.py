"""Reset restores constructor food without changing live food edits."""

import numpy as np

from simulations.c_elegans.environment import AgarPlateEnvironment


def test_reset_restores_consumed_initial_food_and_gradients():
    foods = [(0.002, 0.0, 0.0), (-0.002, 0.0, 0.0)]
    environment = AgarPlateEnvironment(food_positions=foods)
    original_concentration = environment.get_chemical_concentration("NaCl", np.zeros(3))
    environment.post_body_step({"head_position": np.array(foods[0])})
    assert environment.get_active_food_positions() == foods[1:]
    environment.add_food((0.004, 0.0, 0.0))
    observation = environment.reset()
    assert environment.get_active_food_positions() == foods
    assert environment.get_chemical_concentration("NaCl", np.zeros(3)) == original_concentration
    assert observation.chemicals["NaCl"] == original_concentration


def test_replacement_is_live_until_explicit_reset():
    environment = AgarPlateEnvironment(food_positions=[])
    environment.replace_food_sources([(0.002, 0.0, 0.0)])
    environment.step({"head_position": np.zeros(3)})
    environment.post_body_step({"head_position": np.zeros(3)})
    assert environment.get_active_food_positions() == [(0.002, 0.0, 0.0)]
    environment.reset()
    assert environment.get_active_food_positions() == []


def test_initial_food_positions_do_not_alias_caller_list():
    foods = [(0.002, 0.0, 0.0)]
    environment = AgarPlateEnvironment(food_positions=foods)
    foods[:] = [(0.009, 0.0, 0.0)]
    environment.replace_food_sources([])
    environment.reset()
    assert environment.get_active_food_positions() == [(0.002, 0.0, 0.0)]
