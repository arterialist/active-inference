from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from test_hierarchical_body_perturbation import preparation
from test_predictive_receptor import load
from simulations.active_inference.components.body.energy_budget import EnergyBudget
from simulations.active_inference.components.arbitration.energy_feedback import append_energy_feedback, install_energy_feedback
from simulations.active_inference.core.runtime_checkpoint import check_buffer_aliases
from simulations.active_inference.experiments.metabolic_population_probe import BudgetRower, EnergyInputs, record_energy, audit_energy
from simulations.active_inference.experiments.sensory_motor_probe import record_correction, body_measures, audit_motor


def test_budget_conservation_including_failure_and_overflow():
    b = EnergyBudget(energy_j=1., gut_j=0.)
    initial = b.energy_j+b.gut_j
    b.advance(1., 10., 1.)
    assert b.energy_j == 0 and b.unmet_j > 0
    assert b.ingest(100.) == 48.
    for _ in range(100):
        b.advance(.25, 0., 0.)
        E, G, _, demand, unmet, spill, ingested = b.state()
        assert E+G+demand-unmet+spill == pytest.approx(initial+ingested)
    assert b.spill_j > 0
    assert b.demand_j > 40.  # Demand was not erased when the reserve hit zero.


def test_isometric_activation_still_costs_energy():
    a, b = EnergyBudget(gut_j=0.), EnergyBudget(gut_j=0.)
    a.advance(1., 0., 0.); b.advance(1., 0., 1.)
    assert a.energy_j-b.energy_j == pytest.approx(b.params.activation_w)
    with pytest.raises(ValueError):
        b.advance(0., 1., 0.)


@pytest.mark.parametrize('connected', [True, False])
def test_added_body_feedback_is_neural_and_observer_passive(tmp_path, connected):
    original, m = preparation()
    cfg, meta = append_energy_feedback(original, m['motor'], connected=connected)
    other_cfg, _ = append_energy_feedback(original, m['motor'], connected=not connected)
    for key in ('neurons', 'connections', 'external_inputs'):
        assert cfg[key] == other_cfg[key]
    net = load(tmp_path, original)
    oldcells = dict(net.network.neurons)
    fresh = load(tmp_path, cfg)
    install_energy_feedback(SimpleNamespace(network=net), fresh, original, cfg)
    check_buffer_aliases(net)
    assert all(net.network.neurons[n] is cell for n, cell in oldcells.items())
    body = BudgetRower(); body.organs.energy_j = 3.
    other_body = BudgetRower(); other_body.organs.energy_j = 3.
    plain = deepcopy(net)
    m = dict(m, energy_feedback=meta, energy_parameters=vars(body.organs.params),
             timestep=body.model.opt.timestep, fed=False)
    data = record_energy(net, body, m, cfg, 96, fed=False)
    other = record_correction(EnergyInputs(plain, other_body, meta, fed=False), other_body, m, cfg, 96)
    for k, value in other.items():
        np.testing.assert_array_equal(data[k], value, err_msg=k)
    body_measures(data)
    assert audit_energy(data, cfg, m) == 0
    assert audit_motor(data, cfg, m['motor']) == 0
    assert np.any(data['energy_cell_scheduled'][:, 2])
    assert bool(np.any(data['motor_scheduled'][:, :, -1])) == connected
    bad = {k: v.copy() for k, v in data.items()}
    bad['energy_after'][50, 0] += .01
    with pytest.raises(AssertionError):
        audit_energy(bad, cfg, m)
    assert all(n.params.eta_post > 0 and n.params.eta_retro > 0 for n in net.network.neurons.values())
