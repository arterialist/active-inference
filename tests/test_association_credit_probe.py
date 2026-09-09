from copy import deepcopy

import numpy as np
import pytest

from simulations.active_inference.experiments.composition_probe import encode, snapshot
from simulations.active_inference.experiments.population_hierarchy import make_config, cellular
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from simulations.active_inference.experiments.association_credit_probe import CreditRecorder, selected_ports
from neuron.extensions.experimental.bounded_plasticity import BoundedPlasticityNeuron, magnitude_step
from simulations.active_inference.experiments.association_credit_audit import verify_events, expected_magnitude


def test_credit_observer_preserves_replay_and_records_effective_update(tmp_path):
    config, groups, edges = make_config(288)
    for n in config["neurons"]:
        n["metadata"]["bounded_plasticity"] = True
    path = tmp_path/"config.json"
    path.write_text(encode(config))
    net, _, _, _ = fresh(path, 11, BoundedPlasticityNeuron)
    other = deepcopy(net)
    ports = selected_ports({"groups": groups, "edges": edges})
    recorder = CreditRecorder(net, ports, 48)

    def drive(branch):
        states = []
        for t in range(48):
            for nid in groups["vision"]+groups["touch"]:
                if t % 4 == nid % 4:
                    branch.set_external_input(nid, 0, 2.)
            branch.run_tick()
            states.append(cellular(list(branch.network.neurons.values())))
        return np.stack(states)

    reference = drive(other)
    method = BoundedPlasticityNeuron.tick
    with recorder.observe():
        observed = drive(net)
    assert np.array_equal(reference, observed)
    assert encode(snapshot(net)) == encode(snapshot(other))
    assert BoundedPlasticityNeuron.tick is method
    events = recorder.events()
    assert len(events) > 0 and np.any(events["before"] != events["after"])
    assert (events["input"][:, 0] > 0).all() and (events["eta"] > 0).all()
    for e in events:
        expected = magnitude_step(abs(e["before"]), e["error"], int(e["direction"]), e["eta"])
        assert abs(e["after"]) == expected
    # Reconstruct the initial vector from config, not the now-trained clone.
    points = {(p["neuron_id"], p["synapse_id"]): p for p in config["synaptic_points"] if p["type"] == "postsynaptic"}
    starting = np.array([points[n, s]["u_i"]["info"] for n, s, *_ in ports])
    eta = np.array([net.network.neurons[n].params.eta_post for n, *_ in ports])
    caps, decay = np.full(len(ports), 10.), np.full(len(ports), .02)
    trial = {"start": 0, "stop": 48}
    final, residual, _ = verify_events(events, ports, trial, observed, np.full(288, -1), starting, eta, caps, decay)
    assert residual < 5e-12
    assert np.array_equal(final, [net.network.neurons[n].postsynaptic_points[s].u_i.info for n, s, *_ in ports])
    for field, change in (("direction", -1), ("after", .125)):
        corrupt = events.copy()
        if field == "direction":
            corrupt[field][0] *= change
        else:
            corrupt[field][0] += change
        with pytest.raises(ValueError):
            verify_events(corrupt, ports, trial, observed, np.full(288, -1), starting, eta, caps, decay)
    with pytest.raises(RuntimeError):
        with recorder.observe():
            raise RuntimeError("cleanup")
    assert BoundedPlasticityNeuron.tick is method


def test_independent_flow_includes_zero_growth_and_depression():
    q = np.array([.3, .9, 2., 0.])
    e = np.array([.02, .2, 1., .5])
    d = np.array([1, -1, 1, 1])
    eta = np.full(4, .002)
    expected = np.array([magnitude_step(a, b, int(c), z) for a, b, c, z in zip(q, e, d, eta)])
    assert np.allclose(expected_magnitude(q, e, d, eta, np.full(4, 10.), np.full(4, .02)), expected, rtol=1e-14)


def test_minimum_timing_window_cannot_assign_negative_direction_at_max_rate():
    """A constructive bound check, not a claim that all weights must grow.

    t_ref>=2*c whereas at maximal sustained firing the post-spike age is at
    most c-1. Thus all active-input phases receive positive timing direction.
    The separate magnitude decay can still decrease a weight.
    """
    from neuron.neuron import NeuronParameters
    params = NeuronParameters(num_inputs=3, c=3, lambda_param=1., r_base=.6, b_base=.9,
                              eta_post=1e-5, eta_retro=1e-7, gamma=np.ones(2),
                              w_r=np.zeros(2), w_b=np.zeros(2), w_tref=np.array([-100., 0.]))
    n = BoundedPlasticityNeuron(1, params, log_level="CRITICAL", metadata={"bounded_plasticity": True})
    for sid in (0, 1, 2):
        n.add_synapse(sid, 1)
        n.postsynaptic_points[sid].u_i.info = 1. if sid == 0 else .3
        n.postsynaptic_points[sid].u_i.plast = 0.
        n.postsynaptic_points[sid].u_i.adapt[:] = 0.
    # Clamp a local modulator to put the timing window at its smallest allowed
    # value. This is a unit-test condition, not injected reward in the agent.
    n.M_vector[0] = 1.
    spikes, phases = [], set()
    for t in range(40):
        n.input_buffer[0, 0] = 10.
        n.input_buffer[1, 0] = 1.
        n.input_buffer[2, 0] = 1.
        before = n.postsynaptic_points[1].u_i.info
        n.tick({}, t)
        assert n.t_ref == 2*params.c
        if n.O > 0:
            spikes.append(t)
        if np.isfinite(n.t_last_fire):
            age = int(t-n.t_last_fire)
            phases.add(age)
            assert age <= n.t_ref
            assert n.postsynaptic_points[1].u_i.info > before
    assert np.array_equal(np.diff(spikes), np.full(len(spikes)-1, 3))
    assert phases == {0, 1, 2}
