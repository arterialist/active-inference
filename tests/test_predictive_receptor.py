from copy import deepcopy
import math

import numpy as np
import pytest

from simulations.active_inference.experiments.composition_probe import encode, k, snapshot
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from simulations.active_inference.components.learning.predictive_bridge import append_predictive_bridge
from neuron.extensions.experimental.graded_eligibility import GradedEligibilityNeuron
from neuron.extensions.experimental.predictive_receptor import PredictiveReceptorNeuron, predictive_step


def fixture_config():
    neurons, points = [], []
    for nid in (1, 2, 3, 4):
        n = k.neuron(nid, lam=1, c=3, eta_post=1e-7, eta_retro=1e-7,
                     meta={'bounded_plasticity': True, 'graded_gain': 1.})
        n['params']['num_inputs'] = 2; neurons.append(n)
        points += [k.syn(nid, 0, 1., adapt=[0., 0.]), k.syn(nid, 1, 0., adapt=[0., 0.]), k.term(nid)]
    return dict(metadata={}, global_params={'num_inputs': 2, 'num_neuromodulators': 2},
                simulation_params={'max_history': 1}, neurons=neurons, synaptic_points=points,
                connections=[], external_inputs=[k.ext(i, 0) for i in range(1, 5)])


def load(tmp_path, config, cls=PredictiveReceptorNeuron):
    p = tmp_path / 'config.json'; p.write_text(encode(config))
    return fresh(p, 11, cls)[0]


def test_default_path_exact(tmp_path):
    cfg = fixture_config()
    a = load(tmp_path, cfg, GradedEligibilityNeuron)
    b = load(tmp_path, cfg)
    for t in range(64):
        for net in (a, b):
            net.set_external_input(1 + t % 4, 0, .2 + t % 7 / 10)
            net.run_tick()
        assert encode(snapshot(a)) == encode(snapshot(b))


def test_signed_amplitude_and_bounds():
    q, x = np.array([.2, .8]), np.array([.3, .1])
    np.testing.assert_allclose(predictive_step(q, x, .4, .01, 1), q + .004 * x)
    np.testing.assert_allclose(predictive_step(q, x, -.4, .01, 1), q - .004 * x)
    np.testing.assert_array_equal(predictive_step(q, x, 1e5, .1, 1), [1., 1.])
    np.testing.assert_array_equal(predictive_step(q, x, -1e5, .1, 1), [0., 0.])
    for eta in (0, -1, math.inf):
        with pytest.raises(ValueError): predictive_step(q, x, .4, eta, 1)


def test_bridge_preserves_old_graph_and_declares_matching(tmp_path):
    old = fixture_config(); copy = deepcopy(old)
    cfg, groups, edges, selected = append_predictive_bridge(old, [1, 2], [3, 4], fanin=2, consumers=2)
    assert old == copy
    for key in ('neurons', 'synaptic_points', 'connections'):
        assert cfg[key][:len(old[key])] == old[key]
    assert cfg['external_inputs'] == old['external_inputs']
    assert len(selected) == 4
    assert len(cfg['neurons']) == 12
    assert len({n['id'] for n in cfg['neurons']}) == 12
    net = load(tmp_path, cfg)
    assert all(n.params.eta_post > 0 and n.params.eta_retro > 0 for n in net.network.neurons.values())
    for channel, p in enumerate(groups['prediction']):
        assert [e[0] for e in edges if e[1] == p and e[3] == 'positive_teaching'] == [groups['error_positive'][channel]]


def test_error_latency_equation_and_no_direct_membrane_drive(tmp_path):
    cfg, groups, _, _ = append_predictive_bridge(fixture_config(), [1, 2], [3, 4], fanin=2, consumers=2)
    net = load(tmp_path, cfg); predictor = net.network.neurons[groups['prediction'][0]]
    q = np.array([p.u_i.info for sid, p in predictor.postsynaptic_points.items() if sid in predictor.prediction_ports])
    first_error = first_update = None
    previous_context = predictor.prediction_context.copy(); previous_error = 0.
    for t in range(80):
        net.set_external_input(1, 0, .2)
        if t < 40: net.set_external_input(3, 0, .8)
        net.run_tick()
        actual = np.array([predictor.postsynaptic_points[s].u_i.info for s in predictor.prediction_ports])
        d = math.exp(-1/8)
        x = d * previous_context + (1-d) * predictor.prediction_arrivals
        np.testing.assert_array_equal(x, predictor.prediction_context)
        assert predictor.prediction_error_used == previous_error
        expected = np.clip(q + predictor.prediction_eta * previous_error * x, 0, 1)
        np.testing.assert_array_equal(expected, actual)
        if first_error is None and predictor.prediction_error != 0: first_error = t
        if first_update is None and not np.array_equal(q, actual): first_update = t
        assert all(predictor.postsynaptic_points[s].potential == 0 for s, _ in predictor.prediction_error_ports)
        q = actual; previous_context = x; previous_error = predictor.prediction_error
    assert first_error is not None and first_error >= 4
    assert first_update == first_error + 1
    assert previous_error < 0  # Omitted target produces depression, no recall freeze.
    assert predictor.prediction_eta > predictor.params.eta_post


def test_rejects_host_teacher_and_port_overlap(tmp_path):
    cfg, groups, _, _ = append_predictive_bridge(fixture_config(), [1, 2], [3, 4], fanin=2, consumers=2)
    net = load(tmp_path, cfg); n = net.network.neurons[groups['prediction'][0]]
    with pytest.raises(ValueError, match='driven by neurons'):
        n.tick({n.prediction_error_ports[0][0]: {'info': .2}}, 0)
    bad = deepcopy(cfg)
    next(n for n in bad['neurons'] if n['id'] == groups['prediction'][0])['metadata']['prediction_error_ports'][0][0] = 0
    with pytest.raises(ValueError, match='separate positive'): load(tmp_path, bad)


def test_recorded_neural_comparison_and_learning_audit(tmp_path):
    from simulations.active_inference.experiments.predictive_bridge_probe import record, audit_record
    cfg, bridge, _, _ = append_predictive_bridge(fixture_config(), [1,2], [3,4], fanin=2, consumers=2)
    net = load(tmp_path,cfg)
    control = deepcopy(net)
    features = [dict(ticks=96,visual=np.full((96,2),.2),auditory=np.full((96,2),.8))]
    groups = dict(vision=[1,2],touch=[3,4])
    trial = dict(start=0,stop=96,visual_clip=0,audio_clip=0)
    data = record(net,features,groups,bridge,trial)
    from simulations.active_inference.experiments.multimodal_pairing_probe import inputs
    for t in range(96):
        for nid,value in inputs(features,groups,trial,t): control.set_external_input(nid,0,value)
        control.run_tick()
    assert encode(snapshot(net)) == encode(snapshot(control))
    for nid in bridge['prediction']:
        a,b=net.network.neurons[nid],control.network.neurons[nid]
        np.testing.assert_array_equal(a.prediction_context,b.prediction_context)
        assert a.prediction_error==b.prediction_error
    assert audit_record(data,cfg,bridge) < 2e-12
    for field in ('weights','arrivals','error_used','comparator_scheduled','eta'):
        altered = {k:v.copy() for k,v in data.items()}; altered[field][20] += .01
        with pytest.raises(ValueError): audit_record(altered,cfg,bridge)
    continuation = record(net,features,groups,bridge,dict(trial,start=96,stop=192))
    assert audit_record(continuation,cfg,bridge) < 2e-12


@pytest.mark.parametrize('seed',[11,23,44,77])
def test_learned_assignment_survives_activity_reset_with_ongoing_plasticity(tmp_path,seed):
    cfg, bridge, _, selected = append_predictive_bridge(fixture_config(), [1,2], [3,4],
                                                       fanin=2, consumers=2, seed=seed)
    # Heterogeneous birth weights are independent of the presented pairing.
    rng = np.random.default_rng(seed)
    chosen = {(n,s) for n,s,_ in selected}
    for p in cfg['synaptic_points']:
        if p['type']=='postsynaptic' and (p['neuron_id'],p['synapse_id']) in chosen:
            p['u_i']['info'] = float(rng.uniform(.35,.65))
    trajectories = {}
    for swap in (False,True):
        trained = load(tmp_path,cfg)
        for _ in range(6):
            for cue in (0,1):
                for t in range(160):
                    if t<120:
                        trained.set_external_input(1+cue,0,1.)
                        trained.set_external_input(3+(1-cue if swap else cue),0,.7)
                        trained.set_external_input(3+(cue if swap else 1-cue),0,.05)
                    trained.run_tick()
        # Transfer only selected incoming weights into an otherwise fresh
        # object graph. No membrane, trace, queue or terminal memory transfers.
        for cue in (0,1):
            probe = load(tmp_path,cfg)
            for nid,sid,_ in selected:
                probe.network.neurons[nid].postsynaptic_points[sid].u_i.info = trained.network.neurons[nid].postsynaptic_points[sid].u_i.info
            values=[]
            for t in range(96):
                probe.set_external_input(1+cue,0,1.); probe.run_tick()
                values.append([probe.network.neurons[n].O for n in bridge['prediction']])
            trajectories[swap,cue] = np.array(values)
            assert all(probe.network.neurons[n].prediction_eta>0 for n in bridge['prediction'])
    # An assignment-specific contribution is required throughout the late
    # trajectory, not only at the final tick or in a trial average.
    for cue in (0,1):
        delta = trajectories[False,cue]-trajectories[True,cue]
        assert np.min(delta[32:,cue]) > .1
        assert np.max(delta[32:,1-cue]) < -.1
