from copy import deepcopy

import numpy as np
import pytest

from test_predictive_receptor import fixture_config,load
from simulations.active_inference.components.learning.predictive_bridge import append_predictive_bridge
from simulations.active_inference.experiments.predictive_weight_transplant import arrange_weights,install_weights,recording_contrast_groups
from simulations.active_inference.experiments.composition_probe import encode,snapshot


def test_shuffle_preserves_each_predictors_weight_distribution():
    weights=np.arange(32,dtype=float).reshape(4,8)/32
    original=weights.copy();shuffled=arrange_weights(weights,shuffled=True,seed=23)
    np.testing.assert_array_equal(weights,original)
    np.testing.assert_array_equal(np.sort(weights,axis=1),np.sort(shuffled,axis=1))
    assert not np.array_equal(weights,shuffled)
    np.testing.assert_array_equal(shuffled,arrange_weights(weights,shuffled=True,seed=23))


def test_json_key_sorting_does_not_change_recorded_weight_layout():
    import json
    groups=dict(contrast_pool=[5],contrast_relay=[6,7],contrast_above=[8,9],contrast_below=[10,11])
    restored=json.loads(encode(groups))
    assert list(restored)!=list(groups)
    assert list(recording_contrast_groups(restored))==list(groups)
    assert recording_contrast_groups(restored)==groups


def test_only_selected_weights_are_mutated_and_validation_is_atomic(tmp_path):
    cfg,bridge,_,_=append_predictive_bridge(fixture_config(),[1,2],[3,4],fanin=2,consumers=2)
    net=load(tmp_path,cfg)
    for _ in range(13):
        net.set_external_input(1,0,.6);net.set_external_input(3,0,.8);net.run_tick()
    before=deepcopy(net);original=np.array([[net.network.neurons[n].postsynaptic_points[s].u_i.info
        for s in net.network.neurons[n].prediction_ports] for n in bridge['prediction']])
    install_weights(net,bridge,np.array([[.1,.2],[.3,.4]]))
    for n in bridge['prediction']:
        np.testing.assert_array_equal(net.network.neurons[n].prediction_context,before.network.neurons[n].prediction_context)
        assert net.network.neurons[n].prediction_error==before.network.neurons[n].prediction_error
    install_weights(net,bridge,original)
    assert encode(snapshot(net))==encode(snapshot(before))
    with pytest.raises(ValueError):install_weights(net,bridge,np.array([[.1,.2],[.3,1.1]]))
    assert encode(snapshot(net))==encode(snapshot(before))
