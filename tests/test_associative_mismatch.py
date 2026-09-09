from copy import deepcopy

import numpy as np
import pytest

from test_predictive_receptor import fixture_config, load
from simulations.active_inference.components.sensory.population_contrast import append_population_contrast
from simulations.active_inference.components.learning.predictive_bridge import append_predictive_bridge
from simulations.active_inference.experiments.associative_mismatch_probe import audiovisual_trials
from simulations.active_inference.experiments.sensory_timing_transfer import record_timing
from simulations.active_inference.experiments.predictive_bridge_probe import audit_record
from simulations.active_inference.experiments.composition_probe import snapshot, encode
from simulations.active_inference.experiments.associative_mismatch_audit import expected_inputs, familiarity_benefit, contextual_weight_mask


def test_trials_cross_physical_events_without_familiarity_flag():
    trials = audiovisual_trials()
    assert {(t['visual_clip'], t['audio_clip']) for t in trials} == {(0,0),(0,1),(1,0),(1,1)}
    assert all(set(t) == {'start','stop','visual_clip','audio_clip'} for t in trials)
    assert all(t['start'] == 0 and t['stop'] == 300 for t in trials)


@pytest.mark.parametrize('visual,audio', [(0,0),(0,1),(1,0),(1,1)])
def test_audiovisual_recording_is_passive_and_preserves_active_learning(tmp_path, visual, audio):
    cfg, contrast, _ = append_population_contrast(fixture_config(), [1,2])
    cfg, bridge, _, _ = append_predictive_bridge(cfg, contrast['contrast_above']+contrast['contrast_below'], [3,4], fanin=4, consumers=2)
    net = load(tmp_path,cfg); control = deepcopy(net)
    rng = np.random.default_rng(44)
    features = [dict(ticks=32, visual=rng.uniform(.1,.9,(32,2)), auditory=rng.uniform(.1,.9,(32,2))) for _ in (0,1)]
    groups = dict(vision=[1,2],touch=[3,4])
    trial = dict(start=0,stop=32,visual_clip=visual,audio_clip=audio)
    data = record_timing(net,features,groups,bridge,contrast,trial,'native')
    for t in range(32):
        expected = np.zeros(4)
        for ids, values in ((groups['vision'],features[visual]['visual'][t]),
                            (groups['touch'],features[audio]['auditory'][t])):
            for n,v in zip(ids, values):
                if t%4 == n%4:
                    expected[n-1] = 2*v
                    control.set_external_input(n,0,2*v)
        control.run_tick()
        np.testing.assert_array_equal(data['external_information'][t],expected)
    assert encode(snapshot(net)) == encode(snapshot(control))
    assert audit_record(data,cfg,bridge) == 0
    assert (data['eta'] > 0).all()
    assert not np.array_equal(data['weights'][-1],data['start_weights'])
    np.testing.assert_array_equal(data['external_information'],expected_inputs(features,groups,trial))
    mask = contextual_weight_mask(cfg,bridge)
    np.testing.assert_array_equal(data['start_incoming_info'][mask],data['start_weights'].ravel())


def test_familiarity_is_history_relative_for_the_same_input():
    x = np.arange(12).reshape(3,4)
    np.testing.assert_array_equal(familiarity_benefit(x,2*x,0,0),x)
    np.testing.assert_array_equal(familiarity_benefit(x,2*x,1,1),x)
    np.testing.assert_array_equal(familiarity_benefit(x,2*x,0,1),-x)
    np.testing.assert_array_equal(familiarity_benefit(x,2*x,1,0),-x)
