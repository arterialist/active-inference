from copy import deepcopy

import numpy as np

from test_predictive_receptor import fixture_config, load
from simulations.active_inference.components.sensory.population_contrast import append_population_contrast
from simulations.active_inference.components.learning.predictive_bridge import append_predictive_bridge
from simulations.active_inference.experiments.contrast_prediction_probe import record_contrast
from simulations.active_inference.experiments.predictive_bridge_probe import audit_record
from simulations.active_inference.experiments.multimodal_pairing_probe import inputs
from simulations.active_inference.experiments.composition_probe import encode, snapshot


def test_composed_recording_is_passive_with_signals_in_flight(tmp_path):
    old=fixture_config()
    cfg,contrast,_=append_population_contrast(old,[1,2])
    context=contrast['contrast_above']+contrast['contrast_below']
    cfg,bridge,_,_=append_predictive_bridge(cfg,context,[3,4],fanin=4,consumers=2)
    assert cfg['external_inputs']==old['external_inputs']
    net=load(tmp_path,cfg); control=deepcopy(net)
    rng=np.random.default_rng(23)
    features=[dict(ticks=64,visual=rng.uniform(.1,.9,(64,2)),auditory=rng.uniform(.1,.9,(64,2)))]
    groups=dict(vision=[1,2],touch=[3,4])
    for start in (0,64):
        trial=dict(start=start,stop=start+64,visual_clip=0,audio_clip=0)
        data=record_contrast(net,features,groups,bridge,contrast,trial)
        assert audit_record(data,cfg,bridge)==0
        assert data['contrast_weights'].shape[0]==64
        for t in range(start,start+64):
            for nid,value in inputs(features,groups,trial,t):control.set_external_input(nid,0,value)
            control.run_tick()
        assert encode(snapshot(net))==encode(snapshot(control))
        expected=[p.u_i.info for ids in contrast.values() for n in ids
                  for p in control.network.neurons[n].postsynaptic_points.values()]
        np.testing.assert_array_equal(data['contrast_weights'][-1],expected)
        for nid in bridge['prediction']:
            a,b=net.network.neurons[nid],control.network.neurons[nid]
            np.testing.assert_array_equal(a.prediction_context,b.prediction_context)
            assert a.prediction_error==b.prediction_error
