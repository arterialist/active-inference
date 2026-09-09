import numpy as np

from simulations.active_inference.components.sensory.population_contrast import append_population_contrast
from test_predictive_receptor import fixture_config,load


def test_matched_contrast_response_and_default_preservation(tmp_path):
    original=fixture_config()
    cfg,groups,_=append_population_contrast(original,[1,2,3,4])
    assert cfg['neurons'][:4]==original['neurons']
    assert cfg['external_inputs']==original['external_inputs']
    assert len(cfg['neurons'])==17
    net=load(tmp_path,cfg)
    values=np.array([.8,.2,.6,.4]);trace=[]
    for t in range(80):
        for nid,value in enumerate(values,1):net.set_external_input(nid,0,float(value))
        net.run_tick()
        trace.append([[net.network.neurons[n].O for n in groups[role]] for role in ('contrast_above','contrast_below')])
    trace=np.array(trace)
    expected=.999*.99**2*np.stack([np.maximum(0,values-values.mean()),np.maximum(0,values.mean()-values)])
    np.testing.assert_allclose(trace[-1],expected,atol=3e-5,rtol=0)
    assert np.all(trace[:5]==0)
    assert np.any(trace[5]>0)
    assert all(n.params.eta_post>0 and n.params.eta_retro>0 for n in net.network.neurons.values())


def test_uniform_stimulus_does_not_create_large_contrast(tmp_path):
    cfg,groups,_=append_population_contrast(fixture_config(),[1,2,3,4])
    net=load(tmp_path,cfg)
    for _ in range(80):
        for nid in (1,2,3,4):net.set_external_input(nid,0,.6)
        net.run_tick()
        assert max(net.network.neurons[n].O for role in ('contrast_above','contrast_below') for n in groups[role])<1e-4
