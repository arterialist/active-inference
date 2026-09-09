from copy import deepcopy
import numpy as np
import pytest
from simulations.active_inference.experiments.association_balance_probe import balanced_config,record_trial
from simulations.active_inference.experiments.association_continual_audit import inhibitory_flow
from simulations.active_inference.experiments.eligibility_association_probe import protocol
from simulations.active_inference.experiments.composition_probe import encode
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from neuron.extensions.experimental.port_modulation import PortModulationNeuron


@pytest.mark.parametrize('sensitivity',[.25,1.])
def test_native_inhibitory_ledger_and_corruption(tmp_path,sensitivity):
    cfg,g=balanced_config(11)
    for n in cfg['neurons']:
        if n['id'] in g['auditory']:
            n['metadata']['native_port_modulation']=[dict(port=s,sensitivity=sensitivity) for s in range(35,67)]
    path=tmp_path/'config.json';path.write_text(encode(cfg))
    net,*_=fresh(path,11,PortModulationNeuron);masks,_=protocol(g,11,'paired',16)
    raw=record_trial(net,g,masks,dict(cue=0,sound=0,ticks=64))
    q,last,error=inhibitory_flow(raw,np.full((32,32),-.08),np.full(32,-np.inf),0,cfg,g,sensitivity)
    actual=np.array([[net.network.neurons[n].postsynaptic_points[s].u_i.info for s in range(35,67)] for n in g['auditory']])
    np.testing.assert_allclose(q,actual,rtol=0,atol=2e-12)
    assert error<2e-12
    assert np.any(q!=-.08)
    broken=deepcopy(raw);broken['all_input_weights'][4,0,35]-=.001
    with pytest.raises(ValueError,match='Inhibitory local update'):
        inhibitory_flow(broken,np.full((32,32),-.08),np.full(32,-np.inf),0,cfg,g,sensitivity)
