from copy import deepcopy
import numpy as np
import pytest

from simulations.active_inference.experiments.media_drive_audit import ReceptorAudit,physical_values
from simulations.active_inference.experiments.population_hierarchy import make_config,cellular
from simulations.active_inference.experiments.composition_probe import encode
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from neuron.extensions.experimental.bounded_plasticity import BoundedPlasticityNeuron
from neuron.extensions.experimental.graded_eligibility import GradedEligibilityNeuron


@pytest.mark.parametrize('threshold,graded_gain',[(r,0.) for r in (.1,.2,.4,.6,.8,1.6,3.2)]+[(.6,.25)])
def test_receptor_audit_matches_native_dynamics_and_rejects_shifted_drive(tmp_path,threshold,graded_gain):
    cfg,groups,_ = make_config(1152,11)
    cfg['neurons'] = cfg['neurons'][:192]
    for n in cfg['neurons']:
        n['metadata']['bounded_plasticity'] = True
        n['metadata']['graded_gain'] = graded_gain
        n['params'].update(r_base=threshold,b_base=threshold+.25)
    cfg['connections'] = []
    cfg['synaptic_points'] = [p for p in cfg['synaptic_points'] if p['neuron_id']<=192]
    path = tmp_path/'receptors.json'; path.write_text(encode(cfg))
    net,_,neurons,_ = fresh(path,11,GradedEligibilityNeuron if graded_gain else BoundedPlasticityNeuron)
    values = np.random.default_rng(11).uniform(0,1,(32,192));rows=[]
    for t in range(32):
        for index in range(192):
            if t%4==(index+1)%4: net.set_external_input(index+1,0,float(2*values[t,index]))
        net.run_tick(); rows.append(cellular(neurons))
    cells = np.asarray(rows); audit = ReceptorAudit(cfg,groups,threshold,graded_gain)
    observed = audit.check(cells,values)
    assert np.array_equal(observed['receptor_spikes'],np.zeros((32,192),bool) if graded_gain else cells[:,:,1]>0)
    assert np.allclose(audit.q,[n.postsynaptic_points[0].u_i.info for n in neurons],atol=2e-12,rtol=0)
    with pytest.raises(ValueError,match='soma/output'):
        ReceptorAudit(cfg,groups,threshold,graded_gain).check(cells,np.roll(values,1,axis=0))
    bad = cells.copy(); bad[5,0,0] += .1
    with pytest.raises(ValueError,match='soma/output'): ReceptorAudit(cfg,groups,threshold,graded_gain).check(bad,values)


def test_media_source_assignment_preserves_physical_streams():
    features = [dict(visual=np.ones((8,96))*.2,auditory=np.ones((8,96))*.3),
                dict(visual=np.ones((8,96))*.4,auditory=np.ones((8,96))*.5)]
    trial = dict(start=99,stop=107,visual_clip=0,audio_clip=1)
    values = physical_values(features,trial)
    assert (values[:,:96]==.2).all() and (values[:,96:]==.5).all()
    assert not physical_values(features,{**trial,'visual_clip':None})[:,:96].any()
