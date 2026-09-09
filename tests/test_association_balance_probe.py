from copy import deepcopy
import numpy as np
import pytest
from simulations.active_inference.experiments.association_balance_audit import IntegrationAudit
from simulations.active_inference.experiments.association_cue_audit import initial_snapshot
from simulations.active_inference.experiments.association_balance_probe import balanced_config,record_trial
from simulations.active_inference.experiments.eligibility_association_probe import protocol,run_trial,dynamic_snapshot
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from simulations.active_inference.experiments.composition_probe import encode
from neuron.extensions.experimental.eligibility_trace import EligibilityTraceNeuron


def test_balance_graph_is_label_blind_and_delay_matched():
    cfg,g=balanced_config(11)
    assert len(cfg['neurons'])==176
    for n in cfg['neurons']:
        points=[p for p in cfg['synaptic_points'] if p['type']=='postsynaptic' and p['neuron_id']==n['id']]
        assert n['params']['num_inputs']==len(points)
        assert len({p['synapse_id'] for p in points})==len(points)
        assert n['params']['eta_post']>0 and n['params']['eta_retro']>0
    # All-to-all, cue-independent pathways. No additional external control.
    assert len(cfg['connections'])==1280+2048
    assert {e['target_neuron'] for e in cfg['external_inputs']}==set(g['vision']+g['audio'])
    for nid in g['sensory_inhibition']:
        incoming=[c for c in cfg['connections'] if c['target_neuron']==nid]
        assert len(incoming)==32


def test_balance_impulse_observer_is_passive(tmp_path):
    cfg,g=balanced_config(11);path=tmp_path/'config.json';path.write_text(encode(cfg))
    net,*_=fresh(path,11,EligibilityTraceNeuron);other=deepcopy(net)
    masks,_=protocol(g,11,'paired',16);trial=dict(cue=0,sound=None,ticks=12)
    raw=record_trial(net,g,masks,trial);plain=run_trial(other,g,masks,trial)
    assert all(np.array_equal(raw[k],plain[k]) for k in plain)
    assert dynamic_snapshot(net)==dynamic_snapshot(other)
    assert raw['all_arrivals'].shape==(12,32,67)
    assert np.count_nonzero(raw['states'][2,144:176,1])==16
    assert np.count_nonzero(raw['all_arrivals'][3,:,35:])==16*32
    assert not raw['states'][:,96:128,1].any()
    observed=IntegrationAudit(cfg,initial_snapshot(cfg),g).check(raw)
    assert observed.shape==(12,32,4)
    assert (observed[3,:,3]<-1.27).all()
    assert (observed[3,:,0]<observed[3,:,1]).all()
    broken=deepcopy(raw);broken['states'][3,64,1]=1.
    with pytest.raises(ValueError,match='Somatic reconstruction'):
        IntegrationAudit(cfg,initial_snapshot(cfg),g).check(broken)
