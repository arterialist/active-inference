import numpy as np
import pytest
from simulations.active_inference.experiments.media_weight_identity_audit import verify_local_current, first_difference, conditional_cue_effects


def test_current_uses_preupdate_weight_and_native_buffer_precision():
    arrivals=np.array([[.2,.3],[.1,0.]])
    initial=np.array([.7,.8]); weights=np.array([[.9,.6],[.8,.5]])
    expected=arrivals.astype(np.float32)*np.vstack([initial,weights[0]]).astype(np.float32)
    assert np.array_equal(verify_local_current(arrivals,weights,initial,expected),expected)
    with pytest.raises(ValueError):
        verify_local_current(arrivals,weights,initial,arrivals*weights)
    with pytest.raises(ValueError):
        verify_local_current(arrivals,weights,initial,np.ones((1,2)))


def test_latency_is_first_different_tick_not_first_nonzero_output():
    a=np.ones((20,3)); b=a.copy(); b[7,2]=2; b[11,0]=0
    d=first_difference(a,b)
    assert d['first']==7 and d['last']==11
    assert sum(d['different_values_by_tick'])==2


def test_smaller_assignment_contribution_need_not_reverse_total_preference():
    for sign in (-1,1):
        samples = {(name,cue): np.full((10,4),20. if cue==0 else 10.)
                   for name in ('intact','initial','cycle1','cycle2','cycle3') for cue in (0,1)}
        samples['intact',0] += sign
        assert (samples['intact',0]-samples['intact',1] > 0).all()
        effects = conditional_cue_effects(samples)
        assert (effects['selected_adaptation'] == sign).all()
        assert (effects['placement'] == sign).all()


def test_observer_records_native_potentials_without_changing_neural_future(tmp_path):
    import json
    from copy import deepcopy
    from simulations.active_inference.experiments.eligibility_association_probe import config, dynamic_snapshot
    from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
    from simulations.active_inference.experiments.media_weight_identity import PathObserver
    from simulations.active_inference.experiments.eligibility_media_probe import record
    from simulations.active_inference.experiments.population_state_branch import TickDriver
    from neuron.extensions.experimental.eligibility_trace import EligibilityTraceNeuron
    cfg, groups = config(); path = tmp_path/'config.json'; path.write_text(json.dumps(cfg))
    net, _, cells, syns = fresh(path, 11, EligibilityTraceNeuron)
    other = deepcopy(net)
    ports = [(n.id, sid, n.synapse_sources[sid][0]) for n in cells for sid in n.eligibility_ports]
    q = np.array([net.network.neurons[n].postsynaptic_points[sid].u_i.info for n,sid,_ in ports])
    f = [dict(ticks=20, visual=np.ones((20,32)), auditory=np.ones((20,32)))]
    g = dict(groups, touch=groups['audio'])
    trial = dict(start=0,stop=20,visual_clip=0,audio_clip=0)
    observer = PathObserver(cells,syns,net,ports)
    measured = record(net,TickDriver(net),cells,syns,f,g,trial,ports,observer)
    other_cells = list(other.network.neurons.values())
    other_syns = [s for n in other_cells for s in n.postsynaptic_points.values()]
    plain = record(other,TickDriver(other),other_cells,other_syns,f,g,trial,ports)
    assert all(np.array_equal(measured[k],plain[k]) for k in measured)
    assert dynamic_snapshot(net) == dynamic_snapshot(other)
    potentials = np.where(measured['arrivals']>0,np.asarray(observer.potentials),0.)
    assert potentials.any()
    verify_local_current(measured['arrivals'],measured['weights'],q,potentials)
