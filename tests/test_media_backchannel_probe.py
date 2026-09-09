from types import SimpleNamespace
import json

import pytest

from simulations.active_inference.experiments.media_backchannel_probe import selected_feedback
from neuron.neuron import Neuron, RetrogradeSignalEvent
from neuron.extensions.experimental.graded_eligibility import GradedEligibilityNeuron
import numpy as np
from simulations.active_inference.experiments.media_order_audit import compare_backchannel
from simulations.active_inference.experiments.association_route_probe import digest


def test_feedback_lesion_preserves_forward_and_unselected_events(monkeypatch):
    selected=RetrogradeSignalEvent(1,2,3,900,np.array([.1,0.,0.,0.]),0)
    other=RetrogradeSignalEvent(1,4,3,900,np.array([.2,0.,0.,0.]),0)
    forward=(1,900,.7)
    source=SimpleNamespace(id=1)
    target=SimpleNamespace(id=3,presynaptic_points={900:SimpleNamespace(u_o=SimpleNamespace(info=1.))})
    net=SimpleNamespace(current_tick=1,network=SimpleNamespace(neurons={1:source,3:target}))
    def original_tick(n,e,t,dt=1.):return [forward,selected,other]
    def original_retro(n,e):n.presynaptic_points[e.target_terminal_id].u_o.info+=.001
    monkeypatch.setattr(GradedEligibilityNeuron,'tick',original_tick)
    monkeypatch.setattr(Neuron,'process_retrograde_signal',original_retro)
    with selected_feedback(net,[(1,2,3)],True) as (delivered,removed):
        assert GradedEligibilityNeuron.tick(source,{},0)==[forward,other]
        assert len(removed)==1 and not delivered
    assert GradedEligibilityNeuron.tick is original_tick
    assert Neuron.process_retrograde_signal is original_retro
    with pytest.raises(RuntimeError):
        with selected_feedback(net,[(1,2,3)],False) as (delivered,removed):
            assert GradedEligibilityNeuron.tick(source,{},0)==[forward,selected,other]
            Neuron.process_retrograde_signal(target,selected)
            assert delivered[0][5:7]==[1.,1.001] and not removed
            raise RuntimeError('observer failure')
    assert GradedEligibilityNeuron.tick is original_tick
    assert Neuron.process_retrograde_signal is original_retro


def test_backchannel_audit_keeps_path_specific_timing(tmp_path):
    root=tmp_path/'recording';root.mkdir()
    manifest=dict(groups={'visual_core':[1],'tactile_core':[2],'upper_core':[3]},
                  selected_ports=[[2,0,1]],terminal_order=[[1,900],[2,900],[3,900]])
    (root/'manifest.json').write_text(json.dumps(manifest))
    branches=[]
    for reset in (False,True):
        for cut in (False,True):
            cells=np.zeros((4,3,8));term=np.zeros((4,3,3));arrivals=np.zeros((4,1))
            if reset:
                cells[:,1,0]=.1;first=3 if cut else 1;term[first:,0,0]=.01;arrivals[first:,0]=.01
            path=root/f'{reset}-{cut}.npz'
            np.savez_compressed(path,cells=cells,terminals=term,arrivals=arrivals,
                retro_events=np.empty((0,11)) if cut else np.array([[1,2,0,1,900,1.,1.001,.1,0,0,0]]),
                removed_retro=np.array([[0,2,0,1,900,.1,0,0,0]]) if cut else np.empty((0,9)))
            branches.append(dict(reset=reset,cut=cut,file=path.name,sha256=digest(path)))
    (root/'summary.json').write_text(json.dumps(dict(branches=branches,exact_acquisition_replay=True,
                                      intact_prefixes_exact=True,parent_unchanged=True)))
    result=compare_backchannel(root,tmp_path/'audit')['comparisons']
    assert result['intact/visual_core/terminal_info']['first_tick']==1
    assert result['cut/visual_core/terminal_info']['first_tick']==3
    assert result['intact/tactile_core/S']['first_tick']==result['cut/tactile_core/S']['first_tick']==0
