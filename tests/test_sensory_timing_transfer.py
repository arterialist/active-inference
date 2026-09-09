from copy import deepcopy

import numpy as np
import pytest

from test_predictive_receptor import fixture_config,load
from simulations.active_inference.components.sensory.population_contrast import append_population_contrast
from simulations.active_inference.components.learning.predictive_bridge import append_predictive_bridge
from simulations.active_inference.experiments import sensory_timing_transfer as module
from simulations.active_inference.experiments.composition_probe import snapshot,encode


def test_current_physical_frame_no_future_sample_and_no_missing_audio():
    fs=[dict(ticks=8,visual=np.arange(16).reshape(8,2)/20,auditory=np.ones((8,2)))]
    groups=dict(vision=[1,2],touch=[3,4]);tr=dict(start=0,stop=8,visual_clip=0,audio_clip=None)
    assert module.timed_inputs(fs,groups,tr,1,'continuous')==[(1,.05),(2,.075)]
    assert module.timed_inputs(fs,groups,tr,1,'phase_shift_1')==[(2,.3)]
    assert module.timed_inputs(fs,groups,tr,1,'native')==module.native_inputs(fs,groups,tr,1)


@pytest.mark.parametrize('mode',['native','phase_shift_1','continuous'])
def test_adapter_matches_explicit_driver_and_restores_callback(tmp_path,mode):
    cfg,contrast,_=append_population_contrast(fixture_config(),[1,2])
    cfg,bridge,_,_=append_predictive_bridge(cfg,contrast['contrast_above']+contrast['contrast_below'],[3,4],fanin=4,consumers=2)
    net=load(tmp_path,cfg);control=deepcopy(net)
    fs=[dict(ticks=32,visual=np.full((32,2),.6),auditory=np.zeros((32,2)))]
    groups=dict(vision=[1,2],touch=[3,4]);tr=dict(start=0,stop=32,visual_clip=0,audio_clip=None)
    data=module.record_timing(net,fs,groups,bridge,contrast,tr,mode)
    assert module.recorder.inputs is module.native_inputs
    assert module.recorder.audit_record(data,cfg,bridge)==0
    for t in range(32):
        expected=np.zeros(4)
        for n,v in module.timed_inputs(fs,groups,tr,t,mode):
            control.set_external_input(n,0,v);expected[n-1]=v
        control.run_tick()
        np.testing.assert_array_equal(data['external_information'][t],expected)
    assert encode(snapshot(net))==encode(snapshot(control))


def test_adapter_restores_callback_on_failure(monkeypatch):
    def fail(*args):raise RuntimeError('Intentional recorder fault')
    monkeypatch.setattr(module,'record_contrast',fail)
    with pytest.raises(RuntimeError,match='Intentional'):
        module.record_timing(None,None,dict(vision=[1],touch=[2]),None,None,None,'continuous')
    assert module.recorder.inputs is module.native_inputs
