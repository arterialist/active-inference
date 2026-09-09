from copy import deepcopy
import numpy as np
import pytest
from simulations.active_inference.experiments.association_balance_probe import balanced_config,record_trial
from simulations.active_inference.experiments.association_cue_probe import cue_cases
from simulations.active_inference.experiments.eligibility_association_probe import protocol,dynamic_snapshot
from simulations.active_inference.experiments.sensory_schedule import TIMINGS,receptor_schedule,ScheduledReceptors
from simulations.active_inference.experiments.composition_probe import encode
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from neuron.extensions.experimental.port_modulation import PortModulationNeuron
from simulations.active_inference.experiments.association_timing_audit import temporal_response


def test_schedules_preserve_receptor_dose_and_refractory_spacing():
    cfg,g=balanced_config(11);masks,_=protocol(g,11,'paired',16)
    for case in cue_cases(masks,11):
        for timing in TIMINGS:
            a=receptor_schedule(case,masks,11,timing)
            np.testing.assert_array_equal(a,receptor_schedule(case,masks,11,timing))
            expected=np.zeros(32);expected[[n-1 for n in case['selected_receptors']]]=4
            np.testing.assert_array_equal(a.sum(axis=0),expected)
            for n in case['selected_receptors']:assert np.diff(np.flatnonzero(a[:,n-1])).min()>=5


def test_synchronous_driver_matches_existing_full_trace(tmp_path):
    cfg,g=balanced_config(11);path=tmp_path/'config.json';path.write_text(encode(cfg))
    net,*_=fresh(path,11,PortModulationNeuron);control=deepcopy(net)
    masks,_=protocol(g,11,'paired',16);case=next(c for c in cue_cases(masks,11) if c['kind']=='clean')
    schedule=receptor_schedule(case,masks,11,'synchronous')
    driver=ScheduledReceptors(net,schedule)
    a=record_trial(driver,g,masks,dict(cue=None,sound=None,ticks=64))
    b=record_trial(control,g,masks,dict(cue=0,sound=None,ticks=64))
    assert all(np.array_equal(a[k],b[k]) for k in a)
    assert dynamic_snapshot(net)==dynamic_snapshot(control)
    with pytest.raises(ValueError,match='exhausted'):driver.run_tick()


def test_prefix_candidate_is_not_mislabeled_as_late_error():
    schedule=np.zeros((64,32),np.uint8);schedule[0,16:20]=1;schedule[3,:12]=1
    consumer=np.zeros((64,32),bool);consumer[5,16:]=True;consumer[8,:16]=True
    row=temporal_response(consumer,schedule,list(range(16)))[0]
    assert row['aligned_completion']==8
    assert row['prefix_other_spikes']==16 and row['late_status']=='expected_only'
    consumer[8,16:]=True
    assert temporal_response(consumer,schedule,list(range(16)))[0]['late_status']=='both'
