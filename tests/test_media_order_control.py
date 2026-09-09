from collections import Counter
from copy import deepcopy
import numpy as np
import pytest

from simulations.active_inference.experiments.media_order_control import protocol
from simulations.active_inference.experiments.media_order_audit import crossed_effect, verify_protocol, reference_projections, temporal_summary


def test_order_cross_controls_each_sensory_sequence_as_a_multiset():
    histories = {(m,o):protocol(300,4,m,o) for m in ('paired','swapped') for o in (0,1)}
    for sense in ('visual_clip','audio_clip'):
        sequences = {m:Counter(tuple(t[sense] for t in histories[m,o]) for o in (0,1))
                     for m in ('paired','swapped')}
        assert sequences['paired']==sequences['swapped']
    for (mapping,order),trials in histories.items():
        assert trials[-1]['stop']==3168
        assert all(a['stop']==b['start'] for a,b in zip(trials,trials[1:]))
        exp=[t for t in trials if t['phase']=='experience']
        assert Counter(t['visual_clip'] for t in exp)=={0:4,1:4}
        assert Counter(t['audio_clip'] for t in exp)=={0:4,1:4}
        assert all((t['visual_clip']==t['audio_clip'])==(mapping=='paired') for t in exp)
        assert exp[0]['visual_clip']==order
        assert all(t['stop']-t['start']==96 for t in trials if t['phase']=='withdrawal')


@pytest.mark.parametrize('length,repeats,mapping,order',[(299,4,'paired',0),(300,0,'paired',0),(300,4,'other',0),(300,4,'paired',2)])
def test_invalid_protocol_rejected(length,repeats,mapping,order):
    with pytest.raises(ValueError):protocol(length,repeats,mapping,order)


def test_independent_protocol_check_rejects_order_and_duration_changes():
    manifests={(m,o):dict(mapping=m,order=o,repeats=4,trials=protocol(300,4,m,o))
               for m in ('paired','swapped') for o in (0,1)}
    verify_protocol(manifests)
    bad=deepcopy(manifests);bad['paired',0]['trials'][0]['audio_clip']=1
    with pytest.raises(ValueError,match='association'):verify_protocol(bad)
    bad=deepcopy(manifests);bad['swapped',1]['trials'][0]['stop']+=4
    with pytest.raises(ValueError,match='time sequence'):verify_protocol(bad)
    bad=deepcopy(manifests);del bad['swapped',1]
    with pytest.raises(ValueError,match='both assignments'):verify_protocol(bad)


def test_crossed_trace_separates_main_effect_from_order_interaction():
    signal=np.arange(18,dtype=float).reshape(6,3)
    nuisance=np.full((6,3),7.)
    # The two order-specific history offsets cancel without removing signal.
    main,interaction=crossed_effect(signal+nuisance,-signal+nuisance,
                                    signal-nuisance,-signal-nuisance)
    np.testing.assert_array_equal(main,signal)
    np.testing.assert_array_equal(interaction,np.zeros_like(signal))
    main,interaction=crossed_effect(signal,-signal,-signal,signal)
    np.testing.assert_array_equal(main,np.zeros_like(signal))
    np.testing.assert_array_equal(interaction,signal)
    with pytest.raises(ValueError):crossed_effect(signal,signal,signal,np.zeros((2,2)))


def test_common_mode_is_not_misreported_as_ensemble_identity():
    result=reference_projections(np.ones((4,3)),np.array([[2.,2.,2.],[1.,1.,1.]]))
    assert result['spatial'] is None
    np.testing.assert_array_equal(result['common'],np.ones(4))
    np.testing.assert_array_equal(result['raw'],np.ones(4))
    result=reference_projections(np.array([[1.,0.,-1.]]),np.array([[2.,1.,0.],[0.,1.,2.]]))
    assert result['common'] is None
    np.testing.assert_array_equal(result['spatial'],[.5])


def test_excluding_onset_cannot_hide_a_temporal_sign_reversal():
    s=temporal_summary(np.r_[np.full(32,-1.),np.full(268,.01)])
    assert s['post32_mean']>0 and s['full_mean']<0 and s['onset_mean']==-1.
    assert s['negative_ticks']==32 and s['positive_ticks']==268
    assert s['windows'][0]['value']==-1.
