import numpy as np
import pytest

from simulations.active_inference.experiments import context_organization as base
from simulations.active_inference.experiments.magnitude_feedback_probe import (
    configure, ReleaseObserver, stimulus, run)
from neuron.extensions.experimental.magnitude_retrograde import MagnitudeRetrogradeNeuron
from simulations.active_inference.experiments.magnitude_feedback_analysis import verify_returns


def test_observer_exact_and_all_return_sources_recorded(tmp_path):
    p=tmp_path/'config.json';p.write_text(base.encode(configure(True,width=8)))
    a=base.fresh(p,11,MagnitudeRetrogradeNeuron)[0]
    b=base.fresh(p,11,MagnitudeRetrogradeNeuron)[0]
    reference=[]
    for t in range(64):
        for n,v in zip(a.network.neurons.values(),stimulus(t,8)):
            a.set_external_input(n.id,0,float(v))
        a.run_tick();reference.append(base.cellular(list(a.network.neurons.values())))
    measured=[]
    with ReleaseObserver(b,1) as observer:
        for t in range(64):
            for n,v in zip(b.network.neurons.values(),stimulus(t,8)):
                b.set_external_input(n.id,0,float(v))
            b.run_tick();measured.append(base.cellular(list(b.network.neurons.values())))
    np.testing.assert_array_equal(reference,measured)
    data=observer.arrays()
    assert verify_returns(data)==64
    assert data['terminal_info'].shape==(64,9)
    np.testing.assert_array_equal(data['retrograde_offsets'][1:]-data['retrograde_offsets'][:-1],
                                  data['context_terminal'][:,2])
    assert set(data['retrograde_events'][:,1])==set(range(2,10))
    assert np.all(data['retrograde_events'][:,2]==1)
    # Exact numeric reconstruction from serialized per-event values, without
    # treating the ordered return sum as a single update.
    for t in range(64):
        cast={0:float,32:np.float32,64:np.float64}[data['context_before_dtype'][t]]
        u=cast(data['context_terminal'][t,0])
        for row in data['retrograde_events'][data['retrograde_offsets'][t]:data['retrograde_offsets'][t+1]]:
            u=np.clip(u+1e-7*np.float32(row[3]),-100.,100.)
        assert u==data['context_terminal'][t,1]
    broken={k:v.copy() for k,v in data.items()}
    broken['retrograde_events'][0,3]+=1.
    with pytest.raises(ValueError,match='recurrence differs'):verify_returns(broken)
    broken={k:v.copy() for k,v in data.items()}
    broken['retrograde_offsets'][-1]-=1
    with pytest.raises(ValueError,match='Invalid ordered return'):verify_returns(broken)


def test_chunked_smoke(tmp_path):
    result=run(tmp_path/'smoke',True,ticks=24,width=8)
    assert result['executed_ticks']==24
    with np.load(tmp_path/'smoke'/result['chunks'][0]['file']) as z:
        assert z['cells'].shape[0]==24
        assert z['weights'].shape==(24,9,2)
