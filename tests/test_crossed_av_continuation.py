import numpy as np
import random

from simulations.active_inference.experiments import context_organization as base
from simulations.active_inference.experiments import crossed_av_world as world
from simulations.active_inference.experiments.crossed_av_continuation import restore_acquired,expression
from simulations.active_inference.experiments.opponent_context import AfferentDelay
from simulations.active_inference.experiments.temporal_verification import configure,verify_learning


def test_continuation_and_diagnostic_branches_preserve_acquisition(tmp_path):
    cfg,g,s=configure(width=8);p=tmp_path/'config.json';p.write_text(base.encode(cfg))
    net,_,_,_=base.fresh(p,11,base.PredictiveReceptorNeuron)
    birth=tmp_path/'birth.paula';base.save_checkpoint(net,birth)
    rng=np.random.default_rng(60)
    features=[dict(visual=rng.random((300,96)),auditory=rng.random((300,96))) for _ in (0,1)]
    arm=base.Arm();delay=AfferentDelay(64)
    first=world.course(net,arm,features,g,s,0,0,ticks=144,delay=delay)
    assert verify_learning(first)==0
    cp=tmp_path/'mid.paula';physical=tmp_path/'body.npz'
    base.save_checkpoint(net,cp);np.savez_compressed(physical,state=arm.state(),delay=delay.state())
    untouched,body2,delay2=restore_acquired(cp,physical)
    python_state,numpy_state=random.getstate(),np.random.get_state()
    expression(birth,net,features,g,s,1,0,False)
    assert random.getstate()==python_state
    after=np.random.get_state()
    assert after[0]==numpy_state[0] and np.array_equal(after[1],numpy_state[1]) and after[2:]==numpy_state[2:]
    a=world.course(net,arm,features,g,s,0,1,ticks=96,delay=delay)
    b=world.course(untouched,body2,features,g,s,0,1,ticks=96,delay=delay2)
    assert set(a)==set(b)
    assert all(np.array_equal(a[k],b[k]) for k in a)
    assert verify_learning(a)==0 and np.all(a['eta']>0)
