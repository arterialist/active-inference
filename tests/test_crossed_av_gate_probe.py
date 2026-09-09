import numpy as np

from simulations.active_inference.experiments import context_organization as base
from simulations.active_inference.experiments import crossed_av_world as world
from simulations.active_inference.experiments.crossed_av_gate_probe import observed_course
from simulations.active_inference.experiments.opponent_context import AfferentDelay
from simulations.active_inference.experiments.temporal_verification import configure


def test_terminal_observer_preserves_full_course_and_reconstructs_retrograde(tmp_path):
    cfg,groups,selected=configure(width=8)
    path=tmp_path/'config.json';path.write_text(base.encode(cfg))
    net,_,_,_=base.fresh(path,11,base.PredictiveReceptorNeuron)
    checkpoint=tmp_path/'birth.paula';base.save_checkpoint(net,checkpoint)
    features=[dict(visual=np.ones((300,96))*.2,auditory=np.ones((300,96))*.1) for _ in (0,1)]
    expected=world.course(net,base.Arm(),features,groups,selected,0,0,ticks=96,delay=AfferentDelay(64))
    branch=base.load_checkpoint(checkpoint,trusted=True).network
    actual,terminal=observed_course(branch,base.Arm(),AfferentDelay(64),features,groups,selected,0,0)
    assert all(np.array_equal(actual[k],expected[k]) for k in actual)
    assert terminal.shape==(96,4)
    assert terminal[:,2].max()==8
    assert np.any(terminal[:,1]<terminal[:,0])
