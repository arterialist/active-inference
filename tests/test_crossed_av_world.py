import numpy as np
import pytest

from simulations.active_inference.experiments import context_organization as base
from simulations.active_inference.experiments import crossed_av_world as world
from simulations.active_inference.experiments.opponent_context import AfferentDelay,verify_afferents
from simulations.active_inference.experiments.temporal_verification import configure,verify_learning


def features():
    rng=np.random.default_rng(800)
    return [dict(visual=rng.random((300,96)),auditory=rng.random((300,96))) for _ in (0,1)]


def test_crossed_world_has_no_single_sense_solution_or_pair_label():
    f=features()
    for tick in (0,17,63,299):
        rows={(v,a):world.world_drive(f,v,a,tick) for v in (0,1) for a in (0,1)}
        for v in (0,1):
            assert np.array_equal(rows[v,0][0][:96],rows[v,1][0][:96])
            assert rows[v,0][1]==-rows[v,1][1]
        for a in (0,1):
            assert np.array_equal(rows[0,a][0][96:192],rows[1,a][0][96:192])
            assert rows[0,a][1]==-rows[1,a][1]
        for v,a in rows:
            other,force=world.world_drive(f,v,a,tick,reverse=True)
            assert np.array_equal(rows[v,a][0][:194],other[:194])
            assert rows[v,a][1]==-force
            assert np.array_equal(other[192:194],[1,0])
    assert world.world_drive(f,0,0,300)[1]==0
    assert all(sorted(block)==[(0,0),(0,1),(1,0),(1,1)] for block in world.schedule())


@pytest.mark.parametrize('presentation,first,second',[('vision',(0,0),(0,1)),('audio',(0,0),(1,0))])
def test_missing_sense_gives_identical_brain_history_before_somatic_arrival(tmp_path,presentation,first,second):
    cfg,g,s=configure(width=8);path=tmp_path/'brain.json';path.write_text(base.encode(cfg))
    net,_,_,_=base.fresh(path,11,base.PredictiveReceptorNeuron)
    cp=tmp_path/'birth.paula';base.save_checkpoint(net,cp)
    traces=[];f=features()
    for video,audio in (first,second):
        branch=base.load_checkpoint(cp,trusted=True).network
        z=world.course(branch,base.Arm(),f,g,s,video,audio,presentation=presentation,
                       ticks=80,delay=AfferentDelay(64))
        assert verify_learning(z)==base.verify_physics(z)==verify_afferents(z)==0
        assert np.all(z['eta']>0)
        traces.append(z)
    assert np.array_equal(traces[0]['drive'][:64],traces[1]['drive'][:64])
    for field in ('cells','weights','arrivals','errors'):
        assert np.array_equal(traces[0][field][:64],traces[1][field][:64])
    assert np.array_equal(traces[0]['body'][:,4],-traces[1]['body'][:,4])
    assert not np.array_equal(traces[0]['drive'][64:],traces[1]['drive'][64:])


def test_invalid_or_hidden_input_channels_rejected():
    with pytest.raises(ValueError):world.crossed_features(features(),0,1,'pair-label')
    with pytest.raises(ValueError):world.crossed_features(features(),0,1,offset=300)
    f=features();f[0]['visual'][0,0]=np.nan
    with pytest.raises(ValueError):world.crossed_features(f,0,1)


def test_independent_stimulus_audit_rejects_wrong_load_and_sensory_source():
    from simulations.active_inference.experiments.crossed_av_analysis import verify_stimuli
    f=features();row=dict(video=0,audio=1,presentation='vision',offset=64)
    samples=[world.world_drive(f,0,1,t,presentation='vision',offset=64,reverse=True) for t in range(304)]
    z=dict(drive=np.array([x[0] for x in samples]),body=np.zeros((304,5)))
    z['body'][:,4]=[x[1] for x in samples]
    verify_stimuli(z,f,row,True)
    z['body'][0,4]*=-1
    with pytest.raises(ValueError,match='physical assignment'):verify_stimuli(z,f,row,True)
    z['body'][0,4]*=-1;z['drive'][30,96]=.1
    with pytest.raises(ValueError,match='sensory evidence'):verify_stimuli(z,f,row,True)


def test_analysis_rejects_empty_or_distinguishable_controls(tmp_path):
    from simulations.active_inference.experiments.crossed_av_analysis import analyze,verify_missing_sense
    with pytest.raises(ValueError,match='No experimental conditions'):analyze([],tmp_path/'empty')
    a={k:np.zeros((64,2)) for k in ('drive','cells','weights','arrivals','errors')}
    a['body']=np.zeros((64,5));a['body'][:,4]=1
    b={k:v.copy() for k,v in a.items()};b['body'][:,4]=-1
    verify_missing_sense(a,b)
    b['cells'][40,1]=.001
    with pytest.raises(ValueError,match='cells'):verify_missing_sense(a,b)


def test_acquisition_audit_rejects_body_and_credit_resets():
    from simulations.active_inference.experiments.crossed_av_analysis import verify_continuity
    prior=dict(physical_states=np.ones((3,4)),weights=np.ones((3,2,5)),
        delay_final=np.ones((64,4)),errors=np.ones((3,2,3)),
        context_initial=np.ones((2,5)),arrivals=np.ones((3,2,5)))
    current=dict(body_initial=np.ones(4),weights_initial=np.ones((2,5)),
        delay_initial=np.ones((64,4)),error_initial=np.ones(2),context_initial=np.ones((2,5)))
    verify_continuity(prior,current)
    current['body_initial'][0]=0
    with pytest.raises(ValueError,match='body_initial'):verify_continuity(prior,current)
    current['body_initial'][0]=1;current['context_initial'][0,0]=0
    with pytest.raises(ValueError,match='credit trace'):verify_continuity(prior,current)


def test_conditional_capacity_certificates_distinguish_conflict_from_capacity():
    from simulations.active_inference.experiments.crossed_av_capacity import margin_certificate,filtered_context
    conflict=margin_certificate(np.array([[1.,0],[-1.,0]]))
    assert abs(conflict['upper_bound'])<1e-9
    distinct=margin_certificate(np.eye(2))
    assert distinct['feasible_minimum']==pytest.approx(1)
    assert distinct['upper_bound']==pytest.approx(1)
    trace=filtered_context(np.array([[1.],[0.],[0.]]))
    np.testing.assert_allclose(trace[:,0],[0,.2475,.185625])
    with pytest.raises(ValueError):margin_certificate(np.zeros((0,2)))


def test_native_predictor_rounding_is_measured_not_hidden_by_tolerance():
    from simulations.active_inference.experiments.crossed_av_capacity import filtered_context,native_predictor,rounding_guard
    rng=np.random.default_rng(45)
    for scale in (.0001,1.,10.):
        incoming=(scale*rng.random((64,16))).astype(np.float32).astype(float)
        q=rng.random(16)
        difference=abs(native_predictor(incoming,q)-filtered_context(incoming)@q).max()
        assert difference>0
        assert difference<rounding_guard([incoming])
    with pytest.raises(ValueError,match='unclamped'):rounding_guard([np.full((64,16),100.)])
