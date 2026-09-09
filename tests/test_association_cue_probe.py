from simulations.active_inference.experiments.association_cue_probe import cue_cases


def test_partial_and_mixed_cues_have_declared_physical_content():
    masks={'vision':[list(range(1,17)),list(range(17,33))]}
    cases=cue_cases(masks,11)
    assert len(cases)==24 and len({c['name'] for c in cases})==24
    assert cases==cue_cases(masks,11)
    for c in cases:
        values=set(c['selected_receptors']);own=set(masks['vision'][c['cue']]);other=set(masks['vision'][1-c['cue']])
        assert len(values)==len(c['selected_receptors'])==c['own_count']+c['foreign_count']
        assert len(values&own)==c['own_count'] and len(values&other)==c['foreign_count']
        if c['kind']=='balanced':assert c['own_count']==c['foreign_count']
        elif c['kind']=='silence':assert not values
        else:assert c['own_count']>c['foreign_count']


def test_consumer_criterion_does_not_count_one_cell_or_ambiguity_as_completion():
    import numpy as np
    from simulations.active_inference.experiments.association_cue_audit import classify_response
    x=np.zeros((64,32),bool);target=list(range(16))
    assert classify_response(x,target,'clean')['status']=='silent'
    x[5,0]=True;assert classify_response(x,target,'clean')['status']=='partial_selective'
    x[5,:16]=True;assert classify_response(x,target,'clean')['status']=='complete_selective'
    x[5,16]=True;assert classify_response(x,target,'replace25')['status']=='mixed'
    assert classify_response(x,target,'balanced')['status']=='balanced_both'
    assert classify_response(x,target,'silence')['status']=='spontaneous'
