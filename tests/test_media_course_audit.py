import numpy as np
import pytest

from simulations.active_inference.experiments.media_course_audit import context_interactions, check_segments, recruitment_trajectories
from simulations.active_inference.experiments.media_order_audit import crossed_effect


def test_additive_sensory_responses_cancel_but_nonlinearity_is_not_prediction():
    level = {None: 0., 0: 1., 1: 2.}
    additive = {(v, a): np.full((4, 3), 7+level[v]+3*level[a]) for v in level for a in level}
    inter, diagonal = context_interactions(additive)
    assert not diagonal.any() and all(not v.any() for v in inter.values())
    nonlinear = {(v, a): x+level[v]*level[a] for (v, a), x in additive.items()}
    _, diagonal = context_interactions(nonlinear)
    assert diagonal.any()  # A fixed nonlinear sensor alone can do this.
    main, _ = crossed_effect(diagonal, diagonal, diagonal, diagonal)
    assert not main.any()
    main, _ = crossed_effect(2*diagonal, diagonal, 2*diagonal, diagonal)
    assert main.any()  # History-dependent scalar gain also produces an effect.
    with pytest.raises(ValueError):
        context_interactions({k: v for k, v in additive.items() if k != (None, None)})


def test_independent_segment_check_rejects_visual_continuation_or_wrong_clock():
    p = dict(kind='context', visual=0, audio=1, segments=[
        dict(start=20, stop=84, visual_clip=0, audio_clip=None),
        dict(start=84, stop=384, visual_clip=None, audio_clip=1)])
    check_segments(p, 20)
    p['segments'][1]['visual_clip'] = 0
    with pytest.raises(ValueError):
        check_segments(p, 20)


def test_recruitment_removes_background_and_checks_preintervention_identity():
    blank = np.zeros((12, 3, 4)); blank[:, 1, 1] = 1
    cue = blank[:10].copy(); cue[:, 0, 1] = 1
    withdrawn = blank.copy(); withdrawn[:5, 0, 1] = 1
    d = recruitment_trajectories(cue, withdrawn, blank, prefix=4)
    assert not d['cue_minus_blank'][:, 1].any()
    assert d['cue_minus_blank'][:, 0].sum() == 10
    assert d['withdrawn_minus_blank'][4:, 0].sum() == 1
    withdrawn[0, 2, 3] = 1  # A modulation difference is a broken prefix too.
    with pytest.raises(ValueError):
        recruitment_trajectories(cue, withdrawn, blank, prefix=4)
