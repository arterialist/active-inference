import json

import numpy as np
import pytest

from simulations.active_inference.experiments.competition_course_comparison import contrast, pair_events, load_audit
from simulations.active_inference.experiments import context_organization as base


def test_silence_is_not_correct_and_transients_are_retained():
    x = np.zeros((96, 4))
    x[8:12] = [1, -1, -1, 1]
    x[12:17] = -1
    x[18] = [1, -1, -1, 1]
    e = pair_events(x)
    assert e['all_pairs_correct'] == [[8,12],[18,19]]
    assert e['negative_bias'] == [[12,17]]
    assert e['any_pair_silent'] == [[0,8],[17,18],[19,96]]


def test_prediction_and_body_utility_remain_separate():
    a = np.zeros((96,15)); b = a.copy()
    b[:,8] = .2; b[:,1] = -.3
    d = contrast(a,b)
    assert np.all(d[:,0] > 0) and np.all(d[:,1] > 0)
    b[42,4] = .2
    with pytest.raises(ValueError, match='load'): contrast(a,b)


def test_incomplete_or_nonfinite_trajectories_rejected():
    with pytest.raises(ValueError): pair_events(np.ones((96,3)))
    with pytest.raises(ValueError): contrast(np.zeros((95,15)),np.zeros((96,15)))
    with pytest.raises(ValueError): pair_events(np.full((96,4),np.nan))


def test_complete_family_required_and_audited_sources_must_match(tmp_path):
    sources = []; cases = []
    for seed in (11,23,44,77):
        root = tmp_path/str(seed); root.mkdir()
        (root/'manifest.json').write_text(json.dumps(dict(seed=seed,schedule=[])))
        (root/'completed-block-16.json').write_text('{}')
        sources.append(dict(root=str(root), manifest_sha256=base.digest(root/'manifest.json'),
                            progress_sha256=base.digest(root/'completed-block-16.json')))
        for block in range(1,17):
            controls = [('resting','learned')]
            if block in (4,8,12,16): controls += [('acquired','learned'),('acquired','reset')]
            for kind, weights in controls:
                for v,a in ((0,0),(0,1),(1,0),(1,1)):
                    cases.append(dict(seed=seed,blocks=block,kind=kind,weights=weights,video=v,audio=a))
    data = dict(blocks=16,sources=sources,cases=cases)
    p = tmp_path/'summary.json'; p.write_text(json.dumps(data))
    assert len(load_audit(tmp_path)[1]) == 384
    data['cases'] = cases[:-1]; p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Incomplete'): load_audit(tmp_path)
    data['cases'] = cases + [cases[0]]; p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Duplicate probe'): load_audit(tmp_path)
    data['cases'] = cases; p.write_text(json.dumps(data))
    (tmp_path/'11'/'completed-block-16.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='source changed'): load_audit(tmp_path)
