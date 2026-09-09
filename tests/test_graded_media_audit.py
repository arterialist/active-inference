from copy import deepcopy
import gzip
import json
import numpy as np
import pytest
from simulations.active_inference.experiments.graded_media_audit import check_config,verify_rates
from simulations.active_inference.experiments.graded_media_probe import configure
from simulations.active_inference.experiments.population_hierarchy import make_config
from simulations.active_inference.experiments.graded_media_audit import compare_recall_factors
from simulations.active_inference.experiments.association_route_probe import digest


def test_unrelated_synapse_change_is_not_a_sensory_release_intervention():
    original,groups,_=make_config(1152,11)
    cfg=configure(original,groups,'graded');m=dict(groups=groups,condition='graded')
    assert check_config(cfg,original,m)==.25
    bad=deepcopy(cfg);bad['synaptic_points'][1]['u_i']['info']+=.1
    with pytest.raises(ValueError,match='Undeclared graph'):check_config(bad,original,m)


def test_learning_gain_must_follow_previous_local_modulator():
    cfg,groups,_=make_config(1152,11);nid=groups['visual_core'][0]
    cells=np.zeros((4,1152,8));cells[:,nid-1,3]=[.1,.2,.3,.4]
    cells[:,nid-1,7]=1+499*np.array([0.,.1,.2,.3])/(.1+np.array([0.,.1,.2,.3]))
    verify_rates(cells,np.zeros(1152),cfg,[(nid,0,1)])
    cells[1,nid-1,7]=cells[2,nid-1,7]
    with pytest.raises(ValueError,match='prior modulator'):verify_rates(cells,np.zeros(1152),cfg,[(nid,0,1)])


def test_factor_comparison_retains_ticks_and_rejects_hidden_state_change(tmp_path):
    root, source, donor = [tmp_path/name for name in ('factors', 'graded', 'spiking')]
    for p in (root, source, donor):p.mkdir()
    state = {'neurons': {'2': {'synapses': {'0': [.3, 0., [0., 0.], 0.]} }}, 'presynaptic_wheel': []}
    for path in (source/'trained-state.json.gz', source/'initial-state.json.gz', donor/'trained-state.json.gz'):
        with gzip.open(path, 'wt') as f:json.dump(state, f)
    groups = {'tactile_core': [1, 2], 'upper_core': [1, 2]}
    manifest = dict(source=str(source), donor=str(donor), groups=groups, selected_ports=[[2, 0, 1]])
    (root/'manifest.json').write_text(json.dumps(manifest))
    refs = []; probes = []
    def sample(clip):
        cells = np.zeros((40, 2, 8));cells[:, clip, 1] = 1.
        return cells
    for stage in ('initial', 'continuation'):
        for clip in (0, 1):
            path = source/f'{stage}-{clip}.npz';np.savez_compressed(path, cells=sample(clip))
            refs.append(dict(state=stage, sense='audio', clip=clip, gain=1., file=path.name, sha256=digest(path)))
    (source/'summary.json').write_text(json.dumps(dict(probes=refs)))
    for weight in ('graded', 'spiking', 'initial'):
        for expression in ('graded', 'spiking'):
            for clip in (0, 1):
                path = root/f'{weight}-{expression}-{clip}.npz';np.savez_compressed(path, cells=sample(clip))
                with gzip.open(root/(path.stem+'-start.json.gz'), 'wt') as f:json.dump(state, f)
                probes.append(dict(weights=weight, expression=expression, clip=clip, file=path.name,
                                   sha256=digest(path), auditory_spikes_by_tick=[1]*40))
    (root/'summary.json').write_text(json.dumps(dict(probes=probes)))
    out = tmp_path/'analysis';result = compare_recall_factors(root, out)
    assert all(p['total_spikes']==40 and p['first_spike']==0 for p in result['probes'])
    with np.load(out/'trajectories.npz') as z:
        np.testing.assert_array_equal(z['tactile_core/continuation/graded/graded/projection'], np.ones(40))
    state['presynaptic_wheel'] = [[9, 'invented event']]
    with gzip.open(root/'graded-graded-0-start.json.gz', 'wt') as f:json.dump(state, f)
    with pytest.raises(ValueError, match='Undeclared recorded-state'):
        compare_recall_factors(root, tmp_path/'bad')
