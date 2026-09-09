from simulations.active_inference.experiments.graded_media_probe import configure
from simulations.active_inference.experiments.population_hierarchy import make_config


def test_release_intervention_changes_only_sensory_metadata():
    original,groups,_=make_config(1152,11)
    graded=configure(original,groups,'graded');spiking=configure(original,groups,'spiking')
    assert graded['connections']==spiking['connections']==original['connections']
    assert graded['synaptic_points']==spiking['synaptic_points']==original['synaptic_points']
    sensory=set(groups['vision']+groups['touch'])
    for o,g,s in zip(original['neurons'],graded['neurons'],spiking['neurons']):
        assert o['params']==g['params']==s['params']
        if o['id'] not in sensory:assert o==g==s
        else:
            assert g['metadata']['graded_gain']==.25
            assert s['metadata']['graded_gain']==0
            assert 'graded_gain' not in o['metadata']
