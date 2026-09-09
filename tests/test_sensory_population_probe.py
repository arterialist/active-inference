import numpy as np
from simulations.active_inference.experiments.sensory_population_probe import physical_gain,population_config
from simulations.active_inference.experiments.population_hierarchy import make_config


def test_physical_gain_preserves_latency_and_neutral_transfer():
    rng=np.random.default_rng(11);visual=rng.uniform(0,1,(12,96));visual[:3]=0
    db=rng.uniform(-100,0,(12,32));feature=dict(visual=visual,band_db=db)
    assert np.array_equal(physical_gain(feature,'visual',1)[:,:96],visual)
    assert not physical_gain(feature,'visual',.5)[:3].any()
    assert np.all(physical_gain(feature,'visual',2)[3:,:96]<=visual[3:])
    expected=np.clip((db[:,:,None]-np.array([-65.,-45.,-25.]))/20,0,1).reshape(12,96)
    assert np.array_equal(physical_gain(feature,'audio',1)[:,96:],expected)


def test_sensitivity_comparison_preserves_every_other_parameter():
    original,_,_=make_config(1152,11)
    homogeneous,hc=population_config(original,'homogeneous');diverse,dc=population_config(original,'diverse')
    assert np.array_equal(hc,dc) and len(hc)==1152
    assert homogeneous['connections']==diverse['connections']==[]
    assert homogeneous['synaptic_points']==diverse['synaptic_points']
    for a,b in zip(homogeneous['neurons'],diverse['neurons']):
        assert {k:v for k,v in a['params'].items() if k not in ('r_base','b_base')}=={k:v for k,v in b['params'].items() if k not in ('r_base','b_base')}
