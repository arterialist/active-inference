import numpy as np
import pytest

from simulations.active_inference.experiments.sensory_motor_transfer_inspection import cycles, resource_contrast


def test_cycles_keep_incomplete_intervals_and_use_release_onsets():
    z = dict(body=np.zeros((9,5)), body_initial=np.zeros(5), organs=np.zeros((9,13)),
             organ_initial=np.zeros(13), exchange=np.ones((9,9)))
    z['body'][:,1] = np.arange(9)
    c = cycles(z,[0,1,1,0,0,1,1,0,0])
    assert c['partial_intervals'] == [[0,1],[5,9]]
    assert len(c['cycles']) == 1
    row = c['cycles'][0]
    assert (row['start'],row['end']) == (1,5)
    assert row['positive_angular_travel_rad'] == 4
    assert row['inspired_ml'] == 4
    assert row['exchange_per_positive_rad'] == 1
    assert cycles(z,np.zeros(9))['partial_intervals'] == [[0,9]]
    with pytest.raises(ValueError): cycles(z,[np.nan]*9)


def test_good_endpoint_does_not_hide_earlier_negative_resource_effect():
    a = np.array([[0,0],[-2,1],[-1,2],[3,3]],float)
    out = resource_contrast(a,np.zeros_like(a))
    assert out[0]['negative'] == [[1,3]]
    assert out[0]['positive'] == [[3,4]]
    assert out[0]['indistinguishable'] == [[0,1]]
    assert out[0]['minimum'] == -2
    assert out[1]['negative'] == []
    with pytest.raises(ValueError): resource_contrast(a,np.zeros((3,2)))
