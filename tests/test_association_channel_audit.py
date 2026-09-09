import numpy as np
import pytest
from simulations.active_inference.experiments.association_channel_audit import channel_comparison


def test_matching_spikes_do_not_imply_matching_internal_state():
    a = np.zeros((64, 176, 8)); a[5, 96:112, 1] = 1
    b = a.copy(); b[3, 64, 0] = .75
    result = channel_comparison(a, b)
    assert result['consumer_output_exact'] and result['consumer_recorded_state_exact']
    assert result['auditory_output_exact'] and not result['auditory_recorded_state_exact']


def test_one_tick_difference_cannot_hide_in_equal_spike_counts():
    a = np.zeros((64, 176, 8)); a[5, 96, 1] = 1
    b = np.zeros_like(a); b[6, 96, 1] = 1
    result = channel_comparison(a, b)
    assert not result['consumer_output_exact']
    assert result['differing_consumer_output_ticks'] == [5, 6]


def test_invalid_trace_is_not_evidence_of_equality():
    a = np.zeros((64, 176, 8)); a[0, 0, 0] = np.nan
    with pytest.raises(ValueError): channel_comparison(a, a)
    with pytest.raises(ValueError): channel_comparison(a[:0], a[:0])
