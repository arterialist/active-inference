import numpy as np
from simulations.active_inference.experiments.evidence_consumer_comparison import observations


def test_early_candidate_is_not_a_completed_volley_error():
    schedule = np.zeros((64,32)); schedule[[0,3,8,11,16,19,24,27],0] = 1
    counts = np.zeros((2,64),int); counts[1,[6,14,22,30]] = 16; counts[0,[9,17,25,33]] = 16
    p = dict(case='replace25-cue0-sample0/minority_first',condition='evidence_slow',
             checkpoint=128,state='trained',spikes_by_tick_cue=counts)
    row = observations(p,schedule)
    assert row['competing_spike_ticks'] == [6,14,22,30]
    assert [w['margin'] for w in row['completed_volley_windows']] == [16]*4
    assert row['completed_volley_windows'][0]['start'] == 9


def test_transition_distinguishes_carryover_from_failure_to_switch():
    schedule = np.zeros((96,32)); counts = np.zeros((2,96),int)
    counts[0,[6,30,39,42,45]] = 16; counts[1,[46,49,94]] = 16
    p = dict(case='transition-first0-gap0',condition='evidence_slow',checkpoint=128,
             state='trained',spikes_by_tick_cue=counts)
    row = observations(p,schedule)
    assert row['earliest_new_response'] == 38
    assert row['next_response_latency'] == 14
    assert row['old_responses_after_new_available'] == [39,42,45]
    assert row['old_responses_after_first_new'] == []
