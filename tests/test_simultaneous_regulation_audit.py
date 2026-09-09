import numpy as np
import pytest
from simulations.active_inference.experiments.simultaneous_regulation_audit import interaction, pairing_contrast


def test_additive_inputs_and_history_offsets_do_not_imply_a_pair_interaction():
    samples = {(v, a): np.full((30, 4), 9. + (0 if v is None else 3+v) + (0 if a is None else 7+2*a))
               for v in (None, 0, 1) for a in (None, 0, 1)}
    for v in (0, 1):
        for a in (0, 1):
            assert not interaction(samples, v, a).any()
    samples[0, 1] += .25
    assert np.all(interaction(samples, 0, 1) == .25)
    del samples[None, None]
    with pytest.raises(ValueError):
        interaction(samples, 0, 1)


def test_same_input_contrast_uses_training_assignment_not_physical_clip_identity():
    low, high = np.zeros((30, 4)), np.ones((30, 4))
    for v in (0, 1):
        for a in (0, 1):
            paired, swapped = (low, high) if v == a else (high, low)
            assert np.all(pairing_contrast(paired, swapped, v, a) == 1)


def test_balancing_cancels_additive_history_bias_but_individual_pair_does_not():
    a = np.ones((30, 4))
    contrasts = [pairing_contrast(a, 2*a, v, sound) for v in (0, 1) for sound in (0, 1)]
    assert all(x.any() for x in contrasts)
    assert not sum(contrasts).any()


def test_balanced_raw_and_interaction_contrasts_are_not_independent_evidence():
    rng = np.random.default_rng(14)
    histories = [{(v, a): rng.normal(size=(30, 4)) for v in (None, 0, 1) for a in (None, 0, 1)}
                 for _ in range(2)]
    raw, adjusted = [], []
    for v in (0, 1):
        for a in (0, 1):
            raw.append(pairing_contrast(*(s[v, a] for s in histories), v, a))
            adjusted.append(pairing_contrast(*(interaction(s, v, a) for s in histories), v, a))
    np.testing.assert_allclose(sum(raw), sum(adjusted), atol=3e-15, rtol=0)
