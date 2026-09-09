import pytest

from simulations.active_inference.experiments.media_acquisition_course import probe_segments, validate_checkpoints


def test_context_factorial_changes_only_declared_physical_segments():
    cases = [probe_segments(3168, v, a) for v in (None, 0, 1) for a in (None, 0, 1)]
    assert len(cases) == 9
    for segments in cases:
        first, second = segments
        assert first['start'] == 3168 and first['stop'] == second['start'] == 3232
        assert second['stop'] == 3532
        assert first['audio_clip'] is None and second['visual_clip'] is None
    assert cases[4] == [dict(start=3168, stop=3232, visual_clip=0, audio_clip=None),
                        dict(start=3232, stop=3532, visual_clip=None, audio_clip=0)]


def test_checkpoints_include_the_exact_original_replay_boundary():
    validate_checkpoints((4, 16), 4)
    for checkpoints in ((), (16,), (4, 4), (16, 4), (4, 0), (4, 8.5)):
        with pytest.raises(ValueError):
            validate_checkpoints(checkpoints, 4)
    with pytest.raises(ValueError):
        probe_segments(0, 0, 1, prefix=63)
