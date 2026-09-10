"""More exposure must preserve the original timing and matched controls."""
from simulations.drosophila.memory_feedback.repeated_course import protocol
from simulations.drosophila.memory_feedback.second_order import protocol as original


def test_repetition_preserves_two_pairing_reference_and_matched_time():
    assert protocol(2) == original()
    assert protocol(2, True) == original(True)
    a, b = protocol(8), protocol(8, True)
    assert sum(t for _, _, t in a) == 11880
    for cue in ("A", "B", ""):
        assert sum(t for _, c, t in a if c == cue) == sum(t for _, c, t in b if c == cue)
    assert a[-1] == b[-1] == ("retention", "", 1000)
