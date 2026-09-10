"""Test delayed eligibility against the measured teacher arrival times.

With anatomically balanced codes, B recruits no feedback by itself. Learned A
recruits student dopamine at A ticks 76 and 148. The second event can affect
the KC terminal rule about 171 ticks after the last driven B tick, including
the 20-tick gap, one travel tick and one trace-update tick. Retaining at least
half of a completed B trace over that interval requires tau >= 171/log(2),
about 246.7 ticks. Test the next power of two, 256, from birth instead of 64.

This is an unfitted model timing comparison. It is not a biological time
calibration. The normalized EMA also builds more slowly with larger tau, so
paired/unpaired direct acquisition must be checked again; A survival is not
assumed. Learning rate, dopamine kinetics, efficacy, anatomy and body stay as
specified by the broader primary preparation.
"""
import argparse
from pathlib import Path
from unittest.mock import patch

from . import primary_boundary, terminal_course
from ..connectome import sha256

_primary_build = primary_boundary.build
_terminal_build = terminal_course.build


def build(graph):
    with patch.object(terminal_course, "build", lambda g: _terminal_build(g, tau_kc=256.)):
        prep, selected = _primary_build(graph)
    prep.assumptions["eligibility_window"] = dict(tau_kc=256., previous_tau_kc=64.,
        calibration_record="memory-balanced-second-intact-20260910/soma.npy",
        teacher_DAN_A_ticks=[76, 148], conservative_last_event_interval=171,
        derivation="Next power of two above 171/log(2), retaining half of a completed B trace through the second observed teacher event",
        limit="The normalized EMA also builds more slowly. Direct acquisition must be reassessed. No biological tick-duration or learning-time fit.",
        source_sha256=sha256(Path(__file__)))
    return prep, selected


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("graph", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--unpaired", action="store_true")
    a=p.parse_args()
    with patch.object(primary_boundary, "build", build):
        primary_boundary.run_course(a.graph, a.output, unpaired=a.unpaired)
