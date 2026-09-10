"""Two-factor student viability comparison using existing parameters only.

The prior native student stayed silent and its weak selected inputs collapsed.
Compare native course, slower student receiving adaptation, the existing
32-fold output sensitivity applied to student DANs, and their combination.
Parameters are assigned at birth and never switched by phase. No new neural
equation, cue code, connection, external dopamine drive or weight reset.
This tests student viability, not delayed credit or higher-order learning.
"""
import argparse
import math
from pathlib import Path
from unittest.mock import patch

from . import feeding, student_course
from ..connectome import sha256


def configure(graph, *, slow=False, sensitive=False):
    prep, selected, factor = student_course.configure(graph)
    roles = student_course.groups(graph)
    mbons = {prep.root_to_id[r] for r in roles["MBON04"]}
    student = selected[[int(e[3]) in mbons for e in selected]]
    weights = [prep.network.network.neurons[int(e[3])].postsynaptic_points[int(e[4])].u_i.info for e in student]
    # With at most one initial unit release per tick, the unrewarded negative
    # step is bounded by eta*(1+max_weight). Keep its 140-tick sum below 10%
    # of the weakest birth input. This is a conservative local initialization
    # calculation, not a guarantee under future recurrent/modulated activity.
    upper = .1*min(weights)/(140*(1+max(weights)))
    eta = 10.**math.floor(math.log10(upper))
    if slow:
        for nid in mbons:
            prep.network.network.neurons[nid].params.eta_post = eta
    if sensitive:
        for root in roles["PAM07"]+roles["PAM08"]:
            cell = prep.network.network.neurons[prep.root_to_id[root]]
            cell.params.r_base /= factor; cell.params.b_base /= factor
            cell.r = cell.params.r_base; cell.b = cell.params.b_base
    prep.assumptions["student_viability"] = dict(
        slow_receiving=slow, sensitive_DAN=sensitive,
        student_eta_post=eta if slow else .01,
        conservative_eta_upper=upper, minimum_birth_weight=min(weights), maximum_birth_weight=max(weights),
        derivation="Next lower power of ten satisfying eta*140*(1+max_birth_weight) <= 0.1*min_birth_weight",
        DAN_sensitivity=factor if sensitive else 1.,
        scope="Birth-only existing parameters. eta_post applies to all MBON04 inputs; eta_retro remains positive and unchanged. DAN gain reuses the output-cell sensitivity, not a biological calibration.",
        limit="No delayed eligibility exists in the selected update. Retention and feedback specificity must be checked separately; a live DAN alone is insufficient.",
        source_sha256=sha256(Path(__file__)))
    return prep, selected, factor


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("graph", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--slow", action="store_true"); p.add_argument("--sensitive", action="store_true")
    a = p.parse_args(); original = feeding.configure
    def configured(graph, *, sensitive=True):
        with patch.object(feeding, "configure", original):
            return configure(graph, slow=a.slow, sensitive=a.sensitive)
    with patch.object(feeding, "configure", configured), patch.object(feeding, "groups", student_course.groups):
        feeding.run(a.graph, a.output)
