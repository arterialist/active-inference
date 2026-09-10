"""Test original A-memory availability after omission with new inhibition cut."""
import argparse
from pathlib import Path

import numpy as np

from . import ingestion_course as course, ingestion_comparison as composition
from . import interface_controls as controls
from ..connectome import sha256
from neuron.neuron import setup_neuron_logger


def run(base, source):
    _, manifest, mapping, selected = course.settings(base)
    metadata = course.read(source / "manifest.json")
    parent = base / "memory-coverage-paired-20260910"
    donor = base / "memory-coverage-unpaired-20260910"
    m, ids, roots, checked_selected = controls.pair(parent, donor)
    assert np.array_equal(selected, checked_selected)
    mask = controls.mask_for(m, ids, roots, selected, "A", ("MBON07", "MBON04"))
    targets = [mapping[r] for r in manifest["roles"]["SMP108"]]
    state = source / "retention-omission-intact"
    receipts = {}
    for erased in (False, True):
        label = "erased" if erased else "retained"
        branch, body = course.restore(state)
        before = composition.serialized(branch)
        other = course.load_checkpoint(donor / "retention.paula", trusted=True) if erased else branch
        changes, originals = controls.swap(branch, other, selected, mask)
        start = source / ("availability-start-"+label)
        course.snapshot(branch, body, start)
        for point, value in originals:
            point.u_o.info = value
        assert composition.serialized(branch) == before
        branch, body = course.restore(start)
        gate = composition.action_gate(branch, metadata, targets, True)
        report = course.record(branch, body, manifest, mapping, selected, metadata,
            [("A_food", "A", True, 200)], source / ("availability-"+label), gate=gate)
        receipts[label] = dict(changes=changes, all_other_state_and_rng_exact=True,
            ingested_j=report["ingested_j"], stored_energy_gain_j=report["stored_energy_gain_j"],
            summary_sha256=sha256(source / ("availability-"+label) / "summary.json"))
    result = dict(source_sha256=sha256(Path(__file__)), state_sha256=sha256(state / "state.paula"),
        donor_sha256=sha256(donor / "retention.paula"), conditions=receipts,
        interpretation="Same post-omission state, with new omission-to-action traffic cut in both probes. Replace only original A-to-MBON terminal coefficients with the matched unpaired donor to test whether that memory remains functionally available. This intervention is a causal diagnostic, not an online learning rule.")
    course.write(source / "memory-availability.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    setup_neuron_logger("CRITICAL")
    run(args.base, args.source)
