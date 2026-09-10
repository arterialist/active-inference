"""One fixed physical comparison of cue-by-position nutrient expectation."""
import argparse
from pathlib import Path
import shutil
from unittest.mock import patch

from . import ingestion_course as course, body_context_comparison as composition
from . import signed_terminal_credit as signed
from .feeding import FeedingBody
from ..connectome import sha256
from neuron.neuron import setup_neuron_logger


def acquire(base, output):
    if output.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(base).free < 4*1024**3+700*1024**2:
        raise RuntimeError("Shared-volume reserve")
    parent, manifest, mapping, selected = course.settings(base)
    branch = course.load_checkpoint(parent / "retention.paula", trusted=True)
    receipt = signed.convert(branch)
    organism = FeedingBody()
    organism.restore(parent / "retention-body.npz")
    metadata = composition.append_comparison(branch, mapping, manifest)
    metadata.update(source_sha256=sha256(Path(__file__)), composition_sha256=sha256(Path(composition.__file__)),
        signed_variant=dict(source_sha256=sha256(Path(signed.__file__)), conversion=receipt),
        parent_checkpoint_sha256=sha256(parent / "retention.paula"), parent_body_sha256=sha256(parent / "retention-body.npz"),
        manifest=manifest,
        bound="Same eight A-food 200-tick presentations with 1000 blank ticks each, final 1000-tick retention and common 200-tick A-food lead-in. Then one 2x2 physical comparison: food loss versus continued food, actual body coincidence versus matched body-readout bypass. Each runs 600 ticks, 1000 blank ticks and independent 200-tick A/B food probes. Unit omission-to-SMP108 inhibition is unchanged and enabled throughout both arms.")
    output.mkdir(parents=True)
    course.write(output / "manifest.json", metadata)
    course.snapshot(branch, organism, output / "initial")
    phases = [(f"pair{i}_{label}", cue, well, ticks) for i in range(8)
        for label, cue, well, ticks in (("A_food", "A", True, 200), ("blank", "", False, 1000))]
    phases.append(("retention", "", False, 1000))
    course.record(branch, organism, manifest, mapping, selected, metadata, phases, output / "acquisition")
    course.record(branch, organism, manifest, mapping, selected, metadata, [("A_food_lead_in", "A", True, 200)], output / "comparison-start")


def action(base, output):
    _, manifest, mapping, selected = course.settings(base)
    metadata = course.read(output / "manifest.json")
    branch, organism = course.restore(output / "comparison-start")
    targets = [mapping[r] for r in manifest["roles"]["SMP108"]]
    edges = composition.append_action(branch, metadata, targets)
    course.snapshot(branch, organism, output / "action-start")
    result = dict(action_edges=edges, source_sha256=sha256(Path(__file__)), conditions={})
    for outcome, well in (("omission", False), ("food", True)):
        for bypass in (False, True):
            label = outcome+("-bypass" if bypass else "-body")
            branch, organism = course.restore(output / "action-start")
            intervention = composition.bypass_body(branch, metadata) if bypass else dict(changed_readouts=0, all_other_state_and_rng_exact=True)
            course.snapshot(branch, organism, output / ("start-"+label))
            report = course.record(branch, organism, manifest, mapping, selected, metadata,
                [(outcome, "A", well, 600)], output / ("action-"+label))
            report["body_context_intervention"] = intervention
            retention = output / ("retention-"+label)
            course.record(branch, organism, manifest, mapping, selected, metadata, [("retention", "", False, 1000)], retention)
            report["probes"] = {}
            for cue in ("A", "B"):
                b, body = course.restore(retention)
                report["probes"][cue] = course.record(b, body, manifest, mapping, selected, metadata,
                    [(cue+"_food", cue, True, 200)], output / ("probe-"+label+"-"+cue))
            result["conditions"][label] = report
    course.write(output / "action.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("acquire", "action"))
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    setup_neuron_logger("CRITICAL")
    with patch.object(course, "composition", composition):
        dict(acquire=acquire, action=action)[args.mode](args.base, args.output)
