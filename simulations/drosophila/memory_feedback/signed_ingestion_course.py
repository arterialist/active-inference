"""Run one separately identified signed-domain ingestion comparison.

The failed near-boundary checkpoint supplies only a compatibility check. The
scientific course starts again at the accepted A/B reference, with the same
fixed acquisition and omission/action bounds as the interrupted composition.
"""
import argparse
from pathlib import Path
from unittest.mock import patch

from . import ingestion_course as course, ingestion_comparison as composition
from . import signed_terminal_credit as signed
from ..connectome import sha256
from neuron.neuron import setup_neuron_logger


def compatibility(base, output):
    _, manifest, mapping, selected = course.settings(base)
    source = base / "memory-ingestion-comparison-20260910"
    branch, organism = course.restore(source / "acquisition")
    metadata = course.read(source / "manifest.json")
    receipt = signed.convert(branch)
    metadata["signed_variant"] = dict(receipt=receipt, source_sha256=sha256(Path(signed.__file__)),
        origin_sha256=sha256(source / "acquisition/state.paula"), purpose="Near-boundary compatibility check only; not the scientific acquisition or omission comparison")
    output.mkdir(parents=True, exist_ok=False)
    course.write(output / "manifest.json", metadata)
    course.record(branch, organism, manifest, mapping, selected, metadata,
        [("near_boundary_A_food", "A", True, 200)], output / "check")


def acquire(base, output):
    load = course.load_checkpoint
    append = composition.append_comparison
    receipts = []
    def load_signed(path, **kwargs):
        branch = load(path, **kwargs)
        receipts.append(signed.convert(branch))
        return branch
    def append_labelled(branch, mapping, manifest):
        metadata = append(branch, mapping, manifest)
        metadata["signed_variant"] = dict(source_sha256=sha256(Path(signed.__file__)),
            driver_sha256=sha256(Path(__file__)), conversion=receipts[-1],
            scope="Scientific course starts at the accepted A/B reference, not the failed near-boundary state. Same fixed protocol and all existing parameters; signed pure-depression domain is the sole terminal-rule change.")
        return metadata
    with patch.object(course, "load_checkpoint", load_signed), patch.object(composition, "append_comparison", append_labelled):
        course.acquire(base, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compatibility", "acquire", "verify", "action"))
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    setup_neuron_logger("CRITICAL")
    dict(compatibility=compatibility, acquire=acquire, verify=course.verify, action=course.action)[args.mode](args.base, args.output)
