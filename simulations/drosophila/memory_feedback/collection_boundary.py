"""One collection-boundary move with the retained body-context controller.

Only the physical angle threshold changes. Existing sensory calibration,
coincidence features, neural equations, learning rates and action wiring remain.
The optional continuation is bounded in advance; initial probes never train it.
"""
import argparse
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np

from . import ingestion_course as course, body_context_comparison as composition
from .feeding import FeedingBody, DOSE_J, WELL_ANGLE
from ..connectome import sha256
from neuron.neuron import setup_neuron_logger


RELOCATED_ANGLE = .08


class CollectionBoundaryBody(FeedingBody):
    def __init__(self, well_angle=WELL_ANGLE):
        super().__init__()
        self.well_angle = float(well_angle)

    def offer(self, pump, well):
        pump_j = self.organs.ingest(DOSE_J) if pump else 0.
        well_j = self.organs.ingest(DOSE_J) if well and self.body.data.qpos[0] >= self.well_angle else 0.
        return pump_j, well_j


def restore(path, angle):
    branch = course.load_checkpoint(path / "state.paula", trusted=True)
    body = CollectionBoundaryBody(angle)
    body.restore(path / "body-state.npz")
    return branch, body


def settings(base):
    _, manifest, mapping, selected = course.settings(base)
    source = base / "memory-body-context-20260910"
    metadata = course.read(source / "manifest.json")
    return source, manifest, mapping, selected, metadata


def record(branch, body, manifest, mapping, selected, metadata, phases, output):
    report = course.record(branch, body, manifest, mapping, selected, metadata, phases, output)
    # Neural checkpoints do not serialize the physical transducer. Keep its
    # setting beside each body state, including every independent probe.
    course.write(output / "environment.json", dict(well_angle=body.well_angle,
        source_sha256=sha256(Path(__file__)), contact_sample="Before neural/body step",
        afferent_scale_rad=.05, ingestion_dose_j=DOSE_J))
    return report


def initial(base, output):
    source, manifest, mapping, selected, metadata = settings(base)
    parent = source / "retention-food-body"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    branch, body = restore(parent, RELOCATED_ANGLE)
    cells = branch.network.network.neurons
    aff = cells[metadata["body_context"]["position"]]
    aff_gain = aff.postsynaptic_points[0].u_i.info * aff.params.delta_decay**aff.distances[0] * aff._gg
    angles = []
    for nid in metadata["body_context"]["gates"]:
        gate = cells[nid]
        cs, ct = gate.synapse_sources[0]
        ps, pt = gate.synapse_sources[1]
        cue = cells[cs].presynaptic_points[ct].u_o.info * (gate.postsynaptic_points[0].u_i.info+gate.postsynaptic_points[0].u_i.plast)
        gain = aff_gain*cells[ps].presynaptic_points[pt].u_o.info * (gate.postsynaptic_points[1].u_i.info+gate.postsynaptic_points[1].u_i.plast)
        angles.append(.05*cue/gain)
    predictor = cells[metadata["ids"]["prediction"]]
    receipt = dict(source_sha256=sha256(Path(__file__)), parent=str(parent),
        parent_checkpoint_sha256=sha256(parent / "state.paula"), parent_body_sha256=sha256(parent / "body-state.npz"),
        composition_sha256=sha256(Path(composition.__file__)), original_angle=WELL_ANGLE, relocated_angle=RELOCATED_ANGLE,
        saturation_angles_rad=dict(min=float(min(angles)), max=float(max(angles))),
        saturation_scope="Steady unit KC event at retained coefficients. Common gate dendritic attenuation cancels. Transient traces, delays and native coefficient changes remain; this is not a proof of trajectory equivalence.",
        predictor_context_tau_ticks=predictor.prediction_tau_context, predictor_error_tau_ticks=predictor.prediction_tau_error,
        predictor_eta_post=predictor.params.eta_post, predictor_rate_boost=predictor.prediction_boost,
        bound="One .04 to .08 rad collection-boundary move; unchanged sensory scaling, features and controller. Initial 200-tick A probe from saved retention-food-body, separate from any continuation. If warranted, exactly eight A-food200/blank1000 exposures plus final blank1000 in relocated and original-boundary arms; independent post-retention A/B200 probes in relocated environment. No threshold or gain search.",
        interpretation="This can test coarse experience-dependent feeding adjustment, not learning a precise location beyond the coincidence-feature saturation. Immediate useful feeding would weaken the case that this move requires adaptation.")
    course.write(output / "manifest.json", receipt)
    record(branch, body, manifest, mapping, selected, metadata,
        [("initial_A_relocated", "A", True, 200)], output / "initial-A")


def exposure(base, output, relocated):
    source, manifest, mapping, selected, metadata = settings(base)
    if shutil.disk_usage(base).free < 4*1024**3+800*1024**2:
        raise RuntimeError("Shared-volume reserve")
    label = "relocated" if relocated else "original"
    branch, body = restore(source / "retention-food-body", RELOCATED_ANGLE if relocated else WELL_ANGLE)
    phases = [(f"pair{i}_{label}", cue, well, ticks) for i in range(8)
        for label, cue, well, ticks in (("A_food", "A", True, 200), ("blank", "", False, 1000))]
    phases.append(("retention", "", False, 1000))
    path = output / ("exposure-"+label)
    record(branch, body, manifest, mapping, selected, metadata, phases, path)
    for cue in ("A", "B"):
        branch, body = restore(path, RELOCATED_ANGLE)
        record(branch, body, manifest, mapping, selected, metadata,
            [(cue+"_relocated", cue, True, 200)], output / ("probe-"+label+"-"+cue))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("initial", "relocated", "original"))
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    setup_neuron_logger("CRITICAL")
    with patch.object(course, "composition", composition):
        if args.mode == "initial":
            initial(args.base, args.output)
        else:
            exposure(args.base, args.output, args.mode == "relocated")
