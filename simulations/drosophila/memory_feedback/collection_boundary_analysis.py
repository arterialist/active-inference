"""Audit the initial relocation screen without claiming learned relocation."""
import argparse
from pathlib import Path

import numpy as np

from . import collection_boundary as experiment, ingestion_course as course
from .ingestion_analysis import checked_record
from ..connectome import sha256


def measurement(path, parent, angle):
    summary = checked_record(path)
    body = np.load(path / "body.npy", mmap_mode="r")
    comparison = np.load(path / "comparison.npy", mmap_mode="r")
    weights = np.load(path / "prediction_weights.npy", mmap_mode="r")
    organism = experiment.CollectionBoundaryBody(angle)
    organism.restore(parent / "body-state.npz")
    positions = np.r_[organism.body.data.qpos[0], body[:-1, 0]]
    hits = np.flatnonzero(body[:, 5] > 0)
    first = int(hits[0]) if len(hits) else len(body)
    assert np.all(body[positions < angle, 5] == 0)
    return dict(record=str(path), summary_sha256=sha256(path / "summary.json"),
        body_sha256=sha256(path / "body.npy"), comparison_sha256=sha256(path / "comparison.npy"),
        prediction_weights_sha256=sha256(path / "prediction_weights.npy"),
        ingested_j=summary["ingested_j"], stored_energy_gain_j=summary["stored_energy_gain_j"],
        first_ingestion_tick=first if len(hits) else None,
        maximum_angle_rad=float(body[:, 0].max()),
        pre_first_contact_negative_mean=float(comparison[:first, 3].mean()) if first else None,
        maximum_predictor_weight_change_before_first_ingestion=float(np.abs(weights[first]-weights[0]).max()),
        maximum_predictor_weight_change_during_probe=float(np.abs(weights[-1]-weights[0]).max()),
        adaptation_active=True, ticks=len(body))


def run(base, output):
    source = base / "memory-collection-boundary-20260910"
    old = base / "memory-body-context-20260910"
    parent = old / "retention-food-body"
    metadata = course.read(source / "manifest.json")
    assert metadata["source_sha256"] == sha256(Path(experiment.__file__))
    assert metadata["parent_checkpoint_sha256"] == sha256(parent / "state.paula")
    assert metadata["parent_body_sha256"] == sha256(parent / "body-state.npz")
    environment = course.read(source / "initial-A/environment.json")
    assert environment["well_angle"] == metadata["relocated_angle"] == .08
    assert environment["source_sha256"] == metadata["source_sha256"]
    # Original reference is an independent probe of the very same parent.
    for name in ("prediction_weights", "release", "weights"):
        assert np.array_equal(np.load(source / f"initial-A/{name}.npy", mmap_mode="r")[0],
            np.load(old / f"probe-food-body-A/{name}.npy", mmap_mode="r")[0])
    result = dict(source_sha256=sha256(Path(__file__)), manifest_sha256=sha256(source / "manifest.json"),
        environment_sha256=sha256(source / "initial-A/environment.json"), screen=metadata,
        original=measurement(old / "probe-food-body-A", parent, .04),
        relocated_initial=measurement(source / "initial-A", parent, .08),
        initial_selected_and_predictor_coefficients_exact=True,
        completed_exposure_courses=0,
        decision="Withdraw this move as a demonstrated spatial-relearning challenge. It already permits useful feeding in the initial probe; no after-experience or frozen-policy causal contrast was performed.",
        representation="The min(KC, position) readout has a steady unit-event ceiling near .051062 rad. It does not supply a resolved positional basis between that angle and the .08 rad boundary. The position afferent, delays, membrane state, predictor context/error traces and native adaptation still carry history. Saturation alone therefore does not prove full-state indistinguishability or an inability to adapt.",
        consequence="A later change could reflect coarse nutrient-expectation updating or other ongoing adaptation, not necessarily learning a relocated collection boundary. The 200-tick probe establishes immediate useful closed-loop feeding with learning active, not success of a frozen controller. It is not a negative learning-rule result.",
        remaining_question="Experience-dependent adjustment in this environment remains untested. The optional matched exposure modes are unexecuted; neither another boundary nor a new spatial representation was tried.")
    output.mkdir(parents=True, exist_ok=True)
    course.write(output / "collection-boundary-screen.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.base, args.output)
    print(result["decision"])
