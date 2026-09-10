"""Constant-context feeding with finite portions replenished by withdrawal.

The environment never schedules an action phase. Only physical disengagement
after depletion restores food. Existing neural/body state continues throughout.
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


PORTION_J = 100*DOSE_J
RELEASE_ANGLE = WELL_ANGLE/2
TICKS = 5000
ENVIRONMENT_FIELDS = ("pre_step_angle", "remaining_before_j", "accepted_j",
    "remaining_after_j", "portions_loaded", "replenished", "depleted")


class PortionBody(FeedingBody):
    def __init__(self):
        super().__init__()
        self.remaining_j = PORTION_J
        self.portions_loaded = 1
        self.samples = 0
        self.recordings = None

    def offer(self, pump, well):
        if pump:
            raise ValueError("Continuous feeding has no pump route")
        angle = float(self.body.data.qpos[0])
        replenished = self.remaining_j == 0. and angle <= RELEASE_ANGLE
        if replenished:
            self.remaining_j = PORTION_J
            self.portions_loaded += 1
        before = self.remaining_j
        accepted = self.organs.ingest(min(DOSE_J, before)) if well and angle >= WELL_ANGLE and before > 0 else 0.
        self.remaining_j = max(0., before-accepted)
        if self.remaining_j < 1e-12:
            self.remaining_j = 0.
        depleted = before > 0 and self.remaining_j == 0.
        if self.recordings is not None:
            self.recordings[self.samples] = (angle, before, accepted, self.remaining_j,
                self.portions_loaded, replenished, depleted)
        self.samples += 1
        return 0., accepted

    def save(self, path):
        np.savez_compressed(path, state=self.body.state(), energy=self.organs.state(), muscle=self.muscle,
            remaining_j=self.remaining_j, portions_loaded=self.portions_loaded, samples=self.samples,
            portion_j=PORTION_J, release_angle=RELEASE_ANGLE, collection_angle=WELL_ANGLE)

    def restore(self, path):
        super().restore(path)
        with np.load(path, allow_pickle=False) as z:
            if "remaining_j" in z:
                assert float(z["portion_j"]) == PORTION_J
                assert float(z["release_angle"]) == RELEASE_ANGLE
                assert float(z["collection_angle"]) == WELL_ANGLE
                self.remaining_j = float(z["remaining_j"])
                self.portions_loaded = int(z["portions_loaded"])
                self.samples = int(z["samples"])


def settings(base):
    _, manifest, mapping, selected = course.settings(base)
    source = base / "memory-body-context-20260910"
    return source, manifest, mapping, selected, course.read(source / "manifest.json")


def restore(path):
    branch = course.load_checkpoint(path / "state.paula", trusted=True)
    body = PortionBody()
    body.restore(path / "body-state.npz")
    return branch, body


def prepare(base, output):
    if output.exists():
        raise FileExistsError(output)
    source, _, _, _, metadata = settings(base)
    parent = source / "retention-food-body"
    branch, body = restore(parent)
    # Inspect completed omission without treating its finite window as a
    # proof about indefinitely continuing neural dynamics.
    path = source / "action-omission-body"
    saved = np.load(path / "body.npy", mmap_mode="r")
    passive = FeedingBody()
    passive.restore(path / "body-state.npz")
    release_ticks = {}
    for tick in range(1000):
        angle = passive.step(0., 0., 0.)[0]
        for name, threshold in (("collection", WELL_ANGLE), ("replenishment", RELEASE_ANGLE)):
            if name not in release_ticks and angle <= threshold:
                release_ticks[name] = tick+1
    points = [branch.network.network.neurons[metadata["ids"]["prediction"]].postsynaptic_points[s]
        for s in branch.network.network.neurons[metadata["ids"]["prediction"]].prediction_ports]
    original = [p.u_i.info for p in points]
    before = composition.serialized(branch)
    for point, value in zip(points, original, strict=True):
        point.u_i.info = type(value)(0.)
    for point, value in zip(points, original, strict=True):
        point.u_i.info = value
    assert composition.serialized(branch) == before
    output.mkdir(parents=True)
    course.snapshot(branch, body, output / "start-retained")
    for point, value in zip(points, original, strict=True):
        point.u_i.info = type(value)(0.)
    course.snapshot(branch, body, output / "start-erased")
    receipt = dict(source_sha256=sha256(Path(__file__)), composition_sha256=sha256(Path(composition.__file__)),
        parent=str(parent), parent_checkpoint_sha256=sha256(parent / "state.paula"),
        parent_body_sha256=sha256(parent / "body-state.npz"),
        portion_j=PORTION_J, collection_angle=WELL_ANGLE, release_angle=RELEASE_ANGLE, ticks=TICKS,
        environmental_rule="One .8 J portion, offered in the existing .008 J contact quanta at angle >=.04 rad. After depletion, reaching angle <=.02 rad replenishes one portion. No clock-based refill, action phase, neural/body reset, or controller access to inventory/refill flags.",
        bound="Two independent 5000-tick continuous A presentations from the same retained body-context state. Stored predictor weights retained versus zeroed once initially; all other state/RNG exact, same native learning and unit omission brake active. No blank periods, cue changes, gain search or retries.",
        criterion="Repeated acquisition requires intake from at least two physically separated portions, with replenishment caused by withdrawal. Useful continuing feeding additionally requires positive total energy+gut gain over the fixed window. More useful cycles in the retained arm would support a contribution from the stored expectation; a cycle alone would not exclude fixed feedback oscillation or establish a learned sequence.",
        control=dict(original_prediction_weights=[float(v) for v in original], changed_weights=sum(v != 0 for v in original),
            restored_whole_branch_and_rng_exact=True, original_KC_MBON_memory_preserved=True,
            limit="Zeroed predictor weights can learn again during the continuous task. This isolates the initial stored expectation, not all possible learning or all non-memory feedback."),
        feasibility=dict(omission_body_sha256=sha256(path / "body.npy"),
            omission_minimum_angle=float(saved[:, 0].min()), omission_final_angle=float(saved[-1, 0]),
            omission_late_mean_command=float(saved[-150:, 3].mean()),
            corresponding_static_equilibrium_rad=float(saved[-150:, 3].mean()*.2/.15),
            command_balancing_spring_at_contact=WELL_ANGLE*.15/.2,
            zero_neural_output_body_only_diagnostic=dict(initial_body_sha256=sha256(path / "body-state.npz"),
                threshold_crossing_ticks=release_ticks, remaining_muscle_filter_preserved=True,
                scope="Physics feasibility only: no neural simulation or learning result. Existing spring permits disengagement when positive output ceases.")),
        environment_fields=ENVIRONMENT_FIELDS)
    course.write(output / "manifest.json", receipt)


def run(base, output, label):
    if shutil.disk_usage(base).free < 4*1024**3+600*1024**2:
        raise RuntimeError("Shared-volume reserve")
    _, manifest, mapping, selected, metadata = settings(base)
    receipt = course.read(output / "manifest.json")
    assert receipt["source_sha256"] == sha256(Path(__file__))
    assert receipt["composition_sha256"] == sha256(Path(composition.__file__))
    path = output / label
    environment = output / (label+"-environment.npy")
    if path.exists() or environment.exists():
        raise FileExistsError(path)
    branch, body = restore(output / ("start-"+label))
    assert body.samples == 0
    body.recordings = np.lib.format.open_memmap(environment, mode="w+", dtype=np.float64,
        shape=(TICKS, len(ENVIRONMENT_FIELDS)))
    result = course.record(branch, body, manifest, mapping, selected, metadata,
        [("constant_A_finite_portions", "A", True, TICKS)], path)
    body.recordings.flush()
    result.update(environment_sha256=sha256(environment), samples=body.samples,
        portions_loaded=body.portions_loaded, remaining_j=body.remaining_j)
    course.write(output / (label+".json"), result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "retained", "erased"))
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    setup_neuron_logger("CRITICAL")
    with patch.object(course, "composition", composition):
        if args.mode == "prepare":
            prepare(args.base, args.output)
        else:
            run(args.base, args.output, args.mode)
