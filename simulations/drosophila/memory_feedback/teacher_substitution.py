"""Remove only stored alpha1 A release memory before continuing B-before-A.

The paired receiver keeps all receiving weights, other terminal coefficients,
traces, queues, RNG and body state. Only alpha1 A terminal release comes from
the matched unpaired checkpoint. Adaptation continues during acquisition and
expression. The intervention is a causal diagnostic, not a learning mechanism.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from . import continuing_course
from ..connectome import sha256
from ...active_inference.core.runtime_checkpoint import load_checkpoint, save_checkpoint
from neuron.neuron import setup_neuron_logger


def run(receiver, donor, output):
    setup_neuron_logger("CRITICAL")
    receiver, donor, output = map(Path, (receiver, donor, output))
    if output.exists():
        raise FileExistsError(output)
    m, dm = [json.loads((p/"manifest.json").read_text()) for p in (receiver, donor)]
    for key in ("graph_sha256", "factor", "motor_gain", "well_angle", "codes", "roles", "assumptions"):
        if m[key] != dm[key]:
            raise ValueError("Unmatched teacher-memory donors: "+key)
    with np.load(receiver/"identities.npz") as z:
        selected = z["selected"]
        mapping = dict(zip(z["roots"].tolist(), z["cells"].tolist(), strict=True))
    with np.load(donor/"identities.npz") as z:
        if not np.array_equal(selected, z["selected"]):
            raise ValueError("Different memory terminals")
    mask = np.isin(selected[:, 1], [mapping[r] for r in m["codes"]["A"]]) & np.isin(
        selected[:, 3], [mapping[r] for r in m["roles"]["MBON07"]])
    if not np.any(mask):
        raise ValueError("No alpha1 A terminals")
    branch = load_checkpoint(receiver/"retention.paula", trusted=True)
    source = load_checkpoint(donor/"retention.paula", trusted=True)
    changes = []
    for row, pre, terminal, post, sid in selected[mask]:
        target = branch.network.network.neurons[int(pre)].presynaptic_points[int(terminal)]
        value = source.network.network.neurons[int(pre)].presynaptic_points[int(terminal)].u_o.info
        changes.append(dict(source_row=int(row), pre=int(pre), terminal=int(terminal),
                            before=float(target.u_o.info), after=float(value)))
        target.u_o.info = float(value)
    del source
    target_dir = output/"receiver"
    target_dir.mkdir(parents=True)
    intervention = dict(receiver=receiver.name, donor=donor.name, compartment="alpha1", cue="A",
        changes=changes, changed_receiving=0,
        receiver_checkpoint_sha256=sha256(receiver/"retention.paula"),
        donor_checkpoint_sha256=sha256(donor/"retention.paula"), source_sha256=sha256(Path(__file__)))
    m["teacher_substitution"] = intervention
    (target_dir/"manifest.json").write_text(json.dumps(m, indent=2)+"\n")
    for name in ("identities.npz", "retention-body.npz"):
        shutil.copyfile(receiver/name, target_dir/name)
    save_checkpoint(branch, target_dir/"retention.paula", sources=[__file__])
    del branch
    result = continuing_course.run(target_dir, output/"continuation")
    intervention["substituted_checkpoint_sha256"] = result["input_checkpoint_sha256"]
    intervention["continuation_summary_sha256"] = sha256(output/"continuation"/"summary.json")
    (output/"intervention.json").write_text(json.dumps(intervention, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receiver", type=Path); p.add_argument("donor", type=Path); p.add_argument("output", type=Path)
    a=p.parse_args(); run(a.receiver, a.donor, a.output)
