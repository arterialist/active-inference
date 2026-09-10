"""Read retained records to decide whether outcome revaluation is identifiable.

This does not execute a continuation, alter a checkpoint, or establish that an
arbitrary recurrent circuit cannot represent an expectation.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from . import feeding, student_interface, terminal_credit
from ..connectome import sha256
from ...active_inference.core.runtime_checkpoint import load_checkpoint
from neuron.neuron import setup_neuron_logger


def run(base, output):
    base, output = Path(base), Path(output)
    parent = base / "memory-coverage-eight-intact-20260910"
    manifest_path = base / "memory-coverage-paired-20260910/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["assumptions"]["student_interface"]["source_sha256"] == sha256(Path(student_interface.__file__))
    with np.load(parent / "identities.npz") as z:
        roots = z["roots"].tolist()
        ids = dict(zip(roots, z["cells"].tolist(), strict=True))
        selected = z["selected"]
    setup_neuron_logger("CRITICAL")
    branch = load_checkpoint(parent / "retention.paula", trusted=True)
    cells = branch.network.network.neurons
    result = dict(
        source_sha256=sha256(Path(__file__)),
        checkpoint_sha256=sha256(parent / "retention.paula"),
        manifest_sha256=sha256(manifest_path),
        identities_sha256=sha256(parent / "identities.npz"),
        neuron_classes=dict(Counter(type(c).__name__ for c in cells.values())),
        configured_prediction_cells=sum(bool(getattr(c, "prediction_ports", ())) for c in cells.values()),
        minimum_eta_post=min(c.params.eta_post for c in cells.values()),
        minimum_eta_retro=min(c.params.eta_retro for c in cells.values()),
        maximum_retained_B_eligibility=max(getattr(cells[ids[r]], "terminal_credit_kc", 0.) for r in manifest["codes"]["B"]),
        source_hashes={Path(mod.__file__).name: sha256(Path(mod.__file__))
                       for mod in (feeding, student_interface, terminal_credit)},
        probes={},
    )
    assert result["minimum_eta_post"] > 0 and result["minimum_eta_retro"] > 0
    del branch, cells
    probes = (
        ("A_food", parent / "probe-A.npz", "B"),
        ("B_dry", parent / "probe-B.npz", "A"),
        ("A_dry_after_hop", base / "memory-reuse-intact-20260910/course/A-dry/trace.npz", "B"),
    )
    for name, path, absent in probes:
        rows = [roots.index(r) for r in manifest["codes"][absent]]
        mask = np.isin(selected[:, 1], [ids[r] for r in manifest["codes"][absent]])
        with np.load(path) as z:
            result["probes"][name] = dict(
                record=str(path.relative_to(base)), trace_sha256=sha256(path),
                absent_code=absent,
                absent_code_output_sum=float(z["soma"][:, rows, 1].sum()),
                absent_code_max_terminal_change=float(np.max(np.abs(z["release"][-1, mask]-z["release"][0, mask]))),
                maximum_selected_receiving_weight_change=float(np.max(np.abs(z["weights"][-1]-z["weights"][0]))),
            )
    result["mechanism_assessment"] = {
        "selected_terminal_rule": "q' = q exp(-eta*d*x), with d and x nonnegative. Missing dopamine halts this depression; it does not reverse its sign. Neural feedback can supply dopamine even without nutrients.",
        "available_body_input": "Actual accepted ingestion drives PAM11/07/08. Cost, energy reserve, outcome expectation and outcome identity are not supplied by student_interface.step. Gut capacity can indirectly affect accepted ingestion.",
        "native_error": "The base receiving error is incoming release minus receiving coefficient, with a timing-dependent direction. Its name does not establish expected-minus-received bodily consequence.",
        "expression_routes": "KC/MBON inhibition and SMP108 drive already let terminal memory alter action. Receiving weights, intrinsic state and shared feedback remain adaptive, so unchanged B release does not imply unchanged B expression.",
        "identifiability": "No dedicated outcome predictor/comparison is configured. These finite traces show no cross-cue reactivation at the tested states. They do not prove that every possible recurrent state lacks expectation coding.",
    }
    result["decision"] = "Do not run A-only omission as a test of worse-than-expected updating in the present preparation. First establish an independently verified neural outcome prediction and opponent comparison, using existing PAULA components. Preserve the accepted A-to-B result."
    repository = Path(__file__).resolve().parents[3]
    sources = (
        "simulations/active_inference/components/body/energy_budget.py",
        "simulations/active_inference/components/learning/predictive_bridge.py",
        "simulations/active_inference/components/learning/temporal_verification.py",
        "simulations/active_inference/components/arbitration/energy_feedback.py",
        "../neuron-model/neuron/neuron.py",
        "../neuron-model/neuron/extensions/graded.py",
        "../neuron-model/neuron/extensions/experimental/predictive_receptor.py",
    )
    result["assessed_component_hashes"] = {name: sha256(repository / name) for name in sources}
    result["next_composition"] = {
        "minimum_information_circuit": "One independent ingestion afferent, one cue-driven PredictiveReceptorNeuron and two ordinary graded comparison cells. Compare observed/predicted release with opposing dendritic signs. Their releases reach the predictor's existing signed error receptors. Context arrives from actual cue KCs without cue labels in the rule.",
        "independent_evidence": "Drive the added afferent only with actual accepted ingestion. Existing PAM activity is unsuitable as independent observation because learned feedback also drives it.",
        "constraints": "Keep terminal_credit and the accepted circuit intact. Reuse existing predictor dynamics with prediction_boost=0; do not inherit the bridge factory's 499 boost, supply host-computed errors, or tune gains. Added cells/edges are an explicit engineered extension, not measured FlyWire anatomy.",
        "timing": "Start with an explicitly defined prediction of the ongoing nutrient-contact rate. Match comparison-path delays and audit onset/offset transients. Do not call a cue-onset transient a failed future reward; delayed-outcome expectation requires separate temporal verification.",
        "expression_after_verification": "Only after the predictor distinguishes learned expectation plus no contact from unlearned no contact, test its learned cue-conditioned output as a declared neural input to the existing action pathway. This supports cue-local expression updating; A-only transfer to B still requires demonstrated retrieval or a shared outcome representation.",
        "body_criterion": "Nutrient prediction is an information prerequisite, not net-utility learning. Judge any later motor coupling by energy plus gut change and demand, alongside intake and retained A/B function. Sensitivity to energetic cost would additionally require an independent physical cost afferent; a fixed reserve brake does not supply outcome expectation.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.base, args.output)
