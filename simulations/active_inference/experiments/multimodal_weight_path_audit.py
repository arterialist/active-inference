"""Data-only verification and readout of selective information-weight probes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .multimodal_pairing_audit import contrast, readout


def audit(paired, swapped):
    paths = [Path(paired), Path(swapped)]
    summaries = [json.loads((path/"summary.json").read_text()) for path in paths]
    if [s["mapping"] for s in summaries] != ["paired", "swapped"]:
        raise ValueError("Supply paired and swapped probes in that order")
    recordings = [Path(s["source_recording"]) for s in summaries]
    manifests = [json.loads((r/"manifest.json").read_text()) for r in recordings]
    checks = {"same_seed": summaries[0]["seed"] == summaries[1]["seed"],
              "same_config": (recordings[0]/"config.json").read_bytes() == (recordings[1]/"config.json").read_bytes(),
              "same_selected_ports": summaries[0]["selected_ports"] == summaries[1]["selected_ports"],
              "same_index": summaries[0]["synapse_index"] == summaries[1]["synapse_index"],
              "same_runtime_sources": manifests[0]["source_hashes"] == manifests[1]["source_hashes"],
              "same_probe_source": summaries[0]["observer_source_sha256"] == summaries[1]["observer_source_sha256"],
              "source_records_unchanged": True, "exact_full_replay": True,
              "only_declared_weights_changed": True, "finite_states": True,
              "initial_visual_probes_identical": True,
              "donor_records_unchanged": True}
    donor_mode = [s.get("donor") is not None for s in summaries]
    if donor_mode[0] != donor_mode[1]:
        raise ValueError("Both interventions must use the same donor/non-donor design")
    states = []
    for path, summary, recording in zip(paths, summaries, recordings):
        for name, expected in summary["source_files_sha256"].items():
            checks["source_records_unchanged"] &= hashlib.sha256((recording/name).read_bytes()).hexdigest() == expected
        selected = {tuple(x) for x in summary["selected_ports"]}
        mask = np.array([tuple(port) in selected for port in summary["synapse_index"]])
        with np.load(recording/"parameters.npz") as raw:
            initial, learned = raw["initial_info"], raw["learned_info"]
        expected_weights = {"learned_all": learned,
                            "learned_selected_only": np.where(mask, learned, initial),
                            "learned_remainder_only": np.where(mask, initial, learned)}
        if donor_mode[0]:
            donor = Path(summary["donor"]["recording"])
            checks["donor_records_unchanged"] &= hashlib.sha256((donor/"parameters.npz").read_bytes()).hexdigest() == summary["donor"]["parameters_sha256"]
            other = recordings[1] if recording == recordings[0] else recordings[0]
            checks["donor_records_unchanged"] &= donor.resolve() == other.resolve()
            with np.load(donor/"parameters.npz") as raw:
                donor_weights = raw["learned_info"]
            expected_weights = {"learned_all": learned,
                                "selected_from_donor": np.where(mask, donor_weights, learned),
                                "remainder_from_donor": np.where(mask, learned, donor_weights)}
        current = {}
        for condition, expected in expected_weights.items():
            current[condition] = []
            for clip in (0, 1):
                with np.load(path/f"{condition}-visual-{clip}.npz") as raw:
                    cells, final = raw["cells"], raw["incoming_info_after"]
                    checks["only_declared_weights_changed"] &= np.array_equal(raw["incoming_info_before"], expected)
                checks["finite_states"] &= np.isfinite(cells).all() and np.isfinite(final).all()
                if condition == "learned_all":
                    with np.load(recording/f"probe-learned_info-visual-{clip}.npz") as ref:
                        checks["exact_full_replay"] &= np.array_equal(cells, ref["cells"]) and np.array_equal(final, ref["incoming_info_after"])
                current[condition].append(cells)
        current["initial"] = []
        for clip in (0, 1):
            with np.load(recording/f"probe-initial-visual-{clip}.npz") as raw:
                current["initial"].append(raw["cells"])
        states.append(current)
    checks["initial_visual_probes_identical"] = all(np.array_equal(a, b) for a, b in zip(states[0]["initial"], states[1]["initial"]))
    if donor_mode[0]:
        checks["complementary_hybrids_bit_identical"] = all(
            np.array_equal(a, b) for first, second in (("selected_from_donor", "remainder_from_donor"),
                                                      ("remainder_from_donor", "selected_from_donor"))
            for a, b in zip(states[0][first], states[1][second]))
    # JSON bool, not NumPy's scalar type.
    checks = {key: bool(value) for key, value in checks.items()}
    references = []
    for clip in (0, 1):
        with np.load(recordings[0]/f"probe-initial-audio-{clip}.npz") as raw:
            references.append(raw["cells"])
    results = {}
    for role in ("tactile_core", "upper_core"):
        ids = manifests[0]["groups"][role]
        results[role] = {}
        for kind in ("mean_rate", "population_centered_rate", "state_covariance", "population_centered_covariance"):
            refs = np.array([readout(cells, ids, kind) for cells in references])
            contrasts = [{key: contrast(np.array([readout(cells, ids, kind) for cells in samples]), refs)
                          for key, samples in state.items()} for state in states]
            differences = {condition: None if any(c[condition] is None for c in contrasts)
                           else contrasts[0][condition]-contrasts[1][condition] for condition in states[0]}
            interaction = None
            if not donor_mode[0] and all(value is not None for value in differences.values()):
                interaction = (differences["learned_all"] - differences["learned_selected_only"] -
                               differences["learned_remainder_only"] + differences["initial"])
            results[role][kind] = {"paired_contrasts": contrasts[0], "swapped_contrasts": contrasts[1],
                                  "paired_minus_swapped": differences, "factorial_interaction": interaction}
    return {"structural_checks": checks, "structurally_valid": all(checks.values()), "results": results,
            "design": "opposite-assignment weight exchange" if donor_mode[0] else "learned/initial factorial",
            "limits": "These are descriptive nonlinear weight-intervention effects. A pathway's stored changes can interact with changes elsewhere. No statistical, semantic or consciousness claim follows. The source association audit must also be consulted."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--swapped", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(audit(args.paired, args.swapped), allow_nan=False, indent=2)+"\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(result)
    print(result)
