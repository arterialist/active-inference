"""Export audited PAULA recordings to bounded, offline browser replay chunks.

This is a reader, never a simulator. Raw records retain their original numbers
and queues. Each compressed chunk contains one trial and its preceding state;
the browser keeps only six chunks. Derived membrane values are separate from
the recorded post-reset state. Nothing is temporally averaged or interpolated.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

from .association_analysis import audit
from .composition_analysis import encode, rows

ROOT = Path(__file__).resolve().parents[3]
RUNS = {
    "credit-wide": ("20260908_association_correction1/corrective", "Unregulated credit window"),
    "credit-narrow": ("20260908_association_correction_window1", "Modulated credit window"),
    "feedback-fast": ("20260908_association_confirmation1/case18_seed44_corrective_window", "Fast terminal adaptation"),
    "feedback-slow": ("20260908_association_resolved_confirmation1/case20_seed44_corrective_window_resolved_retro", "Slower, still adapting"),
    "recall": ("20260908_association_timing_recovery1", "Timing transfer and return"),
}


def derived_state(previous, record, manifest, distances):
    """Independent reconstruction following the audited discrete update order."""
    t = record["executed_tick"]
    result = {}
    for nid, n in record["state"]["neurons"].items():
        old = previous["neurons"][nid]
        p = manifest["resolved"][nid]["parameters"]
        arrivals = [a for a in old["dendritic_queue"] if a[0] <= t]
        drive = sum(a[2]*p["delta_decay"]**distances[nid, str(a[3])] for a in arrivals)
        pre_reset = max(-1000., min(1000., old["S"]+(-old["S"]+drive)/p["lambda_param"]))
        elapsed = math.inf if old["t_last_fire"] is None else t-old["t_last_fire"]
        threshold = n["b"] if elapsed <= p["c"] else n["r"]
        if abs(pre_reset) < .005:
            threshold = n["r"]
        expected_spike = pre_reset >= threshold and elapsed >= p["c"]
        assert expected_spike == (n["O"] > 0)
        assert abs(n["S"]-(0. if expected_spike else pre_reset)) < 1e-5
        age = None if n["t_last_fire"] is None else t-n["t_last_fire"]
        result[nid] = {
            "pre_reset": pre_reset, "threshold": threshold, "drive": drive,
            "age": age, "eligible": age is not None and age <= n["t_ref"],
            "arrivals": arrivals,
            "weight_delta": {sid: v[0]-old["synapses"][sid][0] for sid, v in n["synapses"].items()},
        }
    return result


def write_payload(path, key, payload):
    raw = encode(payload).encode()
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    encoded = base64.b64encode(compressed).decode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"PAULA_DATA.receive({json.dumps(key)},\"{encoded}\");\n")
    return len(compressed)


def export_run(source, destination, key, label):
    checked = audit(source)  # Recompute equations and classifications, not just pass flags.
    manifest = json.loads((source / "manifest.json").read_text())
    config = json.loads((source / "config.json").read_text())
    trace_hash = hashlib.file_digest((source / "ticks.jsonl.gz").open("rb"), "sha256").hexdigest()
    distances = {(str(p["neuron_id"]), str(p["synapse_id"])): p["distance_to_hillock"]
                 for p in config["synaptic_points"] if p["type"] == "postsynaptic"}
    stream = rows(source)
    initial = next(stream)["state"]
    previous = initial
    chunk = []
    trial_previous = initial
    trial_counts = Counter()
    trials = []
    bytes_written = 0
    for record in stream:
        enriched = dict(record, derived=derived_state(previous, record, manifest, distances))
        chunk.append(enriched)
        trial_counts.update(nid for nid, n in record["state"]["neurons"].items() if n["O"] > 0)
        trial = manifest["trials"][record["trial"]]
        previous = record["state"]
        if record["executed_tick"] == trial["stop"]-1:
            result = dict(trial, cue_period=trial.get("cue_period", 2), counts=dict(trial_counts))
            if trial["outcome"] is None:
                result["category"] = "silence"
            else:
                good, bad = trial_counts[str(5+trial["outcome"])], trial_counts[str(6-trial["outcome"])]
                result["category"] = ("ambiguous" if good and bad else "correct_only" if good >= 2 and not bad
                                      else "wrong_only" if bad else "silent_or_insufficient")
            result["weights"] = {nid: [previous["neurons"][nid]["synapses"][sid][0] for sid in ("0", "1")]
                                 for nid in ("5", "6")}
            trials.append(result)
            bytes_written += write_payload(destination / key / f"trial-{trial['index']}.js", f"{key}/{trial['index']}",
                                           {"previous": trial_previous, "rows": chunk})
            chunk, trial_counts, trial_previous = [], Counter(), previous
    assert not chunk and len(trials) == len(manifest["trials"])
    assert trace_hash == hashlib.file_digest((source / "ticks.jsonl.gz").open("rb"), "sha256").hexdigest()
    metadata = {"key": key, "label": label, "manifest": manifest, "config": config, "initial": initial,
                "trials": trials, "audit": checked, "trace_sha256": trace_hash,
                "source": str(source.relative_to(ROOT)), "ticks": checked["ticks"]}
    bytes_written += write_payload(destination / key / "meta.js", f"{key}/meta", metadata)
    return {"key": key, "label": label, "ticks": checked["ticks"], "trace_sha256": trace_hash,
            "compressed_bytes": bytes_written}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/embodied-assessment/dynamics/data")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for key, (relative, label) in RUNS.items():
        result = export_run(ROOT / ".live/research" / relative, args.output, key, label)
        results.append(result)
        print(encode(result), flush=True)
    catalog = {"schema": 1, "runs": results,
               "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "semantics": "All completed ticks; exact recorded state after tick; independently derived pre-reset membrane; no interpolation."}
    (args.output / "catalog.json").write_text(encode(catalog)+"\n")


if __name__ == "__main__":
    main()
