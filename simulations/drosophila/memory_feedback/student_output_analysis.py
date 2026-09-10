"""Audit active student outputs and their causal inhibition of A feedback."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..connectome import sha256


def read(path):
    return json.loads(Path(path).read_text())


def run(base, output):
    base, output = Path(base), Path(output)
    parent = base/"memory-student-output-paired-20260910"
    m = read(parent/"manifest.json")
    with np.load(parent/"identities.npz") as z:
        selected = z["selected"]
        mapping = dict(zip(z["roots"].tolist(), z["cells"].tolist(), strict=True))
    masks = {role+"_"+cue: np.isin(selected[:, 1], [mapping[r] for r in m["codes"][cue]]) &
        np.isin(selected[:, 3], [mapping[r] for r in m["roles"][role]])
        for role in ("MBON07", "MBON04") for cue in ("A", "B", "C")}
    result = dict(source_sha256=sha256(Path(__file__)), graph_sha256=m["graph_sha256"],
        operating_point=m["assumptions"]["student_output"], courses={}, continuation={}, inhibition={})
    for name in ("paired", "unpaired"):
        p = base/f"memory-student-output-{name}-20260910"
        phases = read(p/"summary.json")
        assert phases[-1]["end"] == 10120
        retention = next(ph["end"] for ph in phases if ph["name"] == "retention")
        q = np.load(p/"release.npy", mmap_mode="r")
        result["courses"][name] = dict(record=p.name, phases=phases,
            retained_release={key: float(q[retention, mask].mean()) for key, mask in masks.items()},
            artifacts={f: sha256(p/f) for f in ("manifest.json", "summary.json", "identities.npz", "retention.paula", "release.npy", "weights.npy", "soma.npy", "body.npy")})
        pp = base/f"memory-output-inhibition-{name}-20260910"
        r = read(pp/"summary.json")
        for k, v in r["probes"].items():
            assert sha256(pp/(k+".npz")) == v["trace_sha256"]
        r["record"] = pp.name
        result["inhibition"][name] = r
    for name in ("intact", "cut", "unpaired"):
        p = base/f"memory-student-output-second-{name}-20260910"
        r = read(p/"summary.json")
        assert r["branch_rng_preserved"] and r["nutrient_j"] == 0 and r["phases"][-1]["end"] == 3720
        for f, digest in r["artifacts"].items():
            assert sha256(p/f) == digest
        q = np.load(p/"release.npy", mmap_mode="r")
        r["record"] = p.name
        r["summary_sha256"] = sha256(p/"summary.json")
        r["retained_release"] = {key: float(q[-1, mask].mean()) for key, mask in masks.items()}
        r["course_spikes"] = {role: sum(ph["spikes"][role] for ph in r["phases"]) for role in m["roles"]}
        result["continuation"][name] = r
    def equal(name, other, file, mask=None):
        a, b = [np.load(base/f"memory-student-output-second-{n}-20260910"/file, mmap_mode="r") for n in (name, other)]
        if mask is not None:
            a, b = a[:, mask], b[:, mask]
        return bool(all(np.array_equal(a[i:i+200], b[i:i+200]) for i in range(0, len(a), 200)))
    result["comparisons"] = dict(intact_unpaired_B_release_exact=equal("intact", "unpaired", "release.npy", masks["MBON04_B"]),
        intact_cut_body_exact=equal("intact", "cut", "body.npy"))
    result["conclusion"] = (
        "Halving native student-output thresholds recruits MBON04 and preserves useful A feeding, but removes A-period student dopamine recruitment. "
        "The output-specific dry-probe intervention tests whether MBON04 inhibition causes that lost teacher response. "
        "This preparation has not established learned B behavior. Student viability and teacher efficacy must coexist in the recurrent circuit; each isolated gain fix is insufficient.")
    output.mkdir(parents=True, exist_ok=True)
    (output/"student-output.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path); p.add_argument("output", type=Path)
    a=p.parse_args(); run(a.records, a.output)
