"""TOPO_LIVE — regenerate the wiring diagram to match the CONFIGURATION actually running.

The 3D brain page ships a payload that was baked once by build_brain_page.py from a single default
build. Structural gates (trise, vac, accum, w_peg, pb_eb_bridge, w_lgi) build or omit whole POPULATIONS, so with a
static payload the diagram silently misrepresents the network: cells exist in the sim that the picture
has never heard of, and vice versa.

This module rebuilds that payload on demand:

    export_brain.py (subprocess, AIF_KWARGS)  ->  topology JSON  ->  pack()  ->  the page's wire format

It is a SUBPROCESS on purpose. export_brain.py is a top-level script whose layout solver, region
predicates and anatomical seats all run at import; importing it a second time in-process would rebuild
a second agent inside the server and fight the worker thread for the GL context.

The pack format is byte-identical to build_brain_page.py's, because the page's parser is the contract.
The ONE deliberate difference: build_brain_page asserts that its hand-written SYSTEMS legend covers
every region, which is correct for a curated artifact and fatal for a live rebuild -- a newly enabled
population would crash the regeneration instead of appearing. Here an unknown region is placed in a
system by prefix, and anything still unmatched lands in "Unassigned" rather than raising.
"""
import base64, json, os, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Which system a region belongs to. Regions added later (opt-in populations) are matched here so the
# legend stays meaningful; anything unmatched falls through to Unassigned.
SYSTEM_OF = {
    "retina PR": "Visual cortex", "chroma": "Visual cortex", "ON/OFF": "Visual cortex",
    "V1": "Visual cortex", "V2": "Visual cortex", "EMD motion": "Visual cortex",
    "HS wide-field": "Visual cortex", "AZ": "Visual cortex", "salience": "Visual cortex",
    "visual ring": "Visual cortex",
    "compass ring": "Central complex", "compass GI": "Central complex",
    "Delta-7 inhibitors": "Central complex", "PG readout": "Central complex",
    "P-EN shift": "Central complex", "CD speed cells": "Central complex",
    "CPU4 ladders": "Central complex", "CPU1 / opponent": "Central complex",
    "P-EG maintenance": "Central complex",
    "PB update relays": "Central complex", "PB maintenance tracts": "Central complex",
    "PB phase-gated update": "Central complex", "PB phase clock": "Central complex",
    "phase vestibular afferents": "Central complex",
    "PI analog (XACC/YACC)": "Central complex", "PI accumulator": "Central complex",
    "PI accum inhibitor": "Central complex", "PI accum (magnitude)": "Central complex",
    "antennal lobe": "Mushroom body", "PN": "Mushroom body", "Kenyon cells": "Mushroom body",
    "APL": "Mushroom body", "MBON": "Mushroom body", "AVOID": "Mushroom body",
    "vAC colour PN": "Mushroom body", "vAC Kenyon cells": "Mushroom body", "vAC APL": "Mushroom body",
    "odour sensors": "Chemosensation", "toxin sensors": "Chemosensation",
    "sting latch": "Chemosensation", "toxin pool": "Chemosensation",
    "toxin RISE trend": "Chemosensation",
    "obstacle range": "Mechanosensation", "obstacle onset": "Mechanosensation",
    "obstacle reflex": "Motor plant",
    "CPG": "Motor plant", "engine": "Motor plant", "motor relays": "Motor plant",
    "steering": "Motor plant", "RISE trend": "Motor plant", "muscles": "Motor plant",
    "belief core": "Active inference", "prediction error": "Active inference",
    "hunger": "Active inference", "uncertainty": "Active inference",
    "arbiter modes": "Active inference", "sleep mode": "Active inference",
    "metabolic afferents": "Active inference",
}
SYSTEM_DESC = {
    "Visual cortex": "optic lobe: photoreceptors -> ON/OFF -> V1 -> V2, plus colour, motion and salience",
    "Central complex": "heading ring attractor, angular-velocity shift and the path-integration home vector",
    "Mushroom body": "learning: antennal lobe -> PN -> Kenyon cells -> APL -> MBON (plus the visual calyx)",
    "Chemosensation": "bilateral odour and toxin receptors, the sting reflex latch and the toxin rise detector",
    "Motor plant": "pacemakers, gating relays and the graded muscles that touch the body",
    "Active inference": "belief core, prediction error, uncertainty, interoception and PAULA mode arbitration",
    "Mechanosensation": "bilateral physical whisker/range transducers and delayed obstacle onset",
    "Unassigned": "cells that belong to no named area",
}
SYSTEM_ORDER = ["Visual cortex", "Central complex", "Mushroom body", "Chemosensation",
                "Mechanosensation", "Motor plant", "Active inference", "Unassigned"]


# Regions that only exist when a STRUCTURAL GATE is on. The legend marks these so it is obvious which
# circuits are optional extras rather than part of the baseline brain -- otherwise enabling a gate just
# makes unexplained new areas appear in the hierarchy.
OPTIONAL_REGION = {
    "toxin pool": "trise", "toxin RISE trend": "trise",
    "vAC colour PN": "vac", "vAC Kenyon cells": "vac", "vAC APL": "vac",
    "PI accumulator": "accum", "PI accum inhibitor": "accum", "PI accum (magnitude)": "accum",
    "P-EG maintenance": "w_peg",
    "PB update relays": "pb_eb_bridge", "PB maintenance tracts": "pb_eb_bridge",
    "PB phase-gated update": "pb_phase_update", "PB phase clock": "pb_phase_update",
    "phase vestibular afferents": "pb_phase_update",
    "sleep mode": "metabolic_sleep", "metabolic afferents": "metabolic_sleep",
    "obstacle range": "obstacle_detour", "obstacle onset": "obstacle_detour",
    "obstacle reflex": "obstacle_detour",
}


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def pack(d):
    """topology dict -> the exact wire format brain_live.html's main() parses."""
    regions = d["regions"]
    # group regions into systems; unknown -> Unassigned (build_brain_page asserts here instead)
    subs = {s: [] for s in SYSTEM_ORDER}
    for r in regions:
        subs.setdefault(SYSTEM_OF.get(r["name"], "Unassigned"), []).append(r["name"])
    d["systems"] = [{"name": s, "d": SYSTEM_DESC.get(s, ""),
                     "subs": [{"name": s, "regs": subs[s]}]}
                    for s in SYSTEM_ORDER if subs.get(s)]

    REGI = {r["name"]: i for i, r in enumerate(regions)}
    GRPI = {r["name"]: {g["name"]: j for j, g in enumerate(r["g"])} for r in regions}
    nodes, edges = d["nodes"], d["edges"]
    POS = 600.0
    pal = sorted({e[2] for e in edges})
    DFLT = pal.index(1.0) if 1.0 in pal else 0
    order = sorted(range(len(edges)), key=lambda k: (edges[k][0], edges[k][1]))
    eu = [edges[k][0] for k in order]
    ev = [edges[k][1] for k in order]
    ed = [pal.index(edges[k][2]) for k in order]
    cnt = np.bincount(np.array(eu, dtype=int), minlength=len(nodes)).astype("<u2")
    assert cnt.sum() == len(edges)
    nz = [i for i, x in enumerate(ed) if x != DFLT]
    ids = np.array([n["id"] for n in nodes], dtype=np.int64)
    dif = np.diff(ids)
    assert dif.min() > 0 and dif.max() < 65536, "node ids must be sorted and gap < 65536"
    assert len(nodes) < 65536 and len(pal) < 256
    return {
        "regions": [dict({"name": r["name"], "c": r["c"], "n": r["n"],
                          "g": [{"name": g["name"], "n": g["n"]} for g in r["g"]]},
                         **({"opt": OPTIONAL_REGION[r["name"]]}
                            if r["name"] in OPTIONAL_REGION else {}))
                    for r in regions],
        "systems": d["systems"],
        "nn": len(nodes), "ne": len(edges), "pos": POS, "pal": pal, "dflt": DFLT,
        "id0": int(ids[0]),
        "idd": _b64(dif.astype("<u2")),
        "reg": _b64(np.array([REGI[n["r"]] for n in nodes], dtype="<u1")),
        "grp": _b64(np.array([GRPI[n["r"]][n["g"]] for n in nodes], dtype="<u1")),
        "pa":  _b64(np.round(np.array([n["p"] for n in nodes]) * POS).astype("<i2")),
        "pw":  _b64(np.round(np.array([n["q"] for n in nodes]) * POS).astype("<i2")),
        "cnt": _b64(cnt),
        "ev":  _b64(np.array(ev, dtype="<u2")),
        "edx": _b64(np.array(nz, dtype="<u2")),
        "edp": _b64(np.array([ed[i] for i in nz], dtype="<u1")),
    }


def regenerate(params, iters=200, timeout=900, version="v1"):
    """Run export_brain with `params`, then pack. Returns (payload_dict, info_str).

    `iters` trades layout polish for turnaround: the shipped artifact uses 900, a live rebuild does
    not need that -- the point is that the right CELLS are present and wired, not a settled aesthetic.
    """
    kw = {k: v for k, v in (params or {}).items()}
    # bool gates must cross the process boundary as real bools, not 1.0
    for b in ("trise", "vac", "accum", "sh_bank", "d7", "pb_eb_bridge", "pb_phase_update", "metabolic_sleep"):
        if b in kw:
            kw[b] = bool(round(kw[b]))
    if "d_emd" in kw:
        kw["d_emd"] = int(kw["d_emd"])
    kw.pop("seed", None)

    out = os.path.join(tempfile.gettempdir(), f"topo_live_{os.getpid()}.json")
    env = dict(os.environ, AIF_KWARGS=json.dumps(kw), AIF_TOPO_OUT=out,
               AIF_TOPO_ITERS=str(iters), AIF_VERSION=str(version))
    t0 = time.time()
    r = subprocess.run([sys.executable, "export_brain.py"], cwd=HERE, env=env,
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"export_brain failed: {(r.stderr or r.stdout)[-400:]}")
    d = json.load(open(out))
    try:
        os.unlink(out)
    except OSError:
        pass
    payload = pack(d)
    return payload, f"{payload['nn']} neurons, {payload['ne']} edges, {len(payload['regions'])} regions in {time.time()-t0:.0f}s"


if __name__ == "__main__":
    p, info = regenerate(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}, iters=120)
    print(info)
    print("systems:", [s["name"] for s in p["systems"]])
