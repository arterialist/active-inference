"""Declared input-subset controls over the unchanged selected ALPN population.

Model sign is not a measured receptor effect. A VP label is not a complete
modality annotation. These panels diagnose artificial stimulation, not odors.
"""
from __future__ import annotations

import re
import numpy as np

PANELS = ("all_selected", "without_inhibitory_drive", "without_vp_drive",
          "without_inhibitory_or_vp_drive")
VP_LABEL = re.compile(r"^VP[1-5][dlm]?[+_]")


def make_panel(graph, name="all_selected"):
    if name not in PANELS:
        raise ValueError(f"Unknown PN input panel: {name}")
    catalog = []
    for root in graph.selected:
        a = graph.nodes[root]["annotation"]
        if a["cell_class"] != "ALPN":
            continue
        signs = np.unique(graph.edges[graph.edges[:, 0] == int(root), 5])
        if len(signs) != 1 or signs[0] not in (-1, 1):
            raise ValueError(f"PN needs a consistent declared source-model sign: {root}")
        vp = any(VP_LABEL.match(label.strip()) for label in a["hemibrain_type"].split(","))
        inhibitory = signs[0] == -1
        excluded = ((name in {"without_inhibitory_drive", "without_inhibitory_or_vp_drive"} and inhibitory)
                    or (name in {"without_vp_drive", "without_inhibitory_or_vp_drive"} and vp))
        catalog.append({"root": root, "model_sign": int(signs[0]), "explicit_vp_label": vp,
            "externally_driven": not excluded,
            **{key: a.get(key, "") for key in ("hemibrain_type", "cell_type", "cell_sub_class",
                                              "known_nt", "known_nt_source", "top_nt", "top_nt_conf")}})
    if not catalog:
        raise ValueError("Input panel needs selected ALPN providers")
    return {"schema": "flywire-pn-input-panel-v1", "name": name,
        "drive_roots": [r["root"] for r in catalog if r["externally_driven"]],
        "excluded_roots": [r["root"] for r in catalog if not r["externally_driven"]],
        "catalog": catalog,
        "rules": {"inhibitory": "Original pair table model sign -1, not inferred from firing or fitted to this result",
                  "vp": "At least one comma-separated hemibrain label matches " + VP_LABEL.pattern,
                  "dose": "Same current per driven cell; total dose is not renormalized",
                  "scope": "Only dedicated experimental current changes. Every selected cell and all graph connections remain."},
        "limitations": ["Input-subset diagnostic, not an odor or a thermal/humidity stimulus",
            "VP labels can identify mixed-input cells; absence of VP is not proof of exclusive olfactory input",
            "Removing experimental drive does not silence a cell recruited through the neural network"]}


def recorded_drive_rows(protocol, roots):
    """Resolve old all-PN records and explicitly selected new records."""
    all_rows = protocol["pn_rows"]
    rows = protocol.get("driven_pn_rows", all_rows)
    if (not isinstance(rows, list) or any(type(i) is not int for i in rows)
            or rows != sorted(set(rows)) or not set(rows) <= set(all_rows)):
        raise ValueError("Invalid driven PN rows")
    panel = protocol.get("pn_drive_panel")
    if "driven_pn_rows" in protocol and panel is None:
        raise ValueError("Selected PN drive lacks its declared panel")
    if panel is not None:
        if (panel.get("schema") != "flywire-pn-input-panel-v1" or panel.get("name") not in PANELS
                or panel["drive_roots"] != np.asarray(roots)[rows].tolist()
                or [r["root"] for r in panel["catalog"]] != np.asarray(roots)[all_rows].tolist()):
            raise ValueError("PN input panel and recorded columns disagree")
        # Re-evaluate the declared selection without treating its name as proof.
        for entry in panel["catalog"]:
            if type(entry["model_sign"]) is not int or entry["model_sign"] not in (-1, 1):
                raise ValueError("Invalid panel source sign")
            vp = any(VP_LABEL.match(label.strip()) for label in entry["hemibrain_type"].split(","))
            excluded = ((panel["name"] in {"without_inhibitory_drive", "without_inhibitory_or_vp_drive"} and entry["model_sign"] == -1)
                        or (panel["name"] in {"without_vp_drive", "without_inhibitory_or_vp_drive"} and vp))
            if (entry["explicit_vp_label"] != vp or entry["externally_driven"] != (not excluded)
                    or (entry["root"] in panel["drive_roots"]) != (not excluded)
                    or (entry["root"] in panel["excluded_roots"]) != excluded):
                raise ValueError("PN panel selection contradicts its declared rule")
    return np.asarray(rows, dtype=int)
