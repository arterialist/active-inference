"""Build-time routing of PAULA activity feedback, with a matched control.

No runtime controller, decoded activity target or stimulus identity. The same
detector populations observe the same regions in both conditions. Only the
alignment of their outgoing modulatory projections differs. The shuffled
control preserves each source's fan-out and each target's fan-in.
"""
from copy import deepcopy

import numpy as np


def regional_regulation(config, groups, edges, *, seed=11, routing="regional"):
    if routing not in ("regional", "shuffled"):
        raise ValueError("Use regional or shuffled routing")
    result, result_edges = deepcopy(config), deepcopy(edges)
    rng = np.random.default_rng(seed+8741)
    roles = ("visual_core", "tactile_core", "upper_core")
    regulators = list(groups["activity_regulator"])
    if len(regulators) < 6:
        raise ValueError("Need at least two regulator cells per region")
    scopes = {role: list(map(int, block)) for role, block in zip(roles, np.array_split(regulators, 3))}
    detector_scope = {nid: role for role, ids in scopes.items() for nid in ids}
    core_scope = {nid: role for role in roles for nid in groups[role]}
    by_port = {(c["target_neuron"], c["target_synapse"]): c for c in result["connections"]}
    observe_edges, mod_edges = {}, {}
    for i, (_, tgt, _, family, present) in enumerate(result_edges):
        if not present:
            continue
        if family == "observed_activity":
            observe_edges.setdefault(tgt, []).append(i)
        elif family == "excitability_modulation":
            mod_edges.setdefault(tgt, []).append(i)
    for tgt, positions in observe_edges.items():
        source = rng.choice(groups[detector_scope[tgt]], len(positions), replace=False)
        for i, src in zip(positions, source):
            edge = result_edges[i]
            edge[0] = int(src)
            by_port[(edge[1], edge[2])]["source_neuron"] = int(src)
    mod_positions = []
    for tgt, positions in mod_edges.items():
        source = rng.choice(scopes[core_scope[tgt]], len(positions), replace=False)
        for i, src in zip(positions, source):
            result_edges[i][0] = int(src)
            mod_positions.append(i)
    swaps = 0
    if routing == "shuffled":
        # Degree-preserving swaps randomize region alignment without changing
        # signal gain, input count, output count, ports, delays or detector input.
        occupied = {(result_edges[i][0], result_edges[i][1]) for i in mod_positions}
        for _ in range(20*len(mod_positions)):
            i, j = map(int, rng.choice(mod_positions, 2, replace=False))
            a, b = result_edges[i], result_edges[j]
            if a[0] == b[0] or a[1] == b[1] or (a[0], b[1]) in occupied or (b[0], a[1]) in occupied:
                continue
            occupied.remove((a[0], a[1]))
            occupied.remove((b[0], b[1]))
            a[0], b[0] = b[0], a[0]
            occupied.update(((a[0], a[1]), (b[0], b[1])))
            swaps += 1
    for i in mod_positions:
        src, tgt, sid, _, _ = result_edges[i]
        by_port[(tgt, sid)]["source_neuron"] = src
    for n in result["neurons"]:
        nid = n["id"]
        if nid in detector_scope:
            n["metadata"]["observed_region"] = detector_scope[nid]
            n["params"]["r_base"], n["params"]["b_base"] = .2, .45
        if nid in core_scope:
            n["params"]["w_r"][1] = 12.
            n["params"]["w_b"][1] = 12.
    alignment = sum(detector_scope[result_edges[i][0]] == core_scope[result_edges[i][1]] for i in mod_positions)/len(mod_positions)
    result["metadata"] = {**result.get("metadata", {}), "regulation_routing": routing}
    return result, result_edges, {"routing": routing, "scopes": scopes,
        "aligned_projection_fraction": alignment, "degree_preserving_swaps": swaps,
        "detector_thresholds": [.2, .45], "M1_threshold_gain": 12.}
