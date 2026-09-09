"""Build-time current-capacity matching, not an online firing-rate controller.

For each selected target, match the magnitude of inhibitory and excitatory
mean current under each source's maximal sustained firing rate, 1/c. Include
initial terminal amplitude, postsynaptic throughput and dendritic attenuation.
This is a declared operating-regime intervention, not a biological plasticity
rule, a guarantee of actual E/I balance, or evidence of associative memory.
All runtime plasticity, thresholds, terminals, wiring and delays are unchanged.
"""
from copy import deepcopy
import math


def match_inhibitory_capacity(config, targets):
    result = deepcopy(config)
    neurons = {n["id"]: n for n in result["neurons"]}
    points = {(p["neuron_id"], p["synapse_id"]): p for p in result["synaptic_points"]
              if p["type"] == "postsynaptic"}
    terminals = {(p["neuron_id"], p["terminal_id"]): p for p in result["synaptic_points"]
                 if p["type"] == "presynaptic"}
    targets = set(targets)
    if not targets or not targets <= neurons.keys():
        raise ValueError("Select existing target neurons")
    if any(e["target_neuron"] in targets for e in result["external_inputs"]):
        raise ValueError("External current capacity is not specified")
    grouped = {n: [] for n in targets}
    seen = set()
    for c in result["connections"]:
        target, sid = c["target_neuron"], c["target_synapse"]
        if target not in targets:
            continue
        if (target, sid) in seen:
            raise ValueError("Multiple sources per input need explicit capacity semantics")
        seen.add((target, sid))
        p = points[target, sid]
        if p["u_i"]["plast"] != 0:
            raise ValueError("This intervention only rescales pure info throughput")
        source = neurons[c["source_neuron"]]
        release = terminals[c["source_neuron"], c["source_terminal"]]["u_o"]["info"]
        cooldown = source["params"]["c"]
        if release < 0 or cooldown <= 0:
            raise ValueError("Need nonnegative release and positive source cooldown")
        attenuation = neurons[target]["params"]["delta_decay"] ** p["distance_to_hillock"]
        capacity = release * p["u_i"]["info"] * attenuation / cooldown
        if not math.isfinite(capacity):
            raise ValueError("Nonfinite current capacity")
        grouped[target].append((p, capacity))
    rows = []
    for target in sorted(targets):
        incoming = grouped[target]
        excitatory = sum(v for _, v in incoming if v > 0)
        inhibitory = -sum(v for _, v in incoming if v < 0)
        if excitatory <= 0 or inhibitory <= 0:
            raise ValueError("Both current signs must have positive capacity")
        scale = excitatory / inhibitory
        changed = []
        for p, capacity in incoming:
            if capacity >= 0:
                continue
            before = p["u_i"]["info"]
            after = before*scale
            cap = neurons[target]["metadata"].get("plasticity_magnitude_cap", 10.)
            if abs(after) > cap:
                raise ValueError("Matched capacity exceeds the declared plasticity bound")
            p["u_i"]["info"] = after
            changed.append([p["synapse_id"], before, after])
        rows.append({"target": target, "excitatory_capacity": excitatory,
                     "inhibitory_capacity_before": inhibitory, "scale": scale,
                     "changed_ports": changed})
    return result, {"condition": "matched_initial_capacity", "targets": rows,
                    "limits": "Equal maximal mean capacities, not equal actual currents or pointwise bounds. Native adaptation can change the balance. No runtime gain controller."}
