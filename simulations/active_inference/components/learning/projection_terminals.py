"""Build-time separation of native PAULA output-terminal adaptation.

One shared terminal pools retrograde updates from every attached target. This
candidate allocates a terminal per projection family, with identical initial
release vectors, target weights and delays. A shuffled control preserves each
terminal's fan-out while mixing which projection families share adaptation.

No new neuron rule, runtime routing, label or stimulus-dependent wiring. This
is a structural hypothesis, not an accepted associative-memory component.
Biological motivation: Reyes et al. 1998, doi:10.1038/1092, report target-specific
presynaptic behavior along one axon. This grouping is not their cellular model.
"""
from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np


def projection_terminals(config, edges, *, mode="shared", seed=11):
    if mode not in ("shared", "family", "shuffled"):
        raise ValueError(mode)
    result = deepcopy(config)
    original_count = sum(s["type"] == "presynaptic" for s in config["synaptic_points"])
    if mode == "shared":
        return result, {"mode": mode, "terminals_before": original_count,
                        "terminals_after": original_count, "added_terminals": 0}
    families = {}
    for src, tgt, sid, family, present in edges:
        if present:
            key = (src, tgt, sid)
            if key in families:
                raise ValueError("Ambiguous edge family")
            families[key] = family
    terminals, used = {}, defaultdict(set)
    for point in result["synaptic_points"]:
        nid = point["neuron_id"]
        sid = point["terminal_id"] if point["type"] == "presynaptic" else point["synapse_id"]
        if sid in used[nid]:
            raise ValueError("Input/output distance IDs already collide")
        used[nid].add(sid)
        if point["type"] == "presynaptic":
            terminals[nid, sid] = point
    groups = defaultdict(list)
    for i, connection in enumerate(result["connections"]):
        src, tid = connection["source_neuron"], connection["source_terminal"]
        family = families[src, connection["target_neuron"], connection["target_synapse"]]
        groups[src, tid].append((i, family))
    rng = np.random.default_rng(seed+9071)
    added, routing = [], []
    for (src, old_tid), members in groups.items():
        labels = sorted({family for _, family in members})
        ids = {labels[0]: old_tid}
        for family in labels[1:]:
            new_tid = next((i for i in range(900, 4096) if i not in used[src]), None)
            if new_tid is None:
                raise ValueError(f"Terminal ID budget exhausted for neuron {src}")
            used[src].add(new_tid)
            point = deepcopy(terminals[src, old_tid])
            point["terminal_id"] = new_tid
            added.append(point)
            ids[family] = new_tid
        assigned = [family for _, family in members]
        if mode == "shuffled":
            rng.shuffle(assigned)
        counts = defaultdict(Counter)
        for (index, family), assignment in zip(members, assigned):
            tid = ids[assignment]
            result["connections"][index]["source_terminal"] = tid
            counts[tid][family] += 1
        routing.append({"source": src, "original_terminal": old_tid,
                        "terminal_families": {tid: dict(c) for tid, c in counts.items()}})
    result["synaptic_points"].extend(added)
    result["metadata"] = {**result.get("metadata", {}), "projection_terminals": mode}
    return result, {"mode": mode, "terminals_before": original_count,
                    "terminals_after": original_count+len(added), "added_terminals": len(added),
                    "routing": routing,
                    "scope": "Initial transmission per edge is unchanged. Adaptive terminal state is no longer pooled across projection families. More terminals also emit more release events; no extra event reaches any individual target."}


def contact_terminals(config, *, source_ids):
    """Give each outgoing connection of selected neurons its own native terminal.

    Source selection is a build-time anatomical choice. Empty selection is an
    exact copy. Forward weights, delays, neurons and each edge's initial release
    are preserved. Each new terminal receives only its connection's native
    retrograde events; shared somatic and modulatory dynamics remain coupled.
    This increases adaptive capacity and event count, not input conductance.
    """
    selected=set(source_ids);result=deepcopy(config)
    known={n['id'] for n in config['neurons']}
    if not selected<=known:raise ValueError('Unknown terminal source neuron')
    if not selected:return result,dict(source_ids=[],added_terminals=0,routing=[])
    used=defaultdict(set);terms={}
    for p in result['synaptic_points']:
        nid=p['neuron_id'];sid=p['terminal_id'] if p['type']=='presynaptic' else p['synapse_id']
        if sid in used[nid]:raise ValueError('Input/output IDs collide')
        used[nid].add(sid)
        if p['type']=='presynaptic':terms[nid,sid]=p
    grouped=defaultdict(list)
    targets=set()
    for c in result['connections']:
        target=c['target_neuron'],c['target_synapse']
        if target in targets:raise ValueError('Ambiguous receiving synapse')
        targets.add(target)
        if c['source_neuron'] in selected:grouped[c['source_neuron'],c['source_terminal']].append(c)
    added=[];routing=[]
    for (src,old),connections in grouped.items():
        if (src,old) not in terms:raise ValueError('Missing source terminal')
        for i,c in enumerate(connections):
            tid=old
            if i:
                tid=next((j for j in range(900,4096) if j not in used[src]),None)
                if tid is None:raise ValueError(f'Terminal ID budget exhausted for neuron {src}')
                used[src].add(tid);point=deepcopy(terms[src,old]);point['terminal_id']=tid;added.append(point)
            c['source_terminal']=tid
            routing.append(dict(source=src,old_terminal=old,terminal=tid,
                                target=c['target_neuron'],synapse=c['target_synapse']))
    result['synaptic_points'].extend(added)
    result['metadata']={**result.get('metadata',{}),'contact_terminal_sources':sorted(selected)}
    return result,dict(source_ids=sorted(selected),added_terminals=len(added),routing=routing,
        scope='One native terminal per outgoing connection of selected neurons; all initial per-edge release and incoming conductance preserved. Shared cell dynamics and neural feedback remain.')


def mean_pooled_return_rates(config, *, source_ids):
    """Control for mean return-update dose without splitting terminal state.

    With identical incoming error streams and unclipped linear terminal updates,
    the shared terminal follows the mean of the contact-specific terminals.
    Closed-loop streams and cellular rate modulation may subsequently differ.
    """
    result=deepcopy(config);selected=set(source_ids);ns={n['id']:n for n in result['neurons']}
    if not selected<=ns.keys():raise ValueError('Unknown terminal source neuron')
    fanout=Counter(c['source_neuron'] for c in config['connections'])
    active=defaultdict(set)
    for c in config['connections']:active[c['source_neuron']].add(c['source_terminal'])
    rates={}
    for nid in selected:
        if len(active[nid])!=1 or fanout[nid]<1:raise ValueError('Mean-rate control requires one active source terminal')
        before=ns[nid]['params']['eta_retro']
        if before<=0:raise ValueError('Return adaptation must remain positive')
        ns[nid]['params']['eta_retro']=before/fanout[nid]
        rates[nid]=dict(before=before,after=before/fanout[nid],fanout=fanout[nid])
    if selected:result['metadata']={**result.get('metadata',{}),'mean_pooled_return_sources':sorted(selected)}
    return result,dict(source_ids=sorted(selected),rates=rates,
        scope='Shared terminal state retained; only selected source eta_retro divided by its outgoing connection count. This matches a mean update under equal error streams, not future closed-loop dynamics.')
