from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from simulations.active_inference.components.learning.projection_terminals import projection_terminals, contact_terminals, mean_pooled_return_rates
from simulations.active_inference.experiments.composition_probe import encode
from simulations.active_inference.experiments.population_hierarchy import make_config
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from neuron.neuron import RetrogradeSignalEvent


def release_vectors(config):
    terms = {(p["neuron_id"], p["terminal_id"]): p for p in config["synaptic_points"] if p["type"] == "presynaptic"}
    return [(terms[c["source_neuron"], c["source_terminal"]]["u_o"],
             terms[c["source_neuron"], c["source_terminal"]]["distance_from_hillock"])
            for c in config["connections"]]


def test_default_is_exact_and_split_preserves_forward_wiring():
    config, _, edges = make_config(1152)
    before = encode(config)
    shared, _ = projection_terminals(config, edges)
    assert shared == config and shared is not config
    for mode in ("family", "shuffled"):
        result, report = projection_terminals(config, edges, mode=mode)
        assert encode(config) == before
        assert result["neurons"] == config["neurons"]
        assert result["external_inputs"] == config["external_inputs"]
        assert release_vectors(result) == release_vectors(config)
        assert report["added_terminals"] > 0
        assert [p for p in result["synaptic_points"] if p["type"] == "postsynaptic"] == [p for p in config["synaptic_points"] if p["type"] == "postsynaptic"]
        for old, new in zip(config["connections"], result["connections"]):
            assert {k: v for k, v in old.items() if k != "source_terminal"} == {k: v for k, v in new.items() if k != "source_terminal"}
        ids = [(p["neuron_id"], p.get("synapse_id", p.get("terminal_id"))) for p in result["synaptic_points"]]
        assert len(ids) == len(set(ids))


def test_shuffled_control_matches_capacity_and_fanout_but_mixes_families():
    config, _, edges = make_config(1152)
    aligned, am = projection_terminals(config, edges, mode="family")
    shuffled, sm = projection_terminals(config, edges, mode="shuffled")
    assert aligned["synaptic_points"] == shuffled["synaptic_points"]
    fanout = lambda cfg: Counter((c["source_neuron"], c["source_terminal"]) for c in cfg["connections"])
    assert fanout(aligned) == fanout(shuffled)
    assert all(len(families) == 1 for row in am["routing"] for families in row["terminal_families"].values())
    assert any(len(families) > 1 for row in sm["routing"] for families in row["terminal_families"].values())
    repeated, _ = projection_terminals(config, edges, mode="shuffled")
    assert repeated == shuffled


def test_retrograde_update_is_local_to_declared_projection_terminal(tmp_path):
    config, _, edges = make_config(288)
    split, report = projection_terminals(config, edges, mode="family")
    row = next(r for r in report["routing"] if len(r["terminal_families"]) >= 2)
    src = row["source"]
    path = tmp_path/"split.json"; path.write_text(encode(split))
    net, _, _, _ = fresh(path, 11)
    neuron = net.network.neurons[src]
    first, second = list(row["terminal_families"])[:2]
    before = {tid: deepcopy(point.u_o.info) for tid, point in neuron.presynaptic_points.items()}
    event = RetrogradeSignalEvent(source_neuron_id=999, source_synapse_id=0,
        target_neuron_id=src, target_terminal_id=first, error_vector=np.array([10., 0., 0., 0.]), timestamp=0)
    neuron.process_retrograde_signal(event)
    assert neuron.presynaptic_points[first].u_o.info != before[first]
    assert neuron.presynaptic_points[second].u_o.info == before[second]
    assert all(n.params.eta_post > 0 and n.params.eta_retro > 0 for n in net.network.neurons.values())


def test_ambiguous_edges_are_rejected():
    config, _, edges = make_config(288)
    with pytest.raises(ValueError, match="Ambiguous"):
        projection_terminals(config, edges+[edges[0]], mode="family")


def test_contact_partition_preserves_forward_drive_and_unselected_sources():
    cfg,groups,_=make_config(288)
    empty,report=contact_terminals(cfg,source_ids=[])
    assert empty==cfg and empty is not cfg and report['added_terminals']==0
    selected=set(groups['visual_core']);split,report=contact_terminals(cfg,source_ids=selected)
    assert split['neurons']==cfg['neurons'] and split['external_inputs']==cfg['external_inputs']
    assert release_vectors(split)==release_vectors(cfg)
    assert [p for p in split['synaptic_points'] if p['type']=='postsynaptic']==[p for p in cfg['synaptic_points'] if p['type']=='postsynaptic']
    fanout=Counter((c['source_neuron'],c['source_terminal']) for c in split['connections'] if c['source_neuron'] in selected)
    assert set(fanout.values())=={1}
    assert all(c==d for c,d in zip(cfg['connections'],split['connections']) if c['source_neuron'] not in selected)
    assert report['added_terminals']==sum(c['source_neuron'] in selected for c in cfg['connections'])-len(selected)
    with pytest.raises(ValueError,match='Unknown'):contact_terminals(cfg,source_ids=[999999])


def test_contact_feedback_changes_only_its_receiving_edge(tmp_path):
    cfg,groups,_=make_config(288);split,report=contact_terminals(cfg,source_ids=groups['visual_core'])
    source=groups['visual_core'][0];rows=[r for r in report['routing'] if r['source']==source]
    path=tmp_path/'contacts.json';path.write_text(encode(split));net,_,_,_=fresh(path,11)
    n=net.network.neurons[source];before={t:float(p.u_o.info) for t,p in n.presynaptic_points.items()}
    r=rows[0]
    n.process_retrograde_signal(RetrogradeSignalEvent(r['target'],r['synapse'],source,r['terminal'],np.array([1.,0.,0.,0.]),0))
    assert n.presynaptic_points[r['terminal']].u_o.info!=before[r['terminal']]
    assert all(p.u_o.info==before[t] for t,p in n.presynaptic_points.items() if t!=r['terminal'])


def test_mean_pooled_control_only_changes_positive_return_rate():
    cfg,groups,_=make_config(288);selected=groups['visual_core']
    result,report=mean_pooled_return_rates(cfg,source_ids=selected)
    assert result['synaptic_points']==cfg['synaptic_points'] and result['connections']==cfg['connections']
    for old,new in zip(cfg['neurons'],result['neurons']):
        if old['id'] not in selected:assert old==new
        else:
            expected=deepcopy(old);expected['params']['eta_retro']/=report['rates'][old['id']]['fanout']
            assert new==expected and new['params']['eta_retro']>0
    # Linear, unclipped terminal updates: the pooled control equals contact mean.
    eta=1e-5;errors=np.array([.2,-.4,.3]);contacts=np.ones(3)+eta*errors
    pooled=1.+(eta/len(errors))*errors.sum()
    assert abs(pooled-contacts.mean())<1e-15


def test_independent_audit_detects_gain_and_control_fanout_changes():
    from simulations.active_inference.experiments.terminal_partition_audit import wiring_checks, matched_control_checks
    config, _, edges = make_config(288)
    aligned, _ = projection_terminals(config, edges, mode="family")
    control, _ = projection_terminals(config, edges, mode="shuffled")
    assert all(wiring_checks(aligned, config, edges, "family").values())
    assert all(wiring_checks(control, config, edges, "shuffled").values())
    assert all(matched_control_checks(aligned, control).values())
    wrong = deepcopy(aligned)
    point = next(p for p in wrong["synaptic_points"] if p["type"] == "presynaptic")
    point["u_o"]["info"] += .1
    assert not wiring_checks(wrong, config, edges, "family")["same_initial_edge_release"]
    wrong = deepcopy(control)
    wrong["connections"][0]["source_terminal"] += 1
    assert not matched_control_checks(aligned, wrong)["control_same_terminal_fanout"]
