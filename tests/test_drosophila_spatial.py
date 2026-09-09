"""Small spatial fixtures test identities, not physiological acceptance."""
from copy import deepcopy
import gzip
import hashlib
import json

import numpy as np
import pytest
from scipy.sparse.csgraph import dijkstra

from simulations.drosophila.spatial import bind_pair_rows, compare_pairs, load_snapshot, parse_spatial, root_from_name
from simulations.drosophila.connectome import sha256
from test_drosophila_connectome import ROOTS, graph


def preparation():
    compact = [
        [[100, None, 3, 0, 0, 0, 100, 5],
         [101, 100, 3, 3000, 0, 0, 100, 5],
         [102, 100, 3, 0, 4000, 0, 100, 5]],
        [[101, 10, 1, 3001, 0, 0], [102, 11, 0, 0, 4001, 0]], {}, [], []]
    connectors = {"connectors": [[10, 3001, 0, 0, 5, 3, 3, 0, 0],
                                   [11, 0, 4001, 0, 5, 3, 3, 0, 0]],
                  "partners": {"10": [[1, 201, 2, 20, 5], [2, 101, 1, 21, 5]],
                               "11": [[3, 102, 1, 20, 5], [4, 301, 3, 21, 5]]}, "tags": {}}
    relations = [{"relation": "presynaptic_to", "relation_id": 20},
                 {"relation": "postsynaptic_to", "relation_id": 21}]
    names = {"1": f"flywire: {ROOTS[1]}", "2": f"flywire: {ROOTS[0]}",
             "3": f"flywire: {ROOTS[2]}"}
    return compact, connectors, relations, names, 1


def test_exact_identity_and_actual_tree_distance():
    result = parse_spatial(*preparation())
    assert result["root"] == int(ROOTS[1]) > 2**53
    assert result["contacts"].tolist() == [
        [10, int(ROOTS[0]), int(ROOTS[1]), 201, 101, -1, 1],
        [11, int(ROOTS[1]), int(ROOTS[2]), 102, 301, 2, -1]]
    assert result["cable_length_nm"] == 7000
    assert dijkstra(result["cable"], indices=1)[2] == 7000
    assert np.linalg.norm(result["xyz_nm"][1] - result["xyz_nm"][2]) == 5000
    assert result["components"] == result["tree_roots"] == 1


@pytest.mark.parametrize("name", ["flywire: 720575940000000001.0", "cell 720575940000000001", "flywire: 0720575940000000001"])
def test_ambiguous_names_are_not_guessed(name):
    with pytest.raises(ValueError):
        root_from_name(name)


def test_polyad_and_autapse_retain_separate_attachments_without_double_counting():
    compact, connectors, relations, names, skeleton = preparation()
    # Add a postsynaptic APL attachment to its own output connector.
    compact[1].append([101, 11, 1, 0, 4001, 0])
    connectors["partners"]["11"].append([5, 101, 1, 21, 5])
    result = parse_spatial(compact, connectors, relations, names, skeleton)
    assert len(result["contacts"]) == 3
    autapse = result["contacts"][-1]
    assert autapse[1] == autapse[2] == result["root"]
    assert autapse[5:7].tolist() == [2, 1]


def test_missing_partner_is_preserved_not_invented():
    inputs = preparation()
    inputs[1]["partners"]["10"].pop(0)
    result = parse_spatial(*inputs)
    assert result["contacts"][0].tolist() == [10, 0, int(ROOTS[1]), 0, 101, -1, 1]
    comparison = compare_pairs(result["contacts"], result["root"], graph())
    assert comparison["unpaired_contacts"] == 1
    assert comparison["spatial_paired_contacts"] == 1
    assert comparison["all_counts_match"] is False


@pytest.mark.parametrize("change,match", [
    (lambda x: x[0][0].append(x[0][0][0]), "Duplicate treenode"),
    (lambda x: x[0][0][1].__setitem__(0, 101.0), "Invalid exact integer"),
    (lambda x: x[0][0][1].__setitem__(3, float('nan')), "Nonfinite"),
    (lambda x: x[0][0][1].__setitem__(1, 101), "Self-parent"),
    (lambda x: x[0][1].append(x[0][1][0]), "Duplicate local"),
    (lambda x: x[0][1][0].__setitem__(2, 2), "Unsupported local"),
    (lambda x: x[0][1][0].__setitem__(0, 999), "outside morphology"),
    (lambda x: x[1]["connectors"][0].__setitem__(1, 3002), "representations disagree"),
    (lambda x: x[1]["connectors"].pop(), "Connector sets disagree"),
    (lambda x: x[1]["partners"]["10"][0].__setitem__(2, 999), "Missing partner identity"),
    (lambda x: x[1]["partners"]["10"].append(x[1]["partners"]["10"][0]), "Duplicate partner link"),
    (lambda x: x[1]["partners"]["10"][1].__setitem__(1, 102), "representations disagree"),
    (lambda x: x[1]["partners"]["10"].pop(1), "Missing local attachments"),
])
def test_inconsistent_sources_fail_instead_of_being_repaired(change, match):
    inputs = preparation()
    change(inputs)
    with pytest.raises(ValueError, match=match):
        parse_spatial(*inputs)


def test_cycle_rejected_but_disconnected_anatomy_is_reported():
    inputs = preparation()
    inputs[0][0][0][1] = 102
    with pytest.raises(ValueError, match="Cycle"):
        parse_spatial(*inputs)
    inputs = preparation()
    inputs[0][0][2][1] = None
    result = parse_spatial(*inputs)
    assert result["components"] == result["tree_roots"] == 2
    assert np.isinf(dijkstra(result["cable"], indices=1)[2])


def test_pair_comparison_exposes_every_missing_extra_and_changed_pair():
    result = parse_spatial(*preparation())
    g = graph()
    before = deepcopy(g.edges)
    comparison = compare_pairs(result["contacts"], result["root"], g)
    assert np.array_equal(g.edges, before)
    assert comparison["all_counts_match"] is False
    assert comparison["pair_table_synapses"] == 7
    assert comparison["spatial_paired_contacts"] == 2
    assert comparison["differences"] == [
        {"pre_root": ROOTS[0], "post_root": ROOTS[1], "pair_table": 3, "spatial_snapshot": 1},
        {"pre_root": ROOTS[1], "post_root": ROOTS[0], "pair_table": 4, "spatial_snapshot": 0},
        {"pre_root": ROOTS[1], "post_root": ROOTS[2], "pair_table": 0, "spatial_snapshot": 1}]


def test_extended_public_link_rows_are_explicitly_supported():
    inputs = preparation()
    for links in inputs[1]["partners"].values():
        for link in links:
            link.extend([3, 100.0, 101.0])
    assert len(parse_spatial(*inputs)["contacts"]) == 2


def snapshot(tmp_path):
    compact, connectors, relations, names, skeleton = preparation()
    raw = {"projects.json.gz": [{"id": 1, "title": "FlyWire m783"}],
           "identity.json.gz": {"totalRecords": 1, "entities": [
               {"name": names["1"], "skeleton_ids": [skeleton]}]},
           "compact.json.gz": compact, "connectors.json.gz": connectors,
           "relations.json.gz": relations, "names-000000.json.gz": names}
    manifest = {"schema": "flywire-catmaid-spatial-v1", "status": "complete",
                "materialization": 783, "project_id": 1,
                "root": ROOTS[1], "skeleton_id": skeleton, "files": {}}
    for filename, value in raw.items():
        payload = json.dumps(value).encode()
        (tmp_path / filename).write_bytes(gzip.compress(payload))
        manifest["files"][filename] = {"sha256": sha256(tmp_path / filename),
            "response_sha256": hashlib.sha256(payload).hexdigest(), "response_bytes": len(payload)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_recorded_snapshot_round_trip_keeps_large_ids(tmp_path):
    original = snapshot(tmp_path)
    manifest, raw, names = load_snapshot(tmp_path)
    assert manifest == original
    assert names["1"] == f"flywire: {ROOTS[1]}"
    assert parse_spatial(raw["compact.json.gz"], raw["connectors.json.gz"],
                         raw["relations.json.gz"], names, 1)["root"] == int(ROOTS[1])


@pytest.mark.parametrize("change,match", [
    (lambda m: m.__setitem__("status", "incomplete"), "Incomplete"),
    (lambda m: m.__setitem__("materialization", 630), "Incomplete"),
    (lambda m: m.__setitem__("root", ROOTS[0]), "identity disagreement"),
    (lambda m: m["files"]["compact.json.gz"].__setitem__("sha256", "wrong"), "Changed spatial"),
    (lambda m: m["files"]["compact.json.gz"].__setitem__("response_sha256", "wrong"), "Changed response"),
])
def test_snapshot_verification_checks_release_identity_and_bytes(tmp_path, change, match):
    manifest = snapshot(tmp_path)
    change(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        load_snapshot(tmp_path)


def test_contact_join_refuses_different_counts_and_preserves_open_connectors():
    parsed = parse_spatial(*preparation())
    g = graph()
    with pytest.raises(ValueError, match="pair counts disagree"):
        bind_pair_rows(parsed["contacts"], parsed["root"], g)
    # Three KC->APL and four APL->KC contacts, plus one open input connector.
    rows = np.array([[i, int(ROOTS[0]), int(ROOTS[1]), 201, 101, -1, 1] for i in range(3)]
                    + [[i + 10, int(ROOTS[1]), int(ROOTS[0]), 102, 301, 2, -1] for i in range(4)]
                    + [[30, 0, int(ROOTS[1]), 0, 101, -1, 1]], dtype=np.int64)
    assert bind_pair_rows(rows, parsed["root"], g).tolist() == [1, 1, 1, 2, 2, 2, 2, -1]
