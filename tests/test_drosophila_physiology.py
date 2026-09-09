"""Adversarial measurement-unit and identity checks, no online data needed."""
from copy import deepcopy
import json

import pytest

from simulations.drosophila.prisco import (
    GLOMERULI, digest, dump_new, glomerulus_candidates, number,
    paired_peaks, pn_observations, pooled_reconciliation, verify_file,
)


def data(rows, file_id=1282868):
    return {"workbooks": {str(file_id): {"sheets": [{"name": "Sheet1", "rows": rows}]}}}


def pn_rows():
    return [[v for g in GLOMERULI for v in (g, None)], ["Mch", "Oct"]*9] + [
        [0.0, float(i)]*9 for i in range(10)]


def test_glomerulus_observations_preserve_zero_identity_and_original_cell():
    rows = pn_rows()
    original = deepcopy(rows)
    result = pn_observations(data(rows), 1282868)
    assert len(result) == 180
    assert result[0]["peak_dff_percent"] == 0
    assert result[-1]["cell"] == "R12"
    assert result[-1]["glomerulus"] == "DC1"
    assert result[-1]["source_row"] == 12
    assert rows == original


@pytest.mark.parametrize("value", [None, "0", True, float("nan"), float("inf")])
def test_missing_and_nonnumeric_values_are_not_zero(value):
    with pytest.raises(ValueError, match="Missing/non-numeric"):
        number(value)


def test_reordered_columns_fail_instead_of_silently_changing_glomeruli():
    rows = pn_rows()
    rows[0][0], rows[0][2] = rows[0][2], rows[0][0]
    with pytest.raises(ValueError, match="headers"):
        pn_observations(data(rows), 1282868)


def test_missing_glomerular_value_does_not_become_silent_drive():
    rows = pn_rows()
    rows[3][2] = None
    with pytest.raises(ValueError, match="Missing"):
        pn_observations(data(rows), 1282868)


def test_ambiguous_subtypes_multiglomerular_and_driver_uncertainty_survive():
    types = ["DM6_adPN", "VA1d_adPN", "VA1v_vPN", "DM6+DL1_adPN",
             "DM6_adPN,DM6_vPN", "DL1_adPN", "DM6_adPN"]
    roots = tuple(str(720575940000000010+i) for i in range(len(types)))
    nodes = {r: {"global_index": i, "annotation": {
        "cell_class": "ALPN" if i != 6 else "Kenyon_Cell",
        "cell_sub_class": "uniglomerular" if i != 3 else "multiglomerular",
        "hemibrain_type": name, "side": "left"}}
        for i, (r, name) in enumerate(zip(roots, types, strict=True))}
    result = glomerulus_candidates(nodes, roots)
    assert result["candidate_count"] == 2
    assert result["selected_alpn_count"] == 6
    assert result["candidate_neurons"]["VA1"] == []
    assert len(result["unresolved"]) == 4
    assert result["candidate_neurons"]["DM6"][0]["root"] == roots[0]
    assert result["candidate_neurons"]["DM6"][0]["driver_expression_verified"] is False


def test_pooled_columns_are_unpaired_and_missing_values_do_not_inflate_n():
    d = data([["Mean", "N"], [10, 2], [30, 1]], 1)
    d["workbooks"].update(data([["pooled"], [8], [12], [None], [30]], 2)["workbooks"])
    result = pooled_reconciliation(d, 1, 2, 2, 0, 1, 2, 0)
    assert result["n_animals"] == 2
    assert result["pooled_roi_values"] == 3
    assert result["counts_match"]
    assert result["difference"] == pytest.approx(0)
    assert result["roi_to_animal_identity_available"] is False
    d["workbooks"]["2"]["sheets"][0]["rows"].append([50])
    assert not pooled_reconciliation(d, 1, 2, 2, 0, 1, 2, 0)["counts_match"]


def test_paired_peaks_keep_each_animal_and_report_roi_count_separately():
    d = data([["Mean", "N", "Mean", "N"], [5, 100, 7, 200], [4, 99, 5, 80]], 1)
    c = paired_peaks(d, 1, 2, 0, 2, ("Mch", "Oct"), "fixture", 2)
    assert c["n_animal_rows"] == 2
    assert [r["second_minus_first"] for r in c["animals"]] == [2, 1]
    assert c["animals"][1]["right_cell"] == "C3"
    with pytest.raises(ValueError, match="animal rows"):
        paired_peaks(d, 1, 2, 0, 2, ("Mch", "Oct"), "fixture", 199)


def test_sources_are_verified_and_outputs_are_never_overwritten(tmp_path):
    p = tmp_path / "source"
    p.write_bytes(b"original")
    spec = {"bytes": p.stat().st_size, "sha256": digest(p)}
    verify_file(p, spec)
    p.write_bytes(b"modified")
    with pytest.raises(ValueError, match="bytes differ"):
        verify_file(p, spec)
    out = tmp_path / "result.json"
    dump_new(out, {"a": 1})
    with pytest.raises(FileExistsError):
        dump_new(out, {"a": 2})
    assert json.loads(out.read_text()) == {"a": 1}
