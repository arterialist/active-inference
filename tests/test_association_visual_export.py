"""Replay integrity checks. No agents or simulations are started."""
import base64
import gzip
import json
from pathlib import Path

import pytest

from simulations.active_inference.experiments.association_visual_export import (
    ROOT, RUNS, derived_state, write_payload,
)
from simulations.active_inference.experiments.composition_analysis import rows

DATA = ROOT / "docs/embodied-assessment/dynamics/data"


def decode(path):
    content = path.read_text()
    assert content.startswith("PAULA_DATA.receive(") and content.endswith(");\n")
    key, encoded = json.loads("["+content[len("PAULA_DATA.receive("):-3]+"]")
    return key, json.loads(gzip.decompress(base64.b64decode(encoded)))


def test_lossless_script_envelope(tmp_path):
    payload = {"tiny": 1e-15, "negative": -1.23456789123456, "text": "<test>"}
    path = tmp_path / "chunk.js"
    write_payload(path, "run/1", payload)
    assert decode(path) == ("run/1", payload)


def test_derived_membrane_is_not_post_reset_state():
    old = {"neurons": {"5": {"S": 0., "t_last_fire": None, "dendritic_queue": [[3,"hillock",2.,0]], "synapses":{"0":[1.]}}}}
    row = {"executed_tick": 3, "state":{"neurons":{"5":{"S":0.,"O":1.,"b":1.2,"r":.6,"t_ref":4.,"t_last_fire":3,"synapses":{"0":[1.1]}}}}}
    manifest = {"resolved":{"5":{"parameters":{"delta_decay":.95,"lambda_param":2.,"c":2}}}}
    derived = derived_state(old,row,manifest,{("5","0"):1})["5"]
    assert derived["pre_reset"] == .95
    assert derived["threshold"] == .6
    assert derived["age"] == 0 and derived["eligible"]
    assert derived["weight_delta"]["0"] == pytest.approx(.1)


@pytest.mark.parametrize("key", RUNS)
def test_every_exported_tick_matches_the_source(key):
    source = ROOT / ".live/research" / RUNS[key][0]
    if not (source / "ticks.jsonl.gz").exists() or not (DATA / key / "meta.js").exists():
        pytest.skip("Local research recording not installed")
    _, metadata = decode(DATA / key / "meta.js")
    stream = rows(source)
    previous = next(stream)["state"]
    assert metadata["initial"] == previous
    count = 0
    for trial in metadata["trials"]:
        chunk_key, chunk = decode(DATA / key / f"trial-{trial['index']}.js")
        assert chunk_key == f"{key}/{trial['index']}"
        assert chunk["previous"] == previous
        assert len(chunk["rows"]) == trial["stop"]-trial["start"]
        for enriched in chunk["rows"]:
            raw = next(stream)
            assert {k:v for k,v in enriched.items() if k != "derived"} == raw
            assert enriched["executed_tick"] == count
            previous = raw["state"]
            count += 1
    assert next(stream, None) is None
    assert count == metadata["ticks"]


def test_known_temporal_credit_counterexample():
    if not (DATA / "credit-wide/trial-74.js").exists():
        pytest.skip("Local research replay export not installed")
    _, wide = decode(DATA / "credit-wide/trial-74.js")
    _, narrow = decode(DATA / "credit-narrow/trial-74.js")
    a,b = wide["rows"][30],narrow["rows"][30]
    assert a["executed_tick"] == b["executed_tick"] == 11870
    assert a["derived"]["5"]["age"] == b["derived"]["5"]["age"] == 5
    assert a["derived"]["5"]["weight_delta"]["0"] > 0
    assert b["derived"]["5"]["weight_delta"]["0"] < 0


def test_recall_failure_and_return_are_not_mislabeled():
    if not (DATA / "recall/meta.js").exists():
        pytest.skip("Local research replay export not installed")
    _, metadata = decode(DATA / "recall/meta.js")
    probes = metadata["trials"]
    assert probes[68]["category"] == "correct_only"
    assert all(p["category"] == "silent_or_insufficient" for p in probes[74:92])
    assert all(p["category"] == "correct_only" for p in probes[92:98])
    assert all(not p["paired"] for p in probes[74:98])
