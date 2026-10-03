"""Cook adjacency extraction and explicit cache refresh contracts."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from simulations.c_elegans import connectome
from simulations.connectome_loader import ConnectomeData, NeuronInfo


@pytest.mark.parametrize("muscle_names", [[], ["MDL01", "MDR01"]])
def test_gap_junctions_follow_neuron_adjacency_once(monkeypatch, muscle_names):
    names = ["A", "B", "C"]
    nodes = names + muscle_names
    chemical = np.zeros((len(nodes), len(nodes)))
    chemical[0, 1] = 4
    for index in range(3, len(nodes)):
        chemical[0, index] = index
    gap = np.zeros_like(chemical)
    gap[0, 1] = gap[1, 0] = 2
    gap[0, 2] = gap[2, 0] = 5

    reader = ModuleType("cect.readers.Cook2019HermReader")
    reader.get_instance = lambda: SimpleNamespace(
        nodes=nodes, connections={"Generic_CS": chemical, "Generic_GJ": gap}
    )
    cells = ModuleType("cect.Cells")
    cells.PREFERRED_HERM_NEURON_NAMES_COOK = names
    cells.SENSORY_NEURONS_COOK = ["A"]
    cells.INTERNEURONS_COOK = ["B"]
    cells.MOTORNEURONS_COOK = ["C"]
    cect = ModuleType("cect")
    cect.__path__ = []
    readers = ModuleType("cect.readers")
    readers.__path__ = []
    readers.Cook2019HermReader = reader
    cect.readers = readers
    for name, module in {
        "cect": cect,
        "cect.Cells": cells,
        "cect.readers": readers,
        "cect.readers.Cook2019HermReader": reader,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    parsed = connectome._parse_from_cect()

    assert [(e.pre_name, e.post_name, e.weight) for e in parsed.gap_junction_edges] == [
        ("A", "B", 2), ("A", "C", 5)
    ]
    assert [(e.pre_name, e.post_name, e.weight) for e in parsed.chemical_edges] == [
        ("A", "B", 4)
    ]
    assert [e.post_name for e in parsed.muscle_edges] == muscle_names


def test_explicit_parse_refreshes_memory_cache(monkeypatch, tmp_path):
    old = ConnectomeData([NeuronInfo("old", "unknown", 0)], [], [])
    fresh = ConnectomeData([NeuronInfo("fresh", "unknown", 0)], [], [])
    path = tmp_path / "connectome.json"
    connectome._save_to_cache(old, path)
    monkeypatch.setattr(connectome, "_CACHE_PATH", path)
    monkeypatch.setattr(connectome, "_connectome_memory_cache", old)
    monkeypatch.setattr(connectome, "_parse_from_cect", lambda: fresh)

    assert connectome.load_connectome(use_cache=False) is fresh
    assert connectome.load_connectome() is fresh
    assert connectome._load_from_cache(path).neurons[0].name == "fresh"


def test_existing_cache_is_not_automatically_reparsed(monkeypatch, tmp_path):
    old = ConnectomeData([NeuronInfo("historical", "unknown", 0)], [], [])
    path = tmp_path / "connectome.json"
    connectome._save_to_cache(old, path)
    before = path.read_bytes()
    monkeypatch.setattr(connectome, "_CACHE_PATH", path)
    monkeypatch.setattr(connectome, "_connectome_memory_cache", None)

    def unexpected_parse():
        pytest.fail("Existing research cache must remain unchanged")

    monkeypatch.setattr(connectome, "_parse_from_cect", unexpected_parse)
    assert connectome.load_connectome().neurons[0].name == "historical"
    assert path.read_bytes() == before
