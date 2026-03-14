"""
C. elegans connectome loader.

Downloads (or loads from cache) the Cook et al. 2019 hermaphrodite connectome
via the OpenWorm cect library and converts it into a ConnectomeData object
suitable for build_paula_network().

Neuron classification follows Cook et al. 2019:
  - SENSORY_NEURONS_COOK  (83 neurons)
  - INTERNEURONS_COOK     (89 neurons)
  - MOTORNEURONS_COOK     (123 neurons)
  - Remaining 7 are classified as 'unknown'
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from loguru import logger

from simulations.connectome_loader import ConnectomeData, NeuronInfo, SynapticEdge


# Cache file so we don't hit the Excel reader on every run
_CACHE_PATH = Path(__file__).resolve().parents[3] / "data" / "c_elegans" / "connectome_cache.json"

# In-memory cache: avoids repeated loads during evolution (keeps RAM flat)
_connectome_memory_cache: ConnectomeData | None = None


def load_connectome(use_cache: bool = True) -> ConnectomeData:
    """
    Load the complete C. elegans hermaphrodite connectome.

    Args:
        use_cache: If True, load from local JSON cache when available.
                   Falls back to parsing the cect Excel data if cache missing.

    Returns:
        ConnectomeData with 302 neurons, chemical synapses, and gap junctions.
    """
    global _connectome_memory_cache
    if use_cache and _CACHE_PATH.exists():
        if _connectome_memory_cache is not None:
            return _connectome_memory_cache
        logger.info(f"Loading connectome from cache: {_CACHE_PATH}")
        _connectome_memory_cache = _load_from_cache(_CACHE_PATH)
        return _connectome_memory_cache

    logger.info("Parsing connectome from cect Cook2019HermReader …")
    data = _parse_from_cect()

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _save_to_cache(data, _CACHE_PATH)
    logger.info(f"Connectome cached to {_CACHE_PATH}")

    return data


# -----------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------


def _parse_from_cect() -> ConnectomeData:
    """Parse connectome from the cect Cook2019HermReader."""
    import cect.Cook2019HermReader as reader_module
    from cect.Cells import (
        PREFERRED_HERM_NEURON_NAMES_COOK,
        SENSORY_NEURONS_COOK,
        INTERNEURONS_COOK,
        MOTORNEURONS_COOK,
    )

    instance = reader_module.get_instance()

    # Neuron list (302 hermaphrodite neurons)
    herm_neurons: list[str] = list(PREFERRED_HERM_NEURON_NAMES_COOK)

    # Classification sets
    sensory_set: set[str] = set(SENSORY_NEURONS_COOK)
    inter_set: set[str] = set(INTERNEURONS_COOK)
    motor_set: set[str] = set(MOTORNEURONS_COOK)

    def _classify(name: str) -> str:
        if name in sensory_set:
            return "sensory"
        if name in motor_set:
            return "motor"
        if name in inter_set:
            return "interneuron"
        return "unknown"

    neurons = [
        NeuronInfo(
            name=name,
            neuron_type=_classify(name),
            paula_id=idx,
        )
        for idx, name in enumerate(herm_neurons)
    ]

    logger.info(
        f"Classified neurons: "
        f"sensory={sum(1 for n in neurons if n.neuron_type=='sensory')}, "
        f"motor={sum(1 for n in neurons if n.neuron_type=='motor')}, "
        f"interneuron={sum(1 for n in neurons if n.neuron_type=='interneuron')}, "
        f"unknown={sum(1 for n in neurons if n.neuron_type=='unknown')}"
    )

    # Adjacency matrices from cect
    cs_matrix: np.ndarray = instance.connections["Generic_CS"]
    gj_matrix: np.ndarray = instance.connections["Generic_GJ"]
    all_nodes: list[str] = instance.nodes

    # Build index map from cect node list
    node_index: dict[str, int] = {n: i for i, n in enumerate(all_nodes)}

    # Build neuron-to-neuron index lookup (only among herm neurons)
    herm_set = set(herm_neurons)

    chemical_edges: list[SynapticEdge] = []
    gap_junction_edges: list[SynapticEdge] = []

    # Extract chemical synapses between herm neurons
    for pre_name in herm_neurons:
        if pre_name not in node_index:
            continue
        pre_idx = node_index[pre_name]
        row_cs = cs_matrix[pre_idx]
        row_gj = gj_matrix[pre_idx]

        for post_name in herm_neurons:
            if post_name not in node_index:
                continue
            post_idx = node_index[post_name]

            cs_weight = float(row_cs[post_idx])
            if cs_weight > 0:
                chemical_edges.append(
                    SynapticEdge(
                        pre_name=pre_name,
                        post_name=post_name,
                        synapse_type="chemical",
                        weight=cs_weight,
                    )
                )

            # Gap junctions: include each directed pair once (pre < post
            # in sorted order to avoid double-counting in this loop).
            gj_weight = float(row_gj[post_idx])
            if gj_weight > 0 and pre_name < post_name:
                gap_junction_edges.append(
                    SynapticEdge(
                        pre_name=pre_name,
                        post_name=post_name,
                        synapse_type="gap_junction",
                        weight=gj_weight,
                    )
                )

    logger.info(
        f"Extracted {len(chemical_edges)} chemical synapses, "
        f"{len(gap_junction_edges)} gap junctions "
        f"(each gap junction will yield 2 directed PAULA connections)"
    )

    return ConnectomeData(
        neurons=neurons,
        chemical_edges=chemical_edges,
        gap_junction_edges=gap_junction_edges,
    )


# -----------------------------------------------------------------------
# Cache helpers
# -----------------------------------------------------------------------


def _save_to_cache(data: ConnectomeData, path: Path) -> None:
    payload = {
        "neurons": [
            {
                "name": n.name,
                "neuron_type": n.neuron_type,
                "paula_id": n.paula_id,
            }
            for n in data.neurons
        ],
        "chemical_edges": [
            {
                "pre_name": e.pre_name,
                "post_name": e.post_name,
                "synapse_type": e.synapse_type,
                "weight": e.weight,
            }
            for e in data.chemical_edges
        ],
        "gap_junction_edges": [
            {
                "pre_name": e.pre_name,
                "post_name": e.post_name,
                "synapse_type": e.synapse_type,
                "weight": e.weight,
            }
            for e in data.gap_junction_edges
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def _load_from_cache(path: Path) -> ConnectomeData:
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load connectome cache from {path}") from e

    neurons = [
        NeuronInfo(
            name=n["name"],
            neuron_type=n["neuron_type"],
            paula_id=n["paula_id"],
        )
        for n in payload["neurons"]
    ]
    chemical_edges = [
        SynapticEdge(
            pre_name=e["pre_name"],
            post_name=e["post_name"],
            synapse_type=e["synapse_type"],
            weight=e["weight"],
        )
        for e in payload["chemical_edges"]
    ]
    gap_junction_edges = [
        SynapticEdge(
            pre_name=e["pre_name"],
            post_name=e["post_name"],
            synapse_type=e["synapse_type"],
            weight=e["weight"],
        )
        for e in payload["gap_junction_edges"]
    ]
    return ConnectomeData(
        neurons=neurons,
        chemical_edges=chemical_edges,
        gap_junction_edges=gap_junction_edges,
    )


# -----------------------------------------------------------------------
# Convenience statistics
# -----------------------------------------------------------------------


def print_connectome_summary(data: ConnectomeData) -> None:
    """Print a human-readable summary of the connectome."""
    type_counts = Counter(n.neuron_type for n in data.neurons)

    print(f"\n{'='*50}")
    print(f"C. elegans Connectome Summary (Cook et al. 2019)")
    print(f"{'='*50}")
    print(f"Total neurons : {data.n_neurons}")
    for t, c in sorted(type_counts.items()):
        print(f"  {t:12s}: {c}")
    print(f"Chemical synapses   : {len(data.chemical_edges)}")
    print(f"Gap junctions       : {len(data.gap_junction_edges)}")
    print(f"Total directed edges: {len(data.all_edges)}")

    weights = [e.weight for e in data.chemical_edges]
    if weights:
        mean_weight = sum(weights) / len(weights)
        print(f"Chem weight stats   : min={min(weights):.0f}, "
              f"max={max(weights):.0f}, "
              f"mean={mean_weight:.1f}")
    print(f"{'='*50}\n")
