"""Experimental assembly of local receptor-to-release coupling, defaults intact."""
from unittest.mock import patch

import numpy as np

from . import paula
from .pn_current_steps import prepare
from neuron.extensions.experimental.presynaptic_inhibition import PresynapticInhibitionNeuron


def prepare_inhibited(graph, intrinsic, tail, inhibitory_root, gain, decay_ticks, *, spatial=None):
    ordinary = paula.Neuron
    orn_roots = [r for r in graph.selected if graph.nodes[r]["annotation"]["hemibrain_type"] == "ORN_DL5"]
    orn_ids = {graph.nodes[r]["global_index"] for r in orn_roots}

    def factory(nid, *args, **kwargs):
        return (PresynapticInhibitionNeuron if nid in orn_ids else ordinary)(nid, *args, **kwargs)

    # Scope the constructor choice to assembly. Running neural dynamics have no
    # experiment-specific dispatch, root comparisons or global feedback rule.
    with patch.object(paula, "Neuron", factory):
        prep, pn = prepare(graph, intrinsic, tail, spatial=spatial)
    bindings = []
    for root in orn_roots:
        cell = prep.network.network.neurons[prep.root_to_id[root]]
        incoming = sorted(graph.edges[graph.edges[:, 1] == int(root)], key=lambda e: int(e[8]))
        outgoing = sorted(graph.edges[graph.edges[:, 0] == int(root)], key=lambda e: int(e[8]))
        ports = [i for i,e in enumerate(incoming) if str(e[0]) == inhibitory_root]
        terminals = [i for i,e in enumerate(outgoing)
                     if graph.nodes[str(e[1])]["annotation"]["cell_class"] == "ALPN"]
        if not ports or not terminals:
            raise ValueError("Each declared ORN must have the measured inhibitory input and ALPN output")
        cell.configure_presynaptic_inhibition(np.array(ports), np.array(terminals), gain, decay_ticks)
        bindings.append({"root": root, "ports": ports, "terminals": terminals,
            "receiving_rows": [int(incoming[i][8]) for i in ports],
            "release_rows": [int(outgoing[i][8]) for i in terminals]})
    prep.assumptions["presynaptic_inhibition"] = {
        "source_root": inhibitory_root, "gain": gain, "decay_ticks": decay_ticks,
        "bindings": bindings,
        "mechanism": "native inhibitory receptor drive, ordinary delay/attenuation, exponential intracellular decay, release fraction 1/(1+gain*state)",
        "scope": "all ALPN-directed terminals of selected DL5 ORNs, including absent boundary targets; ordinary somatic action retained",
        "status": "effective receptor-to-release hypothesis; no fitted receptor kinetics, target-specific receptor evidence or compartment localization",
        "assembly": "temporary ordinary-cell constructor factory; reference builder source and default behavior unchanged"}
    return prep, pn
