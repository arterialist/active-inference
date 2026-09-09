"""A short instrumented execution check, explicitly NOT an odor experiment.

Two artificial current pulses exercise actual KC/APL wiring. Record every
selected cell and postsynaptic input at every tick, including weak ongoing
adaptation. This is a trace, not an executable continuation checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import typer

from .connectome import Subgraph, sha256
from .paula import Dynamics, Neuron, build_paula
from simulations.paula_loader import ensure_paula_available

SOMA_FIELDS = ("S", "O", "F_avg", "r", "b", "t_ref", "t_last_fire")


def run_probe(graph: Subgraph, output: Path, dynamics: Dynamics = Dynamics()) -> dict:
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    preparation = build_paula(graph, dynamics)
    net = preparation.network
    cells = list(net.network.neurons.values())
    cell_rows = {cell.id: i for i, cell in enumerate(cells)}
    roots = tuple(preparation.root_to_id)
    kc = [r for r in roots if graph.nodes[r]["annotation"]["cell_class"] == "Kenyon_Cell"]
    apl = [r for r in roots if graph.nodes[r]["annotation"]["hemibrain_type"] == "APL"]
    if not kc or len(apl) != 1:
        raise ValueError("Execution probe requires KCs and exactly one APL")
    stimulated = kc[::max(1, len(kc) // 32)][:32]
    ticks = 32
    offsets = np.cumsum([0] + [c.params.num_inputs for c in cells])
    ports = [(c.id, sid) for c in cells for sid in c.postsynaptic_points]
    terminals = [(c.id, tid) for c in cells for tid in c.presynaptic_points]
    soma = np.zeros((ticks + 1, len(cells), len(SOMA_FIELDS)), dtype=np.float64)
    modulator = np.zeros((ticks + 1, len(cells), 2), dtype=np.float64)
    inputs = np.zeros((ticks, len(ports), 4), dtype=np.float32)
    local_potential = np.zeros((ticks, len(ports)), dtype=np.float64)
    post_weight = np.zeros((ticks + 1, len(ports)), dtype=np.float64)
    terminal_info = np.zeros((ticks + 1, len(terminals)), dtype=np.float64)
    external = np.zeros((ticks, len(cells)), dtype=np.float64)
    queue_sizes = np.zeros((ticks + 1, len(cells)), dtype=np.int64)

    def snapshot(t):
        for i, c in enumerate(cells):
            soma[t, i] = [getattr(c, field) for field in SOMA_FIELDS]
            modulator[t, i] = c.M_vector
            queue_sizes[t, i] = len(c.propagation_queue)
        post_weight[t] = [c.postsynaptic_points[s].u_i.info for c in cells for s in c.postsynaptic_points]
        terminal_info[t] = [c.presynaptic_points[s].u_o.info for c in cells for s in c.presynaptic_points]

    original_tick = Neuron.tick

    def observed_tick(cell, external_inputs, current_tick, dt=1.0):
        i = cell_rows[cell.id]
        section = slice(offsets[i], offsets[i + 1])
        inputs[current_tick, section] = cell.input_buffer
        active = np.flatnonzero(cell.input_buffer[:, 0] > 0)
        events = original_tick(cell, external_inputs, current_tick, dt)
        # The native code records the actual local product before plasticity.
        # Inactive synapses retain old `potential`, so explicitly exclude them.
        for sid in active:
            local_potential[current_tick, offsets[i] + sid] = cell.postsynaptic_points[sid].potential
        return events

    snapshot(0)
    with patch.object(Neuron, "tick", observed_tick):
        for t in range(ticks):
            driven = stimulated if t == 4 else apl if t == 20 else []
            for root in driven:
                preparation.stimulate(root, 40.0)
                external[t, cell_rows[preparation.root_to_id[root]]] = 40.0
            net.run_tick()
            snapshot(t + 1)

    result = {
        "claim": "execution/instrumentation check only; no physiological or behavioral replication",
        "ticks": ticks, "anatomy": graph.summary(), "anatomical_provenance": graph.provenance,
        "assumptions": preparation.assumptions,
        "runtime": {"python": sys.version, "numpy": np.__version__},
        "code_sha256": {str(p): sha256(p) for p in (
            Path(__file__), Path(__file__).with_name("paula.py"),
            Path(__file__).with_name("connectome.py"),
            ensure_paula_available() / "neuron/neuron.py",
            ensure_paula_available() / "neuron/network.py",
            ensure_paula_available() / "neuron/extensions/graded.py")},
        "protocol": {"kc_pulse_tick": 4, "kc_roots": stimulated, "apl_pulse_tick": 20,
                     "apl_root": apl[0], "input_amplitude": 40.0, "stimulus": "artificial single-tick current, not odor or thermal activation"},
        "recording": {"soma_fields": SOMA_FIELDS, "soma_time": "initial then after each tick",
                      "inputs_time": "actual buffer at entry to native Neuron.tick, after cleft and external delivery",
                      "local_potential_time": "actual native product before postsynaptic update, zero when no positive input",
                      "weights_time": "initial then after each tick; input and terminal columns have explicit ID pairs",
                      "omitted": ["full delayed-event queues", "retrograde error vectors", "terminal modulator vectors, initially zero here"],
                      "not_a_checkpoint": True},
        "checks": {"soma_finite_except_last_fire_sentinels": bool(np.isfinite(soma[:, :, :6]).all()),
                   "post_weights_finite": bool(np.isfinite(post_weight).all()),
                   "positive_terminal_coefficients": bool((terminal_info > 0).all()),
                   "negative_local_potential_samples": int((local_potential < 0).sum()),
                   "changed_postsynaptic_coefficients": int(np.any(post_weight[1:] != post_weight[0], axis=0).sum()),
                   "changed_terminal_coefficients": int(np.any(terminal_info[1:] != terminal_info[0], axis=0).sum())},
        "runtime_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "trace.npz").open("xb") as out:
        np.savez_compressed(out, soma=soma, modulator=modulator, inputs=inputs,
                            local_potential=local_potential, post_weight=post_weight,
                            terminal_info=terminal_info, external=external,
                            queue_sizes=queue_sizes, cell_ids=np.asarray([c.id for c in cells]),
                            root_ids=np.asarray(roots), postsynaptic_ports=np.asarray(ports, dtype=np.int64),
                            terminals=np.asarray(terminals, dtype=np.int64).reshape(-1, 2),
                            edge_bindings=preparation.edge_bindings,
                            incoming_boundary_ports=preparation.incoming_boundary_ports,
                            outgoing_boundary_terminals=preparation.outgoing_boundary_terminals)
    result["trace_sha256"] = sha256(output / "trace.npz")
    with (output / "manifest.json").open("x") as out:
        json.dump(result, out, indent=2)
    return result


def main(source: Path, output: Path):
    result = run_probe(Subgraph.load(source), output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    typer.run(main)
