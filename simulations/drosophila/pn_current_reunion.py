"""Reunite a DL5 current hypothesis with the unchanged selected PN/KC/APL graph.

All graph edges remain. The only imposed signal is a train at one actual ORN
boundary port. Record every cell's soma and every target-PN input each tick.
This tests propagation into neural consumers, not odor selectivity or learning.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import gc
import inspect
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

import numpy as np

from .connectome import Subgraph
from .paula import Dynamics, PNCurrentKernel, build_paula, Neuron
from .pn_current import dl5_cut
from .prisco import digest, dump_new
from neuron.extensions.experimental.input_current import InputCurrentNeuron
from neuron.extensions.experimental.passive_cable import LocalCableGradedNeuron
from neuron.network import NeuronNetwork


def run(graph_path: Path, fit_path: Path, output: Path, *, spatial=None, ticks=320):
    if output.exists():
        raise FileExistsError(output)
    if type(ticks) is not int or ticks < 240:
        raise ValueError("Need at least 240 ticks for the 200-tick train and release tail")
    graph = Subgraph.load(graph_path)
    root, cut = dl5_cut(graph)
    fit = json.loads(fit_path.read_text())["fits"]["2"]["all_cells"]
    taus, fractions = np.array(fit["decay_ms"], dtype=float), np.array(fit["peak_fractions"], dtype=float)
    # Explicit provisional 1 ms/tick conversion for current kinetics alone.
    gain = float(np.sum(fractions/-np.expm1(-1/taus)))
    source_row = min(int(e[8]) for e in cut.edges if str(e[1]) == root and
                     cut.nodes[str(e[0])]["annotation"]["hemibrain_type"] == "ORN_DL5")
    dynamics = Dynamics(weight_per_count=.075,
                        apl_representation="local_cable" if spatial is not None else "global_graded")
    output.mkdir(parents=True)
    results = []
    reference_bindings = None
    for condition in ("native", "area", "peak", "native_charge_matched"):
        started = time.perf_counter()
        random.seed(0)
        spec = None if condition.startswith("native") else PNCurrentKernel(
            root, "ORN_DL5", tuple(taus), tuple(fractions), condition)
        prep = build_paula(graph, dynamics, spatial=spatial, current_kernel=spec)
        bindings = (prep.edge_bindings, prep.incoming_boundary_ports, prep.outgoing_boundary_terminals)
        if reference_bindings is None:
            reference_bindings = tuple(a.copy() for a in bindings)
        else:
            for expected, actual in zip(reference_bindings, bindings, strict=True):
                np.testing.assert_array_equal(expected, actual)
        cells = list(prep.network.network.neurons.values())
        roots = list(prep.root_to_id)
        target = prep.network.network.neurons[prep.root_to_id[root]]
        pn_row = roots.index(root)
        found = prep.incoming_boundary_ports[prep.incoming_boundary_ports[:, 0] == source_row]
        if found.shape != (1, 4):
            raise ValueError("Requested ORN must be an absent boundary source in this preparation")
        port = int(found[0, 3])
        nports = target.params.num_inputs
        soma = np.zeros((ticks+1, len(cells), 3))
        inputs = np.zeros((ticks, nports, 4), dtype=np.float32)
        impulse = np.zeros((ticks, nports))
        current = np.zeros_like(impulse)
        weights = np.zeros((ticks+1, nports))
        terminal_weights = np.zeros((ticks+1, len(target.presynaptic_points)))
        components = np.zeros((ticks, nports, 2))
        apl = next((c for r, c in zip(roots, cells, strict=True)
                    if graph.nodes[r]["annotation"]["hemibrain_type"] == "APL"), None)
        apl_terminals = np.zeros((ticks, len(apl.presynaptic_points) if apl else 0))
        apl_peak_voltage = np.zeros(ticks)

        def snapshot(t):
            soma[t] = [[float(c.S), float(c.O), float(c.F_avg)] for c in cells]
            weights[t] = [p.u_i.info for p in target.postsynaptic_points.values()]
            terminal_weights[t] = [p.u_o.info for p in target.presynaptic_points.values()]

        original = type(target)._hillock_current

        def observed(cell, tick, dt):
            if cell is not target:
                return original(cell, tick, dt)
            inputs[tick] = cell.input_buffer
            for arrival, _, value, sid in cell.propagation_queue:
                if arrival <= tick:
                    impulse[tick, sid] += float(value*cell.params.delta_decay**cell.distances[sid])
            result = original(cell, tick, dt)
            if spec is None:
                current[tick] = impulse[tick]
            else:
                current[tick] = cell.last_port_current
                components[tick, cell.current_ports] = cell.current_state
                np.testing.assert_allclose(impulse[tick], cell.arrived_port_impulse, atol=1e-10, rtol=1e-6)
            if not np.isclose(result, current[tick].sum(), atol=1e-6, rtol=1e-6):
                raise ValueError("Observed current and native integration disagree")
            return result

        snapshot(0)
        with patch.object(type(target), "_hillock_current", observed):
            for tick in range(ticks):
                if tick < 200 and tick % 5 == 0:
                    target.input_buffer[port, 0] = gain if condition == "native_charge_matched" else 1.
                prep.network.run_tick()
                snapshot(tick+1)
                if apl is not None and hasattr(apl, "cable"):
                    apl_terminals[tick] = apl.terminal_release
                    apl_peak_voltage[tick] = float(apl.cable.voltage.max())
                if (tick+1) % 64 == 0:
                    print(f"{condition}: {tick+1}/{ticks} ticks", flush=True)
        if not np.isfinite(soma).all() or not np.isfinite(current).all():
            raise ValueError("Nonfinite recorded state")
        kc_rows = [i for i, r in enumerate(roots) if graph.nodes[r]["annotation"]["cell_class"] == "Kenyon_Cell"]
        pn_rows = [i for i, r in enumerate(roots) if graph.nodes[r]["annotation"]["cell_class"] == "ALPN"]
        trace_path = output / f"{condition}.npz"
        with trace_path.open("xb") as stream:
            np.savez_compressed(stream, soma=soma, pn_inputs=inputs, pn_impulse=impulse,
                pn_current=current, pn_components=components, pn_post_weight=weights,
                pn_terminal_weight=terminal_weights, roots=np.array(roots),
                apl_terminal_release=apl_terminals, apl_peak_voltage=apl_peak_voltage)
        result = {"condition": condition, "runtime_seconds": time.perf_counter()-started,
            "root": root, "source_row": source_row, "port": port,
            "target_spike_ticks": np.flatnonzero(soma[1:, pn_row, 1] > 0).tolist(),
            "kc_cells_fired": int(np.any(soma[1:, kc_rows, 1] > 0, axis=0).sum()),
            "kc_spikes": int((soma[1:, kc_rows, 1] > 0).sum()),
            "all_pn_spikes": int((soma[1:, pn_rows, 1] > 0).sum()),
            "nonstimulated_input_ports_ever_active": np.flatnonzero(np.any(inputs[:, :, 0] > 0, axis=0) & (np.arange(nports) != port)).tolist(),
            "apl_max_local_release": float(apl_terminals.max(initial=0)),
            "apl_max_compartment_voltage": float(apl_peak_voltage.max(initial=0)),
            "target_post_coefficients_changed": int(np.count_nonzero(weights[-1] != weights[0])),
            "target_terminal_coefficients_changed": int(np.count_nonzero(terminal_weights[-1] != terminal_weights[0])),
            "assumptions": prep.assumptions, "trace_sha256": digest(trace_path)}
        results.append(result)
        dump_new(output / f"{condition}.json", result)
        # Next build must not overlap the previous large cable/network in RAM.
        del prep, cells, target, apl
        gc.collect()
    report = {"schema": 1, "anatomy": graph.summary(), "conditions": results,
              "protocol": {"ticks": ticks, "releases": "one boundary ORN port every five ticks during 0..199",
                           "stimulus_type": "imposed receptor release, not simulated odor or ORN",
                           "peak_kernel_discrete_charge_gain": gain, "ms_per_tick_hypothesis": 1,
                           "calibrated_physical_clock": None, "all_connections_preserved": True,
                           "ongoing_adaptation": True},
              "recording": {"soma_fields": ["S", "O", "F_avg"], "soma_time": "initial then after each tick",
                            "inputs": "actual target PN inputs at the hillock hook, before native plasticity",
                            "omitted": ["other cells' input currents and per-tick weights", "full APL compartment arrays", "full event queues", "individual retrograde signals"],
                            "not_a_checkpoint": True},
              "source_hashes": {str(p): digest(p) for p in (graph_path / "manifest.json", fit_path,
                 Path(__file__), Path(__file__).with_name("paula.py"), Path(inspect.getfile(Neuron)),
                 Path(inspect.getfile(InputCurrentNeuron)), Path(inspect.getfile(NeuronNetwork)),
                 Path(inspect.getfile(LocalCableGradedNeuron)), Path(__file__).with_name("pn_current.py"))},
              "claim": "Connected input-mechanism diagnostic with existing anatomy; not physiological, odor, behavior or learning acceptance"}
    dump_new(output / "analysis.json", report)
    return report


def analyze(graph_path: Path, record_path: Path, isolated_path: Path, output: Path):
    """Open-loop exact replay and input-port lesions, never decoded feedback.

    Replaying recorded neural inputs can locate a PN's changed response. A
    lesion here does not allow the rest of the network to react to that lesion;
    it is explicitly not a closed-loop network intervention.
    """
    if output.exists():
        raise FileExistsError(output)
    graph = Subgraph.load(graph_path)
    root, cut = dl5_cut(graph)
    manifest = json.loads((record_path / "analysis.json").read_text())
    if manifest["source_hashes"][str(graph_path / "manifest.json")] != digest(graph_path / "manifest.json"):
        raise ValueError("Reunion and analysis must use the same graph")
    catalog = sorted((e for e in cut.edges if str(e[1]) == root), key=lambda e: int(e[8]))
    results = []
    with np.load(isolated_path / "per_tick.npz") as isolated:
        for condition in manifest["conditions"]:
            path = record_path / f"{condition['condition']}.npz"
            if digest(path) != condition["trace_sha256"]:
                raise ValueError("Recorded trace hash changed")
            with np.load(path) as trace:
                inputs, observed = trace["pn_inputs"], trace["soma"]
                weights = trace["pn_post_weight"]
                target_row = trace["roots"].tolist().index(root)
                actual = observed[1:, target_row, :2]
                if np.any(inputs[:, :, 2:] != 0):
                    raise ValueError("This replay does not reconstruct external neuromodulation")
                assumptions = condition["assumptions"]
                dynamics = replace(Dynamics(**assumptions["parameters"]), apl_representation="global_graded")
                config = assumptions.get("input_current")
                spec = PNCurrentKernel(**{f.name: config[f.name] for f in fields(PNCurrentKernel)}) if config else None
                feedback = condition["nonstimulated_input_ports_ever_active"]

                def replay(block):
                    prep = build_paula(cut, dynamics, current_kernel=spec)
                    cell = prep.network.network.neurons[prep.root_to_id[root]]
                    state = np.zeros_like(actual)
                    for t, row in enumerate(inputs):
                        cell.input_buffer[:] = row
                        cell.input_buffer[block] = 0
                        cell.tick({}, t)
                        state[t] = [cell.S, cell.O]
                    learned = np.array([p.u_i.info for p in cell.postsynaptic_points.values()])
                    return state, learned

                replayed, learned = replay([])
                np.testing.assert_array_equal(replayed, actual)
                np.testing.assert_array_equal(learned, weights[-1])
                original = isolated[f"ms1_{condition['condition']}_train_5"][:len(actual), 5:7]
                difference = np.flatnonzero(np.any(actual != original, axis=1))
                spike_difference = np.flatnonzero(actual[:, 1] != original[:, 1])
                lesions = []
                for block in ([[p] for p in feedback] + ([feedback] if len(feedback) > 1 else [])):
                    changed, _ = replay(block)
                    lesions.append({"blocked_ports": block,
                        "spike_ticks": np.flatnonzero(changed[:, 1] > 0).tolist(),
                        "spike_difference_ticks_vs_connected": np.flatnonzero(changed[:, 1] != actual[:, 1]).tolist(),
                        "exact_isolated_soma_output_recovered": bool(np.array_equal(changed, original))})
                port_evidence = []
                for p in feedback:
                    edge = catalog[p]
                    source = str(edge[0])
                    active = np.flatnonzero(inputs[:, p, 0] > 0)
                    arrived = np.flatnonzero(trace["pn_impulse"][:, p] != 0)
                    port_evidence.append({"port": p, "source_row": int(edge[8]), "source_root": source,
                        "annotation": graph.nodes[source]["annotation"], "counted_contacts": int(edge[4]),
                        "source_model_sign": int(edge[5]), "first_receptor_tick": int(active[0]),
                        "first_current_tick": int(arrived[0]) if len(arrived) else None,
                        "signed_current_sum": float(trace["pn_current"][:, p].sum())})
                results.append({"condition": condition["condition"], "exact_recorded_soma_output_replay": True,
                    "exact_final_post_coefficients": True, "replayed_ticks": len(actual),
                    "first_difference_from_isolation": int(difference[0]) if len(difference) else None,
                    "spike_difference_ticks_from_isolation": spike_difference.tolist(),
                    "feedback_inputs": port_evidence, "open_loop_input_lesions": lesions})
    report = {"schema": 1, "conditions": results,
        "source_hashes": {str(p): digest(p) for p in (record_path / "analysis.json",
            isolated_path / "per_tick.npz", graph_path / "manifest.json", Path(__file__))},
        "claim": "Exact target-PN replay and open-loop input attribution, not a closed-loop lesion or physiological validation"}
    dump_new(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("graph", type=Path)
    run_parser.add_argument("fit", type=Path)
    run_parser.add_argument("output", type=Path)
    run_parser.add_argument("--spatial", type=Path)
    check = sub.add_parser("analyze")
    check.add_argument("graph", type=Path)
    check.add_argument("record", type=Path)
    check.add_argument("isolated", type=Path)
    check.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args.graph, args.fit, args.output, spatial=args.spatial)
    else:
        analyze(args.graph, args.record, args.isolated, args.output)


if __name__ == "__main__":
    main()
