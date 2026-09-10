"""Paired current injections over identified, boundary-preserving chemical pairs.

An explicitly hypothetical passive electrical contact is tested separately from
the retained chemical graph. This is a mechanism assay, not a quantitative fit
to Yaksi and Wilson's driver-defined eLN population or a whole-network repair.
No trial labels or measured outcomes enter the PAULA cells. Learning stays on.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import inspect
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from .connectome import Subgraph
from .electrophysiology import CurrentElectrode
from .paula import Neuron
from .pn_current_steps import prepare
from .prisco import digest, dump_new
from neuron.neuron import RetrogradeSignalEvent
from neuron.network import NeuronNetwork
from neuron.extensions.experimental.electrical import ElectricalCoupling
from neuron.extensions.experimental.input_current import InputCurrentNeuron

PN = "720575940617207185"
LN = "720575940633483807"
FIELDS = ("S_before", "S_after", "O", "F_avg", "t_ref", "r", "b",
          "chemical_current", "electrical_current", "total_current")
WEIGHTS = ("post_info", "post_plast", "terminal_info", "terminal_mod0", "terminal_mod1")


def pair_cut(graph, roots):
    if len(roots) != 2 or len(set(roots)) != 2 or not set(roots) <= set(graph.selected):
        raise ValueError("Need two selected cells, not promoted boundary stubs")
    selected = np.array(roots, dtype=np.int64)
    edges = graph.edges[np.isin(graph.edges[:, 0], selected) | np.isin(graph.edges[:, 1], selected)].copy()
    endpoints = set(roots) | {str(r) for r in edges[:, :2].flat}
    cut = Subgraph(tuple(roots), {r: graph.nodes[r] for r in endpoints}, edges,
        {"parent_provenance": graph.provenance, "cut": "paired recording; all incident ports retained"})
    cut.validate()
    return cut


def course(level, trials=2, duration=500, period=5000, baseline=500):
    if (not np.isfinite(level) or level == 0 or any(type(v) is not int or v < 1
            for v in (trials, duration, period, baseline)) or period <= duration):
        raise ValueError("Need finite nonzero current, positive timing and recovery between steps")
    command = np.zeros(baseline+trials*period)
    epochs = []
    for i in range(trials):
        start = baseline+i*period
        command[start:start+duration] = level
        epochs.append({"trial": i, "start": start, "stop": start+duration,
                       "recovery_stop": start+period, "level": float(level)})
    return command, epochs


def all_weights(cells):
    return np.array([float(v) for c in cells for p in c.postsynaptic_points.values()
                     for v in (p.u_i.info, p.u_i.plast, *p.u_i.adapt)] +
                    [float(v) for c in cells for p in c.presynaptic_points.values()
                     for v in (p.u_o.info, *p.u_o.mod, p.u_i_retro)])


def simulate(graph, intrinsic, tail, source, level, g, blocked, *, trials=2,
             duration=500, period=5000, baseline=500, ordinary=False):
    roots = graph.selected
    if len(roots) != 2 or source not in roots or not np.isfinite(g) or g < 0:
        raise ValueError("Invalid pair, source or conductance")
    if ordinary and g != 0:
        raise ValueError("Ordinary chemical control cannot have a junction")
    prep, _ = prepare(graph, intrinsic, tail, electrical_roots=() if ordinary else roots)
    cells = [prep.network.network.neurons[prep.root_to_id[r]] for r in roots]
    net = prep.network
    coupling = None if ordinary else ElectricalCoupling(net, [(cells[0].id, cells[1].id, g)])
    pulse, epochs = course(level, trials, duration, period, baseline)
    command = np.zeros((len(pulse), 2)); command[:, roots.index(source)] = pulse
    bindings = prep.edge_bindings
    data = {"trace": np.zeros((len(pulse), 2, len(FIELDS))), "command": command,
            "inputs": np.zeros((len(pulse), len(bindings), 4), dtype=np.float32),
            "weights": np.zeros((len(pulse), len(bindings), len(WEIGHTS))),
            "events": np.zeros((len(pulse), 2, 3), dtype=np.int64),
            "initial_weights": all_weights(cells), "edge_bindings": bindings,
            "incoming_boundary_ports": prep.incoming_boundary_ports,
            "outgoing_boundary_terminals": prep.outgoing_boundary_terminals}
    cell_rows = {c.id: i for i, c in enumerate(cells)}
    receiving = {c.id: [(j, int(e[4])) for j, e in enumerate(bindings) if int(e[3]) == c.id] for c in cells}
    native_tick = Neuron.tick

    def observed(c, external, tick, dt=1.):
        rows = receiving[c.id]
        recorded = c.input_buffer[[p for _, p in rows]]
        if np.count_nonzero(c.input_buffer) != np.count_nonzero(recorded):
            raise ValueError("Undeclared boundary or electrode synaptic input")
        for (j, _), values in zip(rows, recorded):
            data["inputs"][tick, j] = values
        events = native_tick(c, external, tick, dt)
        if any(not isinstance(e, RetrogradeSignalEvent) and not (isinstance(e, tuple) and len(e) == 3) for e in events):
            raise TypeError("Unknown native event")
        forward = sum(isinstance(e, tuple) for e in events)
        data["events"][tick, cell_rows[c.id]] = [forward, 0 if blocked else forward, len(events)-forward]
        return [e for e in events if isinstance(e, RetrogradeSignalEvent)] if blocked else events

    with ExitStack() as stack:
        stack.enter_context(patch.object(Neuron, "tick", observed))
        electrodes = [stack.enter_context(CurrentElectrode(c, command[:, i])) for i, c in enumerate(cells)]
        for tick in range(len(pulse)):
            before = [float(c.S) for c in cells]
            (net if ordinary else coupling).run_tick()
            for i, c in enumerate(cells):
                chemical = electrodes[i].native_current[tick] if ordinary else c.electrical_native_current
                gap = 0. if ordinary else c.electrical_current
                data["trace"][tick, i] = [before[i], c.S, c.O, c.F_avg, c.t_ref, c.r, c.b,
                    chemical, gap, electrodes[i].total_current[tick]]
            for j, (_, pre, terminal, post, port) in enumerate(bindings):
                a = net.network.neurons[int(pre)].presynaptic_points[int(terminal)].u_o
                b = net.network.neurons[int(post)].postsynaptic_points[int(port)].u_i
                data["weights"][tick, j] = [b.info, b.plast, a.info, *a.mod]
    data["final_weights"] = all_weights(cells)
    if any(c.params.eta_post <= 0 or c.params.eta_retro <= 0 or c._ablation for c in cells):
        raise ValueError("Native learning disabled")
    if any(not np.isfinite(a).all() for a in data.values()):
        raise ValueError("Nonfinite recording")
    audit(data, g, [c.params.lambda_param for c in cells], blocked)
    return data, {"roots": list(roots), "source": source, "g": g, "chemical_release_blocked": blocked,
        "epochs": epochs, "assumptions": prep.assumptions,
        "lambda_ticks": [c.params.lambda_param for c in cells],
        "changed_weights": int(np.count_nonzero(data["initial_weights"] != data["final_weights"])),
        "protocol": {"trials": trials, "duration": duration, "period": period, "baseline": baseline}}


def audit(data, g, lambdas, blocked):
    """Independent exchange/current/membrane/event checks, not just final firing."""
    a = data["trace"]; command = data["command"]
    np.testing.assert_array_equal(a[:, :, 0], np.vstack([np.zeros((1, 2)), a[:-1, :, 1]]))
    expected_gap = g*(a[:, ::-1, 0]-a[:, :, 0])
    np.testing.assert_array_equal(a[:, :, 8], expected_gap)
    np.testing.assert_allclose(a[:, :, 9], a[:, :, 7]+expected_gap+command, atol=1e-14, rtol=1e-14)
    raw = a[:, :, 0]+(-a[:, :, 0]+a[:, :, 9])/np.asarray(lambdas)
    if np.max(np.abs(raw)) >= 100:
        raise ValueError("Unexpected clipping, not accepted as stabilization")
    spikes = np.zeros(raw.shape)
    for i in range(2):
        last = -np.inf
        for tick, value in enumerate(raw[:, i]):
            threshold = a[tick, i, 6 if tick-last <= 3 else 5]
            if abs(value) < .005:
                threshold = a[tick, i, 5]
            if abs(value-threshold) < 5e-7:
                raise ValueError("Threshold ambiguity in independent float64 audit")
            if value >= threshold and tick-last >= 3:
                spikes[tick, i] = 1.; last = tick
    np.testing.assert_array_equal(a[:, :, 2], spikes)
    # Ordinary PAULA can accumulate a numpy.float32 chemical current; the audit
    # stores float64 and does not silently claim identical intermediate types.
    np.testing.assert_allclose(a[:, :, 1], np.where(spikes, 0., raw), atol=3e-7, rtol=0)
    np.testing.assert_array_equal(data["events"][:, :, 1], 0 if blocked else data["events"][:, :, 0])
    if blocked:
        np.testing.assert_array_equal(data["inputs"], 0)


def summarize(data, meta):
    a = data["trace"]; src = meta["roots"].index(meta["source"]); dst = 1-src
    summaries = []
    for e in meta["epochs"]:
        start, stop, end = e["start"], e["stop"], e["recovery_stop"]
        baseline = a[max(0, start-meta["protocol"]["baseline"]):start, :, 1].mean(axis=0)
        delta = a[start:stop, :, 1].mean(axis=0)-baseline
        quiet = not a[start:stop, :, 2].any()
        summaries.append({**e, "mean_delta_S": delta.tolist(),
            "subthreshold_transfer_ratio": float(delta[dst]/delta[src]) if quiet and abs(delta[src]) > 1e-12 else None,
            "ratio_exclusion": None if quiet else "spike-reset soma is not the experimental low-pass membrane waveform",
            "step_spikes": a[start:stop, :, 2].sum(axis=0).astype(int).tolist(),
            "recovery_spikes": a[stop:end, :, 2].sum(axis=0).astype(int).tolist(),
            "last_recovery_max_abs_S": np.max(np.abs(a[max(stop, end-100):end, :, 1]), axis=0).tolist(),
            "chemical_charge": a[start:stop, :, 7].sum(axis=0).tolist(),
            "electrical_charge": a[start:stop, :, 8].sum(axis=0).tolist()})
    return summaries


def run(graph_path, intrinsic_path, tail_path, output, *, roots=(PN, LN), conductances=(0., .003, .03, .3),
        levels=(-.5, .5, 2.), trials=2, duration=500, period=5000, baseline=500):
    if output.exists():
        raise FileExistsError(output)
    intrinsic = json.loads(intrinsic_path.read_text())
    for p, expected in intrinsic["source_hashes"].items():
        if digest(Path(p)) != expected:
            raise ValueError(f"Changed calibration source: {p}")
    graph = pair_cut(Subgraph.load(graph_path), roots)
    tail = json.loads(tail_path.read_text())["fits"]["2"]["all_cells"]
    files = [graph_path/"manifest.json", intrinsic_path, tail_path, Path(__file__),
        Path(__file__).with_name("paula.py"), Path(__file__).with_name("pn_current_steps.py"),
        Path(inspect.getfile(CurrentElectrode)), Path(inspect.getfile(Neuron)),
        Path(inspect.getfile(NeuronNetwork)), Path(inspect.getfile(ElectricalCoupling)),
        Path(inspect.getfile(InputCurrentNeuron))]
    hashes = {str(p.resolve()): digest(p) for p in files}
    output.mkdir(parents=True); graph.save(output/"graph")
    records = []; start = time.perf_counter()
    for g in conductances:
        for blocked in (False, True):
            for source in roots:
                for level in levels:
                    data, meta = simulate(graph, intrinsic, tail, source, level, g, blocked,
                        trials=trials, duration=duration, period=period, baseline=baseline)
                    parity = None
                    if g == 0:
                        ordinary, _ = simulate(graph, intrinsic, tail, source, level, g, blocked, ordinary=True,
                            trials=trials, duration=duration, period=period, baseline=baseline)
                        for key in data:
                            np.testing.assert_array_equal(data[key], ordinary[key])
                        parity = sum(a.size for a in data.values())
                    name = f"course-{len(records):03d}.npz"
                    with (output/name).open("xb") as f:
                        np.savez_compressed(f, **data)
                    records.append({"file": name, "sha256": digest(output/name), **meta,
                        "ordinary_parity_values": parity, "responses": summarize(data, meta)})
                    dump_new(output/f"course-{len(records)-1:03d}.json", records[-1])
                    print(f"{len(records)}: g={g:g} block={blocked} source={roots.index(source)} I={level:g}; "
                          f"{time.perf_counter()-start:.1f}s", flush=True)
    if any(digest(Path(p)) != expected for p, expected in hashes.items()):
        raise ValueError("Source changed during recording")
    result = {"schema": 1, "source_hashes": hashes, "anatomy": graph.summary(), "records": records,
        "runtime_seconds": time.perf_counter()-start,
        "claim": "Identified chemical-pair mechanism assay with an unmeasured electrical hypothesis, not physiological acceptance",
        "limits": ["The literature's driver-defined eLN population is not identified with this FlyWire lLN2T_e root.",
            "No electrical contact or conductance was inferred from chemical counts; all tested g values are assumptions.",
            "Nominal 1 ms/tick; LN dynamics, relative leak and voltage scale are not biologically calibrated.",
            "Two repeats by default, not the paper's 40-50 trials; deterministic stimuli do not represent biological samples.",
            "Current levels are model units, not the paper's per-cell adjustment to approximately 40 mV.",
            "Acute model release block is not cadmium pharmacology or a developmental gap-junction mutation.",
            "Spike-reset somatic voltage cannot reproduce measured depolarizing coupling coefficients.",
            "All incident ports remain, but boundary cells are absent and undriven. No whole-network repair is claimed.",
            "Every tick retains both somata, currents, internal chemical inputs and pair weights; final all-port weights are retained, not full queue checkpoints."],
        "reference": "https://doi.org/10.1016/j.neuron.2010.08.041"}
    dump_new(output/"analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "intrinsic", "tail", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.graph, args.intrinsic, args.tail, args.output)


if __name__ == "__main__":
    main()
