"""Targeted onset controls in the complete reunited preparation, not a repair.

Keep every measured pair. Compare native release blockade of the first
positive-sign LN recruited in the original trace with a separately declared
signed-current sensitivity control for curated GABA annotations. The latter
does not implement target-specific GABA receptors, electrical coupling or MIP.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import inspect
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

import numpy as np

from .antennal_identity import identity_audit
from .connectome import Subgraph
from .orn_train import pulse_course
from .paula import ORNReleaseDepression, Neuron
from .pn_current_steps import prepare
from .prisco import digest, dump_new
from neuron.neuron import RetrogradeSignalEvent
from neuron.extensions.experimental.release_depression import DepressingReleaseNeuron
from neuron.extensions.experimental.input_current import InputCurrentNeuron
from neuron.extensions.experimental.passive_cable import LocalCableGradedNeuron
from neuron.network import NeuronNetwork

CONDITIONS = ("intact", "first_positive_ln_block", "curated_gaba_negative_control")
# Chosen from the original no-depression trace, not from intervention outcomes.
FIRST_POSITIVE_LN = "720575940633483807"
WATCH = ("720575940628343634", FIRST_POSITIVE_LN, "720575940618757666", "720575940623636701")
TICKS = 1600


def negative_gaba_control(prep, graph):
    """Before tick zero, change signed receiving coefficients, never the graph."""
    roots = identity_audit(graph)["gaba_positive_candidates"]
    ids = {prep.root_to_id[r] for r in roots}
    records = []
    if prep.network.current_tick != 0:
        raise ValueError("A sign sensitivity control must be initialized before the first tick")
    for row, pre, terminal, post, port in prep.edge_bindings:
        if int(pre) not in ids:
            continue
        cell = prep.network.network.neurons[int(post)]
        old = cell.postsynaptic_points[int(port)].u_i.info
        if old <= 0:
            raise ValueError("Candidate receiving coefficient was not positive")
        cell.postsynaptic_points[int(port)].u_i.info = -old
        records.append([int(row), int(pre), int(terminal), int(post), int(port), float(old), float(-old)])
    return {"source_roots": roots, "changed_receiving_coefficients": records,
        "outside_targets": "No boundary cell is instantiated; its absent receptor is not changed.",
        "scope": "All internal targets of curated GABA / positive-source-model ALLNs; unchanged magnitudes.",
        "claim": "Sensitivity to assumed fast-current polarity, not established receptor physiology."}


def run(graph_path, intrinsic_path, tail_path, spatial, reference, output, condition):
    if condition not in CONDITIONS or output.exists():
        raise ValueError("Unknown condition or existing output")
    started = time.perf_counter()
    graph = Subgraph.load(graph_path)
    intrinsic = json.loads(intrinsic_path.read_text())
    tail = json.loads(tail_path.read_text())["fits"]["2"]["all_cells"]
    ref = json.loads((reference / "analysis.json").read_text())
    if ref["condition"] != "no_depression" or ref["ticks"] != 4200:
        raise ValueError("Need the completed original no-depression experiment")
    for p, h in {**intrinsic["source_hashes"], **ref["source_hashes"]}.items():
        if digest(Path(p)) != h:
            raise ValueError(f"Changed source: {p}")
    file_sources = [Path(__file__), Path(__file__).with_name("antennal_identity.py"),
        graph_path / "manifest.json", intrinsic_path, tail_path, spatial / "analysis.json",
        reference / "analysis.json", Path(inspect.getfile(NeuronNetwork))]
    hashes = {**ref["source_hashes"], **{str(p.resolve()): digest(p) for p in file_sources}}
    orns = tuple(r for r in graph.selected if graph.nodes[r]["annotation"]["hemibrain_type"] == "ORN_DL5")
    random.seed(0)
    prep, target = prepare(graph, intrinsic, tail, spatial=spatial,
        release_depression=ORNReleaseDepression(orns, 0., 893.))
    roots = list(prep.root_to_id)
    cells = [prep.network.network.neurons[prep.root_to_id[r]] for r in roots]
    with np.load(reference / "structure.npz") as s:
        np.testing.assert_array_equal(s["roots"], roots)
        for key in ("edge_bindings", "incoming_boundary_ports", "outgoing_boundary_terminals"):
            np.testing.assert_array_equal(s[key], getattr(prep, key))
    intervention = negative_gaba_control(prep, graph) if condition == CONDITIONS[2] else None
    source_cols = {prep.root_to_id[r]: i for i, r in enumerate(orns)}
    watch = [prep.network.network.neurons[prep.root_to_id[r]] for r in WATCH]
    replays = [copy.deepcopy(c) for c in watch]
    watch_cols = {c.id: i for i, c in enumerate(watch)}
    ports = np.r_[0, np.cumsum([c.params.num_inputs for c in watch])]
    command, _ = pulse_course(len(orns))
    block_id = prep.root_to_id[FIRST_POSITIVE_LN] if condition == CONDITIONS[1] else None
    apl = next(c for r, c in zip(roots, cells) if graph.nodes[r]["annotation"]["hemibrain_type"] == "APL")
    output.mkdir(parents=True)
    with (output / "structure.npz").open("xb") as f:
        np.savez_compressed(f, roots=np.array(roots), watch_roots=np.array(WATCH), watch_offsets=ports,
            edge_bindings=prep.edge_bindings, incoming_boundary_ports=prep.incoming_boundary_ports,
            outgoing_boundary_terminals=prep.outgoing_boundary_terminals)
    native_hillock, native_tick = Neuron._hillock_current, Neuron.tick
    arrays, start, replaying = {}, 0, False

    def current(cell, tick, dt):
        col = None if replaying else watch_cols.get(cell.id)
        if col is not None:
            arrays["inputs"][tick-start, ports[col]:ports[col+1]] = cell.input_buffer
        total = native_hillock(cell, tick, dt)
        if col is not None:
            arrays["current"][tick-start, col] = total
        source = None if replaying else source_cols.get(cell.id)
        return total if source is None or command[tick, source] == 0 else total + command[tick, source]

    def released(cell, inputs, tick, dt=1.):
        events = native_tick(cell, inputs, tick, dt)
        if replaying or cell.id != block_id:
            return events
        if any(not isinstance(e, RetrogradeSignalEvent) and not (isinstance(e, tuple) and len(e) == 3) for e in events):
            raise TypeError("Unknown native event")
        forward = sum(isinstance(e, tuple) for e in events)
        arrays["block_events"][tick-start] = [forward, 0, len(events)-forward]
        return [e for e in events if isinstance(e, RetrogradeSignalEvent)]

    chunks = []
    with ExitStack() as stack:
        stack.enter_context(patch.object(Neuron, "_hillock_current", current))
        stack.enter_context(patch.object(Neuron, "tick", released))
        for start in range(0, TICKS, 400):
            stop = min(TICKS, start+400); n = stop-start
            arrays = {"soma": np.zeros((n, len(cells), 3)),
                "inputs": np.zeros((n, int(ports[-1]), 4), dtype=np.float32),
                "current": np.zeros((n, len(watch))), "weights": np.zeros((n, int(ports[-1]))),
                "intrinsic": np.zeros((n, len(watch), 4)), "apl_max": np.zeros(n),
                "block_events": np.zeros((n, 3), dtype=np.int64)}
            for tick in range(start, stop):
                i = tick-start
                prep.network.run_tick()
                arrays["soma"][i] = [[float(c.S), float(c.O), float(c.F_avg)] for c in cells]
                arrays["apl_max"][i] = apl.terminal_release.max(initial=0)
                for col, cell in enumerate(watch):
                    sec = slice(ports[col], ports[col+1])
                    arrays["weights"][i, sec] = [p.u_i.info for p in cell.postsynaptic_points.values()]
                    arrays["intrinsic"][i, col] = [cell.t_ref, cell.r, cell.b, cell.params.lambda_param]
                    # Replay current receiving history, not routed return history.
                    other = replays[col]; other.input_buffer[:] = arrays["inputs"][i, sec]
                    replaying = True
                    try:
                        other.tick({}, tick)
                    finally:
                        replaying = False
                    np.testing.assert_array_equal([other.S, other.O, other.F_avg], [cell.S, cell.O, cell.F_avg])
                    np.testing.assert_array_equal([p.u_i.info for p in other.postsynaptic_points.values()], arrays["weights"][i, sec])
                    np.testing.assert_array_equal([other.t_ref, other.r, other.b, other.params.lambda_param], arrays["intrinsic"][i, col])
            if any(not np.isfinite(a).all() for a in arrays.values()):
                raise ValueError("Nonfinite recording")
            filename = f"ticks-{start:06d}-{stop:06d}.npz"
            with (output / filename).open("xb") as f:
                np.savez_compressed(f, **arrays)
            chunks.append({"file": filename, "start": start, "stop": stop, "sha256": digest(output / filename)})
            print(f"{condition}: {stop}/{TICKS}, {time.perf_counter()-started:.1f}s", flush=True)
    if any(c.params.eta_post <= 0 or c.params.eta_retro <= 0 or c._ablation for c in cells):
        raise ValueError("Adaptation disabled")
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("Source changed during experiment")
    report = {"schema": 1, "condition": condition, "ticks": TICKS,
        "intervention": intervention, "blocked_root": FIRST_POSITIVE_LN if block_id is not None else None,
        "watch_roots": list(WATCH), "receiving_replays_exact": TICKS*len(WATCH),
        "chunks": chunks, "structure_sha256": digest(output / "structure.npz"),
        "source_hashes": hashes, "runtime_seconds": time.perf_counter()-started,
        "reference": str(reference.resolve()), "anatomy": graph.summary(), "assumptions": prep.assumptions,
        "limits": ["Same nominal 1-ms/tick pulse course, no ORN depression; not odor stimulation.",
            "Stops after 400 ticks of the first recovery; cannot establish long-run stability or test 50-Hz recruitment.",
            "Only four interneurons have full receiving-port histories; whole-network soma is recorded every tick.",
            "Replays check soma, receiving weights and intrinsic fields, not outgoing return-history or checkpoints.",
            "GABA current sign control is a hypothesis; curated identity is not target receptor physiology."],
        "claim": "Causal onset diagnostic with preserved anatomy, not physiological or behavioral acceptance"}
    dump_new(output / "analysis.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "intrinsic", "tail", "spatial", "reference", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("condition", choices=CONDITIONS)
    a = p.parse_args()
    run(a.graph, a.intrinsic, a.tail, a.spatial, a.reference, a.output, a.condition)


if __name__ == "__main__":
    main()
