"""Direct sensory versus lateral-cell drive in an identified ORN/PN/LN cut.

An LN current command is a controlled lateral boundary, not a public odor or
an already validated upstream circuit. All native spikes and learning remain
active; attempted pulse rates are not relabeled as observed neural rates.
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
from .ln_input_replay import cut_cells
from .paula import Neuron
from .paired_recording import all_weights
from .pn_current_steps import prepare
from .prisco import digest, dump_new
from neuron.neuron import RetrogradeSignalEvent

PN = "720575940617207185"
LN = "720575940623000858"
LESIONS = ("intact", "LN_to_ORN_block", "LN_to_PN_block", "LN_release_block")
FIELDS = ("S", "O", "F_avg", "t_ref", "r", "b")


def select(graph):
    """Anatomically selected before observing any functional outcome."""
    orns = tuple(r for r in graph.selected if graph.nodes[r]["annotation"]["hemibrain_type"] == "ORN_DL5")
    if not orns or not {PN, LN} <= set(graph.selected):
        raise ValueError("Missing selected identities")
    if graph.nodes[LN]["annotation"]["hemibrain_type"] != "lLN2F_b":
        raise ValueError("Changed LN identity")
    return cut_cells(graph, (*orns, PN, LN))


def commands(n, direct, lateral, seed, *, duration=1000, recovery=1000, baseline=200, trials=2):
    if (type(n) is not int or n < 1 or not np.isfinite([direct, lateral]).all()
            or min(direct, lateral) < 0 or max(direct, lateral) > 100
            or any(type(x) is not int or x < 1 for x in (duration, recovery, baseline, trials))):
        raise ValueError("Invalid declared stimulation")
    rng = np.random.default_rng(seed)
    # Random source-to-phase assignment, repeated exactly in the second trial.
    phases = rng.permutation(n)/n
    result = np.zeros((baseline+trials*(duration+recovery), n+1))
    epochs = []
    for trial in range(trials):
        start = baseline+trial*(duration+recovery); stop = start+duration
        epochs.append({"trial": trial, "start": start, "stop": stop, "recovery_stop": stop+recovery})
        for col in range(n+1):
            rate = direct if col < n else lateral
            if rate == 0: continue
            period = 1000/rate
            phase = phases[col] if col < n else .5
            times = np.floor(np.arange(phase*period, duration, period)).astype(int)+start
            result[times, col] = 40.
    return result, epochs


def blocked_terminals(prep, graph, lesion):
    if lesion not in LESIONS: raise ValueError("Unknown pathway lesion")
    orn_ids = {prep.root_to_id[r] for r in prep.root_to_id
               if graph.nodes[r]["annotation"]["hemibrain_type"] == "ORN_DL5"}
    terminal_targets = {int(e[2]): int(e[3]) for e in prep.edge_bindings if int(e[1]) == prep.root_to_id[LN]}
    if lesion == "LN_release_block":
        return set(prep.network.network.neurons[prep.root_to_id[LN]].presynaptic_points)
    return {t for t, target in terminal_targets.items()
            if (lesion == "LN_to_ORN_block" and target in orn_ids)
            or (lesion == "LN_to_PN_block" and target == prep.root_to_id[PN])}


def simulate(graph, intrinsic, tail, direct, lateral, seed, lesion, *, inhibition_gain=None, inhibition_decay=100., **timing):
    if inhibition_gain is None:
        prep, pn = prepare(graph, intrinsic, tail)
    else:
        from .presynaptic_prepare import prepare_inhibited
        prep, pn = prepare_inhibited(graph, intrinsic, tail, LN, inhibition_gain, inhibition_decay)
    roots = list(prep.root_to_id)
    orns = [r for r in roots if graph.nodes[r]["annotation"]["hemibrain_type"] == "ORN_DL5"]
    if roots != orns+[PN, LN]: raise ValueError("Unexpected population order")
    cells = [prep.network.network.neurons[prep.root_to_id[r]] for r in roots]
    source_ids = [prep.root_to_id[r] for r in orns+[LN]]
    source_cols = {nid: i for i, nid in enumerate(source_ids)}
    ln_id = prep.root_to_id[LN]; blocked = blocked_terminals(prep, graph, lesion)
    command, epochs = commands(len(orns), direct, lateral, seed, **timing)
    ticks = len(command)
    # Native receiving-port identity is determined by all incident rows.
    pn_edges = sorted(graph.edges[graph.edges[:, 1] == int(PN)], key=lambda e: int(e[8]))
    orn_ports = np.array([i for i, e in enumerate(pn_edges) if str(e[0]) in orns], dtype=int)
    ln_ports = np.array([i for i, e in enumerate(pn_edges) if str(e[0]) == LN], dtype=int)
    data = {"soma": np.zeros((ticks, len(cells), len(FIELDS))), "command": command,
        "source_native_current": np.zeros_like(command), "source_total_current": np.zeros_like(command),
        "pn_inputs": np.zeros((ticks, pn.params.num_inputs, 4), dtype=np.float32),
        "pn_current": np.zeros((ticks, pn.params.num_inputs)),
        "pn_weights": np.zeros((ticks, pn.params.num_inputs)),
        "ln_events": np.zeros((ticks, 3), dtype=np.int64),
        "initial_weights": all_weights(cells), "roots": np.array(roots),
        "orn_ports": orn_ports, "ln_ports": ln_ports,
        "edge_bindings": prep.edge_bindings,
        "incoming_boundary_ports": prep.incoming_boundary_ports,
        "outgoing_boundary_terminals": prep.outgoing_boundary_terminals}
    if inhibition_gain is not None:
        data["inhibition"] = np.zeros((ticks, len(orns), 5))
    native_hillock, native_tick = Neuron._hillock_current, Neuron.tick
    pn_hillock = type(pn)._hillock_current

    def electrode(cell, tick, dt):
        current = native_hillock(cell, tick, dt)
        col = source_cols.get(cell.id)
        if col is None: return current
        total = current if command[tick, col] == 0 else current+command[tick, col]
        data["source_native_current"][tick, col] = current
        data["source_total_current"][tick, col] = total
        return total

    def measure(cell, tick, dt):
        if cell is not pn: return pn_hillock(cell, tick, dt)
        data["pn_inputs"][tick] = cell.input_buffer
        current = pn_hillock(cell, tick, dt)
        data["pn_current"][tick] = cell.last_port_current
        return current

    def release(cell, inputs, tick, dt=1.):
        events = native_tick(cell, inputs, tick, dt)
        if cell.id != ln_id: return events
        if any(not isinstance(e, RetrogradeSignalEvent) and not (isinstance(e, tuple) and len(e) == 3) for e in events):
            raise TypeError("Unknown event type")
        admitted = [e for e in events if not isinstance(e, tuple) or e[1] not in blocked]
        forward = sum(isinstance(e, tuple) for e in events)
        data["ln_events"][tick] = [forward, sum(isinstance(e, tuple) for e in admitted), len(events)-forward]
        return admitted

    with ExitStack() as stack:
        stack.enter_context(patch.object(Neuron, "_hillock_current", electrode))
        stack.enter_context(patch.object(type(pn), "_hillock_current", measure))
        stack.enter_context(patch.object(Neuron, "tick", release))
        for t in range(ticks):
            prep.network.run_tick()
            data["soma"][t] = [[c.S, c.O, c.F_avg, c.t_ref, c.r, c.b] for c in cells]
            data["pn_weights"][t] = [p.u_i.info for p in pn.postsynaptic_points.values()]
            if inhibition_gain is not None:
                data["inhibition"][t] = [[c.inhibition_state, c.inhibition_fraction, c.inhibition_arriving_drive,
                    sum(c.inhibition_last_native.values()), sum(c.inhibition_last_effective.values())]
                    for c in cells[:len(orns)]]
    data["final_weights"] = all_weights(cells)
    if any(c.params.eta_post <= 0 or c.params.eta_retro <= 0 or c._ablation for c in cells):
        raise ValueError("Native adaptation disabled")
    if any(not np.isfinite(a).all() for a in data.values() if a.dtype.kind != "U"):
        raise ValueError("Nonfinite recording")
    # Blocked pathways cannot deliver same-run input at the PN. Other cut
    # boundary inputs are silent, with their ports retained in the cell.
    other = np.ones(pn.params.num_inputs, dtype=bool); other[orn_ports] = False; other[ln_ports] = False
    np.testing.assert_array_equal(data["pn_inputs"][:, other], 0)
    if lesion in ("LN_to_PN_block", "LN_release_block"):
        np.testing.assert_array_equal(data["pn_inputs"][:, ln_ports], 0)
    if lesion == "intact":
        np.testing.assert_array_equal(data["ln_events"][:, 0], data["ln_events"][:, 1])
    return data, {"direct_command_hz": direct, "lateral_command_hz": lateral, "seed": seed,
        "inhibition_gain": inhibition_gain, "inhibition_decay_ticks": inhibition_decay if inhibition_gain is not None else None,
        "lesion": lesion, "epochs": epochs, "blocked_terminals": sorted(blocked),
        "assumptions": prep.assumptions,
        "changed_weights": int(np.count_nonzero(data["initial_weights"] != data["final_weights"]))}


def summarize(data, meta):
    n = len(data["roots"])-2; soma = data["soma"]; reports = []
    for e in meta["epochs"]:
        a, b, end = e["start"], e["stop"], e["recovery_stop"]
        duration = (b-a)/1000
        reports.append({**e, "ORN_mean_hz": float(soma[a:b, :n, 1].sum()/n/duration),
            "LN_hz": float(soma[a:b, -1, 1].sum()/duration), "PN_hz": float(soma[a:b, -2, 1].sum()/duration),
            "PN_first_200_spikes": int(soma[a:min(a+200,b), -2, 1].sum()),
            "PN_last_200_spikes": int(soma[max(a,b-200):b, -2, 1].sum()),
            "PN_recovery_spikes": int(soma[b:end, -2, 1].sum()),
            "ORN_recovery_spikes": int(soma[b:end, :n, 1].sum()),
            "LN_recovery_spikes": int(soma[b:end, -1, 1].sum()),
            "PN_ORN_charge_with_recovery": float(data["pn_current"][a:end, data["orn_ports"]].sum()),
            "PN_LN_charge_with_recovery": float(data["pn_current"][a:end, data["ln_ports"]].sum()),
            "PN_final_100_max_abs_S": float(np.abs(soma[end-100:end, -2, 0]).max())})
    return reports


def run(graph_path, intrinsic_path, tail_path, output, *, direct_rates=(0., 2., 5., 10., 20., 50., 100.),
        lateral_rates=(0., 20., 80.), seeds=(11,), lesions=("intact",), inhibition_gain=None, inhibition_decay=100.):
    if output.exists(): raise FileExistsError(output)
    intrinsic = json.loads(intrinsic_path.read_text())
    for p, h in intrinsic["source_hashes"].items():
        if digest(Path(p)) != h: raise ValueError(f"Changed calibration source: {p}")
    graph = select(Subgraph.load(graph_path))
    tail = json.loads(tail_path.read_text())["fits"]["2"]["all_cells"]
    files = [Path(__file__), Path(__file__).with_name("paula.py"), Path(__file__).with_name("pn_current_steps.py"),
             Path(inspect.getfile(Neuron)), graph_path/"manifest.json", intrinsic_path, tail_path]
    if inhibition_gain is not None:
        from .presynaptic_prepare import PresynapticInhibitionNeuron
        files += [Path(__file__).with_name("presynaptic_prepare.py"), Path(inspect.getfile(PresynapticInhibitionNeuron))]
    hashes = {str(p.resolve()): digest(p) for p in files}
    output.mkdir(parents=True); graph.save(output/"graph")
    records = []; start = time.perf_counter()
    for seed in seeds:
        for lesion in lesions:
            for direct in direct_rates:
                for lateral in lateral_rates:
                    data, meta = simulate(graph, intrinsic, tail, direct, lateral, seed, lesion,
                        inhibition_gain=inhibition_gain, inhibition_decay=inhibition_decay)
                    name = f"course-{len(records):03d}.npz"
                    with (output/name).open("xb") as f: np.savez_compressed(f, **data)
                    record = {"file": name, "sha256": digest(output/name), **meta, "responses": summarize(data, meta)}
                    dump_new(output/f"course-{len(records):03d}.json", record); records.append(record)
                    print(f"{len(records)}: {lesion} direct={direct:g} lateral={lateral:g} "
                          f"PN={[r['PN_hz'] for r in record['responses']]}; {time.perf_counter()-start:.1f}s", flush=True)
    if any(digest(Path(p)) != h for p,h in hashes.items()): raise ValueError("Source changed during recording")
    result = {"schema": 1, "source_hashes": hashes, "anatomy": graph.summary(), "records": records,
        "LN_annotation": graph.nodes[LN]["annotation"], "runtime_seconds": time.perf_counter()-start,
        "claim": "Controlled direct/lateral-cell transfer assay, not public-odor normalization or physiological acceptance",
        "limits": ["Lateral current injection replaces absent upstream drive, not an odor transduction or public ORN population.",
            "LN GABA identity is predicted and its intrinsic/receptor parameters are not calibrated.",
            "Without the opt-in extension, LN-to-ORN inputs act at the soma only; the extension adds a declared effective terminal coupling.",
            "Two identical trials share ongoing adaptation; they are not biological replicates.",
            "Rates use a nominal 1-ms tick. Pulse commands are not clamped spike trains.",
            "Most cut boundary cells are absent and undriven; reciprocal paths inside the cut remain intact."]}
    dump_new(output/"analysis.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "intrinsic", "tail", "output"): p.add_argument(name, type=Path)
    p.add_argument("--lesions", nargs="+", choices=LESIONS, default=["intact"])
    p.add_argument("--seeds", nargs="+", type=int, default=[11])
    p.add_argument("--direct-rates", nargs="+", type=float, default=[0,2,5,10,20,50,100])
    p.add_argument("--lateral-rates", nargs="+", type=float, default=[0,20,80])
    p.add_argument("--inhibition-gain", type=float)
    p.add_argument("--inhibition-decay", type=float, default=100.)
    a = p.parse_args()
    run(a.graph, a.intrinsic, a.tail, a.output, direct_rates=a.direct_rates,
        lateral_rates=a.lateral_rates, seeds=a.seeds, lesions=a.lesions,
        inhibition_gain=a.inhibition_gain, inhibition_decay=a.inhibition_decay)


if __name__ == "__main__": main()
