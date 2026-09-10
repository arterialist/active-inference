"""Target-only PN calibration reunited with native, closed-loop KC/APL feedback.

Selected one-second electrical steps, not the unavailable biological step
records. All synaptic learning remains enabled. Record bounded chunks and
replay every target input to locate effects hidden by equal spike counts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

import numpy as np

from .connectome import Subgraph
from .electrophysiology import CurrentElectrode
from .paula import Dynamics, PNCurrentKernel, build_paula, Neuron
from .pn_current import dl5_cut
from .prisco import digest, dump_new
from neuron.extensions.experimental.input_current import InputCurrentNeuron
from neuron.extensions.experimental.passive_cable import LocalCableGradedNeuron
from neuron.network import NeuronNetwork
from neuron.neuron import RetrogradeSignalEvent

CONDITIONS = ("isolated", "intact", "apl_release_block")
TRACE_COLUMNS = ["tick", "electrode_pa", "native_current", "total_current", "S_before",
                 "S_after", "O", "F_avg", "t_ref", "r", "b"]


def step_course(levels=(25., 50., 75., 100.), duration=1000, gap=1000):
    if (type(duration) is not int or duration < 1 or type(gap) is not int or gap < 1
            or not len(levels) or not np.isfinite(levels).all() or np.any(np.asarray(levels) <= 0)):
        raise ValueError("Need finite positive levels and positive integer durations")
    command = np.zeros(gap + len(levels)*(duration+gap))
    epochs = [{"start": 0, "stop": gap, "current_pa": 0., "phase": "baseline"}]
    for i, level in enumerate(levels):
        start = gap+i*(duration+gap)
        command[start:start+duration] = level
        epochs.extend([{"start": start, "stop": start+duration, "current_pa": float(level), "phase": "step"},
                       {"start": start+duration, "stop": start+duration+gap, "current_pa": 0., "phase": "gap"}])
    return command, epochs


def prepare(graph, intrinsic, tail, *, spatial=None, release_depression=None, electrical_roots=()):
    """Change only the identified PN, before any tick, and document overrides."""
    root, _ = dl5_cut(graph)
    if intrinsic["root"] != root:
        raise ValueError("Calibration belongs to a different neuron")
    proposal = intrinsic["proposals"]["pooled_control"]
    gain = intrinsic["joint_unitary"]["per_count_weight"]
    if (not np.isfinite([proposal["lambda_ms"], proposal["rheobase_pa"], gain]).all()
            or proposal["lambda_ms"] < 1 or proposal["rheobase_pa"] <= 0 or gain <= 0):
        raise ValueError("Invalid target calibration")
    dynamics = Dynamics(weight_per_count=.075, apl_representation="local_cable" if spatial else "global_graded")
    spec = PNCurrentKernel(root, "ORN_DL5", tuple(tail["decay_ms"]), tuple(tail["peak_fractions"]), "peak")
    prep = build_paula(graph, dynamics, spatial=spatial, current_kernel=spec,
                       release_depression=release_depression, electrical_roots=electrical_roots)
    target = prep.network.network.neurons[prep.root_to_id[root]]
    target.params.lambda_param = proposal["lambda_ms"]  # Explicit nominal 1 ms/tick hypothesis.
    mappings = {int(e[0]): int(e[4]) for e in prep.edge_bindings if int(e[3]) == target.id}
    mappings.update({int(e[0]): int(e[3]) for e in prep.incoming_boundary_ports if int(e[2]) == target.id})
    changes = []
    for edge in sorted(graph.edges, key=lambda e: int(e[8])):
        if str(edge[1]) == root and graph.nodes[str(edge[0])]["annotation"]["hemibrain_type"] == "ORN_DL5":
            port = mappings[int(edge[8])]
            weight = float(edge[6]*gain)
            if abs(weight) > 100:
                raise ValueError("Calibrated weight exceeds native bounds")
            target.postsynaptic_points[port].u_i.info = weight
            changes.append({"source_row": int(edge[8]), "port": port, "weight": weight})
    if [c["port"] for c in changes] != list(target.current_ports):
        raise ValueError("Current filtering and calibrated partner ports differ")
    prep.assumptions["target_only_override"] = {
        "root": root, "lambda_ticks": target.params.lambda_param,
        "pa_per_model_current": proposal["rheobase_pa"], "orn_per_count_weight": gain,
        "orn_ports": changes, "applied_before_tick": 0,
        "others_unchanged": True, "clock": "nominal 1 ms/tick; not whole-circuit physiological calibration"}
    return prep, target


@contextmanager
def release_observer(cell, blocked, on_tick):
    """Filter only the outermost forward events; retain return events and state."""
    if cell is None:
        yield
        return
    original = type(cell).tick

    def observed(other, inputs, tick, dt=1.):
        events = original(other, inputs, tick, dt)
        if other is not cell:
            return events
        if any(not isinstance(e, RetrogradeSignalEvent) and
               not (isinstance(e, tuple) and len(e) == 3) for e in events):
            raise TypeError("Unknown neural event format")
        forward = sum(isinstance(e, tuple) for e in events)
        on_tick(tick, forward, 0 if blocked else forward, len(events)-forward)
        return [e for e in events if isinstance(e, RetrogradeSignalEvent)] if blocked else events

    with patch.object(type(cell), "tick", observed):
        yield


def audit_trace(trace, command, proposal, *, first_tick=0, last_fire=-np.inf, initial_S=0.):
    """Check electrical command and native Euler/reset equations on every tick."""
    a = np.asarray(trace)
    if a.shape != (len(command), len(TRACE_COLUMNS)) or not np.isfinite(a).all():
        raise ValueError("Invalid target trace")
    np.testing.assert_array_equal(a[:, 0], np.arange(first_tick, first_tick+len(a)))
    np.testing.assert_array_equal(a[:, 1], command)
    np.testing.assert_array_equal(a[:, 3], a[:, 2]+command/proposal["rheobase_pa"])
    np.testing.assert_array_equal(a[:, 4], np.r_[initial_S, a[:-1, 5]])
    before_reset = a[:, 4]+(-a[:, 4]+a[:, 3])/proposal["lambda_ms"]
    if np.max(np.abs(before_reset)) >= 100:
        raise ValueError("Unexpected membrane clipping; audit must model it explicitly")
    spikes = np.zeros(len(a))
    for i, value in enumerate(before_reset):
        tick = first_tick+i
        threshold = a[i, 10] if tick-last_fire <= 3 else a[i, 9]
        if abs(value) < .005:
            threshold = a[i, 9]
        if value >= threshold and tick-last_fire >= 3:
            spikes[i] = 1.
            last_fire = tick
    np.testing.assert_array_equal(a[:, 6], spikes)
    np.testing.assert_allclose(a[:, 5], np.where(spikes > 0, 0., before_reset), atol=1e-12, rtol=0)
    return last_fire


def sources(graph_path, intrinsic_path, tail_path, spatial):
    files = [graph_path/"manifest.json", intrinsic_path, tail_path, Path(__file__),
             Path(__file__).with_name("paula.py"), Path(inspect.getfile(CurrentElectrode)),
             Path(inspect.getfile(Neuron)), Path(inspect.getfile(InputCurrentNeuron)),
             Path(inspect.getfile(LocalCableGradedNeuron)), Path(inspect.getfile(NeuronNetwork))]
    if spatial:
        files += [spatial/"analysis.json"]
    return {str(p.resolve()): digest(p) for p in files}


def run(graph_path, intrinsic_path, tail_path, output, condition, *, spatial=None, levels=(25.,50.,75.,100.), chunk=1000):
    if condition not in CONDITIONS or type(chunk) is not int or chunk < 1:
        raise ValueError("Invalid condition or chunk length")
    if condition != "isolated" and spatial is None:
        raise ValueError("Connected experiment requires the verified spatial APL")
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    graph = Subgraph.load(graph_path)
    intrinsic = json.loads(intrinsic_path.read_text())
    for source, expected in intrinsic["source_hashes"].items():
        if digest(Path(source)) != expected:
            raise ValueError(f"Intrinsic calibration source changed: {source}")
    tail = json.loads(tail_path.read_text())["fits"]["2"]["all_cells"]
    root, cut = dl5_cut(graph)
    hashes = sources(graph_path, intrinsic_path, tail_path, spatial)
    random.seed(0)
    prep, target = prepare(cut if condition == "isolated" else graph, intrinsic, tail,
                           spatial=spatial if condition != "isolated" else None)
    roots = list(prep.root_to_id)
    cells = [prep.network.network.neurons[prep.root_to_id[r]] for r in roots]
    apl = next((c for r, c in zip(roots, cells) if graph.nodes[r]["annotation"]["hemibrain_type"] == "APL"), None)
    output.mkdir(parents=True)
    command, epochs = step_course(levels)
    proposal = intrinsic["proposals"]["pooled_control"]
    command_model = command/proposal["rheobase_pa"]
    catalog = [{"source_row": int(e[8]), "source_root": str(e[0]),
                "source_type": graph.nodes[str(e[0])]["annotation"]["hemibrain_type"],
                "sign": int(e[5]), "contacts": int(e[4])}
               for e in sorted(cut.edges, key=lambda e: int(e[8])) if str(e[1]) == root]
    with (output/"structure.npz").open("xb") as f:
        np.savez_compressed(f, roots=np.array(roots), edge_bindings=prep.edge_bindings,
            incoming_boundary_ports=prep.incoming_boundary_ports,
            outgoing_boundary_terminals=prep.outgoing_boundary_terminals)
    initial_post = np.array([p.u_i.info for p in target.postsynaptic_points.values()])
    initial_terminal = np.array([p.u_o.info for p in target.presynaptic_points.values()])
    records, last_fire, previous_S = [], -np.inf, 0.
    original = type(target)._hillock_current
    arrays, start = {}, 0

    def observed(cell, tick, dt):
        if cell is not target:
            return original(cell, tick, dt)
        row = tick-start
        arrays["inputs"][row] = cell.input_buffer
        arrays["trace"][row, 4] = float(cell.S)
        arrays["trace"][row, 9:] = [cell.r, cell.b]
        current = original(cell, tick, dt)
        arrays["port_current"][row] = cell.last_port_current
        np.testing.assert_allclose(cell.last_port_current.sum(), current, atol=1e-8, rtol=1e-6)
        return current

    def apl_events(tick, attempted, delivered, returned):
        arrays["apl"][tick-start, 4:] = [attempted, delivered, returned]

    with patch.object(type(target), "_hillock_current", observed), \
            CurrentElectrode(target, command_model) as electrode, \
            release_observer(apl, condition == "apl_release_block", apl_events):
        for start in range(0, len(command), chunk):
            stop = min(start+chunk, len(command))
            n = stop-start
            arrays = {"trace": np.zeros((n, len(TRACE_COLUMNS))),
                "soma": np.zeros((n, len(cells), 3)),
                "inputs": np.zeros((n, target.params.num_inputs, 4), dtype=np.float32),
                "port_current": np.zeros((n, target.params.num_inputs)),
                "post_weight": np.zeros((n, len(initial_post))),
                "terminal_weight": np.zeros((n, len(initial_terminal))),
                "apl": np.zeros((n, 7))}
            for tick in range(start, stop):
                row = tick-start
                prep.network.run_tick()
                arrays["trace"][row, :4] = [tick, command[tick], electrode.native_current[tick], electrode.total_current[tick]]
                arrays["trace"][row, 5:9] = [float(target.S), float(target.O), float(target.F_avg), float(target.t_ref)]
                arrays["soma"][row] = [[float(c.S), float(c.O), float(c.F_avg)] for c in cells]
                arrays["post_weight"][row] = [p.u_i.info for p in target.postsynaptic_points.values()]
                arrays["terminal_weight"][row] = [p.u_o.info for p in target.presynaptic_points.values()]
                if apl is not None:
                    arrays["apl"][row, :4] = [float(apl.S), float(apl.O),
                        float(apl.cable.voltage.max()), float(apl.terminal_release.max(initial=0))]
            last_fire = audit_trace(arrays["trace"], command[start:stop], proposal,
                                    first_tick=start, last_fire=last_fire, initial_S=previous_S)
            previous_S = float(target.S)
            if any(not np.isfinite(a).all() for a in arrays.values()):
                raise ValueError("Nonfinite recording")
            path = output/f"ticks-{start:06d}-{stop:06d}.npz"
            with path.open("xb") as f:
                np.savez_compressed(f, **arrays)
            records.append({"file": path.name, "start": start, "stop": stop, "sha256": digest(path)})
            print(f"{condition}: {stop}/{len(command)} ticks, {time.perf_counter()-started:.1f} s", flush=True)
    if any(c.params.eta_post <= 0 or c.params.eta_retro <= 0 or c._ablation for c in cells):
        raise ValueError("Unexpected disabled adaptation")
    if hashes != sources(graph_path, intrinsic_path, tail_path, spatial):
        raise ValueError("Sources changed during experiment")
    result = {"schema": 1, "condition": condition, "root": root, "anatomy": graph.summary(),
        "selected_cells": len(roots), "assumptions": prep.assumptions, "epochs": epochs,
        "runtime_seconds": time.perf_counter()-started, "source_hashes": hashes,
        "chunks": records, "structure_sha256": digest(output/"structure.npz"),
        "initial_post_weight": initial_post.tolist(), "initial_terminal_weight": initial_terminal.tolist(),
        "target_inputs": catalog, "trace_columns": TRACE_COLUMNS,
        "recording": {"soma": ["S", "O", "F_avg"], "apl": ["S", "O", "max_compartment_voltage", "max_local_release", "attempted_forward_events", "delivered_forward_events", "returned_events"],
            "time": "after each tick except input buffers, S_before and r/b at the current hook",
            "omitted": ["full APL compartment arrays", "other cells' input currents and weights", "full event queues", "individual retrograde events"], "checkpoint": False},
        "protocol": {"nominal_ms_per_tick": 1, "physical_clock_established_for_full_graph": False,
            "duration_ticks": 1000, "gap_ticks": 1000, "current_levels_pa": list(levels),
            "step_source": "Gugel et al. 2023 Figure 7B caption, selected levels instead of 5-pA increments",
            "measured_step_responses_available": False, "adaptation_enabled": True,
            "blockade": "all APL forward events removed after native outer tick; state and return events retained" if condition == "apl_release_block" else None,
            "no_odor_no_spontaneous_boundary_input": True},
        "claim": "Closed-loop feedback sensitivity of one PN's effective calibration, not physiological or behavioral acceptance"}
    dump_new(output/"analysis.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("graph", "intrinsic", "tail", "output"):
        p.add_argument(arg, type=Path)
    p.add_argument("condition", choices=CONDITIONS)
    p.add_argument("--spatial", type=Path)
    p.add_argument("--levels", nargs="+", type=float, default=[25., 50., 75., 100.])
    args = p.parse_args()
    run(args.graph, args.intrinsic, args.tail, args.output, args.condition, spatial=args.spatial, levels=args.levels)


if __name__ == "__main__":
    main()
