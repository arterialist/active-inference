"""Conditional numerical experiment on a recorded APL's first nonzero input.

Hold the recorded arriving current over one simulator tick and refine only the
passive integration. This does not rerun the neural network or its learning.
Every node voltage and terminal readout is retained at every substep. The
continuous-time interpretation of that held current is an explicit hypothesis.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import typer

from .connectome import sha256
from .intervention_analysis import inspect_record
from simulations.paula_loader import ensure_paula_available

ensure_paula_available()
from neuron.extensions.experimental.passive_cable import PassiveCable


def refine_first_response(directory: Path, output: Path, divisions=(1, 2, 4, 8, 16, 32, 64)):
    if (not divisions or divisions[0] != 1 or any(type(n) is not int or n < 1 for n in divisions)
            or any(b <= a for a, b in zip(divisions, divisions[1:]))):
        raise ValueError("Use strictly increasing positive integer divisions starting at one")
    if output.exists():
        raise FileExistsError(output)
    report = inspect_record(directory)[0]
    if "cable" not in report:
        raise ValueError("A recorded local cable is required")
    manifest_path = directory / "manifest.json"
    m = json.loads(manifest_path.read_text())
    params = m["assumptions"]["parameters"]
    spatial = m["assumptions"]["spatial"]
    anatomy_path = Path(spatial["analysis_path"]).parent / "anatomy.npz"
    with np.load(anatomy_path, allow_pickle=False) as g:
        node_ids, parents, xyz, radius, contacts, pair_rows = (g[k] for k in
            ("node_ids", "parents", "xyz_nm", "radius_nm", "contacts", "pair_source_rows"))
    with np.load(directory / "columns.npz", allow_pickle=False) as c:
        apl_id = int(c["cell_ids"][m["protocol"]["apl_row"]])
        ports, terminals = c["cable_input_ports"], c["cable_terminals"]
        incoming = {int(r[0]): int(r[4]) for r in c["edge_bindings"] if r[3] == apl_id}
        incoming.update({int(r[0]): int(r[3]) for r in c["incoming_boundary_ports"] if r[2] == apl_id})
        outgoing = {int(r[0]): int(r[2]) for r in c["edge_bindings"] if r[1] == apl_id}
        outgoing.update({int(r[0]): int(r[2]) for r in c["outgoing_boundary_terminals"] if r[1] == apl_id})
    input_mask = (contacts[:, 2] == int(spatial["root"])) & (pair_rows >= 0)
    in_rows = np.array([incoming[int(r)] for r in pair_rows[input_mask]])
    in_nodes = contacts[input_mask, 6]
    in_counts = np.bincount(in_rows, minlength=len(ports))
    uniform_port = int(np.flatnonzero(in_counts == 0)[0])
    output_mask = contacts[:, 1] == int(spatial["root"])
    all_outputs = contacts[output_mask, 5]
    linked_outputs = output_mask & (pair_rows >= 0)
    out_rows = np.array([outgoing[int(r)] for r in pair_rows[linked_outputs]])
    out_nodes = contacts[linked_outputs, 5]
    out_counts = np.bincount(out_rows, minlength=len(terminals))
    selected = None
    for chunk in m["recording"]["chunks"]:
        with np.load(directory / chunk["file"], allow_pickle=False) as data:
            currents = data["cable_current"]
            nonzero = np.flatnonzero(np.any(currents != 0, axis=1))
            if len(nonzero):
                i = int(nonzero[0])
                voltages = data["cable_voltage"]
                if voltages[i].any():
                    raise ValueError("First-input experiment requires an exactly resting cable")
                selected = (chunk["start"] + i, currents[i].copy(), voltages[i+1].copy(),
                            data["cable_arrived_current"][i].copy())
                break
    if selected is None:
        raise ValueError("No nonzero cable input in this recording")
    tick, current, recorded_voltage, arrived = selected
    cable = PassiveCable(parents, xyz / 1000, radius / 1000, params["apl_cable_rm_over_ra_um"])
    alpha = 1 / params["lambda_ticks"]
    gain, cap = params["apl_graded_gain"], params["apl_release_max"]

    def readout(voltage):
        release = np.clip(gain * voltage, 0, cap)
        terminal = np.bincount(out_rows, weights=release[out_nodes], minlength=len(terminals)) / out_counts
        return release, terminal

    # First prove that the selected recorded transition is still reproducible.
    cable.step(current, alpha)
    if not np.array_equal(cable.voltage, recorded_voltage):
        raise ValueError("Current source no longer reproduces the original cable transition exactly")
    peak_nodes = np.array([np.argmin(recorded_voltage), np.argmax(recorded_voltage)])
    contributions = []
    for node in peak_nodes:
        unit = np.zeros(len(node_ids))
        unit[node] = alpha
        adjoint = cable._factor.solve(unit, trans="T")
        transfer = np.bincount(in_rows, weights=adjoint[in_nodes] / in_counts[in_rows], minlength=len(ports))
        transfer[uniform_port] = adjoint @ cable.capacity
        contribution = transfer * arrived
        if not np.isclose(contribution.sum(), recorded_voltage[node], atol=1e-9, rtol=1e-11):
            raise ValueError("Conditional per-port decomposition failed")
        contributions.append(contribution)

    output.mkdir(parents=True, exist_ok=False)
    with (output / "inputs.npz").open("xb") as out:
        np.savez_compressed(out, node_ids=node_ids, input_ports=ports, terminals=terminals,
            current=current, arrived=arrived, recorded_voltage=recorded_voltage,
            peak_node_indices=peak_nodes, peak_node_ids=node_ids[peak_nodes],
            per_port_peak_voltage_contribution=np.array(contributions))
    sources = {str(Path(path).resolve()): sha256(Path(path)) for path in
               (__file__, inspect.getfile(PassiveCable), inspect.getfile(inspect_record))}
    rows, chunks = [], []
    previous_voltage = previous_terminal = None
    for n in divisions:
        cable.voltage.fill(0)
        step_alpha = alpha / n
        pending_v, pending_r, times = [cable.voltage.copy()], [readout(cable.voltage)[1]], [0.0]
        max_residual = 0.0
        for step in range(1, n + 1):
            old = cable.voltage.copy()
            cable.step(current, step_alpha)
            residual = (cable.capacity * cable.voltage + step_alpha * (cable.laplacian @ cable.voltage)
                        - (1-step_alpha) * cable.capacity * old - step_alpha * current)
            max_residual = max(max_residual, float(np.max(np.abs(residual))))
            if max_residual > 1e-9:
                raise ValueError("Substep cable equation failed")
            local, terminal = readout(cable.voltage)
            pending_v.append(cable.voltage.copy())
            pending_r.append(terminal.copy())
            times.append(step / n)
            if len(pending_v) == 5 or step == n:
                name = f"n{n:04d}-through-{step:04d}.npz"
                with (output / name).open("xb") as out:
                    np.savez_compressed(out, time=np.array(times), voltage=np.array(pending_v),
                                        terminal_release=np.array(pending_r))
                chunks.append({"file": name, "divisions": n, "first_time": times[0],
                               "last_time": times[-1], "sha256": sha256(output / name)})
                pending_v, pending_r, times = [cable.voltage.copy()], [terminal.copy()], [step / n]
        # Node and output-contact counts are distinct; open output sites remain
        # in the latter count but do not acquire fabricated neural consumers.
        row = {"divisions": n, "minimum_voltage": float(cable.voltage.min()),
               "maximum_voltage": float(cable.voltage.max()),
               "area_weighted_voltage": float(cable.capacity @ cable.voltage),
               "saturated_nodes": int((local >= cap).sum()),
               "saturated_output_contacts": int((local[all_outputs] >= cap).sum()),
               "equation_max_residual": max_residual,
               "terminal_release_mean": float(terminal.mean()),
               "critical_linear_input_multiplier": float(cap / (gain * cable.voltage.max())) if cable.voltage.max() > 0 else None}
        if previous_voltage is not None:
            row["max_voltage_difference_from_previous"] = float(np.max(abs(cable.voltage - previous_voltage)))
            row["max_terminal_difference_from_previous"] = float(np.max(abs(terminal - previous_terminal)))
        rows.append(row)
        print(json.dumps(row), flush=True)
        previous_voltage, previous_terminal = cable.voltage.copy(), terminal.copy()
    result = {"schema": "apl-first-response-refinement-v1", "recording": str(directory.resolve()),
        "manifest_sha256": sha256(manifest_path), "anatomy_sha256": sha256(anatomy_path),
        "source_files": sources, "tick": tick, "parameters": params,
        "inputs_sha256": sha256(output / "inputs.npz"), "chunks": chunks, "endpoints": rows,
        "exact_original_transition": True,
        "scope": "conditional first-input passive-cable numerical refinement, not a network or physiological replication",
        "held_input": "Recorded post-delay per-node currents held constant for one simulator tick from exact rest. Learning in the source network remains active; the network is not run in this conditional analysis.",
        "contribution": "One-step adjoint decomposition by input port, conditional on zero previous branch voltage and recorded arriving potentials; not a closed-loop causal lesion.",
        "limits": ["No physical duration or electrical-unit calibration", "No spatial-mesh convergence test",
                   "Finer steps change the explicit leak approximation too", "No substep outputs fed into the surrounding network",
                   "Critical multiplier describes this linear first response only, not subsequent plastic-network behavior"]}
    if any(sha256(Path(path)) != value for path, value in sources.items()):
        raise ValueError("Source changed during refinement")
    with (output / "analysis.json").open("x") as out:
        json.dump(result, out, indent=2)
    return result


def main(directory: Path, output: Path):
    refine_first_response(directory, output)


if __name__ == "__main__":
    typer.run(main)
