"""Exploratory DL5 current-shape fit and boundary-preserving PAULA assays.

Source columns, not time points, are held out. The fit predicts normalized
postpeak current only. Instantaneous rise, unknown current scale and missing
presynaptic depression prevent treating it as complete ORN-to-PN physiology.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .gugel import SOURCE, sections
from .prisco import digest, dump_new


def kernel(time, taus, fractions):
    return np.exp(-np.asarray(time)[:, None]/np.asarray(taus)) @ np.asarray(fractions)


def fit_shape(time, waves, components):
    """Equal cell weighting, peak-normalized samples, no per-cell fitted gain."""
    time, waves = np.asarray(time), np.asarray(waves)
    if (time.ndim != 1 or waves.ndim != 2 or waves.shape[1] != len(time)
            or not len(waves) or len(time) < 4 or time[0] != 0
            or not np.all(np.diff(time) > 0) or not np.isfinite(waves).all()
            or not np.isfinite(time).all() or not np.allclose(waves[:, 0], 1)):
        raise ValueError("Need finite, peak-normalized postpeak cell traces")
    if components not in (1, 2):
        raise ValueError("Compare one or two components")

    def unpack(p):
        if components == 1:
            return np.array(p), np.ones(1)
        order = np.argsort(p[:2])
        return np.array(p[:2])[order], np.array([p[2], 1-p[2]])[order]

    starts = [[10], [30]] if components == 1 else [[3, 30, .5], [10, 80, .8], [1, 100, .5], [15, 200, .9]]
    bounds = ([.1], [1000]) if components == 1 else ([.1, .1, 0], [1000, 1000, 1])
    solutions = [least_squares(lambda p: (kernel(time, *unpack(p))-waves).ravel(),
                               start, bounds=bounds) for start in starts]
    result = min(solutions, key=lambda r: float(r.fun @ r.fun))
    if not result.success:
        raise RuntimeError(result.message)
    taus, fractions = unpack(result.x)
    return {"decay_ms": taus.tolist(), "peak_fractions": fractions.tolist(),
            "training_rmse": float(np.sqrt(np.mean(result.fun**2))),
            "near_bound": bool(np.any(taus < .101) or np.any(taus > 999)
                               or (components == 2 and np.min(fractions) < .001))}


def normalized_source(data, baseline_end_ms=20):
    epsc = sections(data)["epsc"]
    time, measured = epsc[:, 0], epsc[:, 1:]
    peak = measured.argmin(axis=0)
    if not np.all(peak == peak[0]):
        raise ValueError("Source peak alignment changed; do not silently realign cells")
    baseline = measured[time < baseline_end_ms].mean(axis=0)
    inward = (baseline-measured[peak[0]:]).T
    if np.any(inward[:, 0] <= 0):
        raise ValueError("No inward peak")
    t = time[peak[0]:]-time[peak[0]]
    mask = t <= 100
    return t[mask], (inward/inward[:, [0]])[:, mask]


def compare_shapes(time, waves):
    if len(waves) != 12:
        raise ValueError("Expected seven solvent and five E2-exposed cells")
    fits = {}
    for count in (1, 2):
        all_fit = fit_shape(time, waves, count)
        loo = []
        for hold in range(len(waves)):
            fit = fit_shape(time, np.delete(waves, hold, axis=0), count)
            prediction = kernel(time, fit["decay_ms"], fit["peak_fractions"])
            loo.append({"source_column": "BCDEFGHJKLMN"[hold], **fit,
                        "held_out_rmse": float(np.sqrt(np.mean((prediction-waves[hold])**2)))})
        transfer = fit_shape(time, waves[:7], count)
        prediction = kernel(time, transfer["decay_ms"], transfer["peak_fractions"])
        fits[str(count)] = {"all_cells": all_fit, "leave_one_cell_out": loo,
                           "mean_held_out_rmse": float(np.mean([r["held_out_rmse"] for r in loo])),
                           "solvent_to_exposed": {"training_columns": "BCDEFGH", **transfer,
                               "test_columns": "JKLMN",
                               "per_cell_rmse": np.sqrt(np.mean((prediction-waves[7:])**2, axis=1)).tolist()}}
    return fits


def shape_analysis(extracted: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    data = json.loads(extracted.read_text())
    if data["source"] != SOURCE:
        raise ValueError("Unrecognized source provenance")
    extraction = json.loads((extracted.parent / "extraction.json").read_text())
    if extraction["cells_sha256"] != digest(extracted):
        raise ValueError("Extracted cells no longer match extraction hash")
    time, waves = normalized_source(data)
    comparisons = compare_shapes(time, waves)
    sensitivity = {}
    for baseline in (30, 40):
        t, y = normalized_source(data, baseline)
        sensitivity[str(baseline)] = compare_shapes(t, y)
    report = {"schema": 1, "source": SOURCE, "cells_sha256": digest(extracted),
              "analysis_source_sha256": digest(Path(__file__)),
              "status": "exploratory model comparison, not preregistered physiological acceptance",
              "protocol": {"baseline_end_ms": 20, "postpeak_end_ms": 100,
                           "cell_weighting": "each cell has equal weight; peak normalization uses that cell's measured peak",
                           "tau_bounds_ms": [.1, 1000], "physical_latency_fitted": False,
                           "sign_handling": "signed baseline-subtracted current, no rectification"},
              "fits": comparisons, "baseline_sensitivity": sensitivity,
              "limits": ["No prediction of peak amplitude, prepeak rise, latency or trial variability.",
                         "Exposed cells are a separate condition, so cross-condition transfer is not an equivalence test.",
                         "Two exponentials do not identify receptor subtypes or exclude dendritic/clamp filtering.",
                         "No short-term depression, release saturation or voltage-dependent conductance in this kernel."]}
    output.mkdir(parents=True)
    with (output / "waveforms.npz").open("xb") as stream:
        np.savez_compressed(stream, time_ms=time, normalized_current=waves,
                            source_columns=np.array(list("BCDEFGHJKLMN")))
    dump_new(output / "analysis.json", report)
    return report


def dl5_cut(graph):
    from .connectome import Subgraph
    targets = [r for r in graph.selected if graph.nodes[r]["annotation"]["hemibrain_type"] == "DL5_adPN"]
    if len(targets) != 1:
        raise ValueError("Need one DL5_adPN")
    root = targets[0]
    edges = graph.edges[(graph.edges[:, 0] == int(root)) | (graph.edges[:, 1] == int(root))]
    endpoints = {str(v) for v in edges[:, :2].flat} | {root}
    return root, Subgraph((root,), {r: graph.nodes[r] for r in endpoints}, edges,
                          {"parent_provenance": graph.provenance, "cut": "one DL5 PN, all incident ports retained"})


def probe(graph_path: Path, fit_path: Path, output: Path, *, ticks=800):
    """Same receptor releases through native, peak-matched and area-matched cells.

    The clock hypotheses convert fitted milliseconds to model decay times only.
    They do not calibrate the rest of the neuron or imply physiological Hz.
    """
    from .connectome import Subgraph
    from .paula import Dynamics, PNCurrentKernel, build_paula
    from neuron.neuron import Neuron, RetrogradeSignalEvent
    from neuron.extensions.experimental.input_current import InputCurrentNeuron

    if output.exists():
        raise FileExistsError(output)
    if type(ticks) is not int or ticks < 800:
        raise ValueError("Need at least 800 ticks, including long post-stimulus tail")
    root, cut = dl5_cut(Subgraph.load(graph_path))
    fit = json.loads(fit_path.read_text())["fits"]["2"]["all_cells"]
    records, traces = [], {}
    dynamics = Dynamics(weight_per_count=.075)
    control_preparation = build_paula(cut, dynamics)
    orn_rows = sorted(int(e[8]) for e in cut.edges if str(e[1]) == root and
                      cut.nodes[str(e[0])]["annotation"]["hemibrain_type"] == "ORN_DL5")
    if not orn_rows:
        raise ValueError("No identified ORN_DL5 input")
    sid = int(control_preparation.incoming_boundary_ports[
        control_preparation.incoming_boundary_ports[:, 0] == orn_rows[0]][0, 3])
    for clock in (.5, 1., 2.):
        peak_charge_gain = float(np.sum(np.asarray(fit["peak_fractions"])/
                                       -np.expm1(-clock/np.asarray(fit["decay_ms"]))))
        for normalization in ("native", "area", "peak", "native_charge_matched"):
            spec = None if normalization.startswith("native") else PNCurrentKernel(
                root, "ORN_DL5", tuple(t/clock for t in fit["decay_ms"]),
                tuple(fit["peak_fractions"]), normalization)
            for stimulus in ("single", "train_20", "train_5", "current_step"):
                prep = build_paula(cut, dynamics, current_kernel=spec)
                for name in ("edge_bindings", "incoming_boundary_ports", "outgoing_boundary_terminals"):
                    np.testing.assert_array_equal(getattr(prep, name), getattr(control_preparation, name))
                c = prep.network.network.neurons[prep.root_to_id[root]]
                port = prep.drive_ports[root] if stimulus == "current_step" else sid
                initial = float(c.postsynaptic_points[port].u_i.info)
                rows = []
                for tick in range(ticks):
                    if stimulus == "single":
                        drive = float(tick == 0)
                    elif stimulus == "current_step":
                        drive = 4. if tick < 200 else 0.
                    else:
                        period = int(stimulus.split("_")[1])
                        drive = float(tick < 200 and tick % period == 0)
                    if normalization == "native_charge_matched" and stimulus != "current_step":
                        drive *= peak_charge_gain
                    c.input_buffer[port, 0] = drive
                    # Independent native queue witness, including this tick's
                    # zero-delay input at its actual pre-learning coefficient.
                    incoming = sum(float(v*c.params.delta_decay**c.distances[p])
                        for at, _, v, p in c.propagation_queue if at <= tick)
                    if drive and c.distances[port] == 0:
                        pp = c.postsynaptic_points[port].u_i
                        incoming += float(np.float32(drive)*(pp.info+pp.plast))
                    before = float(c.S)
                    events = c.tick({}, tick)
                    filtered = isinstance(c, InputCurrentNeuron)
                    current = c.total_current if filtered else incoming
                    impulse = float(c.arrived_port_impulse.sum()) if filtered else incoming
                    rows.append([tick, drive, impulse, current, before, float(c.S), float(c.O),
                                 float(c.postsynaptic_points[port].u_i.info),
                                 sum(isinstance(e, RetrogradeSignalEvent) for e in events),
                                 *(c.current_state.sum(axis=0) if filtered else [0., 0.])])
                trace = np.asarray(rows)
                key = f"ms{clock:g}_{normalization}_{stimulus}"
                traces[key] = trace
                records.append({"key": key, "ms_per_tick_hypothesis": clock,
                    "peak_kernel_discrete_charge_gain": peak_charge_gain,
                    "normalization": normalization, "stimulus": stimulus,
                    "spike_ticks": trace[trace[:, 6] > 0, 0].astype(int).tolist(),
                    "total_impulse": float(trace[:, 2].sum()),
                    "total_current": float(trace[:, 3].sum()), "peak_current": float(trace[:, 3].max()),
                    "current_remaining_at_end": float(trace[-1, 3]),
                    "unrecorded_future_current_sum": float(np.sum(c.current_state * c._current_decay /
                        -np.expm1(-1/c.current_decay_ticks))) if filtered else 0.,
                    "weight_initial": initial, "weight_final": float(trace[-1, 7]),
                    "injected_port": port, "parameters": prep.assumptions})
    report = {"schema": 1, "root": root, "source_row": orn_rows[0], "postsynaptic_port": sid,
              "source_hashes": {str(p): digest(p) for p in (fit_path, Path(__file__),
                  graph_path / "manifest.json", Path(inspect.getfile(Neuron)),
                  Path(inspect.getfile(InputCurrentNeuron)), Path(__file__).with_name("paula.py"))},
              "all_incoming_pairs": len(control_preparation.incoming_boundary_ports),
              "all_outgoing_pairs": len(control_preparation.outgoing_boundary_terminals),
              "columns": ["tick", "release", "arrived_impulse", "effective_current", "S_before", "S_after", "O", "weight", "retrograde_event_count", "fast_current", "slow_current"],
              "records": records,
              "claim": "Isolated native-port mechanism assay, not an ORN simulation, voltage-clamp or physiological firing reproduction",
              "paula_current_to_pa": None, "calibrated_physical_clock": None}
    output.mkdir(parents=True)
    with (output / "per_tick.npz").open("xb") as stream:
        np.savez_compressed(stream, **traces)
    dump_new(output / "analysis.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("cells", type=Path)
    fit.add_argument("output", type=Path)
    run = sub.add_parser("probe")
    run.add_argument("graph", type=Path)
    run.add_argument("fit", type=Path)
    run.add_argument("output", type=Path)
    args = parser.parse_args()
    result = shape_analysis(args.cells, args.output) if args.command == "fit" else probe(args.graph, args.fit, args.output)
    print(json.dumps({"output": str(args.output), "schema": result["schema"]}))


if __name__ == "__main__":
    main()
