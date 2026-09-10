"""DL5 ramp firing and unitary current under one explicit scalar calibration.

Fit proposals use an analytic continuous-time approximation, not a surrogate
brain. Every reported simulated response is rerun through actual PAULA ticks
with positive learning rates. The coarse source comparison keeps each fly
separate and does not claim a fitted effective lambda is a membrane measurement.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import inspect
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .connectome import Subgraph
from .electrophysiology import CurrentElectrode
from .gugel import SOURCE, sections, waveform_metrics
from .paula import Dynamics, PNCurrentKernel, build_paula, Neuron
from .pn_current import dl5_cut
from .prisco import digest, dump_new
from neuron.extensions.experimental.input_current import InputCurrentNeuron

RAMP_SLOPE_PA_S = 4.5
WINDOW_MS = 50.
BIN_EDGES_PA = np.arange(5., 100., 5.)  # 5..95, excluding source endpoints.
RAMP_COLUMNS = tuple("BCDEGHIJ")


def load_source(cells_path, article_path):
    data = json.loads(cells_path.read_text())
    audit = json.loads((cells_path.parent / "extraction.json").read_text())
    if data["source"] != SOURCE or audit["cells_sha256"] != digest(cells_path):
        raise ValueError("Unrecognized or changed Gugel extraction")
    article = json.loads(article_path.read_text())
    if article["id"] != "85443" or article["version"] != 2:
        raise ValueError("Need the declared article version")

    def images(value):
        if isinstance(value, dict):
            if value.get("id") == "fig7" and "caption" in value:
                yield value
            for v in value.values():
                yield from images(v)
        elif isinstance(value, list):
            for v in value:
                yield from images(v)

    figure, = list(images(article["body"]))
    caption = " ".join(p["text"] for p in figure["caption"])
    if "4.5 pA/s" not in caption or "50 ms bins with 25 ms overlap" not in caption:
        raise ValueError("Ramp protocol differs from this implementation")
    arrays = sections(data)
    ramp = arrays["ramp"][:890]
    epsc = arrays["epsc"]
    peaks = [waveform_metrics(epsc[:, 0], epsc[:, i])["peak_inward_pa"] for i in range(1, 8)]
    return ramp, np.asarray(peaks), {"source": SOURCE, "cells_sha256": digest(cells_path),
        "article_sha256": digest(article_path), "article_url": "https://api.elifesciences.org/articles/85443/versions/2",
        "figure": "7C", "protocol_from_caption": {"ramp_slope_pa_s": 4.5, "window_ms": 50, "overlap_ms": 25},
        "source_rows": [5, 894], "source_columns": list(RAMP_COLUMNS),
        "excluded_from_fit": "0..5 and 95..100 pA endpoints; the source's final 20 all-zero rows are retained in extraction but not interpreted as physical trials",
        "unresolved": ["Bin center versus edge convention and trial aggregation are not specified in the caption.",
                       "The first nonzero current is 0.100 pA, then steps are 0.1125 pA; do not replace the axis.",
                       "Ascending source rows do not provide a measured descending branch of the triangular ramp.",
                       "Synaptic-current and intrinsic-firing cells are unpaired across experiments."]}


def coarse_curve(current_pa, values):
    x, y = np.asarray(current_pa), np.asarray(values)
    if x.ndim != 1 or y.shape[-1] != len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid current/response arrays")
    result = []
    for lo, hi in zip(BIN_EDGES_PA[:-1], BIN_EDGES_PA[1:], strict=True):
        selected = (x >= lo) & (x < hi)
        if not selected.any():
            raise ValueError("Missing current-range support")
        result.append(y[..., selected].mean(axis=-1))
    return np.stack(result, axis=-1)


def continuous_rate(current_pa, rheobase_pa, lambda_ms):
    """Steady scalar LIF approximation for proposal generation only."""
    x = np.asarray(current_pa, dtype=float)
    result = np.zeros_like(x)
    on = x > rheobase_pa
    result[on] = 1000 / (lambda_ms * np.log(x[on]/(x[on]-rheobase_pa)))
    return np.minimum(result, 1000/3.)


def fit_proposal(current_pa, cell_rates, *, fixed_lambda_ms=None):
    y = np.asarray(cell_rates)
    if y.ndim != 2 or y.shape[1] != len(current_pa) or not len(y) or np.any(y < 0):
        raise ValueError("Keep experimental cells as distinct rows")
    if fixed_lambda_ms is not None and (not np.isfinite(fixed_lambda_ms) or fixed_lambda_ms < 1.1):
        raise ValueError("Invalid fixed integration coefficient")
    observed = coarse_curve(current_pa, y)
    unpack = lambda p: p if fixed_lambda_ms is None else [p[0], fixed_lambda_ms]
    starts = ((15, 30), (25, 60), (40, 100), (60, 200)) if fixed_lambda_ms is None else ((15,), (25,), (40,), (60,))
    bounds = ([.1, 1.1], [95, 500]) if fixed_lambda_ms is None else ([.1], [95])
    fits = [least_squares(lambda p: (coarse_curve(current_pa, continuous_rate(current_pa, *unpack(p)))-
                                    observed).ravel(), start, bounds=bounds)
            for start in starts]
    fit = min(fits, key=lambda f: float(f.fun @ f.fun))
    if not fit.success:
        raise RuntimeError(fit.message)
    parameters = unpack(fit.x)
    return {"rheobase_pa": float(parameters[0]), "lambda_ms": float(parameters[1]),
            "fixed_lambda_ms": fixed_lambda_ms,
            "analytic_training_per_cell_rmse_hz": np.sqrt(np.mean(fit.fun.reshape(len(y), -1)**2, axis=1)).tolist(),
            "status": "exploratory effective scalar fit, not measured membrane properties"}


def window_rates(spike_ms, centers_ms, *, alignment="center"):
    spikes, centers = np.asarray(spike_ms), np.asarray(centers_ms)
    if (spikes.ndim != 1 or centers.ndim != 1 or not np.isfinite(spikes).all()
            or not np.isfinite(centers).all() or np.any(np.diff(spikes) < 0)):
        raise ValueError("Invalid observation times")
    shifts = {"center": -WINDOW_MS/2, "leading": 0., "trailing": -WINDOW_MS}
    if alignment not in shifts:
        raise ValueError("Unknown bin alignment")
    lo = centers + shifts[alignment]
    counts = np.searchsorted(spikes, lo+WINDOW_MS, side="left")-np.searchsorted(spikes, lo, side="left")
    return counts * (1000/WINDOW_MS)


def prepared_cell(cut, root, proposal, clock_ms, tail_fit):
    if tuple(cut.selected) != (root,):
        raise ValueError("Intrinsic calibration applies only to the isolated target PN, not other cells")
    if (not np.isfinite(clock_ms) or clock_ms <= 0
            or not np.isfinite(proposal["rheobase_pa"]) or proposal["rheobase_pa"] <= 0
            or not np.isfinite(proposal["lambda_ms"]) or proposal["lambda_ms"] < clock_ms):
        raise ValueError("Invalid current scale, clock or integration coefficient")
    c_ticks = int(np.ceil(3/clock_ms))
    dynamics = replace(Dynamics(weight_per_count=.075), lambda_ticks=proposal["lambda_ms"]/clock_ms,
                       cooldown_ticks=c_ticks)
    spec = PNCurrentKernel(root, "ORN_DL5", tuple(t/clock_ms for t in tail_fit["decay_ms"]),
                            tuple(tail_fit["peak_fractions"]), "peak")
    prep = build_paula(cut, dynamics, current_kernel=spec)
    return prep, prep.network.network.neurons[prep.root_to_id[root]]


def audit_ramp_trace(trace, proposal, clock_ms, cooldown_ticks):
    """Independent equation check for this zero-chemical-input assay.

    It checks the physical command as well as post-reset voltage. A false
    current on a spike tick could otherwise hide behind the reset to zero.
    """
    a = np.asarray(trace)
    if a.ndim != 2 or a.shape[1] != 9 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Invalid ramp trace")
    expected_time = np.arange(len(a))*clock_ms
    expected_pa = np.maximum(0., 100-np.abs(expected_time*RAMP_SLOPE_PA_S/1000-100))
    np.testing.assert_array_equal(a[:, 0], expected_time)
    np.testing.assert_array_equal(a[:, 1], expected_pa)
    np.testing.assert_array_equal(a[:, 2], 0.)
    np.testing.assert_array_equal(a[:, 3], expected_pa/proposal["rheobase_pa"])
    np.testing.assert_array_equal(a[:, 4], np.r_[0., a[:-1, 5]])
    pre_reset = a[:, 4] + clock_ms/proposal["lambda_ms"]*(-a[:, 4]+a[:, 3])
    last = -np.inf
    expected_O = np.zeros(len(a))
    for tick, voltage in enumerate(pre_reset):
        threshold = 1.2 if tick-last <= cooldown_ticks else 1.
        if abs(voltage) < .005:
            threshold = 1.
        if voltage >= threshold and tick-last >= cooldown_ticks:
            expected_O[tick] = 1.
            last = tick
    np.testing.assert_array_equal(a[:, 6], expected_O)
    expected_S = np.where(expected_O > 0, 0., pre_reset)
    np.testing.assert_allclose(a[:, 5], expected_S, atol=1e-12, rtol=0)
    return {"checked_ticks": len(a), "command_and_spikes_exact": True,
            "maximum_voltage_equation_residual": float(np.max(np.abs(a[:, 5]-expected_S))),
            "scope": "known membrane/reset equations and commanded current, not experimental spike-time agreement"}


def run_ramp(cut, root, proposal, clock_ms, tail_fit, source_current):
    prep, cell = prepared_cell(cut, root, proposal, clock_ms, tail_fit)
    # Include 1 s of descent for window support. Only the measured ascending
    # branch is scored; its padded source zeros are not a command to the neuron.
    time_ms = np.arange(int(np.ceil((100/RAMP_SLOPE_PA_S+1)*1000/clock_ms))) * clock_ms
    current_pa = np.maximum(0., 100-np.abs(time_ms*RAMP_SLOPE_PA_S/1000-100))
    command = current_pa/proposal["rheobase_pa"]
    trace = np.zeros((len(command), 9))
    initial_weights = np.array([p.u_i.info for p in cell.postsynaptic_points.values()])
    with CurrentElectrode(cell, command) as electrode:
        for tick in range(len(command)):
            before = float(cell.S)
            events = cell.tick({}, tick)
            trace[tick] = [time_ms[tick], current_pa[tick], electrode.native_current[tick],
                          electrode.total_current[tick], before, float(cell.S), float(cell.O),
                          float(cell.F_avg), float(cell.t_ref)]
            # The electrode is not a chemical receptor and creates no return events.
            if any(not isinstance(e, tuple) for e in events):
                raise ValueError("Pure electrical injection produced a synaptic return event")
    final_weights = np.array([p.u_i.info for p in cell.postsynaptic_points.values()])
    np.testing.assert_array_equal(initial_weights, final_weights)
    if cell.params.eta_post <= 0 or cell.params.eta_retro <= 0 or cell._ablation:
        raise ValueError("Adaptation must remain enabled")
    spike_ms = trace[trace[:, 6] > 0, 0]
    centers = np.asarray(source_current)/RAMP_SLOPE_PA_S*1000
    observed = {name: window_rates(spike_ms, centers, alignment=name)
                for name in ("center", "leading", "trailing")}
    return trace, observed, {"parameters": prep.assumptions, "proposal": proposal,
        "clock_ms": clock_ms, "actual_cooldown_ms": cell.params.c*clock_ms,
        "equation_audit": audit_ramp_trace(trace, proposal, clock_ms, cell.params.c),
        "spike_count": len(spike_ms), "electrode_is_not_a_synapse": True,
        "unchanged_weights_without_synaptic_input": True, "adaptation_enabled": True}


def joint_synaptic_probe(cut, root, proposal, tail_fit, mean_peak_pa, *, clock_ms=1):
    edges = sorted((e for e in cut.edges if str(e[1]) == root and
                    cut.nodes[str(e[0])]["annotation"]["hemibrain_type"] == "ORN_DL5"), key=lambda e: int(e[8]))
    counts = np.array([int(e[4]) for e in edges])
    attenuation = .95**2
    per_count = mean_peak_pa/(proposal["rheobase_pa"] * counts.mean() * attenuation)
    result, traces = [], {}
    # Sample every anatomical ORN partner. The equal-partner sampling model is
    # an assumption, not established optogenetic recruitment probabilities.
    for edge in edges:
        prep, cell = prepared_cell(cut, root, proposal, clock_ms, tail_fit)
        mappings = {int(row[0]): int(row[3]) for row in prep.incoming_boundary_ports}
        for e in edges:
            cell.postsynaptic_points[mappings[int(e[8])]].u_i.info = float(e[4]*per_count)
        port = mappings[int(edge[8])]
        trace = np.zeros((250, 6))
        for tick in range(len(trace)):
            if tick == 0:
                cell.input_buffer[port, 0] = 1.
            cell.tick({}, tick)
            trace[tick] = [tick*clock_ms, cell.total_current*proposal["rheobase_pa"],
                           float(cell.S), float(cell.O),
                           float(cell.postsynaptic_points[port].u_i.info), float(cell.t_ref)]
        key = f"row_{int(edge[8])}"
        traces[key] = trace
        result.append({"source_row": int(edge[8]), "source_root": str(edge[0]),
            "contacts": int(edge[4]), "port": port, "peak_current_pa": float(trace[:, 1].max()),
            "peak_S": float(trace[:, 2].max()), "spikes": int(np.count_nonzero(trace[:, 3])),
            "weight_initial": float(edge[4]*per_count), "weight_final": float(trace[-1, 4])})
    # Factorial controls separate changed contact gain from changed integration.
    # They share the fitted tail and the original diagnostic release course.
    # No depression or fitted biological spike train is supplied to the cell.
    train_results = []
    for name, coefficient, gain in (("legacy", 20., .075), ("gain_only", 20., per_count),
            ("integration_only", proposal["lambda_ms"], .075), ("joint", proposal["lambda_ms"], per_count)):
        prep, cell = prepared_cell(cut, root, {**proposal, "lambda_ms": coefficient}, clock_ms, tail_fit)
        mappings = {int(row[0]): int(row[3]) for row in prep.incoming_boundary_ports}
        for e in edges:
            cell.postsynaptic_points[mappings[int(e[8])]].u_i.info = float(e[4]*gain)
        port = mappings[int(edges[0][8])]
        train = np.zeros((800, 6))
        for tick in range(len(train)):
            if tick < 200 and tick % 5 == 0:
                cell.input_buffer[port, 0] = 1.
            cell.tick({}, tick)
            train[tick] = [tick*clock_ms, cell.total_current*proposal["rheobase_pa"],
                          float(cell.S), float(cell.O),
                          float(cell.postsynaptic_points[port].u_i.info), float(cell.t_ref)]
        traces[f"train_{name}"] = train
        train_results.append({"condition": name, "lambda_ms": coefficient, "per_count_weight": gain,
                              "spike_ticks": np.flatnonzero(train[:, 3]).tolist()})
    return {"per_count_weight": per_count, "assumed_equal_partner_count_mean": float(counts.mean()),
            "target_mean_peak_pa": mean_peak_pa, "records": result,
            "train": {"source_row": int(edges[0][8]), "port": port,
                      "stimulus": "one release every five ticks during 0..199, then no input through tick 799",
                      "clock_ms": clock_ms, "spike_ticks": train_results[-1]["spike_ticks"],
                      "synaptic_depression_included": False,
                      "factorial_controls": train_results,
                      "current_axis": "All conditions use the same fitted pA unit for comparison; legacy-gain conditions do not match the measured mean current peak under this mapping."},
            "assumption": f"single scalar pA-to-current mapping for soma injection and effective synaptic current; equal sampling of all {len(counts)} identified ORN partners, not a measured sampling distribution",
            "interpretation": "Matching the mean peak sets gain; it is not an independent prediction. S is not calibrated to mV."}, traces


def run(graph_path, cells_path, article_path, tail_fit_path, output):
    if output.exists():
        raise FileExistsError(output)
    ramp, peaks, source = load_source(cells_path, article_path)
    root, cut = dl5_cut(Subgraph.load(graph_path))
    tail_fit = json.loads(tail_fit_path.read_text())["fits"]["2"]["all_cells"]
    current, observed = ramp[:, 0], ramp[:, 1:].T
    control = observed[4:]
    proposals = {"pooled_control": fit_proposal(current, control)}
    proposals["fixed_lambda20_gain_fit"] = fit_proposal(current, control, fixed_lambda_ms=20.)
    for i, col in enumerate(RAMP_COLUMNS[4:]):
        proposals[f"holdout_{col}"] = {**fit_proposal(current, np.delete(control, i, axis=0)),
            "held_out_column": col, "training_columns": [c for c in RAMP_COLUMNS[4:] if c != col]}
    # The previous pair-count gain only supplies a reference, not a calibrated
    # physical current scale. The average of ALL partners replaces the arbitrary
    # first selected axon as the declared mapping convention.
    contacts = [int(e[4]) for e in cut.edges if str(e[1]) == root and
                cut.nodes[str(e[0])]["annotation"]["hemibrain_type"] == "ORN_DL5"]
    proposals["previous_effective_mapping"] = {"rheobase_pa": float(peaks.mean()/(np.mean(contacts)*.075*.95**2)),
        "lambda_ms": 20., "status": "reference under a declared current-unit mapping, not an earlier physiological calibration"}
    output.mkdir(parents=True)
    records = []
    scores = coarse_curve(current, observed)
    for name, p in proposals.items():
        for clock in ((1., .5) if name == "pooled_control" else (1.,)):
            trace, rates, summary = run_ramp(cut, root, p, clock, tail_fit, current)
            key = f"{name}_ms{clock:g}"
            trace_path = output / f"{key}.npz"
            with trace_path.open("xb") as stream:
                np.savez_compressed(stream, trace=trace, source_current_pa=current,
                                    source_rates_hz=observed, **{f"model_{k}_hz": v for k, v in rates.items()})
            comparisons = {}
            for alignment, predicted in rates.items():
                curve = coarse_curve(current, predicted)
                comparisons[alignment] = {"per_cell_rmse_hz": np.sqrt(np.mean((curve-scores)**2, axis=1)).tolist(),
                    "pooled_control_rmse_hz": float(np.sqrt(np.mean((curve-scores[4:].mean(axis=0))**2))),
                    "predicted_coarse_hz": curve.tolist()}
            records.append({"name": name, "key": key, **summary, "comparison": comparisons,
                            "trace_sha256": digest(trace_path)})
            print(f"{key}: {summary['spike_count']} spikes; control-mean RMSE {comparisons['center']['pooled_control_rmse_hz']:.3f} Hz", flush=True)
    joint, unitary = joint_synaptic_probe(cut, root, proposals["pooled_control"], tail_fit, float(peaks.mean()))
    with (output / "unitary.npz").open("xb") as stream:
        np.savez_compressed(stream, **unitary)
    report = {"schema": 1, "source": source, "root": root, "proposals": proposals,
        "record_columns": ["time_ms", "electrode_pa", "native_current", "total_current", "S_before", "S_after", "O", "F_avg", "t_ref"],
        "unitary_columns": ["time_ms", "effective_current_pa", "S", "O", "post_weight", "t_ref"],
        "fit_current_bin_edges_pa": BIN_EDGES_PA.tolist(), "source_coarse_rates_hz": scores.tolist(),
        "records": records, "joint_unitary": joint,
        "source_hashes": {str(p): digest(p) for p in (graph_path / "manifest.json", cells_path,
             article_path, tail_fit_path, Path(__file__), Path(inspect.getfile(CurrentElectrode)),
             Path(inspect.getfile(Neuron)), Path(inspect.getfile(InputCurrentNeuron)), Path(__file__).with_name("paula.py"))},
        "claim": "Exploratory effective scalar calibration with native tick verification, not membrane-biophysics identification, population acceptance, synaptic-depression or odor/behavior reproduction",
        "limits": ["The control-mean fit must be judged alongside all four cells and cell-level holdouts.",
            "No source spike times or trial count were reconstructed from averaged binned rates.",
            "Current-bin aggregation is an analysis choice; endpoints and source padding are excluded, not repaired.",
            "A fitted effective lambda may absorb missing compartments, active currents or recording effects.",
            "No calibration is applied to the connected brain or to other cell types.",
            "Actual neural learning remains enabled; absence of chemical input during current injection is not frozen plasticity."]}
    dump_new(output / "analysis.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "cells", "article", "tail_fit", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.graph, args.cells, args.article, args.tail_fit, args.output)


if __name__ == "__main__":
    main()
