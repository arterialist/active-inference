"""Full-tick tests separating axial propagation from distributed ligand drive.

Vertical stimulation only. All conditions preserve the saved geometry and
observation mapping. Zero axial coupling and tip-only channel activation are
explicit causal interventions, not alternative connectomes or fitted defaults.
"""
from __future__ import annotations

import argparse
import inspect
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags

from .amin_input_physics import preparation, region_voltage, stationary
from .amin_spatial import digest, write_new

ALPHA = .05
ON_TICKS = 200
TICKS = 400
GAIN = 10.


def independent_geometry(geometry, ratio):
    """Build area and axial operator via an edge-incidence matrix."""
    parents, xyz, radius = (geometry[k] for k in ("parents", "xyz_um", "radius_um"))
    child = np.flatnonzero(parents >= 0)
    parent = parents[child]
    length = np.sqrt(np.sum((xyz[child]-xyz[parent])**2, axis=1))
    r = (radius[child]+radius[parent])/2
    area = np.zeros(len(parents))
    np.add.at(area, child, np.pi*r*length)
    np.add.at(area, parent, np.pi*r*length)
    edge_g = ratio*np.pi*r*r/length/area.sum()
    rows = np.arange(len(child))
    incidence = coo_matrix((np.r_[np.ones(len(child)), -np.ones(len(child))],
        (np.r_[rows, rows], np.r_[child, parent])), shape=(len(child), len(parents))).tocsr()
    return area/area.sum(), (incidence.T@diags(edge_g)@incidence).tocsr()


def check_transition(previous, voltage, capacity, laplacian, ligand, alpha):
    """No external current, E=1. Return row-wise relative backward error."""
    lhs = capacity*voltage + alpha*(laplacian@voltage + ligand*voltage)
    rhs = (1-alpha)*capacity*previous + alpha*ligand
    scale = abs(capacity*voltage) + alpha*(abs(laplacian)@abs(voltage)+abs(ligand*voltage)) + abs(rhs)
    residual = abs(lhs-rhs)
    backward = float(np.max(residual/np.maximum(scale, np.finfo(float).tiny)))
    mass_error = float(abs(capacity@voltage-(1-alpha)*(capacity@previous)-alpha*np.sum(ligand*(1-voltage))))
    if backward > 1e-10 or mass_error > 1e-8:
        raise ValueError(f"Full-tick equation mismatch: relative={backward}, mass={mass_error}")
    return backward, mass_error


def run(directory, source_directory, cells_path, output):
    if output.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(directory).free < 3_000_000_000:
        raise ValueError("Need 3 GB free before full-node recording")
    cable, geometry, provenance = preparation(directory, source_directory, cells_path)
    from neuron.extensions.experimental.conductance_cable import ConductanceCable
    from . import amin_input_physics
    density = geometry["density"][:, 1]
    regional_density = np.array([density[geometry["region"] == r].max() for r in range(1, 36)])
    # Select by the measured-stimulus fit alone, never by response or error.
    tip_regions = (np.argsort(regional_density)[-2:]+1).tolist()
    cases = [("intact", 25000., False), ("no-axial", 0., False), ("tip-only", 25000., True)]
    output.mkdir(parents=True)
    with (output/"geometry.npz").open("xb") as stream:
        np.savez_compressed(stream, **geometry)
    files = [{"file": "geometry.npz", "sha256": digest(output/"geometry.npz")}]
    results = []
    start = time.perf_counter()
    for name, ratio, tip_only in cases:
        cell = ConductanceCable(geometry["parents"], geometry["xyz_um"], geometry["radius_um"], ratio)
        ligand = cell.capacity*density*GAIN
        if tip_only:
            ligand[~np.isin(geometry["region"], tip_regions)] = 0
        zero = np.zeros(len(density))
        steady = stationary(cell.capacity, cell.laplacian, zero, ligand)[0]
        states, ticks = [cell.voltage.copy()], [0]
        regional, masses = [region_voltage(cell.voltage, geometry)], [0.]
        transition_errors = []
        for tick in range(1, TICKS+1):
            previous = cell.voltage.copy()
            g = ligand if tick <= ON_TICKS else zero
            cell.step_conductance(zero, ALPHA, g)
            transition_errors.append(check_transition(previous, cell.voltage, cell.capacity, cell.laplacian, g, ALPHA))
            regional.append(region_voltage(cell.voltage, geometry))
            masses.append(float(cell.capacity@cell.voltage))
            states.append(cell.voltage.copy())
            ticks.append(tick)
            if tick == ON_TICKS:
                at_offset = cell.voltage.copy()
            if len(states) == 8 or tick == TICKS:
                filename = f"{name}-through-{tick:04d}.npz"
                with (output/filename).open("xb") as stream:
                    np.savez_compressed(stream, tick=np.array(ticks), voltage=np.array(states))
                files.append({"file": filename, "condition": name, "first_tick": ticks[0],
                              "last_tick": ticks[-1], "sha256": digest(output/filename)})
                states, ticks = [], []
            if tick % 100 == 0:
                print(json.dumps({"condition": name, "tick": tick,
                    "region17_raw_voltage": float(regional[-1][16]),
                    "peak_raw_voltage": float(max(regional[-1]))}), flush=True)
        regional = np.array(regional)
        filename = name+"-inputs-readout.npz"
        with (output/filename).open("xb") as stream:
            np.savez_compressed(stream, ligand=ligand, region_voltage=regional, mass=np.array(masses),
                transition_error=np.array(transition_errors), stationary=steady, at_offset=at_offset)
        files.append({"file": filename, "sha256": digest(output/filename)})
        prediction = regional[ON_TICKS]/regional[ON_TICKS].max()
        results.append({"condition": name, "rm_over_ra_um": ratio, "tip_only": tip_only,
            "ligand_gain": GAIN, "region17_at_offset": float(prediction[16]),
            "normalized_profile_at_offset": prediction.tolist(),
            "onset_to_offset_max_stationary_error": float(max(abs(at_offset-steady))),
            "end_max_voltage": float(max(abs(cell.voltage))),
            "observed_region17": float(geometry["observed"][16, 1]),
            "spatial_rmse_at_offset": float(np.sqrt(np.mean((prediction-geometry["observed"][:, 1])**2)))})
    result = {"schema": "amin-conductance-full-tick-v1", **provenance,
        "source_sha256": digest(__file__), "preparation_source_sha256": digest(inspect.getfile(amin_input_physics)),
        "conductance_source_sha256": digest(inspect.getfile(ConductanceCable)),
        "node_count": len(density), "ticks": TICKS, "on_ticks": ON_TICKS, "alpha": ALPHA,
        "tip_regions": tip_regions, "reversal": 1., "results": results, "files": files,
        "wall_seconds": time.perf_counter()-start,
        "limits": ["Vertical stimulus and one exploratory conductance gain only; not physiology acceptance",
            "200 ticks on then 200 off is a diagnostic input, not measured ATP kinetics or physical time",
            "No axial coupling and tip-only activation are explicit causal interventions",
            "No neural network, synaptic adaptation, learned recall or embodied behavior tested",
            "Normalized calcium is not calibrated voltage; all shared specimen and stimulus uncertainties remain"]}
    write_new(output/"analysis.json", result)
    return result["results"]


def audit(directory):
    m = json.loads((directory/"analysis.json").read_text())
    for item in m["files"]:
        if digest(directory/item["file"]) != item["sha256"]:
            raise ValueError(f"Recording changed: {item['file']}")
    with np.load(directory/"geometry.npz", allow_pickle=False) as f:
        geometry = dict(f)
    results = []
    for condition in m["results"]:
        name = condition["condition"]
        c, l = independent_geometry(geometry, condition["rm_over_ra_um"])
        np.testing.assert_allclose(c, geometry["capacity"], rtol=1e-14, atol=0)
        with np.load(directory/(name+"-inputs-readout.npz"), allow_pickle=False) as f:
            inputs = dict(f)
        expected_g = c*geometry["density"][:, 1]*condition["ligand_gain"]
        if condition["tip_only"]:
            expected_g[~np.isin(geometry["region"], m["tip_regions"])] = 0
        np.testing.assert_allclose(inputs["ligand"], expected_g, rtol=1e-14, atol=0)
        previous, count, worst, worst_mass = None, 0, 0., 0.
        for item in (r for r in m["files"] if r.get("condition") == name):
            with np.load(directory/item["file"], allow_pickle=False) as f:
                ticks, states = f["tick"], f["voltage"]
            if states.shape != (len(ticks), len(c)) or not np.isfinite(states).all():
                raise ValueError("Invalid full-node state")
            if ticks[0] != item["first_tick"] or ticks[-1] != item["last_tick"]:
                raise ValueError("Trace index mismatch")
            for tick, voltage in zip(ticks, states, strict=True):
                if tick != count:
                    raise ValueError("Missing or repeated tick")
                if tick == 0:
                    np.testing.assert_array_equal(voltage, np.zeros(len(c)))
                else:
                    g = expected_g if tick <= m["on_ticks"] else np.zeros(len(c))
                    error, mass_error = check_transition(previous, voltage, c, l, g, m["alpha"])
                    worst, worst_mass = max(worst, error), max(worst_mass, mass_error)
                # Recompute from all node values, not from stored summary fields.
                np.testing.assert_array_equal(region_voltage(voltage, geometry), inputs["region_voltage"][tick])
                np.testing.assert_allclose(c@voltage, inputs["mass"][tick], rtol=1e-13, atol=1e-15)
                previous = voltage.copy()
                count += 1
        if count != m["ticks"]+1:
            raise ValueError("Incomplete recording")
        results.append({"condition": name, "states_checked": count*len(c),
                        "max_relative_backward_error": worst, "max_mass_error": worst_mass})
    return {"numerical_record_audit": results, "scope": "Not a physiological validation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    for name in ("directory", "source_directory", "cells", "output"):
        r.add_argument(name, type=Path)
    a = sub.add_parser("audit")
    a.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = audit(args.directory) if args.command == "audit" else run(args.directory, args.source_directory, args.cells, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
