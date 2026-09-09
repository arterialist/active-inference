"""Matched intracellular input-physics diagnostics on registered Amin APL.

These are conditional stationary cable equations, not a running neural brain.
Anatomy, registration, readout and observed data are held fixed. No parameter
from this exploratory comparison becomes an agent default or a measured value.
"""
from __future__ import annotations

import argparse
import inspect
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from .amin_anatomy import tree_order
from .amin_spatial import CELLS_SHA256, SOURCES, digest, merge_profiles, write_new

# 6,250 / 25,000 / 56,250 correspond to nominal 25 / 50 / 75 um at
# radius 0.2 um. 90,700 and 2,000,000 are the comparison ratios discussed
# in Amin 2020. Intermediate values probe the full cable, not the heuristic.
RATIOS_UM = (6250., 12500., 25000., 56250., 90700., 200000., 500000., 2000000.)
CONDUCTANCE_GAINS = (.1, 1., 10., 100.)
SITES = ("H", "V", "C")


def preparation(directory, source_directory, cells_path):
    from simulations.paula_loader import ensure_paula_available
    ensure_paula_available()
    from neuron.extensions.experimental.passive_cable import PassiveCable

    manifest = json.loads((directory / "extraction.json").read_text())
    if digest(directory / "registered_anatomy.npz") != manifest["geometry_sha256"]:
        raise ValueError("Registered geometry changed")
    if digest(cells_path) != CELLS_SHA256:
        raise ValueError("Observation data changed")
    for name in ("redDyeVoronoiFits.mat", "voronoiIndices.mat"):
        if digest(source_directory / name) != SOURCES[name]:
            raise ValueError(f"Changed source: {name}")
    with np.load(directory / "registered_anatomy.npz", allow_pickle=False) as f:
        parents, xyz, radius, regions = (f[k] for k in
            ("original_parents", "original_xyz_pixels", "original_radius_pixels", "node_region"))
        used = f["sample_used"]
        child, fraction, sample_region = (f[k][used] for k in
            ("sample_child", "sample_fraction", "sample_region"))
    tree_order(parents)
    cable = PassiveCable(parents, xyz*.008, radius*.008, 25000.)
    edge_child = np.flatnonzero(parents >= 0)
    lengths = np.linalg.norm(cable.xyz_um[edge_child] - cable.xyz_um[parents[edge_child]], axis=1)
    length_measure = np.bincount(edge_child, weights=lengths/2, minlength=len(parents))
    length_measure += np.bincount(parents[edge_child], weights=lengths/2, minlength=len(parents))
    length_measure /= length_measure.sum()
    dyes = loadmat(source_directory / "redDyeVoronoiFits.mat")
    density = np.zeros((len(parents), 3))
    for j, site in enumerate(SITES):
        density[regions > 0, j] = dyes[site.lower()+"stimFit"].ravel()[regions[regions > 0]-1]
    idx = loadmat(source_directory / "voronoiIndices.mat")
    hi, vi = (idx[k].ravel().astype(int)-1 for k in ("ctohindices", "ctovindices"))
    book = json.loads(cells_path.read_text())["workbooks"]["elife-56954-fig8-data1-v2.xlsx"]
    observed = []
    for site in SITES:
        rows = book[site+"stim"]["rows"]
        observed.append(merge_profiles([r[1] for r in rows[4:30]], [r[3] for r in rows[4:32]], hi, vi))
    return cable, {
        "parents": parents, "xyz_um": cable.xyz_um, "radius_um": cable.radius_um,
        "region": regions, "capacity": cable.capacity, "length_measure": length_measure,
        "density": density, "sample_child": child, "sample_parent": parents[child],
        "sample_fraction": fraction, "sample_region": sample_region,
        "observed": np.array(observed).T,
    }, {
        "registered_geometry_sha256": manifest["geometry_sha256"],
        "cells_sha256": digest(cells_path), "cable_source_sha256": digest(inspect.getfile(PassiveCable)),
        "stimulus_sha256": SOURCES["redDyeVoronoiFits.mat"],
        "mapping_sha256": SOURCES["voronoiIndices.mat"],
    }


def region_voltage(voltage, geometry):
    f = geometry["sample_fraction"]
    samples = (1-f)*voltage[geometry["sample_parent"]] + f*voltage[geometry["sample_child"]]
    region = geometry["sample_region"]
    return np.bincount(region, weights=samples, minlength=36)[1:] / np.bincount(region, minlength=36)[1:]


def stationary(capacity, laplacian, current, conductance=None, reversal=1.):
    """Solve (C + L + G)v = I + G*E, with G in normalized cable units."""
    c, i = np.asarray(capacity), np.asarray(current)
    g = np.zeros_like(c) if conductance is None else np.asarray(conductance)
    if (c.ndim != 1 or i.shape != c.shape or g.shape != c.shape
            or not all(np.isfinite(v).all() for v in (c, i, g))
            or np.any(c <= 0) or np.any(g < 0) or not np.isfinite(reversal)):
        raise ValueError("Invalid membrane, current or conductance")
    matrix = diags(c+g, format="csc") + laplacian
    rhs = i + g*reversal
    voltage = splu(matrix, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0).solve(rhs)
    residual = float(np.max(abs(matrix@voltage-rhs)))
    mass_error = float(abs(c@voltage - (i+g*(reversal-voltage)).sum()))
    if not np.isfinite(voltage).all() or residual > 1e-9 or mass_error > 1e-8:
        raise ValueError("Stationary equation or mass balance failed")
    return voltage, residual, mass_error


def sweep(directory, source_directory, cells_path, output):
    if output.exists():
        raise FileExistsError(output)
    cable, geometry, provenance = preparation(directory, source_directory, cells_path)
    output.mkdir(parents=True)
    with (output / "geometry.npz").open("xb") as stream:
        np.savez_compressed(stream, **geometry)
    base = cable.capacity[:, None]*geometry["density"]
    length = geometry["length_measure"][:, None]*geometry["density"]
    # Same total injected current per stimulation site. This intervenes on
    # spatial allocation only, not total dose or the membrane/leak measure.
    length *= base.sum(axis=0)/length.sum(axis=0)
    cases = [("membrane-current", None), ("length-current", None)]
    cases += [("membrane-conductance", gain) for gain in CONDUCTANCE_GAINS]
    results, files = [], [{"file": "geometry.npz", "sha256": digest(output / "geometry.npz")}]
    start = time.perf_counter()
    for ratio in RATIOS_UM:
        laplacian = cable.laplacian*(ratio/25000.)
        for mode, gain in cases:
            name = f"r{ratio:g}-{mode}" + (f"-g{gain:g}" if gain is not None else "")
            voltages, currents, conductances, raw_profiles, checks = [], [], [], [], []
            for j, site in enumerate(SITES):
                current = length[:, j] if mode == "length-current" else base[:, j]
                g = np.zeros(len(current))
                if mode == "membrane-conductance":
                    g = gain*current
                    current = np.zeros(len(current))
                v, residual, mass_error = stationary(cable.capacity, laplacian, current, g)
                voltages.append(v)
                currents.append(current)
                conductances.append(g)
                raw_profiles.append(region_voltage(v, geometry))
                checks.append({"site": site, "equation_max_residual": residual,
                               "mass_balance_error": mass_error})
            profiles = np.array(raw_profiles).T
            prediction = profiles/profiles.max(axis=0)
            squared = (prediction-geometry["observed"])**2
            filename = name+".npz"
            with (output / filename).open("xb") as stream:
                np.savez_compressed(stream, voltage=np.array(voltages).T, current=np.array(currents).T,
                    conductance=np.array(conductances).T, region_voltage=profiles,
                    normalized_prediction=prediction, squared_error=squared)
            files.append({"file": filename, "sha256": digest(output / filename)})
            row = {"case": name, "file": filename, "rm_over_ra_um": ratio, "mode": mode,
                "conductance_gain": gain, "reversal": 1., "checks": checks,
                "rmse_by_site": dict(zip(SITES, np.sqrt(squared.mean(axis=0)).tolist(), strict=True)),
                "overall_rmse": float(np.sqrt(squared.mean())),
                "v_stimulation_region17": float(prediction[16, 1]),
                "peak_regions": (np.argmax(profiles, axis=0)+1).tolist()}
            results.append(row)
            print(json.dumps({k: row[k] for k in ("case", "rmse_by_site", "v_stimulation_region17")}), flush=True)
    # This is a leave-one-site-out sensitivity diagnostic on already inspected
    # data, NOT an unseen validation set. Selection never uses the held-out site.
    held_out = []
    for site in SITES:
        selected = min(results, key=lambda r: sum(r["rmse_by_site"][s]**2 for s in SITES if s != site))
        held_out.append({"held_out_site": site, "selected_case": selected["case"],
                         "held_out_rmse": selected["rmse_by_site"][site]})
    result = {"schema": "amin-input-physics-sweep-v1", **provenance,
        "analysis_source_sha256": digest(__file__), "ratios_um": RATIOS_UM,
        "conductance_gains": CONDUCTANCE_GAINS, "node_count": len(cable.capacity),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "wall_seconds": time.perf_counter()-start},
        "results": results, "leave_one_site_out_diagnostic": held_out, "files": files,
        "limits": ["Static conditional equations, not a neural tick simulation or learning result",
            "No calibrated calcium-voltage map, physical tick duration or ATP-to-channel occupancy map",
            "Conductance gain is relative to resting leak per unit fitted dye; it is not measured",
            "Reversal is one in arbitrary voltage units; global scale cancels in normalized steady profiles",
            "Length-current changes drive allocation, not morphology or membrane area",
            "Shared specimen and previously inspected observations, no independent-animal validation",
            "No acceptance cutoff and no changes to connected FlyWire circuit or PAULA defaults"]}
    write_new(output / "analysis.json", result)
    return {"best_descriptive_case": min(results, key=lambda r: r["overall_rmse"]),
            "leave_one_site_out_diagnostic": held_out, "wall_seconds": result["runtime"]["wall_seconds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "source_directory", "cells", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(sweep(args.directory, args.source_directory, args.cells, args.output), indent=2))


if __name__ == "__main__":
    main()
