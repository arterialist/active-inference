"""Recover Amin's registered APL and evaluate its spatial reference operator.

This reads a saved anatomical preparation, not a PAULA network checkpoint.
The saved MATLAB object and the repository's v1.1 CSV are distinct revisions.
No fuzzy anatomical join, reconstructed-cell deletion or new connection is
used to make their counts agree. Sample points are numerical quadrature sites,
not additional neurons or biological synapses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, dijkstra
from scipy.spatial import cKDTree

from .amin_spatial import AUTHOR_COMMIT, CELLS_SHA256, SOURCES, digest, merge_profiles, write_new

PRIMARY = {
    "apl200607.mat": "2370823eeb91f12f62aca1b7316f1731f70ae1c3d9e246d3334217d1a65bcf19",
    "APLskelv1.1.csv": "c6fa7a1c17458645d01f5b9f62dc945d61607b1e4975664570d5d064a43370c3",
    "aplmanualSkel200303.mat": "62e0eb4289636c521db3657421e4cf789b7fae4d0f1f46bc2ef59bbf485eb983",
}


def integer(values, name):
    values = np.asarray(values)
    if not np.isfinite(values).all() or np.any(values != np.floor(values)):
        raise ValueError(f"Noninteger source field: {name}")
    return values.astype(np.int64)


def tree_order(parents):
    parents = np.asarray(parents)
    n = len(parents)
    if (parents.dtype.kind not in "iu" or parents.shape != (n,)
            or np.any(parents < -1) or np.any(parents >= n)):
        raise ValueError("Invalid parent indices")
    roots = np.flatnonzero(parents < 0)
    if len(roots) != 1:
        raise ValueError("Expected exactly one connected tree")
    children = np.flatnonzero(parents >= 0)
    graph = coo_matrix((np.ones(len(children)), (parents[children], children)), shape=(n, n)).tocsr()
    order = breadth_first_order(graph, int(roots[0]), directed=True, return_predecessors=False)
    if len(order) != n:
        raise ValueError("Disconnected or cyclic parent graph")
    return order


def taper_distance(length, ra, rb):
    """Integral of dx/sqrt(r(x)) for a linearly tapered edge, in sqrt(pixels)."""
    length, ra, rb = np.asarray(length), np.asarray(ra), np.asarray(rb)
    if (not all(np.isfinite(v).all() for v in (length, ra, rb))
            or np.any(length <= 0) or np.any(ra <= 0) or np.any(rb <= 0)):
        raise ValueError("Positive finite edge length and radii required")
    return 2 * length / (np.sqrt(ra) + np.sqrt(rb))


def sample_tree(parents, xyz, radius, sample_children, sample_xyz):
    """Subdivide original edges at saved sample positions, without new paths."""
    parents = np.asarray(parents)
    tree_order(parents)
    n, m = len(parents), len(sample_children)
    child = integer(sample_children, "sample child")
    if np.any(child < 0) or np.any(child >= n) or np.any(parents[child] < 0):
        raise ValueError("Sample does not lie on a non-root edge")
    pxyz, cxyz = xyz[parents[child]], xyz[child]
    edge = cxyz - pxyz
    fraction = np.einsum("ij,ij->i", sample_xyz - pxyz, edge) / np.einsum("ij,ij->i", edge, edge)
    error = np.linalg.norm(sample_xyz - (pxyz + fraction[:, None] * edge), axis=1)
    if np.any(fraction <= 0) or np.any(fraction >= 1) or np.max(error, initial=0) > 1e-7:
        raise ValueError("Saved sample is not strictly inside its declared edge")
    out_parents = np.r_[parents, np.full(m, -1, dtype=np.int64)]
    out_xyz = np.vstack((xyz, sample_xyz))
    out_radius = np.r_[radius, radius[parents[child]] + fraction * (radius[child] - radius[parents[child]])]
    sort = np.lexsort((fraction, child))
    previous_child = -1
    previous_node = -1
    for s in sort:
        c = child[s]
        if c != previous_child:
            previous_node = parents[c]
        out_parents[n+s] = previous_node
        out_parents[c] = n+s
        previous_node, previous_child = n+s, c
    order = tree_order(out_parents)
    c = order[1:]
    edge_distance = np.zeros(n+m)
    edge_distance[c] = taper_distance(np.linalg.norm(out_xyz[c] - out_xyz[out_parents[c]], axis=1),
                                      out_radius[c], out_radius[out_parents[c]])
    return out_parents, out_xyz, out_radius, edge_distance, order, fraction, error


def exponential_tree_sum(parents, distance, weights, decay_length, order=None):
    """Exact all-source exp(-tree distance/k) sum in linear storage and work.

    Two tree passes factor the same pairwise equation, with no distance cutoff.
    This is a static reference operator, not a replacement neural controller.
    """
    parents, distance, weights = map(np.asarray, (parents, distance, weights))
    if order is None:
        order = tree_order(parents)
    if (distance.shape != parents.shape or weights.ndim not in (1, 2)
            or len(weights) != len(parents) or not np.isfinite(weights).all()
            or not np.isfinite(distance).all() or np.any(distance < 0)
            or not np.isfinite(decay_length) or decay_length <= 0):
        raise ValueError("Invalid tree kernel input")
    decay = np.exp(-distance / decay_length)
    down = weights.astype(np.float64, copy=True)
    for child in order[:0:-1]:
        down[parents[child]] += decay[child] * down[child]
    total = down.copy()
    for child in order[1:]:
        total[child] += decay[child] * (total[parents[child]] - decay[child] * down[child])
    return total


def extract(directory, output):
    from importlib.metadata import version
    from matio import load_from_mat
    import pandas as pd

    if output.exists():
        raise FileExistsError(output)
    if version("mat-io") != "1.0.0":
        raise ValueError("Use the audited mat-io 1.0.0 reader")
    for name, expected in PRIMARY.items():
        if digest(directory / name) != expected:
            raise ValueError(f"Changed primary source: {name}")
    obj = load_from_mat(directory / "apl200607.mat", raw_data=True)["apl"]
    if obj.classname != "APLskel":
        raise ValueError("Saved object is not APLskel")
    a = obj.properties
    table = a["skel"].properties
    cols = {str(k.item()): v.ravel() for k, v in zip(table["varnames"].ravel(), table["data"].ravel(), strict=True)}
    xyz = np.column_stack([cols[k] for k in ("x", "y", "z")])
    radius = cols["radius"]
    ids = integer(cols["rowId"], "rowId")
    if not np.array_equal(ids, np.arange(1, len(ids)+1)):
        raise ValueError("Saved source row IDs are not consecutive")
    parents = integer(cols["link"], "link")
    parents = np.where(parents < 1, -1, parents-1)
    order = tree_order(parents)
    if not np.array_equal(xyz, np.vstack([v.ravel() for v in a["nodes"]["coords"].ravel()])):
        raise ValueError("Saved node coordinates and skeleton table differ")
    links = integer(a["links"], "links")
    expected_links = np.column_stack((ids, parents+1))
    expected_links[parents < 0] = 0
    if not np.array_equal(links, expected_links):
        raise ValueError("Saved adjacency differs from skeleton table")
    regions = integer(a["nodeVoronoiSegments"].ravel(), "node regions")
    link_regions = integer(a["linkVoronoiSegments"].ravel(), "link regions")
    if not np.array_equal(regions[parents >= 0], link_regions[parents >= 0]):
        raise ValueError("Saved node/link observation conventions differ")
    # Independently recompute the saved region assignment from the authors'
    # original raster mask and documented coordinate normalization.
    mask = load_from_mat(directory / "aplmanualSkel200303.mat", raw_data=True)["aplSkel"].properties["voronoiMask"]
    factor = (np.ptp(xyz, axis=0).max()+1) / 256
    pixel = np.floor((xyz - xyz.min(axis=0)) / factor).astype(int)
    valid = (xyz[:, 0] > 10000) & (xyz[:, 0] < 28000) & (pixel[:, 2] < mask.shape[2])
    recomputed = np.zeros(len(ids), dtype=np.int64)
    selected = mask[pixel[valid, 0], pixel[valid, 1], pixel[valid, 2], :]
    if np.any(selected.sum(axis=1) > 1):
        raise ValueError("Overlapping author Voronoi regions")
    recomputed[valid] = np.where(selected.any(axis=1), selected.argmax(axis=1)+1, 0)
    if not np.array_equal(regions, recomputed):
        raise ValueError(f"Saved registration not reproduced: {np.count_nonzero(regions != recomputed)} nodes")
    del mask, selected
    samples = a["synSet"].ravel()[2]
    sample_xyz = samples["synLocs"]
    sample_child = integer(samples["synLinks"].ravel(), "sample links")-1
    sample_region = integer(samples["voronoiSegments"].ravel(), "sample regions")
    sample_roi = integer(samples["synROIs"].ravel(), "sample ROIs")
    if not np.array_equal(sample_region, link_regions[sample_child]):
        raise ValueError("Sample region differs from saved edge region")
    used = np.isin(sample_roi, [3, 7, 13, 14, 15, 16, 17, 18]) & (sample_xyz[:, 0] > 10000) & (sample_xyz[:, 0] < 28000) & (sample_region > 0)
    pp, xx, rr, ed, sample_order, fraction, projection_error = sample_tree(parents, xyz, radius, sample_child, sample_xyz)
    # Record cross-revision correspondences for diagnosis only. They do not
    # control any geometry, radius, edge or observation mapping used above.
    raw = pd.read_csv(directory / "APLskelv1.1.csv")
    nearest_distance, nearest_index = cKDTree(raw[["x", "y", "z"]].to_numpy()).query(xyz)
    exact = nearest_distance == 0
    comparable = exact & ((parents < 0) | exact[np.maximum(parents, 0)])
    mapped_parent = np.where(parents < 0, -1, nearest_index[np.maximum(parents, 0)])
    raw_parent = integer(raw.link.to_numpy(), "CSV link")
    raw_parent = np.where(raw_parent < 1, -1, raw_parent-1)
    radius_error = np.abs(radius[exact] - raw.radius.to_numpy()[nearest_index[exact]])
    output.mkdir(parents=True)
    with (output / "registered_anatomy.npz").open("xb") as stream:
        np.savez_compressed(stream, source_ids=ids, original_parents=parents,
            original_xyz_pixels=xyz, original_radius_pixels=radius, node_region=regions,
            sample_child=sample_child, sample_xyz_pixels=sample_xyz, sample_region=sample_region,
            sample_roi=sample_roi, sample_used=used, sample_fraction=fraction,
            parents=pp, xyz_pixels=xx, radius_pixels=rr, electrotonic_distance=ed,
            traversal_order=sample_order, sample_nodes=np.arange(len(ids), len(pp)),
            csv_nearest_index=nearest_index, csv_nearest_distance=nearest_distance)
    result = {"schema": "amin-registered-apl-v1", "source_sha256": PRIMARY,
        "author_commit": AUTHOR_COMMIT, "reader_version": version("mat-io"),
        "source_filename_in_object": str(a["filepath"].item()),
        "source_nodes": len(ids), "sample_count": len(sample_xyz), "used_samples": int(used.sum()),
        "subdivided_nodes": len(pp), "all_node_region_assignments_reproduced": True,
        "regions_zero_unassigned": int((regions == 0).sum()),
        "samples_per_region": np.bincount(sample_region[used], minlength=36).tolist(),
        "sample_projection_max_error_pixels": float(projection_error.max()),
        "cross_revision": {"csv_nodes": len(raw), "saved_nodes_with_exact_xyz_match": int(exact.sum()),
            "unique_matched_csv_nodes": len(np.unique(nearest_index[exact])),
            "radius_changes_above_1e_minus_6_pixels": int((radius_error > 1e-6).sum()),
            "radius_max_difference_pixels": float(radius_error.max()),
            "comparable_parent_edges": int(comparable.sum()),
            "different_parent_edges": int(np.sum(mapped_parent[comparable] != raw_parent[nearest_index[comparable]])),
            "used_for_registration": False},
        "geometry_units": "source pixel = 8 nm; radius is also in source pixels",
        "scope": "Authors' saved healed APL reference preparation; not the current CSV revision or FlyWire specimen",
        "geometry_sha256": digest(output / "registered_anatomy.npz"), "source_code_sha256": digest(__file__)}
    write_new(output / "extraction.json", result)
    return result


def benchmark(directory, source_directory, cells_path, output):
    from scipy.io import loadmat
    if output.exists():
        raise FileExistsError(output)
    m = json.loads((directory / "extraction.json").read_text())
    if digest(directory / "registered_anatomy.npz") != m["geometry_sha256"] or digest(cells_path) != CELLS_SHA256:
        raise ValueError("Changed registered geometry or observation cells")
    for name in ("redDyeVoronoiFits.mat", "voronoiIndices.mat"):
        if digest(source_directory / name) != SOURCES[name]:
            raise ValueError(f"Changed stimulus/observation mapping: {name}")
    with np.load(directory / "registered_anatomy.npz", allow_pickle=False) as f:
        parents, distance, order, points, used, region = (f[k] for k in
            ("parents", "electrotonic_distance", "traversal_order", "sample_nodes", "sample_used", "sample_region"))
    tree_order(parents)
    dyes = loadmat(source_directory / "redDyeVoronoiFits.mat")
    indices = loadmat(source_directory / "voronoiIndices.mat")
    hi, vi = (indices[k].ravel().astype(int)-1 for k in ("ctohindices", "ctovindices"))
    book = json.loads(cells_path.read_text())["workbooks"]["elife-56954-fig8-data1-v2.xlsx"]
    stimulus = np.zeros((len(parents), 3))
    for j, key in enumerate(("hstimFit", "vstimFit", "cstimFit")):
        stimulus[points[used], j] = dyes[key].ravel()[region[used]-1]
    # Dijkstra is an independent check on selected actual sample destinations;
    # it uses the full tree and every source, not just a toy tree or near sites.
    child = np.flatnonzero(parents >= 0)
    graph = coo_matrix((np.r_[distance[child], distance[child]],
                       (np.r_[child, parents[child]], np.r_[parents[child], child])), shape=(len(parents), len(parents))).tocsr()
    check_nodes = points[used][np.linspace(0, used.sum()-1, 7, dtype=int)]
    checked_distance = dijkstra(graph, directed=False, indices=check_nodes)
    output.mkdir(parents=True)
    results, files = [], []
    for length, k in ((25, 624), (50, 1249), (75, 1873)):
        total = exponential_tree_sum(parents, distance, stimulus, k, order)
        direct = np.exp(-checked_distance[:, points[used]] / k) @ stimulus[points[used]]
        max_check_error = float(np.max(abs(total[check_nodes] - direct)))
        if not np.allclose(total[check_nodes], direct, rtol=1e-11, atol=1e-11):
            raise ValueError("Tree factorization disagrees with independent all-source distances")
        name = f"kernel-{length}.npz"
        with (output / name).open("xb") as stream:
            np.savez_compressed(stream, total=total, stimulus=stimulus, check_nodes=check_nodes,
                                direct_check=direct)
        files.append({"file": name, "sha256": digest(output / name)})
        for j, site in enumerate(("H", "V", "C")):
            values = total[points[used], j]
            counts = np.bincount(region[used], minlength=36)[1:]
            profile = np.bincount(region[used], weights=values, minlength=36)[1:] / counts
            profile /= profile.max()
            rows = book[site+"stim"]["rows"]
            start = (13 if site == "H" else 8) + 2*((length-25)//25)
            published = merge_profiles([r[start] for r in rows[4:30]], [r[start+1] for r in rows[4:32]], hi, vi)
            observed = merge_profiles([r[1] for r in rows[4:30]], [r[3] for r in rows[4:32]], hi, vi)
            results.append({"site": site, "nominal_length_um": length, "k_sqrt_pixels": k,
                "profile": profile.tolist(), "published_profile": published.tolist(), "observed": observed.tolist(),
                "max_abs_difference_from_published": float(np.max(abs(profile-published))),
                "rmse_against_observed": float(np.sqrt(np.mean((profile-observed)**2))),
                "independent_distance_max_error": max_check_error})
    result = {"schema": "amin-anatomical-kernel-v1", "geometry_sha256": m["geometry_sha256"],
        "source_code_sha256": digest(__file__), "cells_sha256": digest(cells_path),
        "input_samples": int(used.sum()), "rows": results, "files": files,
        "limits": ["Static published reference equation, not a PAULA neuron or dynamic learning result",
                   "Uses the authors' saved healed tree and saved random sample positions",
                   "Saved object differs from the separate v1.1 skeleton CSV",
                   "Normalized spatial calcium means do not calibrate voltage or model time"]}
    write_new(output / "analysis.json", result)
    return {"rows": [{k: v for k, v in row.items() if not isinstance(v, list)} for row in results]}


def cable_experiment(directory, source_directory, cells_path, output):
    """Full registered tree, fixed membrane-density drive, every node/tick.

    A conditional intracellular test of the unchanged PAULA cable operator.
    It does not run a spiking/plastic network, model the ATP puff duration or
    infer calibrated voltage from calcium. Unassigned regions receive no
    stimulus but remain in the sealed tree and can carry axial current.
    """
    import inspect
    import shutil
    from scipy.io import loadmat
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    from simulations.paula_loader import ensure_paula_available
    ensure_paula_available()
    from neuron.extensions.experimental.passive_cable import PassiveCable
    if output.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(directory).free < 3_000_000_000:
        raise ValueError("Need 3 GB free for bounded full-node recordings")
    m = json.loads((directory / "extraction.json").read_text())
    if digest(directory / "registered_anatomy.npz") != m["geometry_sha256"] or digest(cells_path) != CELLS_SHA256:
        raise ValueError("Changed registered geometry or observation cells")
    for name in ("redDyeVoronoiFits.mat", "voronoiIndices.mat"):
        if digest(source_directory / name) != SOURCES[name]:
            raise ValueError(f"Changed stimulus/observation mapping: {name}")
    with np.load(directory / "registered_anatomy.npz", allow_pickle=False) as f:
        parents, xyz, radius, regions, used, sample_child, sample_region, fraction = (f[k] for k in
            ("original_parents", "original_xyz_pixels", "original_radius_pixels", "node_region",
             "sample_used", "sample_child", "sample_region", "sample_fraction"))
    dyes = loadmat(source_directory / "redDyeVoronoiFits.mat")
    idx = loadmat(source_directory / "voronoiIndices.mat")
    hi, vi = (idx[k].ravel().astype(int)-1 for k in ("ctohindices", "ctovindices"))
    book = json.loads(cells_path.read_text())["workbooks"]["elife-56954-fig8-data1-v2.xlsx"]
    # Original source tree, not the sample-subdivided tree. Interpolation reads
    # sample voltage without introducing artificial neuronal compartments.
    cable = PassiveCable(parents, xyz*.008, radius*.008, 25000.)
    child = sample_child[used]
    parent = parents[child]
    f = fraction[used]
    count = np.bincount(sample_region[used], minlength=36)[1:]
    def profile(voltage):
        sample_v = (1-f)*voltage[parent] + f*voltage[child]
        regional = np.bincount(sample_region[used], weights=sample_v, minlength=36)[1:] / count
        return regional
    output.mkdir(parents=True)
    files, results = [], []
    with (output / "geometry.npz").open("xb") as stream:
        np.savez_compressed(stream, parents=parents, xyz_um=xyz*.008, radius_um=radius*.008,
            region=regions, capacity=cable.capacity, sample_parent=parent, sample_child=child,
            sample_fraction=f, sample_region=sample_region[used])
    files.append({"file": "geometry.npz", "sha256": digest(output / "geometry.npz")})
    for site in ("H", "V", "C"):
        drive = dyes[site.lower()+"stimFit"].ravel()
        density = np.zeros(len(parents))
        density[regions > 0] = drive[regions[regions > 0]-1]
        current = cable.capacity*density
        stationary = spsolve(diags(cable.capacity)+cable.laplacian, current)
        cable.voltage.fill(0)
        times, states = [0], [cable.voltage.copy()]
        regional = [profile(cable.voltage)]
        equation_error, masses = [], [0.]
        for tick in range(1, 401):
            old = cable.voltage.copy()
            cable.step(current, .05)
            residual = cable.capacity*cable.voltage + .05*(cable.laplacian@cable.voltage) - .95*cable.capacity*old - .05*current
            equation_error.append(float(np.max(abs(residual))))
            masses.append(float(cable.capacity@cable.voltage))
            times.append(tick)
            states.append(cable.voltage.copy())
            regional.append(profile(cable.voltage))
            if len(states) == 8 or tick == 400:
                name = f"{site.lower()}-through-{tick:04d}.npz"
                with (output / name).open("xb") as stream:
                    np.savez_compressed(stream, tick=np.array(times), voltage=np.array(states))
                files.append({"file": name, "site": site, "first_tick": times[0], "last_tick": times[-1],
                              "sha256": digest(output / name)})
                times, states = [], []
            if tick % 100 == 0:
                print(json.dumps({"site": site, "tick": tick, "max_voltage": float(cable.voltage.max())}), flush=True)
        error = float(np.max(abs(stationary-cable.voltage)))
        if max(equation_error) > 1e-10 or error > 1e-8:
            raise ValueError("Registered cable failed numerical checks or settling")
        predicted = profile(stationary)
        predicted /= predicted.max()
        table = book[site+"stim"]["rows"]
        observed = merge_profiles([r[1] for r in table[4:30]], [r[3] for r in table[4:32]], hi, vi)
        name = f"{site.lower()}-inputs-and-readout.npz"
        with (output / name).open("xb") as stream:
            np.savez_compressed(stream, density=density, current=current, stationary=stationary,
                region_voltage=np.array(regional), equation_residual=np.array(equation_error),
                mass=np.array(masses), observed=observed, normalized_prediction=predicted)
        files.append({"file": name, "sha256": digest(output / name)})
        results.append({"site": site, "normalized_prediction": predicted.tolist(), "observed": observed.tolist(),
            "rmse": float(np.sqrt(np.mean((predicted-observed)**2))), "stationary_max_error": error,
            "equation_max_error": max(equation_error), "peak_region": int(np.argmax(predicted)+1)})
    result = {"schema": "amin-registered-cable-v1", "registered_geometry_sha256": m["geometry_sha256"],
        "analysis_source_sha256": digest(__file__), "cable_source_sha256": digest(inspect.getfile(PassiveCable)),
        "node_count": len(parents), "rm_over_ra_um": 25000., "alpha": .05, "ticks": 400,
        "scope": "Conditional PAULA intracellular operator on the authors' saved registered APL, not the FlyWire circuit",
        "input": "Fitted dye mapped to unit membrane drive density within each saved Voronoi region; held, not ATP kinetics",
        "readout": "Linear voltage interpolation at saved selected sample sites, region mean, then peak normalization",
        "limits": ["No calcium mapping or physical tick duration", "No synaptic/plastic network runs",
                   "Saved reconstruction differs from the newer v1.1 CSV", "No physiological acceptance threshold",
                   "Membrane-area drive differs from the reference kernel's length-sampled point input"],
        "results": results, "files": files}
    write_new(output / "analysis.json", result)
    return {"results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("directory", type=Path)
    ex.add_argument("output", type=Path)
    bm = sub.add_parser("benchmark")
    for name in ("directory", "source_directory", "cells", "output"):
        bm.add_argument(name, type=Path)
    cb = sub.add_parser("cable")
    for name in ("directory", "source_directory", "cells", "output"):
        cb.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.command == "extract":
        result = extract(args.directory, args.output)
    else:
        fn = benchmark if args.command == "benchmark" else cable_experiment
        result = fn(args.directory, args.source_directory, args.cells, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
