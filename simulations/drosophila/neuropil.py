"""Public FlyWire neuropil meshes as observational apertures, not neural inputs.

Use the actual triangle volumes in the same CATMAID project as the APL tree.
No fitted registration, guessed box ROI, anatomical pruning or neural edit.
Overlaps, outside nodes and ray-direction disagreements remain explicit.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .amin_spatial import write_new
from .connectome import sha256
from .spatial import PublicSnapshot

REGIONS = ("MB_CA_L", "MB_PED_L", "MB_VL_L", "MB_ML_L")
DIRECTIONS = np.array([[1., .2, .3], [-.3, 1., .2], [.2, -.3, 1.]])


def acquire(output):
    snapshot = PublicSnapshot(output)
    index = snapshot.get("volumes.json.gz", "/1/volumes/")
    columns = index["columns"]
    volumes = {r[columns.index("name")]: dict(zip(columns, r, strict=True)) for r in index["data"]}
    result = {"schema": "flywire-neuropil-source-v1", "project_id": 1,
        "status": "incomplete", "regions": {}, "files": snapshot.records}
    try:
        for name in REGIONS:
            row = volumes[name]
            if row["project_id"] != 1 or not row["watertight"]:
                raise ValueError(f"Unexpected source mesh: {name}")
            detail = snapshot.get(name+".json.gz", f"/1/volumes/{row['id']}/")
            if detail["name"] != name or detail["project_id"] != 1 or detail["id"] != row["id"]:
                raise ValueError("Volume identity mismatch")
            result["regions"][name] = row
        result["status"] = "complete"
    finally:
        write_new(output/"manifest.json", result)
    return result


def mesh_from_x3d(text):
    import trimesh
    element = ET.fromstring(text)
    if element.tag != "IndexedTriangleSet":
        raise ValueError("Unsupported source mesh encoding")
    coordinate = element.find("Coordinate")
    if coordinate is None:
        raise ValueError("Missing source vertices")
    # Parse every token; fromstring can silently stop at malformed trailing text.
    vertices = np.asarray(coordinate.attrib["point"].split(), dtype=float)
    indices = np.asarray(element.attrib["index"].split(), dtype=float)
    if (len(vertices) % 3 or len(indices) % 3 or not np.isfinite(vertices).all()
            or not np.isfinite(indices).all() or np.any(indices != np.floor(indices))):
        raise ValueError("Invalid mesh numbers")
    vertices, faces = vertices.reshape(-1, 3), indices.astype(np.int64).reshape(-1, 3)
    if len(faces) == 0 or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("Invalid triangle index")
    # X3D repeats identical coordinates per triangle. Merge exact duplicates
    # only; retain every triangle and its winding, with no geometric repair.
    unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
    mesh = trimesh.Trimesh(vertices=unique, faces=inverse[faces], process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent or not mesh.is_volume:
        raise ValueError("Source mesh is not a closed consistently oriented volume")
    return mesh


def classify(mesh, xyz, chunk_size=2048):
    """Three ray-parity queries; do not resolve disagreements by majority vote."""
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("Positive integer chunk size required")
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Invalid coordinates")
    votes = np.zeros((len(xyz), len(DIRECTIONS)), dtype=bool)
    candidate = np.flatnonzero(np.all((xyz >= mesh.bounds[0]) & (xyz <= mesh.bounds[1]), axis=1))
    for start in range(0, len(candidate), chunk_size):
        rows = candidate[start:start+chunk_size]
        for j, direction in enumerate(DIRECTIONS):
            _, ray, _ = mesh.ray.intersects_location(xyz[rows], np.tile(direction, (len(rows), 1)), multiple_hits=True)
            votes[rows, j] = np.bincount(ray, minlength=len(rows)) % 2 == 1
    ambiguous = votes.any(axis=1) != votes.all(axis=1)
    return votes.all(axis=1), ambiguous


def bind(source, anatomy, output):
    import trimesh
    import rtree
    if output.exists():
        raise FileExistsError(output)
    m = json.loads((source/"manifest.json").read_text())
    if m["schema"] != "flywire-neuropil-source-v1" or m["status"] != "complete":
        raise ValueError("Incomplete source meshes")
    for name, record in m["files"].items():
        if Path(name).name != name or sha256(source/name) != record["sha256"]:
            raise ValueError("Changed mesh snapshot")
    a = json.loads((anatomy/"analysis.json").read_text())
    if sha256(anatomy/"anatomy.npz") != a["anatomy_sha256"]:
        raise ValueError("Changed spatial anatomy")
    with np.load(anatomy/"anatomy.npz", allow_pickle=False) as f:
        ids, xyz = f["node_ids"], f["xyz_nm"]
    inside, ambiguous, details = [], [], []
    output.mkdir(parents=True)
    for name in REGIONS:
        with gzip.open(source/(name+".json.gz"), "rt") as stream:
            raw = json.load(stream)
        mesh = mesh_from_x3d(raw["mesh"])
        bbox = np.array([[raw["bbox"][side][axis] for axis in ("x", "y", "z")] for side in ("min", "max")])
        np.testing.assert_allclose(mesh.bounds, bbox, rtol=0, atol=.1)
        mask, uncertain = classify(mesh, xyz)
        inside.append(mask)
        ambiguous.append(uncertain)
        details.append({"name": name, "id": raw["id"], "nodes_inside": int(mask.sum()),
            "ambiguous_nodes": int(uncertain.sum()), "vertices": len(mesh.vertices),
            "faces": len(mesh.faces), "mesh_bounds_nm": mesh.bounds.tolist(),
            "volume_nm3": float(mesh.volume)})
        print(json.dumps(details[-1]), flush=True)
        with (output/(name+"-mesh.npz")).open("xb") as stream:
            np.savez_compressed(stream, vertices_nm=mesh.vertices, faces=mesh.faces)
    inside, ambiguous = np.array(inside).T, np.array(ambiguous).T
    with (output/"node_regions.npz").open("xb") as stream:
        np.savez_compressed(stream, node_ids=ids, xyz_nm=xyz, inside=inside,
                            ambiguous=ambiguous, region_names=np.array(REGIONS), ray_directions=DIRECTIONS)
    result = {"schema": "flywire-apl-neuropil-binding-v1", "root": a["root"],
        "anatomy_sha256": a["anatomy_sha256"], "source_manifest_sha256": sha256(source/"manifest.json"),
        "node_regions_sha256": sha256(output/"node_regions.npz"), "source_code_sha256": sha256(Path(__file__)),
        "versions": {"trimesh": trimesh.__version__, "rtree": rtree.__version__},
        "regions": details, "outside_all": int((~inside.any(axis=1)).sum()),
        "multiple_region_membership": int((inside.sum(axis=1) > 1).sum()),
        "limits": ["Server-provided neuropil annotation, not optical ROIs or same-animal physiology",
            "Same CATMAID project coordinates, no fitted coordinate transformation",
            "Three directions use the same triangle-intersection library, not independent algorithms",
            "Points exactly on a mesh surface have undefined ray parity; disagreements retained",
            "No nodes or contacts are deleted; memberships can overlap or be absent",
            "Meshes only define external readouts, never neural inputs or behavior"]}
    write_new(output/"analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("acquire")
    a.add_argument("output", type=Path)
    b = sub.add_parser("bind")
    for name in ("source", "anatomy", "output"):
        b.add_argument(name, type=Path)
    args = parser.parse_args()
    result = acquire(args.output) if args.command == "acquire" else bind(args.source, args.anatomy, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
