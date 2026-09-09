"""Bind verified FlyWire contact sites to one opt-in PAULA cable neuron."""
from pathlib import Path
import json

import numpy as np

from .connectome import sha256
from .spatial import CONTACT_COLUMNS, bind_pair_rows


def configure_local_apl(cell, root, graph, directory: Path, drive_port, rm_over_ra_um):
    from neuron.extensions.experimental.passive_cable import PassiveCable

    report_path = directory / "analysis.json"
    report = json.loads(report_path.read_text())
    if (report["schema"] != "flywire-spatial-audit-v1" or report["root"] != root
            or report["contact_columns"] != CONTACT_COLUMNS):
        raise ValueError("Spatial analysis does not identify this APL")
    path = directory / "anatomy.npz"
    if sha256(path) != report["anatomy_sha256"]:
        raise ValueError("Changed spatial anatomy")
    with np.load(path, allow_pickle=False) as data:
        ids, parents, xyz, radius, contacts, rows = (data[key] for key in
            ("node_ids", "parents", "xyz_nm", "radius_nm", "contacts", "pair_source_rows"))
    if not np.array_equal(rows, bind_pair_rows(contacts, int(root), graph)):
        raise ValueError("Contact-to-original-row bindings disagree")
    if ids.dtype.kind not in "iu" or ids.shape != parents.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid spatial node identities")
    ordered = graph.edges[np.argsort(graph.edges[:, 8])]
    incoming = {int(row[8]): i for i, row in enumerate(ordered[ordered[:, 1] == int(root)])}
    outgoing = {int(row[8]): i for i, row in enumerate(ordered[ordered[:, 0] == int(root)])}
    if (drive_port != len(incoming) or len(cell.postsynaptic_points) != len(incoming) + 1
            or set(cell.presynaptic_points) != set(range(len(outgoing)))):
        raise ValueError("Spatial binding and neural port construction disagree")
    pre_mask = (contacts[:, 2] == int(root)) & (rows >= 0)
    post_mask = (contacts[:, 1] == int(root)) & (rows >= 0)
    cable = PassiveCable(parents, xyz / 1000, radius / 1000, rm_over_ra_um)
    cell.configure_cable(cable,
        np.array([incoming[int(row)] for row in rows[pre_mask]], dtype=np.int64), contacts[pre_mask, 6],
        np.array([outgoing[int(row)] for row in rows[post_mask]], dtype=np.int64), contacts[post_mask, 5],
        contacts[contacts[:, 1] == int(root), 5], uniform_input_port=drive_port)
    cell.cable_node_ids = ids.copy()
    cell.metadata["spatial_assumptions"] = {
        "root": root, "analysis_path": str(report_path.resolve()), "analysis_sha256": sha256(report_path),
        "anatomy_sha256": report["anatomy_sha256"], "nodes": len(ids),
        "incoming_contacts": int(pre_mask.sum()), "outgoing_contacts": int(post_mask.sum()),
        "unpaired_contacts_not_driven_or_connected": int((rows < 0).sum()),
        "zero_radius_nodes_in_source": int((radius == 0).sum()),
        "radius_assumption": "Each tree edge is a cylinder at its endpoint-mean radius. Source radii stay unchanged; this is not a converged tapered-cable reconstruction.",
        "rm_over_ra_um": rm_over_ra_um,
        "membrane_area_um2": float(cable.area_um2.sum()),
        "input": "Split each aggregate native arriving current equally across that pair's measured contacts; total current conserved",
        "output": "Mean local rectified release across that pair's output contacts, then native terminal coefficient; target pair weight unchanged",
        "S": "membrane-area-weighted branch voltage; not a measured soma voltage",
        "O": "mean local release across all recorded APL output sites, including unlinked sites; not a spike flag",
        "experimental_drive": "The separate APL drive port supplies uniform current per membrane area",
        "limits": "Passive branch exchange; global inherited neuromodulation/timing plasticity; no calcium model, active channels, separate contact weights or finite-volume convergence claim",
    }
    return cell.metadata["spatial_assumptions"]
