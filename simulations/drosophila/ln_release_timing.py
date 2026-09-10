"""Compare early and late single-LN release lesions against an intact history.

This tests dependence on intervention time in one deterministic preparation.
It does not establish an attractor, physiological stability or odor coding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .connectome import Subgraph
from .electrical_analysis import prefix
from .orn_onset import FIRST_POSITIVE_LN
from .pn_current import dl5_cut
from .pn_current_steps import prepare
from .prisco import digest, dump_new


def audit_block(events, column, start):
    """Demand target-only, time-specific withholding of chemical events."""
    if events.ndim != 3 or events.shape[2] != 3 or not 0 <= start < len(events):
        raise ValueError("Invalid event recording or onset")
    if not np.isfinite(events).all() or np.any(events < 0) or np.any(events != np.floor(events)):
        raise ValueError("Invalid event counts")
    expected = events[:, :, 0].copy()
    expected[start:, column] = 0
    np.testing.assert_array_equal(events[:, :, 1], expected)
    return {"withheld_forward_events": int(events[start:, column, 0].sum()),
            "retained_return_events_after_onset": int(events[start:, column, 2].sum())}


def differences(reference, changed):
    if reference.shape != changed.shape:
        raise ValueError("Different neural recording shapes")
    def first(mask):
        indices = np.flatnonzero(mask)
        return int(indices[0]) if len(indices) else None
    return {"first_state_difference": first(np.any(reference != changed, axis=(1, 2))),
            "first_spike_difference": first(np.any(reference[:, :, 1] != changed[:, :, 1], axis=1)),
            "cells_with_changed_spikes": int(np.any(reference[:, :, 1] != changed[:, :, 1], axis=0).sum())}


def run(graph_path, intrinsic_path, tail_path, reference, early, late):
    ticks = 1600
    graph = Subgraph.load(graph_path)
    old_meta, original, structure = prefix(reference, ticks, current_source=False)
    early_meta, onset, early_structure = prefix(early, ticks, current_source=False)
    late_meta, delayed, late_structure = prefix(late, ticks)
    if old_meta["condition"] != "no_depression" or early_meta["blocked_root"] != FIRST_POSITIVE_LN:
        raise ValueError("Wrong reference or early lesion")
    if late_meta["condition"] != "no_depression" or "electrical" in delayed:
        raise ValueError("Unexpected concurrent intervention")
    block = late_meta["protocol"]["single_ln_release_block"]
    if block["root"] != FIRST_POSITIVE_LN:
        raise ValueError("Different lesion target")
    start = block["start_tick"]
    for key in structure:
        np.testing.assert_array_equal(structure[key], late_structure[key])
    for key in ("roots", "edge_bindings", "incoming_boundary_ports", "outgoing_boundary_terminals"):
        np.testing.assert_array_equal(structure[key], early_structure[key])
    parity = 0
    for key in original:
        np.testing.assert_array_equal(original[key][:start], delayed[key][:start])
        parity += original[key][:start].size
    roots = structure["roots"].tolist()
    event_audit = audit_block(delayed["ln_events"], structure["ln_roots"].tolist().index(FIRST_POSITIVE_LN), start)
    intrinsic = json.loads(intrinsic_path.read_text())
    tail = json.loads(tail_path.read_text())["fits"]["2"]["all_cells"]
    _, cut = dl5_cut(graph)
    _, pn = prepare(cut, intrinsic, tail)
    pn_index = roots.index(intrinsic["root"])
    for t in range(ticks):
        pn.input_buffer[:] = delayed["pn_inputs"][t]
        pn.tick({}, t)
        np.testing.assert_array_equal([pn.S, pn.O, pn.F_avg], delayed["soma"][t, pn_index])
        np.testing.assert_array_equal([pn.t_ref, pn.r, pn.b, pn.total_current], delayed["pn_intrinsic"][t])
        np.testing.assert_array_equal(pn.last_port_current, delayed["pn_current"][t])
        np.testing.assert_array_equal([p.u_i.info for p in pn.postsynaptic_points.values()], delayed["pn_post_weight"][t])
    conditions = {}
    for name, data in (("intact", original), ("block_from_0", onset), (f"block_from_{start}", delayed)):
        epochs = []
        for lo, hi in ((200, start), (start, 1200), (1200, 1600)):
            counts = {}
            for cls in ("olfactory", "ALLN", "ALPN", "Kenyon_Cell"):
                cols = [i for i, r in enumerate(roots) if graph.nodes[r]["annotation"]["cell_class"] == cls]
                spikes = data["soma"][lo:hi, cols, 1].sum(axis=0)
                counts[cls] = {"spikes": int(spikes.sum()), "cells_fired": int(np.count_nonzero(spikes))}
            apl = data["apl_max"] if name == "block_from_0" else data["apl"][:, 2]
            epochs.append({"start": lo, "stop": hi, "populations": counts,
                "DL5_spikes": int(data["soma"][lo:hi, pn_index, 1].sum()),
                "target_LN_spikes": int(data["soma"][lo:hi, roots.index(FIRST_POSITIVE_LN), 1].sum()),
                "max_APL_release": float(apl[lo:hi].max())})
        conditions[name] = {"epochs": epochs, "vs_intact": differences(original["soma"], data["soma"])}
    return {"schema": 1, "claim": "State-dependent lesion effect in one finite closed-loop model course",
        "target": FIRST_POSITIVE_LN, "late_onset": start, "conditions": conditions,
        "event_audit": event_audit, "exact_pre_intervention_values": parity,
        "exact_PN_receiving_replay_ticks": ticks,
        "source_hashes": {str(p.resolve()): digest(p) for p in (
            Path(__file__), graph_path/"manifest.json", intrinsic_path, tail_path,
            reference/"analysis.json", early/"analysis.json", late/"analysis.json")},
        "limits": ["A single deterministic first train and 400 recovery ticks; not long-run or multi-odor acceptance.",
            "Late versus early intervention tests dependence on history, not bistability or learned memory.",
            "Return events are counted, not independently replayed through outgoing terminals.",
            "Intact and early-lesion sources are historical; retained artifact hashes and direct arrays are checked.",
            "The current-to-synapse conversion, LN intrinsic dynamics and most receptor effects remain assumptions."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "intrinsic", "tail", "reference", "early", "late", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    result = run(a.graph, a.intrinsic, a.tail, a.reference, a.early, a.late)
    dump_new(a.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
