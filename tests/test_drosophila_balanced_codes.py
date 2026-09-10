"""Cue reassignment must preserve the measured circuit and KC membership."""
from types import SimpleNamespace

import numpy as np

from simulations.drosophila.memory_feedback.balanced_codes import reassign


def test_reassignment_balances_direct_route_without_deleting_connections():
    alpha = [str(n) for n in range(1, 13)]
    gamma = [str(n) for n in range(13, 25)]
    targets = {"101": "MBON07", "102": "MBON04", "103": "SMP108", "104": "APL"}
    nodes = {r: {"annotation": {"hemibrain_type": "KCab-m" if r in alpha else "KCg-m"}} for r in alpha+gamma}
    nodes.update({r: {"annotation": {"hemibrain_type": t}} for r, t in targets.items()})
    codes = {c: alpha[i*4:(i+1)*4]+gamma[i*4:(i+1)*4] for i, c in enumerate("ABC")}
    pairs = [(r, "101" if r in alpha else "102", 4) for r in alpha+gamma]
    pairs += [(r, "103", 1) for r in gamma[4:7]]
    edges = np.array([[int(a), int(b), int(a), int(b), count, 1, count, i, i] for i, (a, b, count) in enumerate(pairs)])
    g = SimpleNamespace(selected=tuple(nodes), nodes=nodes, edges=edges, internal=edges,
                        provenance={"controlled_codes": codes})
    changed = reassign(g)
    assert changed.edges is edges and changed.nodes is nodes and changed.selected is g.selected
    new = changed.provenance["controlled_codes"]
    assert sorted(sum(new.values(), [])) == sorted(alpha+gamma)
    assert g.provenance["controlled_codes"] == codes
    for roots in new.values():
        assert len(set(roots)&set(alpha)) == len(set(roots)&set(gamma)) == 4
        assert len(set(roots)&set(gamma[4:7])) == 1
    assert new == reassign(g).provenance["controlled_codes"]
