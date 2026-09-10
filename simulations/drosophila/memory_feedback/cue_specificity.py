"""Test whether B-terminal memory changes the response to the held-out cue C."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import interface_controls as controls
from ..connectome import sha256


def run(parent,intact,blocked,output):
    parent,intact,blocked,output=map(Path,(parent,intact,blocked,output))
    if output.exists(): raise FileExistsError(output)
    original=controls.mask_for
    def b_mask(m,ids,roots,selected,cue,targets):
        assert cue=="C"
        return original(m,ids,roots,selected,"B",targets)
    result=dict(source_sha256=sha256(Path(__file__)),intervention_cue="B",probe_cue="C",probes={})
    for name,donor in (("sham",intact),("B_memory_replaced",blocked)):
        with patch.object(controls,"mask_for",b_mask):
            r=controls.probe(parent,parent,output/name,state=intact,donor_state=donor,
                             cue="C",well=True,student_only=True)
        r["intervention_cue"]="B";r["specificity_driver_sha256"]=sha256(Path(__file__))
        (output/name/"summary.json").write_text(json.dumps(r,indent=2)+"\n")
        result["probes"][name]=r
    with np.load(output/"sham/trace.npz") as a,np.load(output/"B_memory_replaced/trace.npz") as b:
        result["C_body_exact"]=bool(np.array_equal(a["body"],b["body"]))
        result["C_soma_exact"]=bool(np.array_equal(a["soma"],b["soma"]))
    result["limit"]="Same retained receiver state and C input; only B-to-MBON04 release coefficients are replaced. This tests one held-out controlled cue, not natural odor generalization."
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("parent","intact","blocked","output"): p.add_argument(name,type=Path)
    a=p.parse_args();run(a.parent,a.intact,a.blocked,a.output)
