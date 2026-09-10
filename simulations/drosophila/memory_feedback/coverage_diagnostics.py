"""Causal student-memory probes before extending the coverage preparation.

Compare each acquired state with a B-terminal-only substitution. If those
changes have no physical effect, test a fixed fourfold log-depression
projection as an exposure diagnostic. No imposed state counts as learning.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import interface_controls, credit_projection
from ..connectome import sha256


def read(path): return json.loads(Path(path).read_text())


def run(parent, intact, blocked, output):
    parent,intact,blocked,output=map(Path,(parent,intact,blocked,output))
    if output.exists(): raise FileExistsError(output)
    a,c=read(intact/"summary.json"),read(blocked/"summary.json")
    assert not a["cut"] and c["cut"] and not a["displaced"] and not c["displaced"]
    assert a["nutrient_j"]==c["nutrient_j"]==0
    assert a["input_checkpoint_sha256"]==c["input_checkpoint_sha256"]==sha256(parent/"retention.paula")
    assert a["branch_rng_preserved"] and c["branch_rng_preserved"]
    assert [(p["name"],p["begin"],p["end"]) for p in a["phases"]]==[(p["name"],p["begin"],p["end"]) for p in c["phases"]]
    pairings=a.get("pairings",2);assert pairings==c.get("pairings",2)
    result=dict(source_sha256=sha256(Path(__file__)),parent=parent.name,intact=intact.name,blocked=blocked.name,
        pairings=pairings,probes={})
    for name,state,donor in (("sham",intact,intact),("removed",intact,blocked),("transferred",blocked,intact),("blocked_sham",blocked,blocked)):
        result["probes"][name]=interface_controls.probe(parent,parent,output/name,state=state,donor_state=donor,
                                                      cue="B",well=False,student_only=True)
    result["comparisons"]={}
    for name,reference in (("removed","sham"),("transferred","blocked_sham")):
        with np.load(output/name/"trace.npz") as x,np.load(output/reference/"trace.npz") as y:
            result["comparisons"][name]=dict(body_exact=bool(np.array_equal(x["body"],y["body"])),
                max_angle_difference=float(x["body"][:,0].max()-y["body"][:,0].max()))
    for name,path in (("sham",intact),("blocked_sham",blocked)):
        with np.load(output/name/"trace.npz") as x,np.load(path/"probe-B.npz") as y:
            exact=all(np.array_equal(x[k],y[k]) for k in ("soma","body"))
            assert exact;result["probes"][name]["original_replay_exact"]=exact
    if all(r["body_exact"] for r in result["comparisons"].values()):
        projection=credit_projection.run(parent,intact,blocked,output/"projection")
        projection["projection"].update(base_pairings=pairings,projected_pairings=4*pairings,
            wrapper_sha256=sha256(Path(__file__)),
            limit="Fixed fourfold log-depression extrapolation from this recorded exposure. It changes only B terminal coefficients at the same retained state; the coupled longer course may differ.")
        (output/"projection"/"summary.json").write_text(json.dumps(projection,indent=2)+"\n")
        result["probes"]["projection"]=projection
    result["limit"]="Dry B probes with native adaptation continuing. Terminal substitutions test expression at identical receiver states. A projection is imposed, not acquired behavior. Teacher-memory dependence requires a separate acquisition intervention."
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("parent","intact","blocked","output"): p.add_argument(name,type=Path)
    a=p.parse_args();run(a.parent,a.intact,a.blocked,a.output)
