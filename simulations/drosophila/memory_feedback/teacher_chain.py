"""Test the acquired-A to stored-B to action chain at identical receiver states."""
import argparse
import json
from pathlib import Path

from .interface_controls import probe
from ..connectome import sha256


def run(base,output):
    base,output=Path(base),Path(output)
    if output.exists(): raise FileExistsError(output)
    parent=base/"memory-coverage-paired-20260910"
    intact=base/"memory-coverage-eight-intact-20260910"
    teacher_parent=base/"memory-coverage-teacher-removed-20260910/receiver"
    teacher=base/"memory-coverage-eight-teacher-removed-20260910"
    donors={"teacher":(teacher_parent,teacher),
            "unpaired":(base/"memory-coverage-unpaired-20260910",base/"memory-coverage-eight-unpaired-20260910"),
            "displaced":(parent,base/"memory-coverage-eight-displaced-20260910")}
    jobs=[(name+"-into-intact-dry",parent,dp,intact,ds,False) for name,(dp,ds) in donors.items()]
    jobs.extend((("intact-into-teacher-dry",teacher_parent,parent,teacher,intact,False),
                 ("teacher-into-intact-food",parent,teacher_parent,intact,teacher,True),
                 ("teacher-sham-food",teacher_parent,teacher_parent,teacher,teacher,True),
                 ("intact-into-teacher-food",teacher_parent,parent,teacher,intact,True)))
    result=dict(source_sha256=sha256(Path(__file__)),probes={},
        limit="B-to-MBON04 release-only interventions at each receiver's own retained state, with adaptation continuing. Cross-state body differences are not attributed to B memory.")
    for name,receiver,donor,state,donor_state,well in jobs:
        r=probe(receiver,donor,output/name,state=state,donor_state=donor_state,cue="B",well=well,student_only=True)
        r["receiver_parent_path"]=str(receiver.relative_to(base));r["donor_parent_path"]=str(donor.relative_to(base))
        result["probes"][name]=r
        print(json.dumps(dict(name=name,spikes=r["spikes"]["SMP108"],well_j=r["well_j"],first_contact_tick=r["first_contact_tick"])),flush=True)
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("base",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();run(a.base,a.output)
