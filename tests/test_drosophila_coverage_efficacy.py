"""Uniform coverage efficacy must preserve all other receiving pairs."""
from types import SimpleNamespace as NS
from simulations.drosophila.memory_feedback.coverage_efficacy import scale


def test_coverage_scaling_is_limited_to_measured_source_and_targets():
    points={i:NS(u_i=NS(info=.25)) for i in range(3)}
    cell=NS(postsynaptic_points=points,params=NS(w_max=100))
    prep=NS(root_to_id={"s":1,"d":2,"other":3},network=NS(network=NS(neurons={2:cell,3:cell})),
            edge_bindings=[(10,1,0,2,0),(11,3,0,2,1),(12,1,0,3,2)])
    changes=scale(prep,{"SMP108":["s"],"PAM07":["d"],"PAM08":[]})
    assert [x["source_row"] for x in changes]==[10]
    assert [points[i].u_i.info for i in range(3)]==[1.,.25,.25]
