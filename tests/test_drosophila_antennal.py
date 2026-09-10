import numpy as np
import pytest

from simulations.drosophila.antennal_lobe import upstream_selection, verify_extension, missing_incident_inventory
from simulations.drosophila.connectome import Catalog, extract_subgraph, RAW_COLUMNS
from simulations.drosophila.paula import ORNReleaseDepression, build_paula


def fixture(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    types = [("olfactory","ORN_DL5"),("ALPN","DL5_adPN"),("ALLN",""),("ALLN","lLN1"),
             ("olfactory","ORN_DL5"),("olfactory","ORN_other"),("ALPN","other_PN")]
    roots = tuple(str(i+1) for i in range(len(types)))
    c = Catalog(roots,{r:{"root_id":r,"cell_class":cls,"hemibrain_type":typ,"side":"right" if r=="5" else "left"}
                       for r,(cls,typ) in zip(roots,types)})
    def edge(a,b,n,s,row):
        return [a,b,a-1,b-1,n,s,n*s,row,row]
    rows = np.array([edge(1,2,40,1,0),edge(1,3,2,1,1),edge(3,1,1,-1,2),
                     edge(3,4,2,-1,3),edge(6,3,4,1,4),edge(5,7,5,1,5)],dtype=np.int64)
    path=tmp_path/"source.parquet"
    pq.write_table(pa.table({name:pa.array(rows[:,i]) for i,name in enumerate(RAW_COLUMNS)}),path,row_group_size=1)
    parent=extract_subgraph(path,c,("2",),{})
    return c,path,parent,rows


def test_upstream_includes_untyped_ln_other_side_orn_and_complete_new_ports(tmp_path):
    c,path,parent,rows=fixture(tmp_path)
    selected,orns,lns,witnesses=upstream_selection(path,c,parent)
    assert selected==("1","2","3","5") and orns=={"1","5"} and lns=={"3"}
    # LN4 is two hops away, not silently recursively included; its edge remains a boundary.
    expanded=extract_subgraph(path,c,selected,{})
    assert len(expanded.edges)==6 and witnesses[:,8].tolist()==[1,2]
    check=verify_extension(parent,expanded)
    assert check["prior_incident_pairs_exact"]==1
    records={r["root"]:r for r in missing_incident_inventory(parent,expanded,set(selected)-set(parent.selected))}
    assert records["3"]["incoming_pairs_missing_from_parent_cut"]==2
    assert records["5"]["outgoing_pairs"]==1
    bad=extract_subgraph(path,c,selected,{})
    bad.edges[0,4]+=1;bad.edges[0,6]+=1
    with pytest.raises(AssertionError):verify_extension(parent,bad)


def test_depression_preserves_every_binding_and_only_selects_orn_alpn_terminals(tmp_path):
    c,path,parent,rows=fixture(tmp_path)
    selected,*_=upstream_selection(path,c,parent)
    g=extract_subgraph(path,c,selected,{})
    native=build_paula(g)
    prep=build_paula(g,release_depression=ORNReleaseDepression(("1","5"),.22,893.))
    for name in ("edge_bindings","incoming_boundary_ports","outgoing_boundary_terminals"):
        np.testing.assert_array_equal(getattr(prep,name),getattr(native,name))
    assert prep.assumptions["release_depression"]["source_rows"]=={"1":[0],"5":[5]}
    assert prep.network.network.neurons[0].release_terminals.tolist()==[0]
    assert len(prep.network.network.neurons[0].presynaptic_points)==2  # LN projection remains.
    with pytest.raises(ValueError):
        build_paula(g,release_depression=ORNReleaseDepression(("3",),.22,893.))
