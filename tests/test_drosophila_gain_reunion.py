import numpy as np
import pytest

from simulations.drosophila.connectome import Subgraph
from simulations.drosophila.gain_reunion import reunion_cut,target_bindings,pathway_bindings,filter_forward
from neuron.neuron import RetrogradeSignalEvent
from simulations.drosophila.ln_gain import PN, LN
from simulations.drosophila.paula import build_paula
from simulations.drosophila.gain_reunion_analysis import first_difference,port_groups,audit_pathway_events


def test_scope_preserves_incident_ports_and_makes_other_ln_exclusion_explicit():
    roots=("101",PN,LN,"201","202","203")
    kinds=(("olfactory","ORN_DL5"),("ALPN","DL5_adPN"),("ALLN","lLN2F_b"),
           ("Kenyon_Cell","KC"),("APL","APL"),("ALLN","other_LN"))
    nodes={r:{"global_index":i+1,"annotation":{"root_id":r,"cell_class":kind[0],"hemibrain_type":kind[1]}}
           for i,(r,kind) in enumerate(zip(roots,kinds))}
    edges=np.array([[int(PN),201,2,4,10,1,10,0,0],
                    [201,202,4,5,10,1,10,0,1],
                    [202,201,5,4,10,-1,-10,0,2],
                    [203,int(LN),6,3,10,-1,-10,0,3],
                    [int(LN),203,3,6,10,-1,-10,0,4]],dtype=np.int64)
    graph=Subgraph(roots,nodes,edges,{})
    assert reunion_cut(graph,"full") is graph
    cut=reunion_cut(graph,"consumers")
    assert set(cut.selected)==set(roots)-{"203"}
    assert "203" in cut.nodes
    antennal=reunion_cut(graph,"antennal")
    assert set(antennal.selected)=={"101",PN,LN,"203"}
    assert "201" in antennal.nodes
    np.testing.assert_array_equal(cut.edges,graph.edges)
    whole,part=build_paula(graph),build_paula(cut)
    for r in cut.selected:
        a,b=[p.network.network.neurons[p.root_to_id[r]] for p in (whole,part)]
        assert a.params.num_inputs==b.params.num_inputs
        assert a.distances==b.distances and a.upper_t_ref_bound==b.upper_t_ref_bound
    with pytest.raises(ValueError):reunion_cut(graph,"quiet_subset")
    terminals,ports=target_bindings(whole,[PN,"203"],nodes["201"]["global_index"])
    assert terminals[0]>=0 and ports[0]>=0
    assert terminals[1]==ports[1]==-1


def test_first_difference_does_not_reduce_away_timing_changes():
    a=np.zeros((10,3));b=a.copy();b[2,1]=1;b[7,1]=-1
    assert a[:,1].sum()==b[:,1].sum()
    assert first_difference(a,b)==2
    assert first_difference(a[:,1],b[:,1])==2
    assert first_difference(a,a) is None
    with pytest.raises(ValueError):first_difference(a,a[:5])


def test_current_groups_use_source_identity_and_recorded_model_polarity():
    roots=("101",PN,LN,"201")
    kinds=(("olfactory","ORN_DL5"),("ALPN","DL5_adPN"),("ALLN","lLN2F_b"),("ALLN","APL"))
    nodes={r:{"global_index":i+1,"annotation":{"root_id":r,"cell_class":kind[0],"hemibrain_type":kind[1]}}
           for i,(r,kind) in enumerate(zip(roots,kinds))}
    edges=np.array([[101,int(PN),1,2,10,1,10,0,1],
                    [int(LN),int(PN),3,2,10,-1,-10,0,2],
                    [201,int(PN),4,2,10,-1,-10,0,3]],dtype=np.int64)
    masks=port_groups(Subgraph(roots,nodes,edges,{}))
    np.testing.assert_array_equal(masks["ORN_DL5"],[1,0,0,0])
    np.testing.assert_array_equal(masks["ALLN_negative"],[0,1,0,0])
    np.testing.assert_array_equal(masks["APL"],[0,0,1,0])
    np.testing.assert_array_equal(masks["experimental"],[0,0,0,1])


def test_recurrent_block_selects_only_declared_internal_positive_model_pairs():
    from types import SimpleNamespace
    roots=("101",PN,LN,"201","202","203")
    kinds=(("ALLN","positive_LN"),("ALPN","DL5_adPN"),("ALLN","negative_LN"),
           ("Kenyon_Cell","KC"),("ALLN","APL"),("ALLN","boundary_LN"))
    nodes={r:{"annotation":{"cell_class":c,"hemibrain_type":t}} for r,(c,t) in zip(roots,kinds)}
    edges=np.array([[101,int(LN),1,3,10,1,10,0,0],
        [101,int(PN),1,2,10,1,10,0,1], [int(LN),101,3,1,10,-1,-10,0,2],
        [101,202,1,5,10,1,10,0,3], [202,int(LN),5,3,10,1,10,0,4],
        [101,203,1,6,10,1,10,0,5],
        [int(PN),int(LN),2,3,10,1,10,0,6],
        [int(PN),101,2,1,10,-1,-10,0,7]],dtype=np.int64)
    bindings=np.array([[i,1,i,3,0] for i in (0,1,2,3,4,6,7)],dtype=np.int64)
    graph=Subgraph(roots[:-1],nodes,edges,{})
    prep=SimpleNamespace(edge_bindings=bindings)
    np.testing.assert_array_equal(pathway_bindings(prep,graph,"positive_LN_to_LN"),bindings[[0]])
    np.testing.assert_array_equal(pathway_bindings(prep,graph,"positive_LN_to_target_PN"),bindings[[1]])
    np.testing.assert_array_equal(pathway_bindings(prep,graph,"positive_PN_to_LN"),bindings[[5]])
    np.testing.assert_array_equal(pathway_bindings(prep,graph,"positive_LN_or_PN_to_LN"),bindings[[0,5]])
    assert pathway_bindings(prep,graph,"none").shape==(0,5)
    with pytest.raises(ValueError):pathway_bindings(prep,graph,"all_excitation")
    returning=RetrogradeSignalEvent.__new__(RetrogradeSignalEvent)
    events=[(1,2,3),(1,7,3),returning]
    admitted=filter_forward(events,{2})
    assert admitted==[events[1],returning]
    assert filter_forward(events,set())==events
    with pytest.raises(TypeError):filter_forward([object()],{2})


def test_pathway_audit_rejects_inert_early_and_leaky_interventions():
    events=np.array([[2,2,1],[4,4,2],[3,0,4],[5,0,1]])
    assert audit_pathway_events(events,2)=={"withheld_forward_events":8,"return_events_preserved":5}
    for row,value in ((0,0),(2,1)):
        bad=events.copy();bad[row,1]=value
        with pytest.raises(AssertionError):audit_pathway_events(bad,2)
    with pytest.raises(ValueError):audit_pathway_events(np.zeros((4,3)),2)
    with pytest.raises(ValueError):audit_pathway_events(events,-1)
