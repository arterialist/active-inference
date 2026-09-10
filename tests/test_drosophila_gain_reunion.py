import numpy as np
import pytest

from simulations.drosophila.connectome import Subgraph
from simulations.drosophila.gain_reunion import reunion_cut,target_bindings,pathway_bindings,filter_forward,initialize_feedback_gain,window_lateral_command,initialize_regulator_afferent_gain
from neuron.neuron import RetrogradeSignalEvent
from simulations.drosophila.ln_gain import PN, LN
from simulations.drosophila.paula import build_paula
from simulations.drosophila.gain_reunion_analysis import first_difference,port_groups,audit_pathway_events,audit_feedback_gain,audit_command_shift,regulator_pulses


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


def test_feedback_initialization_changes_only_selected_receiving_weights():
    from types import SimpleNamespace as NS
    nodes={PN:{"annotation":{"cell_class":"ALPN","hemibrain_type":"PN"}},
           LN:{"annotation":{"cell_class":"ALLN","hemibrain_type":"LN"}},
           "101":{"annotation":{"cell_class":"olfactory","hemibrain_type":"ORN_DL5"}}}
    edges=np.array([[int(PN),int(LN),1,2,10,1,10,0,0],[101,int(LN),3,2,20,1,20,0,1]])
    points={0:NS(u_i=NS(info=2.)),1:NS(u_i=NS(info=3.))}
    prep=NS(edge_bindings=np.array([[0,1,0,2,0],[1,3,0,2,1]]),
            network=NS(network=NS(neurons={2:NS(postsynaptic_points=points)})))
    graph=Subgraph((PN,LN,"101"),nodes,edges,{})
    _,before,after=initialize_feedback_gain(prep,graph,1.)
    np.testing.assert_array_equal(before,after)
    _,before,after=initialize_feedback_gain(prep,graph,.25)
    np.testing.assert_array_equal(after,before*.25)
    assert points[0].u_i.info==.5 and points[1].u_i.info==3.
    for bad in (0,-1,2,float("nan")):
        with pytest.raises(ValueError):initialize_feedback_gain(prep,graph,bad)


def test_feedback_audit_checks_actual_weights_against_measured_pairs(tmp_path):
    from simulations.drosophila.prisco import digest
    nodes={PN:{"annotation":{"cell_class":"ALPN","hemibrain_type":"PN"}},
           LN:{"annotation":{"cell_class":"ALLN","hemibrain_type":"LN"}}}
    graph=Subgraph((PN,LN),nodes,np.array([[int(PN),int(LN),1,2,10,1,10,0,0]]),{})
    bindings=np.array([[0,1,0,2,0]])
    np.savez(tmp_path/"feedback-initial.npz",bindings=bindings,reference_weights=[.75],initial_weights=[.375])
    np.savez(tmp_path/"feedback-final.npz",weights=[.37500001])
    spec={"scale":.5,"pairs":1,"changed_by_learning":1,
          "initial_sha256":digest(tmp_path/"feedback-initial.npz"),
          "final_sha256":digest(tmp_path/"feedback-final.npz")}
    meta={"feedback_initialization":spec,"assumptions":{"parameters":{"weight_per_count":.075}}}
    result=audit_feedback_gain(tmp_path,graph,meta,{"edge_bindings":bindings})
    assert result["nonzero_initial_weights"]==result["changed_by_learning"]==1
    spec["scale"]=.25
    with pytest.raises(AssertionError):audit_feedback_gain(tmp_path,graph,meta,{"edge_bindings":bindings})
    spec["scale"]=.5;spec["changed_by_learning"]=0
    with pytest.raises(ValueError):audit_feedback_gain(tmp_path,graph,meta,{"edge_bindings":bindings})


def test_lateral_windows_are_charge_matched_and_do_not_change_sensory_commands():
    from simulations.drosophila.ln_gain import commands
    original,_=commands(42,100,80,11,trials=1)
    assert window_lateral_command(original,None) is original
    early=window_lateral_command(original,(200,300))
    late=window_lateral_command(original,(600,700))
    assert early[:,-1].sum()==late[:,-1].sum()==320.
    assert np.count_nonzero(early[:,-1])==8
    np.testing.assert_array_equal(early[200:300,-1],late[600:700,-1])
    np.testing.assert_array_equal(early[:,:-1],original[:,:-1])
    np.testing.assert_array_equal(late[:,:-1],original[:,:-1])
    assert not np.any(early[300:,-1]) and not np.any(late[:600,-1])
    for bad in ((300,200),(0,3000),(-1,300),(0.,300)):
        with pytest.raises(ValueError):window_lateral_command(original,bad)
    assert audit_command_shift(early,late)=={"pulses":8,"shift_ticks":400,"injected_current_sum":320.}
    bad=late.copy();bad[0,0]=1
    with pytest.raises(AssertionError):audit_command_shift(early,bad)
    bad=late.copy();bad[606,-1]=39
    with pytest.raises(AssertionError):audit_command_shift(early,bad)
    bad=late.copy();bad[606,-1]=0;bad[607,-1]=40
    with pytest.raises(ValueError):audit_command_shift(early,bad)


def test_pulse_annotation_uses_preceding_spike_not_current_spike():
    data={"soma":np.zeros((8,1,3)),"command":np.zeros((8,1)),"source_current":np.zeros((8,1,2))}
    data["soma"][[1,4],0,1]=1
    data["command"][[2,4,7],0]=40
    structure={"roots":np.array([LN]),"source_roots":np.array([LN])}
    rows=regulator_pulses(data,structure,3)
    assert [r["ticks_since_previous_spike"] for r in rows]==[1,3,3]
    assert [r["cooldown_prevents_spike"] for r in rows]==[True,False,False]
    assert [r["fired"] for r in rows]==[False,True,False]


def test_regulator_afferent_scale_changes_only_measured_internal_sensory_receivers():
    from types import SimpleNamespace as NS
    nodes={PN:{"annotation":{"hemibrain_type":"DL5_adPN"}},LN:{"annotation":{"hemibrain_type":"lLN2F_b"}},
           "101":{"annotation":{"hemibrain_type":"ORN_DL5"}},"102":{"annotation":{"hemibrain_type":"ORN_DL5"}}}
    edges=np.array([[101,int(LN),1,2,10,1,10,0,0],[101,int(PN),1,3,20,1,20,0,1],
                    [int(PN),int(LN),3,2,5,1,5,0,2],[102,int(LN),4,2,10,1,10,0,3]])
    ln_ports={0:NS(u_i=NS(info=.75)),1:NS(u_i=NS(info=.375)),2:NS(u_i=NS(info=.75))}
    pn_ports={0:NS(u_i=NS(info=1.5))}
    prep=NS(edge_bindings=np.array([[0,1,0,2,0],[1,1,1,3,0],[2,3,0,2,1]]),
            network=NS(network=NS(neurons={2:NS(postsynaptic_points=ln_ports),3:NS(postsynaptic_points=pn_ports)})))
    graph=Subgraph(("101",LN,PN),nodes,edges,{})
    _,before,after=initialize_regulator_afferent_gain(prep,graph,1.)
    np.testing.assert_array_equal(before,after)
    bindings,before,after=initialize_regulator_afferent_gain(prep,graph,4.)
    assert bindings[:,0].tolist()==[0]
    np.testing.assert_array_equal(after,before*4.)
    assert [p.u_i.info for p in ln_ports.values()]==[3.,.375,.75]
    assert pn_ports[0].u_i.info==1.5
    for bad in (0,-1,float('inf'),float('nan')):
        with pytest.raises(ValueError):initialize_regulator_afferent_gain(prep,graph,bad)


def test_afferent_audit_rejects_a_false_scale_or_pair(tmp_path):
    from simulations.drosophila.prisco import digest
    nodes={"101":{"annotation":{"hemibrain_type":"ORN_DL5"}},LN:{"annotation":{"hemibrain_type":"lLN2F_b"}}}
    graph=Subgraph(("101",LN),nodes,np.array([[101,int(LN),1,2,10,1,10,0,0]]),{})
    bindings=np.array([[0,1,0,2,0]])
    np.savez(tmp_path/'regulator-afferent-initial.npz',bindings=bindings,reference_weights=[.75],initial_weights=[1.5])
    np.savez(tmp_path/'regulator-afferent-final.npz',weights=[1.50001])
    spec={"scale":2.,"pairs":1,"changed_by_learning":1,
          "initial_sha256":digest(tmp_path/'regulator-afferent-initial.npz'),
          "final_sha256":digest(tmp_path/'regulator-afferent-final.npz')}
    meta={"regulator_afferent_initialization":spec,"assumptions":{"parameters":{"weight_per_count":.075}}}
    result=audit_feedback_gain(tmp_path,graph,meta,{"edge_bindings":bindings},afferent=True)
    assert result['scale']==2. and result['changed_by_learning']==1
    spec['scale']=4.
    with pytest.raises(AssertionError):audit_feedback_gain(tmp_path,graph,meta,{"edge_bindings":bindings},afferent=True)
    spec['scale']=2.;spec['pairs']=2
    with pytest.raises(ValueError):audit_feedback_gain(tmp_path,graph,meta,{"edge_bindings":bindings},afferent=True)
