import numpy as np
import pytest

from simulations.drosophila.connectome import Subgraph
from simulations.drosophila.ln_gain import PN, LN, commands, select, simulate
from simulations.drosophila import paula
from simulations.drosophila.ln_gain_analysis import audit_gate, lateral_source_silent


def fixture():
    roots = ("101", PN, LN)
    nodes = {r: {"global_index": i+1, "annotation": {"root_id": r, "cell_class": cls, "hemibrain_type": typ}}
             for i,(r,cls,typ) in enumerate(zip(roots, ("olfactory","ALPN","ALLN"), ("ORN_DL5","DL5_adPN","lLN2F_b")))}
    edges = np.array([[101,int(PN),1,2,40,1,40,0,101],
                      [int(LN),101,3,1,10,-1,-10,0,102],
                      [int(LN),int(PN),3,2,15,-1,-15,0,103]], dtype=np.int64)
    g = Subgraph(roots,nodes,edges,{})
    intrinsic = {"root": PN, "proposals": {"pooled_control": {"rheobase_pa":25.,"lambda_ms":60.}},
                 "joint_unitary":{"per_count_weight":.037},"source_hashes":{}}
    return g,intrinsic,{"decay_ms":[10.,48.],"peak_fractions":[.82,.18]}


def test_commands_repeat_with_explicit_quiet_recovery_and_independent_channels():
    a, epochs = commands(42,10,80,11)
    b,_ = commands(42,10,0,11)
    np.testing.assert_array_equal(a[:,:42],b[:,:42])
    np.testing.assert_array_equal(a[200:1200],a[2200:3200])
    assert not a[:200].any() and not a[1200:2200].any() and not a[3200:].any()
    np.testing.assert_array_equal((a[200:1200,:42]>0).sum(0),10)
    assert (a[200:1200,-1]>0).sum() == 80
    assert epochs[1]["recovery_stop"] == 4200
    with pytest.raises(ValueError): commands(1,-1,20,0)


def test_no_electrode_is_not_a_neural_null():
    data={"roots":np.array([PN,LN]),"soma":np.zeros((4,2,6)),"command":np.zeros((4,2))}
    assert lateral_source_silent(data)
    data["soma"][2,1,1]=1
    assert not lateral_source_silent(data)


def test_zero_gain_exact_native_parity_and_factory_restoration():
    g,i,k = fixture(); original = paula.Neuron
    args = (select(g),i,k,10,80,11,"intact")
    a,ma = simulate(*args,duration=300,recovery=300)
    b,mb = simulate(*args,duration=300,recovery=300,inhibition_gain=0.)
    for name in a: np.testing.assert_array_equal(a[name],b[name])
    assert paula.Neuron is original and mb["changed_weights"] > 0
    np.testing.assert_array_equal(b["inhibition"][:,:,1],1)


def test_terminal_gate_requires_its_measured_pathway_and_preserves_spikes():
    g,i,k = fixture(); args=(select(g),i,k,10,80,11)
    ordinary,_ = simulate(*args,"intact",duration=300,recovery=300)
    gated,meta = simulate(*args,"intact",duration=300,recovery=300,inhibition_gain=1.)
    blocked,_ = simulate(*args,"LN_to_ORN_block",duration=300,recovery=300,inhibition_gain=1.)
    np.testing.assert_array_equal(ordinary["soma"][:,0,1],gated["soma"][:,0,1])
    assert gated["pn_current"][:,gated["orn_ports"]].sum() < ordinary["pn_current"][:,ordinary["orn_ports"]].sum()
    assert gated["inhibition"][:,:,1].min() < 1
    np.testing.assert_array_equal(blocked["inhibition"][:,:,0],0)
    np.testing.assert_array_equal(blocked["inhibition"][:,:,1],1)
    assert meta["changed_weights"] > 0
    audit_gate(gated,1.,100.,[1])
    bad={**gated,"inhibition":gated["inhibition"].copy()}
    bad["inhibition"][250,0,4] += .1
    with pytest.raises(ValueError,match="aggregation"):audit_gate(bad,1.,100.,[1])
    bad={**gated,"inhibition":gated["inhibition"].copy()}
    bad["inhibition"][250,0,0] += .1
    with pytest.raises(AssertionError):audit_gate(bad,1.,100.,[1])
