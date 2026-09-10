"""Read full transfer courses, replay PN receiving histories and expose limits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .connectome import Subgraph
from .ln_gain import PN, LN, simulate, summarize
from .ln_input_replay import cut_cells
from .pn_current_steps import prepare
from .prisco import digest, dump_new
from neuron.neuron import MAX_SYNAPTIC_WEIGHT, MIN_SYNAPTIC_WEIGHT


def read_record(directory, record):
    path = directory/record["file"]
    if digest(path) != record["sha256"]: raise ValueError("Changed recording")
    with np.load(path) as f: data = {k:f[k] for k in f.files}
    for a in data.values():
        if a.dtype.kind != "U" and not np.isfinite(a).all(): raise ValueError("Nonfinite record")
    if summarize(data,record) != record["responses"]:
        raise ValueError("Summary disagrees with full tick record")
    return data


def audit_gate(data, gain, decay, terminal_counts):
    gate = data["inhibition"]
    before = np.concatenate([np.zeros_like(gate[:1,:,0]),gate[:-1,:,0]],axis=0)
    np.testing.assert_allclose(gate[:,:,0],before*np.exp(-1/decay)+gate[:,:,2],rtol=2e-15,atol=1e-15)
    np.testing.assert_array_equal(gate[:,:,1],1/(1+gain*gate[:,:,0]))
    # Historical observations sum native numpy.float32 coefficients, whereas
    # nonunit gated coefficients become float64. Distributing a multiplication
    # across those two sums is not bit-exact. Bound accumulation error by the
    # number of declared terms and the native terminal coefficient bounds,
    # including possible signed cancellation; do not fit an epsilon to data.
    n=np.asarray(terminal_counts,dtype=float)
    if n.shape != (gate.shape[1],) or np.any(n < 1): raise ValueError("Invalid terminal counts")
    epsilon=np.finfo(np.float32).eps
    gamma=n*epsilon/(1-n*epsilon)
    bound=2*gamma*n*max(abs(MIN_SYNAPTIC_WEIGHT),abs(MAX_SYNAPTIC_WEIGHT))
    residual=np.abs(gate[:,:,4]-gate[:,:,3]*gate[:,:,1])
    if np.any(residual > bound): raise ValueError("Release aggregation exceeds floating-point bound")
    return {"max_release_aggregation_residual":float(residual.max()),"max_conservative_bound":float(bound.max())}


def lateral_source_silent(data):
    """An absent electrode command does not imply an inactive neural source."""
    return not np.any(data["soma"][:,data["roots"].tolist().index(LN),1])


def run(baseline, directories, intrinsic_path, tail_path):
    intrinsic = json.loads(intrinsic_path.read_text())
    tail = json.loads(tail_path.read_text())["fits"]["2"]["all_cells"]
    base_meta = json.loads((baseline/"analysis.json").read_text())
    key = lambda r:(r["seed"],r["direct_command_hz"],r["lateral_command_hz"])
    base_records = {key(r):r for r in base_meta["records"] if r["lesion"] == "intact"}
    results = []; pn_replays = parity = gate_ticks = 0
    for directory in (baseline,*directories):
        meta = json.loads((directory/"analysis.json").read_text())
        graph = Subgraph.load(directory/"graph")
        cut = cut_cells(graph,(PN,))
        rows = []
        for record in meta["records"]:
            data = read_record(directory,record)
            pn_col = data["roots"].tolist().index(PN)
            gain = record.get("inhibition_gain")
            if gain is not None:
                audit_gate(data,gain,record["inhibition_decay_ticks"],
                    [len(b["terminals"]) for b in record["assumptions"]["presynaptic_inhibition"]["bindings"]])
                gate_ticks += data["inhibition"].shape[0]*data["inhibition"].shape[1]
            # Every condition's PN is replayed from its actual inputs. This does
            # not reconstruct unrecorded release-terminal return histories.
            _,pn = prepare(cut,intrinsic,tail)
            for t in range(len(data["soma"])):
                pn.input_buffer[:] = data["pn_inputs"][t]
                pn.tick({},t)
                np.testing.assert_array_equal([pn.S,pn.O,pn.F_avg,pn.t_ref,pn.r,pn.b],data["soma"][t,pn_col])
                np.testing.assert_array_equal(pn.last_port_current,data["pn_current"][t])
                np.testing.assert_array_equal([p.u_i.info for p in pn.postsynaptic_points.values()],data["pn_weights"][t])
            pn_replays += len(data["soma"])
            comparison = None
            base = base_records.get(key(record))
            if directory != baseline and base is not None:
                old = read_record(baseline,base)
                n = len(data["roots"])-2
                comparison = {"ORN_spike_tick_differences":int(np.count_nonzero(old["soma"][:,:n,1] != data["soma"][:,:n,1])),
                    "LN_spike_tick_differences":int(np.count_nonzero(old["soma"][:,-1,1] != data["soma"][:,-1,1])),
                    "PN_spike_tick_differences":int(np.count_nonzero(old["soma"][:,-2,1] != data["soma"][:,-2,1]))}
                comparison["native_LN_silent"] = lateral_source_silent(old)
                comparison["variant_LN_silent"] = lateral_source_silent(data)
                if (gain is not None and record["lesion"] == "intact"
                        and lateral_source_silent(old) and lateral_source_silent(data)):
                    for k in old:
                        np.testing.assert_array_equal(old[k],data[k]); parity += old[k].size
                del old
            rows.append({"seed":record["seed"],"direct":record["direct_command_hz"],"lateral":record["lateral_command_hz"],
                "lesion":record["lesion"],"gain":gain,"decay":record.get("inhibition_decay_ticks"),
                "responses":record["responses"],"vs_native_intact":comparison})
            del data
        results.append({"directory":str(directory.resolve()),"manifest_sha256":digest(directory/"analysis.json"),"conditions":rows})
        print(f"Checked {directory.name}: {len(rows)} courses",flush=True)
    # Zero coupling is a separate mechanism-null intervention, not inferred
    # merely from a case with no lateral input.
    graph = Subgraph.load(baseline/"graph"); zero_parity = 0
    for direct in (0.,10.,50.):
        record = base_records[(11,direct,80.)]
        old = read_record(baseline,record)
        new,_ = simulate(graph,intrinsic,tail,direct,80.,11,"intact",inhibition_gain=0.)
        for k in old:
            np.testing.assert_array_equal(old[k],new[k]); zero_parity += old[k].size
    return {"schema":1,"conditions":results,"PN_receiving_replay_ticks":pn_replays,
        "gate_equation_cell_ticks":gate_ticks,"silent_LN_exact_native_values":parity,
        "zero_coupling_exact_native_values":zero_parity,
        "source_hashes":{str(p.resolve()):digest(p) for p in (Path(__file__),Path(__file__).with_name("ln_gain.py"),intrinsic_path,tail_path)},
        "claim":"Conditional neural gain-control mechanism test; neither public-input normalization nor reunited functional acceptance",
        "limits":["Input phase seeds and within-network repeated trials are not biological replicates.",
            "Exact recorded receiving replay checks PN execution, not all-cell terminal learning histories.",
            "Gate equations are imposed cellular assumptions; their numerical agreement is not discovery of a biological mechanism.",
            "Native and zero-gain source versions differ; direct full-array parity is tested without rewriting historical source hashes."]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("baseline","intrinsic","tail","output"):p.add_argument(name,type=Path)
    p.add_argument("directories",nargs="+",type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=run(a.baseline,a.directories,a.intrinsic,a.tail)
    dump_new(a.output,result)
    print({k:v for k,v in result.items() if k not in ("conditions","source_hashes")})


if __name__=="__main__":main()
