"""Two real visual recordings through a 1,441-cell PAULA contrast preparation.

Tests the neural representation, not associative learning. The old 1,152-cell
graph remains active. Added relays/pool/opponent cells have positive adaptation.
Every cellular tick, information terminal, and added input weight is retained.
An independent arithmetic audit reconstructs their delayed graded dynamics.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .composition_probe import encode,fingerprint,k
from .association_route_probe import digest
from .multimodal_pairing_probe import fresh,inputs
from .population_hierarchy import cellular
from .predictive_context_capacity import cone_projection
from ..components.sensory.population_contrast import append_population_contrast
from neuron.extensions.experimental.graded_eligibility import GradedEligibilityNeuron


def audit(data,cfg,added):
    ids=[n['id'] for n in cfg['neurons']]; index={n:i for i,n in enumerate(ids)}
    nodes={n['id']:n for n in cfg['neurons']}; ai=[index[n] for n in added]
    sources={(c['target_neuron'],c['target_synapse']):index[c['source_neuron']] for c in cfg['connections']}
    sizes=[nodes[n]['params']['num_inputs'] for n in added];offset=np.cumsum([0]+sizes)
    src=[sources.get((n,s),-1) for n in added for s in range(nodes[n]['params']['num_inputs'])]
    src=np.array(src);connected=src>=0
    decay=np.array([nodes[n]['params']['delta_decay'] for n in added],dtype=np.float32)
    lam=np.array([nodes[n]['params']['lambda_param'] for n in added],dtype=np.float32)
    previous=data['initial_cells']; terminal=data['initial_terminals'];q=data['initial_added_weights']
    due=np.zeros(len(added),dtype=np.float32); residual=0.
    for t,cells in enumerate(data['cells']):
        old_s=previous[ai,0].astype(np.float32)
        s=old_s+(1/lam)*(-old_s+due)
        err=max(float(np.max(np.abs(s-cells[ai,0]))),float(np.max(np.abs(np.maximum(0,s)-cells[ai,1]))))
        residual=max(residual,err)
        if err!=0:raise ValueError(f'Graded neural transmission/integration differs at tick {t}: {err}')
        arrival=np.zeros(len(src),dtype=np.float32)
        arrival[connected]=(previous[src[connected],1]*terminal[src[connected]]).astype(np.float32)
        local=arrival*q.astype(np.float32)
        # All added dendritic distances are one. Native heap order sorts
        # simultaneous potentials, then accumulates their attenuated values.
        next_due=[]
        for j,(lo,hi) in enumerate(zip(offset[:-1],offset[1:])):
            total=np.float32(0.)
            for v in np.sort(local[lo:hi][arrival[lo:hi]>0]):total=total+v*decay[j]
            next_due.append(total)
        due=np.array(next_due,dtype=np.float32)
        previous=cells;terminal=data['terminals'][t];q=data['added_weights'][t]
    return residual


def run(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    m=json.loads((source/'manifest.json').read_text())
    original=json.loads((source/'config.json').read_text())
    # Use the pre-bridge source to avoid silently including a second predictor.
    if len(original['neurons'])!=1152:raise ValueError('Expected original full population graph')
    cfg,groups,edges=append_population_contrast(original,m['groups']['vision'])
    added=[i for ids in groups.values() for i in ids]
    assert len(cfg['neurons'])==1441
    features=[]
    for p,h in sorted(m['physical_sources'].items()):
        if digest(p)!=h:raise ValueError('Physical recording changed')
        with np.load(p) as z:features.append({k:z[k] for k in z.files})
    hashes=fingerprint()
    for p in (Path(__file__),Path(__file__).parents[1]/'components/sensory/population_contrast.py'):
        hashes[str(p.resolve())]=digest(p)
    output.mkdir(exist_ok=False);path=output/'config.json';path.write_text(encode(cfg)+'\n')
    manifest=dict(source=str(source),source_config_sha256=digest(source/'config.json'),groups={**m['groups'],**groups},
        edges=edges,added_ids=added,physical_sources=m['physical_sources'],source_hashes=hashes,
        scope='Contrast representation only, one graph seed, two physical clips, 600 ticks. No learning acceptance.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    entries=[];responses=[];used=0
    for clip in (0,1):
        net,_,members,_=fresh(path,11,GradedEligibilityNeuron)
        new=[net.network.neurons[n] for n in added]
        points=[n.postsynaptic_points[s] for n in new for s in range(n.params.num_inputs)]
        terminal=lambda:np.array([n.presynaptic_points[k.TERM].u_o.info for n in members])
        weights=lambda:np.array([p.u_i.info for p in points])
        start=dict(initial_cells=cellular(members),initial_terminals=terminal(),initial_added_weights=weights())
        data={key:[] for key in ('cells','terminals','added_weights')}
        trial=dict(start=0,stop=300,visual_clip=clip,audio_clip=None)
        for t in range(300):
            for nid,value in inputs(features,m['groups'],trial,t):net.set_external_input(nid,0,value)
            net.run_tick();data['cells'].append(cellular(members));data['terminals'].append(terminal());data['added_weights'].append(weights())
        data={k:np.array(v) for k,v in data.items()};data.update(start)
        residual=audit(data,cfg,added)
        destination=output/f'visual-{clip}.npz';np.savez_compressed(destination,**data);used+=destination.stat().st_size
        if used>80*1024**2:raise OSError('Contrast recording budget reached')
        response=data['cells'][:,np.array(groups['contrast_above']+groups['contrast_below'])-1,1]
        responses.append(response)
        entries.append(dict(clip=clip,file=destination.name,sha256=digest(destination),equation_residual=residual))
        print(encode(entries[-1]),flush=True)
    x=np.stack([r.mean(axis=0) for r in responses]);y=np.stack([f['auditory'].mean(axis=0) for f in features])
    raw=np.stack([f['visual'].mean(axis=0) for f in features]);capacity={}
    for name,basis in (('all_raw',raw),('neural_contrast',x)):
        capacity[name]={}
        for mapping,target in (('paired',y),('swapped',y[::-1])):
            residuals=[float(np.sum((cone_projection(basis.T,target[:,i])[0]-target[:,i])**2)) for i in range(96)]
            capacity[name][mapping]=dict(infeasible_mean_pairs=sum(r>1e-12 for r in residuals),squared_errors=residuals)
    result=dict(entries=entries,capacity=capacity,bytes=used,ticks=600,
        interpretation='Optimistic static-readout capacity of observed cue means; not an acquired association or a bound on the complete adapting system.')
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Runtime changed')
    (output/'summary.json').write_text(encode(result)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(**vars(p.parse_args()))
