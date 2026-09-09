"""Uniform-image control for the actual four-phase sensory transducer.

The same time-integrated input per receptor arrives either on the existing
four-tick schedule or continuously. A uniform image contains no spatial
contrast. Any periodic contrast response needs a temporal/interface account.
No acquired association or autonomous neural oscillation is claimed here.
"""
import argparse
import inspect
import json
from pathlib import Path
import shutil

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode,fingerprint,k
from .multimodal_pairing_probe import fresh,inputs
from .population_hierarchy import cellular,FIELDS
from .population_contrast_probe import audit
from ..components.sensory.population_contrast import append_population_contrast
from neuron.extensions.experimental.graded_eligibility import GradedEligibilityNeuron


def drive(groups,trial,tick,features,mode):
    if mode=='four_phase':return inputs(features,groups,trial,tick)
    if mode!='continuous_mean':raise ValueError('Unknown sensory timing')
    # Existing encoder emits 2*value once per four ticks. Deliver value/2
    # on every tick for equal per-channel integrated external information.
    values=features[0]['visual'][tick-trial['start']]
    return [(nid,float(value/2)) for nid,value in zip(groups['vision'],values)]


def run(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    m=json.loads((source/'manifest.json').read_text());original=json.loads((source/'config.json').read_text())
    if len(original['neurons'])!=1152:raise ValueError('Expected the original full population graph')
    cfg,groups,edges=append_population_contrast(original,m['groups']['vision'])
    added=[n for ids in groups.values() for n in ids]
    hashes=fingerprint()
    for obj in (run,append_population_contrast,audit,fresh):
        p=Path(inspect.getfile(obj)).resolve();hashes[str(p)]=digest(p)
    if shutil.disk_usage(output.parent).free<750*1024**2:raise OSError('Free-space reserve reached')
    output.mkdir(exist_ok=False);path=output/'config.json';path.write_text(encode(cfg)+'\n')
    features=[dict(ticks=300,visual=np.full((300,len(m['groups']['vision'])),.6),
                   auditory=np.zeros((300,len(m['groups']['touch']))))]
    manifest=dict(source=str(source),source_config_sha256=digest(source/'config.json'),source_hashes=hashes,
        groups={**m['groups'],**groups},added_ids=added,fields=FIELDS,edges=edges,
        physical_control='Synthetic constant uniform image, value 0.6; no sound; 300 ticks per mode',
        modes=['four_phase','continuous_mean'],scope='Temporal interface control, not biological phase fidelity or learning acceptance')
    (output/'manifest.json').write_text(encode(manifest)+'\n');entries=[];used=0
    for mode in manifest['modes']:
        net,_,members,_=fresh(path,11,GradedEligibilityNeuron)
        points=[p for n in added for p in net.network.neurons[n].postsynaptic_points.values()]
        weights=lambda:np.array([p.u_i.info for p in points])
        terminals=lambda:np.array([n.presynaptic_points[k.TERM].u_o.info for n in members])
        data=dict(initial_cells=cellular(members),initial_terminals=terminals(),initial_added_weights=weights())
        rows={key:[] for key in ('cells','terminals','added_weights','external')}
        vi={n:i for i,n in enumerate(m['groups']['vision'])}
        trial=dict(start=0,stop=300,visual_clip=0,audio_clip=None)
        for t in range(300):
            external=np.zeros(len(vi))
            for nid,value in drive(m['groups'],trial,t,features,mode):
                net.set_external_input(nid,0,value);external[vi[nid]]=value
            net.run_tick();rows['cells'].append(cellular(members));rows['terminals'].append(terminals())
            rows['added_weights'].append(weights());rows['external'].append(external)
        data.update({k:np.array(v) for k,v in rows.items()})
        residual=audit(data,cfg,added)
        index={n.id:i for i,n in enumerate(members)}
        response=data['cells'][:,[index[n] for role in ('contrast_above','contrast_below') for n in groups[role]],1]
        # Summaries only index the fully retained trajectories. No diagnostic
        # output or summary is supplied back to the neural preparation.
        row=dict(mode=mode,equation_residual=residual,late_max=float(response[64:].max()),
                 late_mean=float(response[64:].mean()),phase_means=[response[64+p::4].mean(axis=0) for p in range(4)])
        destination=output/(mode+'.npz');np.savez_compressed(destination,**data);used+=destination.stat().st_size
        row.update(file=destination.name,sha256=digest(destination));entries.append(row)
        print(encode({k:v for k,v in row.items() if k!='phase_means'}),flush=True)
        if used>80*1024**2:raise OSError('Interface control raw budget reached')
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Source changed during the control')
    with np.load(output/'four_phase.npz') as a,np.load(output/'continuous_mean.npz') as b:
        if not np.allclose(a['external'].sum(axis=0),b['external'].sum(axis=0),rtol=0,atol=1e-10):
            raise ValueError('Integrated external dose differs')
    result=dict(entries=entries,ticks=600,bytes=used)
    (output/'completion.json').write_text(encode(result)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','output'):p.add_argument('--'+key,type=Path,required=True)
    run(**vars(p.parse_args()))
