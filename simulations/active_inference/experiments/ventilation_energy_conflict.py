"""Does actual energy feedback counteract fixed low-oxygen recruitment?

Match the held-low sensory branch, but zero only the two energy-alarm incoming
muscle information weights at the original composition checkpoint. Preserve all
cells, other weights, body state, native return edges and positive adaptation.
This tests an interaction exposed by the oxygen clamp, not learned arbitration
or a claim that the four-second course requires the energy alarm for survival.
"""
import argparse
from copy import deepcopy
import inspect
import json
from pathlib import Path
import shutil

import numpy as np

from . import context_organization as base
from .ventilation_input_clamp import ClampedOrganDelay,audit
from .ventilation_regulation import record
from .ventilation_regulation_replay import restore
from .ventilation_screen import observations


def run(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free<3*1024**3:raise OSError('Need 3 GiB reserve')
    m=json.loads((root/'manifest.json').read_text());g=m['groups'];meta=m['meta']
    if m['mode']!='feedback' or m['ticks']!=1024:raise ValueError('Need corrected feedback parent')
    sources=dict(m['sources'])
    for name in ('initial.paula','initial-body.npz','manifest.json','config.json'):
        sources[str(root/name)]=base.digest(root/name)
    for obj in (run,audit,restore,record):
        p=str(Path(inspect.getfile(obj)).resolve());sources[p]=base.digest(p)
    for p,h in sources.items():
        if base.digest(p)!=h:raise ValueError('Source changed: '+p)
    net,body,delay,od=restore(root/'initial.paula',root/'initial-body.npz',g)
    cfg=json.loads((root/'config.json').read_text());original=deepcopy(cfg);changed=[]
    for src,nid,sid in meta['energy']['ports']:
        cell=net.network.neurons[nid]
        if cell.synapse_sources[sid][0]!=src or cell.postsynaptic_points[sid].u_i.info!=-2.:
            raise ValueError('Unexpected energy projection')
        cell.postsynaptic_points[sid].u_i.info=0.
        p=next(p for p in cfg['synaptic_points'] if p['type']=='postsynaptic' and
               (p['neuron_id'],p['synapse_id'])==(nid,sid))
        if p['u_i']['info']!=-2.:raise ValueError('Config mismatch')
        p['u_i']['info']=0.;changed.append([nid,sid,-2.,0.])
    if len(changed)!=2:raise ValueError('Expected two antagonist projections')
    # The source metadata describes anatomy; the separate intervention record
    # below declares the actual two changed runtime/config weights.
    od=ClampedOrganDelay(od.state(),.25)
    pm=json.loads((Path(m['parent'])/'manifest.json').read_text())
    with np.load(pm['media']) as z:features={k:z[k] for k in ('visual','auditory')}
    output.mkdir();(output/'config.json').write_text(base.encode(cfg)+'\n')
    protocol=dict(parent=str(root),seed=m['seed'],groups=g,meta=meta,start=net.current_tick,ticks=1024,
                  oxygen_clamp=.25,changed_weights=changed,sources=sources,limits=__doc__)
    (output/'protocol.json').write_text(base.encode(protocol)+'\n')
    z=record(net,body,delay,od,features,g,meta,1024)
    np.savez_compressed(output/'ticks.npz',**z);audit(z,cfg,g,features,.25)
    base.save_checkpoint(net,output/'final.paula',sources=list(sources))
    np.savez_compressed(output/'final-body.npz',state=body.state(),delay=delay.state(),organ=body.organs.state(),
        organ_delay=od.state(),oxygen_clamp=[.25],gate=[body.crossings,body.next_gate])
    if any(base.digest(p)!=h for p,h in sources.items()):raise ValueError('Source changed during conflict test')
    result=dict(protocol,sha256=base.digest(output/'ticks.npz'),observations=observations(z),
        final_checkpoint_sha256=base.digest(output/'final.paula'),physical_sha256=base.digest(output/'final-body.npz'))
    (output/'manifest.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(seed=m['seed'],condition='energy-output-cut',final_organs=z['organs'][-1])),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('output')
    a=p.parse_args();run(a.root,a.output)
