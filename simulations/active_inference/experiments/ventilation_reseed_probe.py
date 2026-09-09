"""Causal construction diagnostic, not an autonomous motor intervention.

Branch the original acquired brain, with no new cells. Compare unchanged
continuation, a stale input-cache rebuild, an explicitly declared second birth
pulse, and a synchronized rebuild. Record actual CPG input buffers and complete
existing sensorimotor observations. The two pairwise identities test a concrete
runtime cause; they do not establish organ-responsive neural regulation.
"""
import argparse
import inspect
import json
from pathlib import Path
import shutil

import numpy as np

from . import context_organization as base
from .active_sweep_memory import restore
from .active_sweep_credit import record_credit
from .ventilation_regulation import CoupledHinge
from ..core.external_input_state import synchronize_quiescent_external_inputs
from neuron.extensions.experimental.cascade_eligibility import CascadeEligibilityNeuron


def run(parent,output,ticks=256):
    parent,output=Path(parent).resolve(),Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    if ticks!=256: raise ValueError('Declared construction diagnostic is 256 ticks')
    if shutil.disk_usage(output.parent).free<3*1024**3: raise OSError('Need 3 GiB reserve')
    m=json.loads((parent/'manifest.json').read_text());g=m['groups']
    row=next(r for r in m['rows'] if r['kind']=='current')
    sources=dict(m['sources'])
    for key,h in (('checkpoint','checkpoint_sha256'),('physical','physical_sha256')):
        sources[str(parent/row[key])]=row[h]
    for obj in (run,CoupledHinge,synchronize_quiescent_external_inputs,record_credit):
        p=str(Path(inspect.getfile(obj)).resolve());sources[p]=base.digest(p)
    for p,h in sources.items():
        if base.digest(p)!=h: raise ValueError('Source changed: '+p)
    with np.load(m['media']) as z: features={k:z[k] for k in ('visual','auditory')}
    output.mkdir();data={};rows=[]
    for mode in ('unchanged','stale-rebuild','explicit-pulse','safe-rebuild'):
        net,oldbody,delay=restore(parent/row['checkpoint'],parent/row['physical'])
        if net.current_tick!=6352: raise ValueError('Wrong acquisition age')
        topo=net.network;birth=(g['cpg'][0],0)
        initial_drive=topo._ext_vec['info'][topo._ext_vec['row_of'][birth]]
        dictionary_drive=topo.external_inputs[birth]['info']
        if initial_drive!=0 or dictionary_drive!=5: raise ValueError('Expected recorded stale bootstrap')
        if mode=='stale-rebuild': topo._ext_vec=None
        elif mode=='explicit-pulse': net.set_external_input(*birth,5.)
        elif mode=='safe-rebuild':
            synchronize_quiescent_external_inputs(topo);topo._ext_vec=None
        body=CoupledHinge(net,g)
        body.restore(oldbody.state(),crossings=oldbody.crossings,next_gate=oldbody.next_gate)
        initial_organs=body.organs.state();observed=[];old_tick=CascadeEligibilityNeuron.tick
        def tick(n,external,t,dt=1.):
            if n.id in g['cpg']:
                if n.id==g['cpg'][0]: observed.append([])
                observed[-1].append(n.input_buffer.copy())
            return old_tick(n,external,t,dt)
        CascadeEligibilityNeuron.tick=tick
        try: z=record_credit(net,body,delay,features,g,ticks)
        finally: CascadeEligibilityNeuron.tick=old_tick
        z.update({k:np.asarray(v) for k,v in body.rows.items()})
        z.update(cpg_inputs=np.asarray(observed),cpg_ids=np.array(g['cpg']),organ_initial=initial_organs)
        expected=np.zeros(ticks,dtype=np.float32)
        if mode in ('stale-rebuild','explicit-pulse'): expected[0]=5.
        ids=list(z['neuron_ids']);terms=list(map(tuple,z['terminal_ids']))
        src,term=topo.neurons[birth[0]].synapse_sources[0]
        expected[1:]+=np.asarray(z['cells'][:-1,ids.index(src),base.FIELDS.index('O')]*
                                z['terminal_info'][:-1,terms.index((src,term))],dtype=np.float32)
        np.testing.assert_array_equal(z['cpg_inputs'][:,0,0,0],expected)
        np.testing.assert_array_equal(z['cpg_inputs'][:,:,0,1:],0.)
        path=output/(mode+'.npz');np.savez_compressed(path,**z);data[mode]=z
        rows.append(dict(mode=mode,path=path.name,sha256=base.digest(path),
                         authoritative_initial_drive=float(initial_drive),stale_dictionary_drive=float(dictionary_drive)))
    for left,right in (('unchanged','safe-rebuild'),('stale-rebuild','explicit-pulse')):
        if set(data[left])!=set(data[right]): raise ValueError('Missing comparison field')
        for k in data[left]: np.testing.assert_array_equal(data[left][k],data[right][k],err_msg=left+'/'+right+'/'+k)
    a,b=data['unchanged'],data['stale-rebuild'];ids=list(a['neuron_ids']);oi=base.FIELDS.index('O')
    first={key:int(np.flatnonzero(np.any(a[key].reshape(ticks,-1)!=b[key].reshape(ticks,-1),axis=1))[0])
           for key in ('cpg_inputs','cells','body','muscles')}
    rasters={mode:{str(n):np.flatnonzero(z['cells'][:,ids.index(n),oi]>0).tolist() for n in g['cpg']}
             for mode,z in data.items()}
    if any(base.digest(p)!=h for p,h in sources.items()): raise ValueError('Source changed during experiment')
    result=dict(seed=m['seed'],parent=str(parent),start=6352,ticks=ticks,rows=rows,sources=sources,
                first_difference=first,rasters=rasters,exact_pairs=[['unchanged','safe-rebuild'],['stale-rebuild','explicit-pulse']],
                compared_fields=sorted(a),limits=__doc__)
    (output/'manifest.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(seed=m['seed'],first_difference=first,exact_pairs=result['exact_pairs'])),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent');p.add_argument('output')
    a=p.parse_args();run(a.parent,a.output)
