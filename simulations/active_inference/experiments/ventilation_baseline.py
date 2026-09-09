"""Unmodified acquired brain from the same state as organ-feedback composition.

Organs are passive observers here. No new neurons, afferents or neural edges.
This distinguishes ongoing changes in the original brain from the effects of
adding cells, changing muscle fan-in or introducing native return pathways.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from . import context_organization as base
from .active_sweep_memory import restore
from .active_sweep_credit import record_credit
from .active_sweep_transfer import audit as audit_predictor
from .ventilation_screen import audit as audit_body
from .ventilation_regulation import CoupledHinge


def run(parent,output):
    parent,output=Path(parent).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free<3*1024**3:raise OSError('Need 3 GiB reserve')
    m=json.loads((parent/'manifest.json').read_text());g=m['groups']
    row=next(r for r in m['rows'] if r['kind']=='current');sources=dict(m['sources'])
    for key,h in (('checkpoint','checkpoint_sha256'),('physical','physical_sha256')):
        sources[str(parent/row[key])]=row[h]
    sources[str(parent/'manifest.json')]=base.digest(parent/'manifest.json')
    sources[str(Path(__file__).resolve())]=base.digest(__file__)
    from . import ventilation_regulation,ventilation_screen
    from ..components.body import ventilation,energy_budget
    for module in (ventilation_regulation,ventilation_screen,ventilation,energy_budget):
        sources[str(Path(module.__file__).resolve())]=base.digest(module.__file__)
    for p,h in sources.items():
        if base.digest(p)!=h:raise ValueError('Source changed: '+p)
    net,oldbody,delay=restore(parent/row['checkpoint'],parent/row['physical'])
    if net.current_tick!=6352:raise ValueError('Wrong initial state')
    body=CoupledHinge(net,g);body.restore(oldbody.state(),next_gate=oldbody.next_gate,crossings=oldbody.crossings)
    initial=body.organs.state();cells_initial=base.cellular(list(net.network.neurons.values()))
    cfg=json.loads(Path(m['config']).read_text())
    with np.load(m['media']) as z:features={k:z[k] for k in ('visual','auditory')}
    output.mkdir();(output/'config.json').write_text(base.encode(cfg)+'\n')
    protocol=dict(parent=str(parent),seed=m['seed'],groups=g,mode='original',ticks=1024,start=6352,
                  sources=sources,limits=__doc__,cell_fields=base.FIELDS)
    (output/'protocol.json').write_text(base.encode(protocol)+'\n')
    z=record_credit(net,body,delay,features,g,1024)
    z.update({k:np.asarray(v) for k,v in body.rows.items()})
    z.update(organ_initial=initial,reg_cells_initial=cells_initial,intervention=np.array([1.,1.,1.]))
    np.savez_compressed(output/'ticks.npz',**z)
    base.save_checkpoint(net,output/'final.paula',sources=list(sources))
    np.savez_compressed(output/'final-body.npz',state=body.state(),delay=delay.state(),
                        organ=body.organs.state(),gate=[body.crossings,body.next_gate])
    audit_predictor(z,cfg,g,features,.8);audit_body(z)
    if any(base.digest(p)!=h for p,h in sources.items()):raise ValueError('Source changed during baseline')
    deficits={name:np.flatnonzero(z['organs'][:,col]>1e-12).tolist() for name,col in [('oxygen',4),('energy',10)]}
    result=dict(protocol,sha256=base.digest(output/'ticks.npz'),deficits=deficits,audited_ticks=1024,
                final_organs=z['organs'][-1])
    (output/'manifest.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(seed=m['seed'],first_deficit={k:(v[0] if v else None) for k,v in deficits.items()})),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.parent,a.output)
