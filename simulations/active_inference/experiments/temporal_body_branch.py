"""Complete-state interventions on acquired temporal-prediction weights."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode
from .temporal_body_probe import record_history,audit_history
from .predictive_bridge_probe import audit_record
from .predictive_weight_transplant import install_weights,arrange_weights
from .proprioceptive_learning_probe import BranchNetwork
from .proprioceptive_learning_audit import verify_physics
from ..components.body.radian_research_rower import RadianResearchRower
from ..core.runtime_checkpoint import load_checkpoint,save_checkpoint
from neuron.neuron import setup_neuron_logger


def run(source,output,*,condition='learned',ticks=512):
    if condition not in ('learned','birth','shuffled') or not 64<=ticks<=1024:
        raise ValueError('Invalid bounded intervention')
    source,output=Path(source).resolve(),Path(output).resolve()
    if shutil.disk_usage(output.parent).free<2*1024**3:raise OSError('Keep 2 GiB free')
    m=json.loads((source/'manifest.json').read_text());cfg=json.loads((source/'config.json').read_text())
    setup_neuron_logger('CRITICAL')
    branch=load_checkpoint(source/'final.neural-checkpoint',trusted=True)
    with np.load(source/'closed-loop.npz') as z:
        weights=z['weights'][-1].copy();birth=z['start_weights'].copy();physical=z['physical_after'][-1].copy()
    actual=np.array([[branch.network.network.neurons[n].postsynaptic_points[s].u_i.info
        for s in branch.network.network.neurons[n].prediction_ports] for n in m['bridge']['prediction']])
    np.testing.assert_array_equal(actual,weights)
    replacement=weights if condition=='learned' else birth if condition=='birth' else arrange_weights(weights,shuffled=True)
    install_weights(branch.network,m['bridge'],replacement)
    body=RadianResearchRower();body.restore(physical)
    output.mkdir(exist_ok=False)
    manifest=dict(source=str(source),condition=condition,mode=m['mode'],ticks=ticks,
        gain=m['gain'],motor=m['motor'],bridge=m['bridge'],old_bridge=m['old_bridge'],basis=m['basis'],fields=m['fields'],
        source_hashes={str(source/f):digest(source/f) for f in ('manifest.json','config.json','closed-loop.npz','final.neural-checkpoint')},
        producer_sha256=digest(__file__),
        interpretation='Selected new prediction weights only. Complete neural/body state preserved, all adaptation remains. '
        'Physical feedback may diverge; compare full traces. One original brain graph, not independent replication.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    save_checkpoint(branch,output/'initial.neural-checkpoint',sources=(__file__,))
    started=time.perf_counter()
    data=record_history(BranchNetwork(branch),body,m['motor'],m['bridge'],m['basis'],ticks,gain=m['gain'])
    raw=output/'closed-loop.npz';np.savez_compressed(raw,**data)
    result=dict(condition=condition,mode=m['mode'],ticks=ticks,
        history_residual=audit_history(data,cfg,m['basis']),prediction_residual=audit_record(data,cfg,m['bridge']),
        physics_residual=verify_physics(data,m['gain']),seconds=time.perf_counter()-started,
        raw_bytes=raw.stat().st_size,raw_sha256=digest(raw))
    save_checkpoint(branch,output/'final.neural-checkpoint',sources=(__file__,))
    (output/'summary.json').write_text(encode(result)+'\n');print(encode(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--condition',choices=('learned','birth','shuffled'),default='learned')
    p.add_argument('--ticks',type=int,default=512)
    run(**vars(p.parse_args()))
