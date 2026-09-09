"""Finish interrupted cached-factor courses without repeating completed probes.

Set logging before checkpoint loading: logging sinks are intentionally local,
not serialized neural state. Existing complete arrays are retained and passed to
the independent audit. Incomplete zip files are preserved under .incomplete.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
from neuron.neuron import setup_neuron_logger
from simulations.active_inference.core.runtime_checkpoint import load_checkpoint
from .association_route_probe import digest
from .composition_probe import encode
from .eligibility_association_probe import dynamic_snapshot
from .eligibility_media_probe import record
from .media_order_audit import load_state
from .media_weight_identity import set_selected_weights,PathObserver
from .runtime_checkpoint_probe import CheckpointDriver


def finish(root):
    setup_neuron_logger('CRITICAL')
    root=Path(root).resolve()
    if (root/'summary.json').exists():raise ValueError('Course is already complete')
    m=json.loads((root/'manifest.json').read_text());source=Path(m['source']);course=Path(m['course'])
    if any(digest(p)!=h for p,h in m['source_hashes'].items()):raise ValueError('Recorded source changed')
    m['continuation_sources']={str(Path(__file__).resolve()):digest(__file__)}
    (root/'manifest.json').write_text(encode(m)+'\n')
    with np.load(root/'weights.npz') as z:
        qsets={name:z[name] for name in m['conditions']};qsets={'control':z['learned'],**qsets}
    cm=json.loads((course/'manifest.json').read_text());features=[]
    for clip in (0,1):
        p=next(Path(p) for p in cm['physical_sources'] if Path(p).name==f'sensory-{clip}.npz')
        with np.load(p) as z:features.append({k:z[k] for k in z.files})
    parent=load_state(course/'checkpoint-16-state.json.gz');checkpoint=source/m['checkpoint']['file']
    if digest(checkpoint)!=m['checkpoint']['sha256']:raise ValueError('Checkpoint changed')
    rows=[];computed=0;reused=0;began=time.perf_counter()
    for condition,q in qsets.items():
        for cue in ([0] if condition=='control' else m['cues']):
            name=f'{condition}-cue-{cue}.npz';path=root/name;existing=False
            if path.exists():
                try:
                    with np.load(path) as z:
                        for key in z.files:z[key]  # Force all compressed members to be read.
                    existing=True
                except (zipfile.BadZipFile,EOFError,ValueError):
                    failed=path.with_suffix('.npz.incomplete')
                    if failed.exists():raise ValueError('An earlier incomplete file already exists')
                    path.rename(failed)
            trial=dict(start=parent['tick'],stop=parent['tick']+300,visual_clip=cue,audio_clip=None)
            if not existing:
                if shutil.disk_usage(root).free<1024**3:raise OSError('Less than 1 GiB free')
                restored=load_checkpoint(checkpoint,trusted=True);net=restored.network
                if json.loads(dynamic_snapshot(net))!=parent:raise ValueError('Restored state differs')
                set_selected_weights(net,m['selected_ports'],q)
                cells=list(net.network.neurons.values());syns=[p for n in cells for p in n.postsynaptic_points.values()]
                if any(n.params.eta_post<=0 or n.params.eta_retro<=0 for n in cells):raise ValueError('Frozen learning')
                observer=PathObserver(cells,syns,net,m['selected_ports'])
                d=record(net,CheckpointDriver(restored),cells,syns,features,m['groups'],trial,m['selected_ports'],observer)
                d.update(selected_potential=np.asarray(observer.potentials),terminals=np.asarray(observer.releases))
                d['selected_local_current']=np.where(d['arrivals']>0,d['selected_potential'],0.)
                if condition=='control':
                    with np.load(source/'intact-cue-0.npz') as z:
                        if set(z.files)!=set(d) or any(not np.array_equal(d[k],z[k]) for k in d):raise ValueError('Resumed control differs')
                else:np.savez_compressed(path,**d)
                computed+=300
                del restored,net,cells,syns,observer,d
            if condition!='control':
                rows.append(dict(condition=condition,cue=cue,trial=trial,file=name,sha256=digest(path),reused=existing))
                reused+=int(existing)
            print(encode(dict(condition=condition,cue=cue,reused=existing)),flush=True)
    if any(digest(p)!=h for p,h in {**m['source_hashes'],**m['continuation_sources']}.items()):raise ValueError('Source changed')
    result=dict(probes=rows,intact_control_exact=True,acquisition_ticks=0,ticks=3900,
        tick_count_scope='Unique completed protocol, not total work across interruptions and repeated validation.',
        resumed_executed_ticks=computed,reused_probes=reused,seconds=time.perf_counter()-began)
    (root/'summary.json').write_text(encode(result)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('recording',type=Path)
    finish(p.parse_args().recording)
