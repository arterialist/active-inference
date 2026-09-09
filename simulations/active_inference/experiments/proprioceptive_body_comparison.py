"""Branch the same 1793-cell executable birth state into two physical bodies.

No neuronal parameter changes. The radians-explicit body is an intervention,
not a replacement of the reference body. Every tick and physical integration
state is retained. This diagnostic precedes an action-consequence learning test.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np

from .association_route_probe import digest
from .composition_probe import encode
from .proprioceptive_loop_probe import record_loop
from .predictive_bridge_probe import audit_record
from ..components.body.research_rower import ResearchRower
from ..components.body.radian_research_rower import RadianResearchRower
from ..core.runtime_checkpoint import load_checkpoint, save_checkpoint


def audit_body_replay(data, body, *, gain):
    body.restore(data['physical_before'][0])
    for t in range(len(data['ticks'])):
        np.testing.assert_array_equal(body.state(),data['physical_before'][t])
        np.testing.assert_array_equal(body.data.qpos[body.positions],data['joint_position'][t])
        np.testing.assert_array_equal(body.sense(),data['joint_input'][t])
        np.testing.assert_array_equal(gain*np.maximum(0.,data['muscle_state'][t]),data['actuator_ctrl'][t])
        body.step(data['muscle_state'][t],gain=gain)
        np.testing.assert_array_equal(body.state(),data['physical_after'][t])
    return 0.


def run(source,output,*,ticks=640,gain=8.):
    source,output=Path(source).resolve(),Path(output).resolve()
    if not 64<=ticks<=1200 or not np.isfinite(gain) or gain<0:
        raise ValueError('Bounded body diagnostic required')
    if shutil.disk_usage(output.parent).free<2*1024**3:
        raise OSError('Keep 2 GiB free')
    m=json.loads((source/'manifest.json').read_text())
    cfg=json.loads((source/'config.json').read_text())
    branch=load_checkpoint(source/'initial.neural-checkpoint',trusted=True)
    output.mkdir(exist_ok=False)
    body=RadianResearchRower()
    manifest=dict(parent=str(source),parent_checkpoint_sha256=digest(source/'initial.neural-checkpoint'),
        parent_manifest_sha256=digest(source/'manifest.json'),mujoco_version=mujoco.__version__,
        fields=m['fields'],neuron_ids=m['neuron_ids'],motor=m['motor'],bridge=m['bridge'],
        ticks=ticks,gain=gain,body='explicit_radians',
        joint_ranges_radians=body.model.jnt_range[body.joint_ids],
        sources={str(Path(p).resolve()):digest(p) for p in (__file__,
            Path(__file__).parents[1]/'components/body/radian_research_rower.py')},
        interpretation='Same executable neural birth state as parent; only body angle convention changes. '
                       'No audiovisual playback. All neural plasticity remains positive.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    started=time.perf_counter()
    data=record_loop(branch.network,body,m['motor'],m['bridge'],ticks,gain=gain)
    np.savez_compressed(output/'closed-loop.npz',**data)
    residual=audit_record(data,cfg,m['bridge'])
    replay=audit_body_replay(data,RadianResearchRower(),gain=gain)
    save_checkpoint(branch.network,output/'final.neural-checkpoint',sources=(__file__,))
    with np.load(source/'closed-loop.npz') as z:
        for key in data:
            if key.startswith('start_'):
                np.testing.assert_array_equal(data[key],z[key])
    scale=np.max(np.abs(body.model.jnt_range[body.joint_ids]),axis=1)
    result=dict(ticks=ticks,seconds=time.perf_counter()-started,neural_residual=residual,
        physical_residual=replay,raw_bytes=(output/'closed-loop.npz').stat().st_size,
        fractional_sensor_ticks=np.sum((data['joint_input']>0)&(data['joint_input']<1),axis=0),
        joint_limit_exceed_ticks=np.sum(np.abs(data['joint_position'])>scale,axis=0),
        max_abs_joint_radians=np.max(np.abs(data['joint_position']),axis=0),
        max_weight_change=float(np.max(np.abs(data['weights']-data['start_weights']))))
    (output/'summary.json').write_text(encode(result)+'\n');print(encode(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--ticks',type=int,default=640);p.add_argument('--gain',type=float,default=8.)
    run(**vars(p.parse_args()))
