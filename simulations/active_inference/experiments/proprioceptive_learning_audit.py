"""Independent full-tick audit, including float64 actuator transduction.

The first body-comparison auditor multiplied recorded float32 muscle arrays
before converting to float64. The physical bridge converts FIRST. Birth
records happened to promote to float64; resumed records exposed the mistake.
This audit retains exact equality and verifies the completed raw records even
when the original post-run auditor failed. It does not rewrite that evidence.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .association_route_probe import digest
from .associative_mismatch_audit import contextual_weight_mask
from .composition_probe import encode
from .predictive_bridge_probe import audit_record
from ..components.body.radian_research_rower import RadianResearchRower


def verify_physics(data,gain):
    body=RadianResearchRower()
    body.restore(data['physical_before'][0])
    for t in range(len(data['ticks'])):
        np.testing.assert_array_equal(body.state(),data['physical_before'][t])
        np.testing.assert_array_equal(body.data.qpos[body.positions],data['joint_position'][t])
        scales=np.max(np.abs(body.model.jnt_range[body.joint_ids]),axis=1)
        p=np.clip(body.data.qpos[body.positions]/scales,-1.,1.)
        expected_sensor=np.maximum(0.,[p[0],-p[0],p[1],-p[1]])
        np.testing.assert_array_equal(expected_sensor,data['joint_input'][t])
        # Same dtype boundary as ResearchRower.step, not a wider tolerance.
        muscle=np.asarray(data['muscle_state'][t],dtype=float)
        np.testing.assert_array_equal(gain*np.maximum(0.,muscle),data['actuator_ctrl'][t])
        body.step(muscle,gain=gain)
        np.testing.assert_array_equal(body.state(),data['physical_after'][t])
    return 0.


def first_difference(a,b):
    different=np.any(a!=b,axis=tuple(range(1,a.ndim))) if a.ndim>1 else a!=b
    ticks=np.flatnonzero(different)
    return int(ticks[0]) if ticks.size else None


def intervals(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return [[int(a),int(b)-1] for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]


def run(learned,birth,shuffled,output):
    paths={k:Path(p).resolve() for k,p in dict(learned=learned,birth=birth,shuffled=shuffled).items()}
    datasets={};manifests={};sources=[];maximum=0.
    for name,path in paths.items():
        m=json.loads((path/'manifest.json').read_text());manifests[name]=m
        if m['condition']!=name:raise ValueError('Wrong branch identity')
        from . import proprioceptive_learning_probe as producer
        if digest(producer.__file__)!=m['source_code_sha256']:raise ValueError('Producer changed')
        for filename,key in (('final.neural-checkpoint','source_checkpoint_sha256'),
                ('closed-loop.npz','source_record_sha256'),('manifest.json','source_manifest_sha256')):
            if digest(Path(m['source'])/filename)!=m[key]:raise ValueError('Source changed')
        cfg=json.loads((Path(m['parent'])/'config.json').read_text())
        with np.load(path/'closed-loop.npz') as z:data={k:z[k] for k in z.files}
        if len(data['ticks'])!=m['ticks'] or not np.array_equal(np.diff(data['ticks']),np.ones(m['ticks']-1)):
            raise ValueError('Missing or nonconsecutive ticks')
        if not np.array_equal(data['neuron_ids'],m['neuron_ids']):raise ValueError('Column identity changed')
        maximum=max(maximum,audit_record(data,cfg,m['bridge']),verify_physics(data,m['gain']))
        datasets[name]=data;sources.append(dict(path=str(path/'closed-loop.npz'),sha256=digest(path/'closed-loop.npz')))
    m=manifests['learned'];a=datasets['learned'];mask=contextual_weight_mask(cfg,m['bridge'])
    index={int(n):i for i,n in enumerate(a['neuron_ids'])}
    pos=[index[n] for n in m['bridge']['error_positive']]
    neg=[index[n] for n in m['bridge']['error_negative']]
    effects=dict(ticks=a['ticks']);report={}
    for name,b in datasets.items():
        other=manifests[name]
        if any(other[k]!=m[k] for k in ('source','parent','gain','source_checkpoint_sha256','ticks')):
            raise ValueError('Branches are not matched')
        for key,value in a.items():
            if key.startswith('start_') and key not in ('start_incoming_info','start_weights'):
                np.testing.assert_array_equal(value,b[key])
        np.testing.assert_array_equal(a['start_incoming_info'][~mask],b['start_incoming_info'][~mask])
        np.testing.assert_array_equal(b['start_incoming_info'][mask],b['start_weights'].ravel())
        np.testing.assert_array_equal(a['physical_before'][0],b['physical_before'][0])
        effects[name+'_error']=b['cells'][:,pos,1]+b['cells'][:,neg,1]
        effects[name+'_prediction']=b['cells'][:,[index[n] for n in m['bridge']['prediction']],1]
        effects[name+'_joint_input']=b['joint_input']
        if name=='learned':continue
        benefit=effects[name+'_error']-effects['learned_error']
        effects[name+'_minus_learned_error']=benefit
        mean=benefit.mean(axis=1)
        report[name]=dict(first_error_difference=first_difference(effects[name+'_error'],effects['learned_error']),
            first_physical_difference=first_difference(a['physical_after'],b['physical_after']),
            max_physical_difference=float(np.max(np.abs(a['physical_after']-b['physical_after']))),
            first_sensory_difference=first_difference(a['joint_input'],b['joint_input']),
            max_sensory_difference=float(np.max(np.abs(a['joint_input']-b['joint_input']))),
            negative_mean_benefit_intervals=intervals(mean<0),
            mean_benefit_min=float(mean.min()),mean_benefit_max=float(mean.max()),
            late64_mean_benefit=float(mean[-64:].mean()),
            negative_ticks_per_channel=np.sum(benefit<0,axis=0),
            late64_benefit_per_channel=benefit[-64:].mean(axis=0))
    output=Path(output).resolve();output.mkdir(exist_ok=False)
    np.savez_compressed(output/'effects-per-tick.npz',**effects)
    result=dict(sources=sources,max_residual=maximum,results=report,
        interpretation='Positive control-minus-learned opponent output means lower neural error with acquired weights. '
        'Full trajectories and every unfavorable interval retained. Same initial neural/body state, '
        'except selected weights. Future physical signals may diverge through native return pathways. '
        'Independent audit corrects float conversion order; original failed post-audit remains documented. '
        'No final runtime checkpoints were written by failed producers; their initial checkpoints and all ticks remain.')
    (output/'summary.json').write_text(encode(result)+'\n');print(encode(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('learned','birth','shuffled','output'):p.add_argument('--'+key,type=Path,required=True)
    run(**vars(p.parse_args()))
