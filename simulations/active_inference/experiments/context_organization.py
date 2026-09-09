"""Context-bound audiovisual learning in an embodied force-compensation task.

This is a new experimental organization, not a new accepted agent version.
Real media drive fixed pixels/filterbank receptors. A visible two-colour context
cue distinguishes two externally imposed mechanical contingencies. Assignment
exists only in the world: cue 0/1 gives opposite pushes, reversed in context 1.
No cue identity, target action, host error or training flag enters neural code.

Two heterogeneous mixed-sensory populations project to shared force predictors.
Ordinary signed neural context input suppresses the incompatible population.
Predictor error comes from actual opponent force receptors and neural comparison.
Prediction drives antagonist muscles; joint receptors provide a weak innate
restoring reflex. All PAULA adaptation remains positive, including during probes.
Context gating is specified anatomy, not learned latent-context discovery.
"""
import argparse
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np

from .composition_probe import encode, fingerprint, k
from .multimodal_pairing_probe import fresh
from .population_hierarchy import cellular, FIELDS
from ..components.learning.predictive_bridge import append_predictive_bridge
from ..core.runtime_checkpoint import save_checkpoint, load_checkpoint
from neuron.extensions.experimental.predictive_receptor import PredictiveReceptorNeuron


XML = '''<mujoco model="context-organism"><compiler angle="radian"/>
<option timestep="0.004" gravity="0 0 0" integrator="implicitfast"/>
<worldbody><body name="arm"><joint name="hinge" type="hinge" axis="0 0 1"
 damping="0.12" armature="0.04"/><geom type="capsule" fromto="0 0 0 .25 0 0"
 size=".02" mass=".3"/></body></worldbody>
<actuator><motor name="muscle" joint="hinge" gear="0.2"/></actuator></mujoco>'''
DT = .004
FORCE = .2


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(seed=11, contextual=True, width=192):
    if width < 8 or width % 2:
        raise ValueError('Need an even mixed population of at least eight cells per context')
    rng = np.random.default_rng(seed)
    groups = {}; cursor = 1
    for role, count in [('vision',96),('audio',96),('context',2),('force',2),
                        ('joint',2),('mixed_0',width),('mixed_1',width)]:
        groups[role] = list(range(cursor,cursor+count)); cursor += count
    cfg = dict(metadata=dict(preparation='context-organization-v1',seed=seed),
               global_params=dict(num_inputs=2,num_neuromodulators=2),
               simulation_params=dict(max_history=1), neurons=[],synaptic_points=[],
               connections=[],external_inputs=[])
    ports = {i:0 for ids in groups.values() for i in ids}
    nodes = {}
    for role, ids in groups.items():
        for nid in ids:
            node = k.neuron(nid,lam=2,c=3,eta_post=1e-7,eta_retro=1e-7,
                delta_decay=.99,meta=dict(role=role,bounded_plasticity=True,
                graded_gain=1.,plasticity_rate_boost=0.))
            cfg['neurons'].append(node);nodes[nid]=node
            cfg['synaptic_points'].append(k.term(nid))
            if not role.startswith('mixed'):
                cfg['synaptic_points'].extend([k.syn(nid,0,1.,adapt=[0.,0.]),
                                              k.syn(nid,1,0.,adapt=[0.,0.])])
                cfg['external_inputs'].append(k.ext(nid,0));ports[nid]=2

    def wire(src,tgt,w):
        sid=ports[tgt];ports[tgt]+=1
        cfg['synaptic_points'].append(k.syn(tgt,sid,float(w),adapt=[0.,0.]))
        cfg['connections'].append(k.conn(int(src),tgt,sid))

    # Paired banks have the same input topology. Context changes access, not
    # stimulus-specific birth weights. Signed features widen the positive cone.
    sensory=groups['vision']+groups['audio']
    for j in range(width):
        sources=rng.choice(sensory,8,replace=False)
        weights=rng.choice([-1.,1.],8)/4
        for ctx in (0,1):
            nid=groups[f'mixed_{ctx}'][j]
            for src,w in zip(sources,weights):wire(src,nid,w)
            wire(groups['context'][1-ctx],nid,-4. if contextual else 0.)
    for nid,node in nodes.items():node['params']['num_inputs']=ports[nid]
    mixed=groups['mixed_0']+groups['mixed_1']
    cfg, bridge, edges, selected=append_predictive_bridge(cfg,mixed,groups['force'],
        seed=seed,fanin=len(mixed),consumers=2)
    groups.update(bridge)
    # A fixed, unfitted zero initial prediction makes a q-reset control exact.
    selected_set={(n,s) for n,s,_ in selected}
    for p in cfg['synaptic_points']:
        if p['type']=='postsynaptic' and (p['neuron_id'],p['synapse_id']) in selected_set:
            p['u_i']['info']=0.
    # Replace only the appended generic consumer's wiring with antagonist motor
    # anatomy. No learned readout or class-to-action mapping is inserted.
    motors=groups.pop('prediction_consumer');groups['muscle']=motors
    cfg['connections']=[c for c in cfg['connections'] if c['target_neuron'] not in motors]
    cfg['synaptic_points']=[p for p in cfg['synaptic_points']
        if not(p['type']=='postsynaptic' and p['neuron_id'] in motors)]
    for j,nid in enumerate(motors):
        for n in cfg['neurons']:
            if n['id']==nid:n['params']['lambda_param']=4.;n['params']['num_inputs']=2;n['metadata']['role']='muscle'
        for sid,(src,w) in enumerate([(groups['prediction'][1-j],1.),(groups['joint'][1-j],.5)]):
            cfg['synaptic_points'].append(k.syn(nid,sid,w,adapt=[0.,0.]))
            cfg['connections'].append(k.conn(src,nid,sid))
    return cfg, groups, selected


class Arm:
    def __init__(self):
        self.model=mujoco.MjModel.from_xml_string(XML)
        self.data=mujoco.MjData(self.model)
        self.spec=mujoco.mjtState.mjSTATE_INTEGRATION
        self.size=mujoco.mj_stateSize(self.model,self.spec)
    def state(self):
        out=np.empty(self.size);mujoco.mj_getState(self.model,self.data,out,self.spec);return out
    def restore(self,value):
        mujoco.mj_setState(self.model,self.data,value,self.spec);mujoco.mj_forward(self.model,self.data)
    def step(self,command,force):
        self.data.ctrl[0]=command;self.data.qfrc_applied[0]=force
        mujoco.mj_step(self.model,self.data)


def physical_drive(features,context,clip,rel,active=300,reverse=False):
    """World stimulus and measured applied load, never an action instruction."""
    values=np.zeros(198)
    values[192+context]=1. # Two opponent photoreceptors viewing the context lamp.
    force=0.
    if rel<active:
        values[:96]=features[clip]['visual'][rel % len(features[clip]['visual'])]
        values[96:192]=features[clip]['auditory'][rel % len(features[clip]['auditory'])]
        force=FORCE*(1 if (clip ^ context ^ int(reverse))==0 else -1)
    values[194]=max(0.,force/FORCE);values[195]=max(0.,-force/FORCE)
    return values,force


def course(net,arm,features,groups,selected,context,clip,ticks=364,reverse=False):
    neurons=list(net.network.neurons.values()); ids=[n.id for n in neurons]
    ext=groups['vision']+groups['audio']+groups['context']+groups['force']+groups['joint']
    predictors=[net.network.neurons[i] for i in groups['prediction']]
    before=arm.state(); q0=np.array([[n.postsynaptic_points[s].u_i.info for s in n.prediction_ports] for n in predictors])
    x0=np.array([n.prediction_context.copy() for n in predictors])
    e0=np.array([n.prediction_error for n in predictors])
    cells=[];body=[];drive=[];weights=[];arrivals=[];errors=[];eta=[];phys=[]
    for rel in range(ticks):
        v,force=physical_drive(features,context,clip,rel,reverse=reverse)
        angle=float(arm.data.qpos[0]);v[196:198]=[max(0.,angle),max(0.,-angle)]
        for nid,value in zip(ext,v):net.set_external_input(nid,0,float(value))
        net.run_tick()
        command=net.network.neurons[groups['muscle'][0]].O-net.network.neurons[groups['muscle'][1]].O
        arm.step(command,force)
        cells.append(cellular(neurons));body.append([arm.data.time,arm.data.qpos[0],arm.data.qvel[0],command,force])
        drive.append(v);phys.append(arm.state())
        weights.append([[n.postsynaptic_points[s].u_i.info for s in n.prediction_ports] for n in predictors])
        arrivals.append([n.prediction_arrivals.copy() for n in predictors])
        errors.append([[n.prediction_error_used,n.prediction_error_arrival,n.prediction_error] for n in predictors])
        eta.append([n.prediction_eta for n in predictors])
    data={key:np.asarray(value) for key,value in dict(cells=cells,body=body,drive=drive,
        weights=weights,arrivals=arrivals,errors=errors,eta=eta,physical_states=phys).items()}
    data.update(neuron_ids=np.array(ids),body_initial=before,weights_initial=q0,
        context_initial=x0,error_initial=e0,
        prediction_ids=np.array(groups['prediction']),
        context_source_ids=np.array([[src for target,_,src in selected if target==nid]
                                    for nid in groups['prediction']]))
    if any(not np.isfinite(a).all() for a in data.values()):raise ValueError('Nonfinite record')
    return data


def verify_physics(data):
    arm=Arm();arm.restore(data['body_initial']);residual=0.
    for i,row in enumerate(data['body']):
        arm.step(float(row[3]),float(row[4]))
        residual=max(residual,float(np.max(abs(arm.state()-data['physical_states'][i]))))
    if residual>2e-12:raise ValueError(f'Physical replay differs: {residual}')
    return residual


def verify_learning(data):
    """Independent recurrence, including strictly positive local rates.

    This checks the selected learning law from actual recorded arrivals. It is
    not a reconstruction of every upstream cell or a functional acceptance test.
    """
    q=data['weights_initial'].copy();x=data['context_initial'].copy();e=data['error_initial'].copy()
    dx=math.exp(-1/8);de=math.exp(-1/4);residual=0.
    for t,a in enumerate(data['arrivals']):
        x=dx*x+(1-dx)*a
        rate=1e-5*(1+499*abs(e)/(.01+abs(e)))
        q=np.minimum(1.,np.maximum(0.,q+rate[:,None]*e[:,None]*x))
        residual=max(residual,float(np.max(abs(q-data['weights'][t]))),
            float(np.max(abs(rate-data['eta'][t]))),
            float(np.max(abs(e-data['errors'][t,:,0]))))
        e=de*e+(1-de)*data['errors'][t,:,1]
        residual=max(residual,float(np.max(abs(e-data['errors'][t,:,2]))))
    if residual>2e-12 or np.any(data['eta']<=0):raise ValueError(f'Learning reconstruction differs: {residual}')
    return residual


def run(output,seed=11,contextual=True,reverse=False,repeats=4):
    output=Path(output).resolve()
    if not 1<=repeats<=8:raise ValueError('Bounded acquisition: 1..8 repetitions per context')
    if shutil.disk_usage(output.parent).free<3*1024**3:raise OSError('Need 3 GiB reserve')
    source=Path(__file__).resolve().parents[3]/'.live/research/20260908_bounded_learning_paired_seed11'
    features=[];physical_sources={}
    for clip in (0,1):
        p=source/f'sensory-{clip}.npz';physical_sources[str(p)]=digest(p)
        with np.load(p) as z:features.append({k:z[k] for k in ('visual','auditory')})
    cfg,groups,selected=configure(seed,contextual)
    output.mkdir(exist_ok=False)
    (output/'config.json').write_text(encode(cfg)+'\n')
    hashes=fingerprint()
    for obj in (run,append_predictive_bridge,cellular,fresh):
        p=Path(inspect.getfile(obj)).resolve();hashes[str(p)]=digest(p)
    m=dict(seed=seed,contextual=contextual,reverse=reverse,repeats=repeats,
        groups=groups,selected=selected,source_hashes=hashes,physical_sources=physical_sources,
        cell_fields=FIELDS,body_fields=['time_s','angle_rad','velocity_rad_s','command','applied_torque_Nm'],
        xml=XML,neural_tick_seconds=DT,assignment='positive iff clip XOR context XOR reverse == 0',
        limitations='Two real clips and two externally signalled contexts. Force sensors directly transduce imposed load. '
        'Context gating is specified, not discovered. Body is a one-joint MuJoCo preparation. '
        'No semantic recognition, autonomous context inference, mammal-level breadth or consciousness claim. '
        'All learning stays positive. Context-blind comparison preserves connections but zeros context inhibition; '
        'it is not an activity-matched control. Display snapshots are not complete intracellular trajectories.')
    (output/'manifest.json').write_text(encode(m)+'\n')
    net,_,_,_=fresh(output/'config.json',seed,PredictiveReceptorNeuron)
    arm=Arm();began=time.perf_counter();rows=[]
    save_checkpoint(net,output/'initial.paula',sources=[__file__])
    for context in (0,1):
        for repeat in range(repeats):
            for clip in ((0,1) if repeat%2==0 else (1,0)):
                data=course(net,arm,features,groups,selected,context,clip,reverse=reverse)
                name=f'train-c{context}-r{repeat}-v{clip}.npz';np.savez_compressed(output/name,**data)
                residual=verify_physics(data)
                rows.append(dict(file=name,sha256=digest(output/name),physics_residual=residual,
                                 learning_residual=verify_learning(data)))
            print(encode(dict(stage='training',context=context,repeat=repeat,seconds=time.perf_counter()-began)),flush=True)
        save_checkpoint(net,output/f'after-context-{context}.paula',sources=[__file__])
        np.savez_compressed(output/f'after-context-{context}-body.npz',state=arm.state())
    save_checkpoint(net,output/'final.paula',sources=[__file__])
    probes=[]
    # Same learned whole state and physical state for all probes. Weight reset
    # touches only acquired prediction ports; membranes/queues/learning survive.
    for intervention in ('intact','reset'):
        for context in (0,1):
            for clip in (0,1):
                branch=load_checkpoint(output/'final.paula',trusted=True).network
                body=Arm();body.restore(arm.state())
                if intervention=='reset':
                    for nid,sid,_ in selected:branch.network.neurons[nid].postsynaptic_points[sid].u_i.info=0.
                data=course(branch,body,features,groups,selected,context,clip,reverse=reverse)
                name=f'probe-{intervention}-c{context}-v{clip}.npz';np.savez_compressed(output/name,**data)
                residual=verify_physics(data)
                probes.append(dict(file=name,sha256=digest(output/name),physics_residual=residual,
                                   learning_residual=verify_learning(data)))
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Source changed during run')
    summary=dict(training=rows,probes=probes,seconds=time.perf_counter()-began,
        neurons=len(net.network.neurons),ticks=(4*repeats+8)*364,
        evidence='Recorded experiment. Behavioral claims require per-tick comparison and neural audit.')
    (output/'summary.json').write_text(encode(summary)+'\n');print(encode(summary),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=11)
    p.add_argument('--blind',action='store_true');p.add_argument('--reverse',action='store_true')
    p.add_argument('--repeats',type=int,default=4)
    a=p.parse_args();run(a.output,a.seed,not a.blind,a.reverse,a.repeats)
