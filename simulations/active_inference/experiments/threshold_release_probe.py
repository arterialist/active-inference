"""Pre-spike observation and single-release intervention in a recorded network.

Diagnostic host instrumentation only, never a neural controller or accepted
brain component. The trace reads the running native frame before spike reset.
The release intervention removes scheduled outgoing events after normal local
learning, preserving RNG draws, the soma's spike and all retrograde messages.
No shared neuron source is edited. Use in a dedicated single-threaded worker.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
import gzip
import inspect
import json
from pathlib import Path
import sys
import textwrap
import time

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode, network_module
from neuron.neuron import Neuron
from neuron.extensions.experimental.eligibility_trace import EligibilityTraceNeuron
from .eligibility_association_probe import dynamic_snapshot
from .eligibility_media_probe import record
from .eligibility_media_state import assignment_values
from .multimodal_pairing_probe import fresh


class ThresholdObserver:
    def __init__(self, target):
        self.target, self.rows = target, []

    @contextmanager
    def observe(self):
        original = Neuron.tick
        lines, offset = inspect.getsourcelines(original)
        tree = ast.parse(textwrap.dedent(''.join(lines)))
        matches = [n.lineno+offset-1 for n in ast.walk(tree)
                   if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'will_fire']
        if len(matches) != 1: raise ValueError('Native firing observation point changed')
        line = matches[0]; observer = self
        def observed(n, ext, tick, dt=1.):
            if n is not observer.target: return original(n,ext,tick,dt)
            if sys.gettrace() is not None: raise RuntimeError('Do not replace an existing debugger/trace')
            captured = []
            def trace(frame,event,arg):
                if frame.f_code is not original.__code__: return None
                if event == 'line' and frame.f_lineno == line:
                    v = frame.f_locals
                    captured.append(dict(tick=int(tick),neuron=n.id,pre_S=float(n.S),old_S=float(v['old_S']),
                        I_t=float(v['I_t']),dS=float(v['dS']),active_threshold=float(v['active_threshold']),
                        r=float(n.r),b=float(n.b),c=int(n.params.c),dt=float(dt),lambda_param=float(n.params.lambda_param),
                        since_last=None if not np.isfinite(v['time_since_last_fire']) else float(v['time_since_last_fire']),
                        will_fire=bool(v['will_fire']),S_dtype=str(np.asarray(n.S).dtype),
                        I_dtype=str(np.asarray(v['I_t']).dtype)))
                return trace
            sys.settrace(trace)
            try: events=original(n,ext,tick,dt)
            finally: sys.settrace(None)
            if len(captured)!=1: raise AssertionError('Expected exactly one pre-spike observation')
            observer.rows.extend(captured)
            return events
        Neuron.tick=observed
        try: yield self
        finally: Neuron.tick=original


class ReleaseDriver:
    def __init__(self, net, source, tick, block=False):
        self.net,self.source,self.tick,self.block=net,source,tick,block
        self.evidence=None

    def do_tick(self):
        tick=self.net.current_tick
        result=self.net.run_tick()
        if tick != self.tick: return result
        before=json.loads(dynamic_snapshot(self.net))
        removed=[]; found=[]
        for slot in self.net.presynaptic_wheel:
            for signal in list(slot):
                e=signal.event
                if isinstance(e,tuple) and len(e)==3 and e[0]==self.source and signal.arrival_tick==tick+1:
                    found.append([signal.arrival_tick,'tuple',list(e)])
                    if self.block:
                        slot.remove(signal);removed.append([signal.arrival_tick,'tuple',list(e)])
        expected=deepcopy(before)
        for row in removed: expected['presynaptic_wheel'].remove(row)
        after=json.loads(dynamic_snapshot(self.net))
        if after != expected: raise AssertionError('Undeclared release intervention')
        self.evidence=dict(tick=tick,source=self.source,block=self.block,found=found,removed=removed,
                           before=before,after=after)
        result['traveling_signals']=sum(map(len,self.net.presynaptic_wheel))+sum(map(len,self.net.retrograde_wheel))
        return result


def read_state(path):
    with gzip.open(path,'rt') as f: return json.load(f)


def run(source, opposite, reference, output, neuron=446, clip=0, probe_tick=25):
    source,opposite,reference,output=map(lambda p:Path(p).resolve(),(source,opposite,reference,output))
    m,other=[json.loads((p/'manifest.json').read_text()) for p in (source,opposite)]
    if m['seed']!=other['seed'] or {m['mapping'],other['mapping']}!={'paired','swapped'} or (source/'config.json').read_bytes()!=(opposite/'config.json').read_bytes():
        raise ValueError('Unmatched acquisition states')
    ref=json.loads((reference/'manifest.json').read_text())
    if Path(ref['source_recording'])!=source or ref['mode']!='factors': raise ValueError('Wrong factor reference')
    if network_module.MIN_CONNECTION_SIGNAL_TRAVEL_TICKS!=1 or network_module.MAX_CONNECTION_SIGNAL_TRAVEL_TICKS!=1:
        raise ValueError('Requires fixed one-tick cleft delay')
    if clip not in (0,1) or not 0<=probe_tick<m['clip_ticks']: raise ValueError('Invalid probe')
    hashes={**m['source_hashes'],str(Path(__file__).resolve()):digest(__file__),
            str(Path(inspect.getfile(assignment_values)).resolve()):digest(inspect.getfile(assignment_values))}
    if any(digest(p)!=h for p,h in hashes.items()): raise ValueError('Acquisition runtime changed')
    files={str(p):digest(p) for root in (source,opposite,reference) for p in (root/'manifest.json',root/'summary.json')}
    for root in (source,opposite):
        for name in ('config.json','training-final-state.json.gz'):files[str(root/name)]=digest(root/name)
    output.mkdir(parents=True,exist_ok=False)
    conditions=('unchanged','opposite_mean','opposite_residual','opposite_selected','residual_block')
    (output/'manifest.json').write_text(encode(dict(source=str(source),opposite=str(opposite),reference=str(reference),
        source_hashes=hashes,source_files_sha256=files,neuron=neuron,clip=clip,probe_tick=probe_tick,conditions=conditions,
        scope='One preselected conditional-expression counterexample. Release block is diagnostic host intervention, not brain code.'))+'\n')
    cfg=source/'config.json';ports=m['selected_ports'];features=[]
    for k in (0,1):
        with np.load(Path(m['source_recording'])/f'sensory-{k}.npz') as z:features.append({key:z[key] for key in z.files})
    net,core,members,points=fresh(cfg,m['seed'],EligibilityTraceNeuron)
    if neuron not in net.network.neurons: raise ValueError('Unknown neuron')
    started=time.perf_counter()
    for index,trial in enumerate(m['trials']):
        data=record(net,core,members,points,features,m['groups'],trial,ports)
        with np.load(source/f'experience-{index:03d}.npz') as z:
            if any(not np.array_equal(data[k],z[k]) for k in data):raise AssertionError('Acquisition replay differs')
        if index%8==7:print(encode(dict(training=index+1,seconds=time.perf_counter()-started)),flush=True)
    parent=dynamic_snapshot(net);original=json.loads(parent);donor=read_state(opposite/'training-final-state.json.gz')
    if original!=read_state(source/'training-final-state.json.gz'):raise AssertionError('Full trained state differs')
    trial=dict(start=net.current_tick,stop=net.current_tick+m['clip_ticks'],visual_clip=clip,audio_clip=None)
    rows=[]
    for condition in conditions:
        branch=deepcopy(net);expected=deepcopy(original)
        factor='opposite_residual' if condition=='residual_block' else condition
        if factor!='unchanged':
            values=({(n,s):donor['neurons'][str(n)]['synapses'][str(s)][0] for n,s,_ in ports} if factor=='opposite_selected'
                    else assignment_values(original,donor,ports,factor))
            for (n,s),v in values.items():
                branch.network.neurons[n].postsynaptic_points[s].u_i.info=v
                expected['neurons'][str(n)]['synapses'][str(s)][0]=v
        if json.loads(dynamic_snapshot(branch))!=expected:raise AssertionError('Wrong branch state')
        members=list(branch.network.neurons.values());points=[p for n in members for p in n.postsynaptic_points.values()]
        observer=ThresholdObserver(branch.network.neurons[neuron])
        driver=ReleaseDriver(branch,neuron,trial['start']+probe_tick,condition=='residual_block')
        with observer.observe():data=record(branch,driver,members,points,features,m['groups'],trial,ports)
        if condition!='residual_block':
            with np.load(reference/f'{condition}-{clip}.npz') as z:
                if any(not np.array_equal(data[k],z[k]) for k in data):raise AssertionError('Observation changed factor replay')
        elif not driver.evidence['removed']:raise AssertionError('Target release did not occur')
        name=condition+'.npz';np.savez_compressed(output/name,**data)
        detail=condition+'.json.gz'
        with gzip.open(output/detail,'wt') as f:f.write(encode(dict(start=expected,thresholds=observer.rows,release=driver.evidence,final=json.loads(dynamic_snapshot(branch))))+'\n')
        rows.append(dict(condition=condition,file=name,sha256=digest(output/name),detail=detail,detail_sha256=digest(output/detail)))
        if dynamic_snapshot(net)!=parent:raise AssertionError('Parent changed')
        print(encode(dict(branch=condition,seconds=time.perf_counter()-started)),flush=True)
    if any(digest(p)!=h for p,h in {**hashes,**files}.items()):raise ValueError('Sources changed')
    result=dict(branches=rows,training_exact=True,observer_controls_exact=True,trial=trial,
                ticks=trial['start']+len(rows)*m['clip_ticks'],seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(encode(result)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','opposite','reference','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--neuron',type=int,default=446);p.add_argument('--clip',type=int,default=0);p.add_argument('--probe-tick',type=int,default=25)
    a=p.parse_args();run(a.source,a.opposite,a.reference,a.output,a.neuron,a.clip,a.probe_tick)
