"""Diagnose a learned context-terminal sign change, without installing a fix.

Branch the complete acquired neural/body/delay state. Replay the original four
probes exactly, then restore ONLY the context terminal's release coefficient
to its birth value. All adaptation remains active. This is a causal lesion
control, not a corrected learning rule or acceptance of the whole preparation.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import context_organization as base
from . import crossed_av_world as world
from .crossed_av_continuation import restore_acquired, isolated_rng
from .opponent_context import verify_afferents
from .temporal_verification import verify_learning


def observed_course(net, arm, delay, features, groups, selected, video, audio):
    context=groups['context'][0]
    neuron=net.network.neurons[context]
    terminals=list(neuron.presynaptic_points)
    if len(terminals)!=1:raise ValueError('Expected the diagnosed single context terminal')
    terminal_id=terminals[0];terminal=neuron.presynaptic_points[terminal_id]
    original=net.run_tick;record=[]
    def observed_tick():
        slot=net.current_tick % net.wheel_size
        events=[s.event for s in net.retrograde_wheel[slot]
                if s.event.target_neuron_id==context and s.event.target_terminal_id==terminal_id]
        before=terminal.u_o.info
        expected=before
        for e in events:
            expected+=neuron.params.eta_retro*e.error_vector[0]
            expected=np.clip(expected,-100.,100.)
        arriving=sum(float(s.event[2]) for s in net.presynaptic_wheel[slot]
                     if isinstance(s.event,tuple) and s.event[:2]==(context,terminal_id))
        result=original()
        after=terminal.u_o.info
        if after!=expected:raise ValueError('Native terminal update reconstruction failed')
        record.append([before,after,len(events),arriving])
        return result
    net.run_tick=observed_tick
    try:
        data=world.course(net,arm,features,groups,selected,video,audio,ticks=96,delay=delay)
    finally:net.run_tick=original
    return data,np.array(record)


def run(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
    if s['blocks']!=16 or m['reverse']:raise ValueError('Need completed normal sixteen-block course')
    for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
        if base.digest(p)!=h:raise ValueError('Source changed')
    checkpoint=s['checkpoints'][-1]
    if (base.digest(root/checkpoint['neural'])!=checkpoint['neural_sha256'] or
            base.digest(root/checkpoint['physical'])!=checkpoint['physical_sha256']):
        raise ValueError('Final executable state changed')
    parent=Path(m['parent'])
    if base.digest(parent/'initial.paula')!=m['parent_evidence']['initial.paula']:
        raise ValueError('Birth reference changed')
    with isolated_rng():birth=base.load_checkpoint(parent/'initial.paula',trusted=True).network
    context=m['groups']['context'][0]
    terminal_id=next(iter(birth.network.neurons[context].presynaptic_points))
    birth_release=birth.network.neurons[context].presynaptic_points[terminal_id].u_o.info
    features=[]
    for clip in (0,1):
        paths=[p for p in m['physical_sources'] if Path(p).name==f'sensory-{clip}.npz']
        if len(paths)!=1:raise ValueError('Ambiguous media')
        with np.load(paths[0]) as z:features.append({k:z[k] for k in ('visual','auditory')})
    output.mkdir()
    manifest=dict(source=str(root),seed=m['seed'],groups=m['groups'],
        context_neuron=context,terminal_id=terminal_id,birth_release=float(birth_release),
        source_manifest_sha256=base.digest(root/'manifest.json'),
        source_summary_sha256=base.digest(root/'summary.json'),producer_sha256=base.digest(__file__),
        terminal_columns=['release_before','release_after','retrograde_event_count','arriving_release'],
        intervention='Restore one presynaptic release coefficient to birth; retain all other acquired state and learning.',
        limits=__doc__)
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')
    probes=[]
    for restored in (False,True):
        for v,a in ((0,0),(0,1),(1,0),(1,1)):
            with isolated_rng():
                net,arm,delay=restore_acquired(root/checkpoint['neural'],root/checkpoint['physical'])
                if restored:net.network.neurons[context].presynaptic_points[terminal_id].u_o.info=birth_release
                data,trace=observed_course(net,arm,delay,features,m['groups'],m['selected'],v,a)
            if not restored:
                ref=next(r for r in s['probes'] if (r['blocks'],r['kind'],r['weights'],r['video'],r['audio'])==
                         (16,'acquired','learned',v,a))
                if base.digest(root/ref['file'])!=ref['sha256']:raise ValueError('Replay reference changed')
                with np.load(root/ref['file']) as z:
                    if set(z.files)!=set(data) or any(not np.array_equal(z[k],data[k]) for k in z.files):
                        raise ValueError('Observer changes original course')
            residuals=dict(learning=verify_learning(data),physics=base.verify_physics(data),
                           afferents=verify_afferents(data))
            name=f'gate{int(restored)}-v{v}-a{a}.npz'
            np.savez_compressed(output/name,**data,context_terminal=trace)
            probes.append(dict(file=name,sha256=base.digest(output/name),restored=restored,
                               video=v,audio=a,residuals=residuals))
    result=dict(probes=probes,executed_ticks=768,unchanged_replays_exact=True)
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(seed=m['seed'],executed_ticks=768,unchanged_replays_exact=True)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.source,a.output)
