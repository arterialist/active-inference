"""Return to added resistance after acquiring a useful released-world correction.

Resume the released/intact brain and body at tick 5328. Restore drag .8,
retaining the spring, native damping, position, velocity, gates, sensory queue,
neural state and active plasticity. Compare current predictive weights, the
actual pre-removal weights at 4304, same-age still-loaded weights at 5328,
and a selected-weight reset. All branches run 1024 ticks; replay 64 released
ticks against the previous uninterrupted record before interpreting the return.

These are local weight interventions in one continuing state, not four complete
organisms with different histories. No change cue or performance signal reaches
the brain. This tests retention/interference and readaptation within a bounded
course, not asymptotic convergence or a formal learning-savings estimate.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import context_organization as base
from .active_sweep_acquisition import load, reserve, replay_prefix, continuation
from .active_sweep_transfer import audit, replace_weights, match_initial
from .active_sweep_memory import restore, reset_selected
from .active_sweep_credit import record_credit
from .active_sweep_memory_analysis import stroke_effect
from .body_state_memory_analysis import intervals
from .crossed_av_continuation import isolated_rng
from .eligibility_reference_intervention_analysis import onset

START, TICKS, REPLAY = 5328, 1024, 64
KINDS = ('current', 'prior', 'same-age', 'reset')
TRACE_COLUMNS = ('physical_time_s', 'joint_angle_rad', 'joint_velocity_rad_s',
                 'neural_motor_command', 'environmental_torque_Nm',
                 'signed_prediction', 'cumulative_crossings', 'next_gate_sign',
                 'crossed_this_tick')


def run(parent, output):
    parent, output = (Path(p).resolve() for p in (parent, output))
    if output.exists():
        raise FileExistsError(output)
    m = json.loads((parent/'manifest.json').read_text()); g = m['groups']
    rows = {r['name']: r for r in m['rows']}
    acquired = Path(m['parent'])/'matched_cascade-to-4304.npz'
    sources = dict(m['sources'])
    sources[str(parent/'manifest.json')] = base.digest(parent/'manifest.json')
    sources[str(Path(__file__).resolve())] = base.digest(__file__)
    for name in ('released-intact', 'loaded-intact', 'readapt-intact'):
        r = rows[name]
        for key, h in (('file','sha256'), ('checkpoint','checkpoint_sha256'), ('physical','physical_sha256')):
            sources[str(parent/r[key])] = r[h]
    if str(acquired) not in sources:
        raise ValueError('Missing original acquisition provenance')
    for path, digest in sources.items():
        if base.digest(path) != digest:
            raise ValueError('Evidence changed: '+path)
    cfg = json.loads(Path(m['config']).read_text()); features = load(m['media'])
    acquired_record = load(parent/rows['released-intact']['file'])
    reference = load(parent/rows['readapt-intact']['file'])
    donors = {'current': acquired_record['weights'][-1],
              'prior': load(acquired)['weights'][-1],
              'same-age': load(parent/rows['loaded-intact']['file'])['weights'][-1]}
    output.mkdir(); reserve(output)
    manifest = dict(parent=str(parent), seed=m['seed'], groups=g, config=m['config'], media=m['media'],
                    sources=sources, start=START, ticks=TICKS, replay_ticks=REPLAY,
                    kinds=KINDS, trace_columns=TRACE_COLUMNS, limits=__doc__)
    (output/'protocol.json').write_text(base.encode(manifest)+'\n')
    results = []
    for kind in ('replay',)+KINDS:
        with isolated_rng():
            r = rows['released-intact']
            net, body, delay = restore(parent/r['checkpoint'], parent/r['physical'])
            if net.current_tick != START:
                raise ValueError('Wrong acquired clock')
            body.drag = 0. if kind == 'replay' else .8
            if kind == 'reset':
                reset_selected(net, g)
            elif kind not in ('current', 'replay'):
                replace_weights(net, g, donors[kind])
            data = record_credit(net, body, delay, features, g, REPLAY if kind == 'replay' else TICKS)
            match_initial(reference, data, weights=kind in ('current', 'replay'))
            if kind == 'reset':
                if np.any(data['weights_initial']): raise ValueError('Missing reset')
            elif not np.array_equal(data['weights_initial'], donors.get(kind, donors['current'])):
                raise ValueError('Wrong weight donor')
            if kind == 'replay':
                replay_prefix(reference, data)
            elif kind == 'current':
                continuation(acquired_record, data)
            audit(data, cfg, g, features, body.drag)
            path = output/f'{kind}.npz'; np.savez_compressed(path, **data)
            row = dict(kind=kind, file=path.name, sha256=base.digest(path), drag=body.drag,
                       start=START, end=net.current_tick)
            if kind != 'replay':
                neural = output/f'{kind}.paula'; physical = output/f'{kind}-body.npz'
                base.save_checkpoint(net, neural, sources=list(sources))
                np.savez_compressed(physical, state=body.state(), delay=delay.state(),
                                    gate=[body.crossings, body.next_gate])
                row.update(checkpoint=neural.name, checkpoint_sha256=base.digest(neural),
                           physical=physical.name, physical_sha256=base.digest(physical))
            results.append(row); reserve(output)
            print(base.encode(dict(seed=m['seed'], kind=kind, tick=net.current_tick,
                                   new_crossings=body.crossings-int(data['gate_initial'][0]))), flush=True)
    if any(base.digest(p) != h for p,h in sources.items()):
        raise ValueError('Evidence changed during run')
    manifest['rows'] = results
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')
    return manifest


def analyze(roots, output):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    sources = {}; arrays = {}; cases = []; seeds = set(); checked = 0
    for root in (Path(p).resolve() for p in roots):
        m = json.loads((root/'manifest.json').read_text()); seed=m['seed']; g=m['groups']
        if seed in seeds: raise ValueError('Duplicate seed')
        seeds.add(seed); parent=Path(m['parent'])
        for p,h in m['sources'].items():
            if base.digest(p) != h: raise ValueError('Evidence changed: '+p)
            sources[p]=h
        pm=json.loads((parent/'manifest.json').read_text()); pr={r['name']:r for r in pm['rows']}
        acquired=load(parent/pr['released-intact']['file'])
        reference=load(parent/pr['readapt-intact']['file'])
        donors={'current':acquired['weights'][-1],
                'prior':load(Path(pm['parent'])/'matched_cascade-to-4304.npz')['weights'][-1],
                'same-age':load(parent/pr['loaded-intact']['file'])['weights'][-1]}
        cfg=json.loads(Path(m['config']).read_text()); features=load(m['media'])
        if len(m['rows']) != 5 or {r['kind'] for r in m['rows']} != {'replay',*KINDS}:
            raise ValueError('Incomplete return family')
        records={}
        for row in m['rows']:
            kind=row['kind']; n=REPLAY if kind=='replay' else TICKS; drag=0. if kind=='replay' else .8
            for key,h in (('file','sha256'),('checkpoint','checkpoint_sha256'),('physical','physical_sha256')):
                if key not in row: continue
                p=root/row[key]
                if base.digest(p) != row[h]: raise ValueError('Artifact changed: '+str(p))
                sources[str(p)]=row[h]
            z=load(root/row['file']); records[kind]=z
            if (len(z['body']),row['start'],row['end'],row['drag']) != (n,START,START+n,drag):
                raise ValueError('Wrong duration or intervention')
            if round(z['body'][0,0]/.004)-1 != START: raise ValueError('Wrong physical clock')
            match_initial(reference,z,weights=kind in ('current','replay'))
            if kind=='reset':
                if np.any(z['weights_initial']): raise ValueError('Missing reset')
            elif not np.array_equal(z['weights_initial'],donors.get(kind,donors['current'])):
                raise ValueError('Wrong donor')
            audit(z,cfg,g,features,drag); checked+=n
            if kind=='replay':
                replay_prefix(reference,z); continue
            ids=list(z['neuron_ids']); pi=[ids.index(i) for i in g['prediction']]
            pred=z['cells'][:,pi[0],base.FIELDS.index('O')]-z['cells'][:,pi[1],base.FIELDS.index('O')]
            arrays[f's{seed}_{kind}']=np.column_stack((z['body'],pred,z['gate']))
        continuation(acquired,records['current'])
        contrasts={}
        for kind in ('prior','same-age','reset'):
            a,b=records['current'],records[kind]
            effect,strokes=stroke_effect(a,b,g)
            aa,bb=arrays[f's{seed}_current'],arrays[f's{seed}_{kind}']
            error=abs(aa[:,5]-aa[:,4]/.2)-abs(bb[:,5]-bb[:,4]/.2)
            arrays[f's{seed}_vs_{kind}_stroke']=effect
            arrays[f's{seed}_vs_{kind}_error']=error
            contrasts[kind]=dict(strokes=strokes,larger_error=intervals(error>0),smaller_error=intervals(error<0),
                greater_advance=intervals(effect>0),lesser_advance=intervals(effect<0),
                first_difference={k:onset(a[k],b[k]) for k in ('weights','cells','body','drive','errors')})
        cases.append(dict(seed=seed,contrasts=contrasts,
            world_change_onset={k:onset(records['current'][k][:len(reference['body'])],reference[k])
                                for k in ('body','raw_afferents','drive','cells','weights','errors')},
            gates={kind:np.flatnonzero(records[kind]['gate'][:,2]).tolist() for kind in KINDS}))
        sources[str(root/'manifest.json')]=base.digest(root/'manifest.json')
    if seeds != {11,23,44,77}: raise ValueError('Need all four declared seeds')
    output.mkdir(); np.savez_compressed(output/'per-tick.npz',**arrays)
    result=dict(cases=cases,seeds=sorted(seeds),sources=sources,checked_ticks=checked,
                exact_replay_ticks=4*REPLAY,trace_columns=TRACE_COLUMNS,analyzer_sha256=base.digest(__file__),limits=__doc__)
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(checked_ticks=checked,exact_replay_ticks=4*REPLAY,cases=cases)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('paths',nargs='+',type=Path)
    p.add_argument('--output',required=True,type=Path); p.add_argument('--analyze',action='store_true'); a=p.parse_args()
    if a.analyze: analyze(a.paths,a.output)
    elif len(a.paths)==1: run(a.paths[0],a.output)
    else: p.error('One completed transfer family per worker')
