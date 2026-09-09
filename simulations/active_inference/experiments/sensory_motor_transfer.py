"""Does retained sensorimotor organization transfer across real sensory backgrounds?

The existing 608-cell phase-dependent brain branches at its acquired tick 6352.
The actual mechanics and normal air stay fixed. Independently replace vision
and sound by the second recorded clip, retaining the original global playback
phase. Cross each background with intact versus zeroed selected predictive
weights. All other acquired state, native returns and positive learning remain.
There is no new circuit, context lamp, teaching signal or behavior criterion in
the brain. Clip identity selects external media only, never a motor action.

Each course is 1024 ticks: a bounded transfer/expression probe with continuing
learning, not enough to establish asymptotic adaptation. These two recordings
are not matched in sensory dose. Differences cannot establish semantic novelty
or audiovisual recognition. The aim is functional transfer of an acquired
sensorimotor organization, not invariance of every sensory neuron to new input.
"""
import argparse
import inspect
import json
from pathlib import Path
import shutil

import numpy as np

from . import context_organization as base
from .active_sweep_memory import reset_selected
from .ventilation_regulation import record
from .ventilation_regulation_replay import restore, exact_prefix
from .ventilation_regulation_audit import audit as audit_full
from .ventilation_changing_air import prefix
from .ventilation_screen import observations


BACKGROUNDS = {'familiar': (0, 0), 'vision-changed': (1, 0),
               'sound-changed': (0, 1), 'both-changed': (1, 1)}
MEMORIES = ('retained', 'reset')
TICKS = 1024


def select_media(features, background):
    v, a = BACKGROUNDS[background]
    result = {'visual': features[v]['visual'], 'auditory': features[a]['auditory']}
    for field, values in result.items():
        if values.shape != (300, 96) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('Invalid recorded sensory stream: '+field)
    return result


def audit(z, cfg, groups, meta, features):
    # Reuse the existing fixed-world full added-path auditor. Its observation
    # list must include the five appended cells, just as the recorder does.
    # This changes analyst metadata only, never runtime cells or equations.
    observed_cfg = dict(cfg, metadata={**cfg['metadata'], 'ventilation_feedback': meta})
    return audit_full(z, observed_cfg, groups, features)


def reserve(output):
    if shutil.disk_usage(output).free < 3*1024**3:
        raise OSError('Three GiB free-space reserve reached')
    if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > 384*1024**2:
        raise OSError('Per-seed 384 MiB artifact budget reached')


def run(parent, media_root, output):
    parent, media_root, output = (Path(p).resolve() for p in (parent, media_root, output))
    if output.exists():
        raise FileExistsError(output)
    m = json.loads((parent/'manifest.json').read_text())
    r = next(r for r in m['rows'] if r['condition'] == 'authorized')
    folder = parent/'authorized'; cfg = json.loads((folder/'config.json').read_text())
    groups, meta = m['groups'], r['meta']
    if m['start'] != 6352 or m['ticks'] != 2048 or len(cfg['neurons']) != 608:
        raise ValueError('Need the completed full-size phase-composition parent')
    sources = dict(m['sources'])
    sources[str(parent/'manifest.json')] = base.digest(parent/'manifest.json')
    for name, h in r['files'].items():
        sources[str(folder/name)] = h
    for obj in (run, audit_full, record, restore, reset_selected, prefix):
        p = str(Path(inspect.getfile(obj)).resolve()); sources[p] = base.digest(p)
    media_manifest = json.loads((media_root/'manifest.json').read_text())
    sources[str(media_root/'manifest.json')] = base.digest(media_root/'manifest.json')
    features = []
    for i, media in enumerate(media_manifest['media']):
        path = media_root/f'sensory-{i}.npz'
        sources[str(path)] = base.digest(path)
        sources[media['path']] = media['sha256']
        with np.load(path) as z:
            if str(z['source_sha256']) != media['sha256']:
                raise ValueError('Encoded stimulus has a different source')
            features.append({k: z[k] for k in ('visual', 'auditory')})
    if len(features) != 2:
        raise ValueError('Need the two original real-media recordings')
    for p, h in sources.items():
        if base.digest(p) != h:
            raise ValueError('Changed evidence: '+p)
    output.mkdir(); reserve(output)
    protocol = dict(parent=str(parent), seed=m['seed'], start=6352, ticks=TICKS,
        groups=groups, meta=meta, media_root=str(media_root), media=media_manifest['media'],
        backgrounds=BACKGROUNDS, memories=MEMORIES, sources=sources, limits=__doc__)
    (output/'protocol.json').write_text(base.encode(protocol)+'\n')
    (output/'config.json').write_text(base.encode(cfg)+'\n')
    rows = []
    for background in BACKGROUNDS:
        stimulus = select_media(features, background)
        for memory in MEMORIES:
            reserve(output)
            net, body, delay, od = restore(folder/'initial.paula', folder/'initial-body.npz', groups)
            if net.current_tick != 6352 or any(n.params.eta_post <= 0 or n.params.eta_retro <= 0
                                             for n in net.network.neurons.values()):
                raise ValueError('Wrong acquired state or frozen learning')
            tag = background+'-'+memory
            changed = reset_selected(net, groups) if memory == 'reset' else np.empty((0, 3))
            data = record(net, body, delay, od, stimulus, groups, meta, TICKS)
            path = output/(tag+'.npz'); np.savez_compressed(path, **data)
            # Keep evidence before checking, including a failed neural audit.
            residual = audit(data, cfg, groups, meta, stimulus)
            exact = 0
            if background == 'familiar' and memory == 'retained':
                with np.load(folder/'ticks.npz') as z:
                    old = {k: z[k] for k in z.files if k != 'air_fraction'}
                exact_prefix(old, prefix(data, 512)); exact = 512
                del old
            neural = output/(tag+'.paula'); physical = output/(tag+'-body.npz')
            base.save_checkpoint(net, neural, sources=list(sources))
            np.savez_compressed(physical, state=body.state(), delay=delay.state(), organ=body.organs.state(),
                organ_delay=od.state(), gate=[body.crossings, body.next_gate])
            item = dict(background=background, memory=memory, file=path.name, sha256=base.digest(path),
                checkpoint=neural.name, checkpoint_sha256=base.digest(neural), physical=physical.name,
                physical_sha256=base.digest(physical), reset_weights=changed,
                exact_previous_ticks=exact, residual=residual, observations=observations(data))
            rows.append(item)
            (output/'progress.json').write_text(base.encode(dict(completed=[r['file'] for r in rows]))+'\n')
            print(base.encode(dict(seed=m['seed'], background=background, memory=memory,
                first_debt={k: v for k, v in item['observations'].items() if k.endswith('_tick')})), flush=True)
            del net, body, delay, od, data
    if any(base.digest(p) != h for p, h in sources.items()):
        raise ValueError('Source or input changed during course')
    result = dict(protocol, rows=rows, checked_ticks=TICKS*len(rows), config_sha256=base.digest(output/'config.json'))
    (output/'manifest.json').write_text(base.encode(result)+'\n')
    return result


def analyze(roots, output):
    """Retain every resource contrast, neural onset and failed interval."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .ventilation_clamp_analysis import first

    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    cases, arrays, sources, seen = [], {}, {}, set()
    figures = {name: plt.subplots(4,4,figsize=(14,9),sharex=True,sharey='row',layout='constrained')
               for name in BACKGROUNDS}
    for column, root in enumerate(map(lambda p:Path(p).resolve(), roots)):
        if column >= 4: raise ValueError('Expected four seeds')
        m = json.loads((root/'manifest.json').read_text()); seed = m['seed']; g = m['groups']
        if seed in seen or m['ticks'] != TICKS: raise ValueError('Repeated seed or wrong duration')
        seen.add(seed)
        expected = {(b,w) for b in BACKGROUNDS for w in MEMORIES}
        if len(m['rows']) != 8 or {(r['background'],r['memory']) for r in m['rows']} != expected:
            raise ValueError('Incomplete transfer family')
        for p,h in m['sources'].items():
            if base.digest(p) != h: raise ValueError('Changed source: '+p)
            sources[p] = h
        sources[str(root/'manifest.json')] = base.digest(root/'manifest.json')
        if base.digest(root/'config.json') != m['config_sha256']: raise ValueError('Changed graph')
        cfg = json.loads((root/'config.json').read_text()); features=[]
        for i in (0,1):
            with np.load(Path(m['media_root'])/f'sensory-{i}.npz') as f:
                features.append({k:f[k] for k in ('visual','auditory')})
        familiar, results = {}, []
        for background in BACKGROUNDS:
            selected, initials = {}, {}
            fig, axes = figures[background]
            for memory,color,style in (('retained','#126ca4','-'),('reset','#a65324','--')):
                r = next(r for r in m['rows'] if (r['background'],r['memory']) == (background,memory))
                for k,h in (('file','sha256'),('checkpoint','checkpoint_sha256'),('physical','physical_sha256')):
                    p = root/r[k]
                    if base.digest(p) != r[h]: raise ValueError('Changed artifact: '+str(p))
                    sources[str(p)] = r[h]
                with np.load(root/r['file']) as f: z={k:f[k] for k in f.files}
                if len(z['body']) != TICKS: raise ValueError('Truncated record')
                residual = audit(z,cfg,g,m['meta'],select_media(features,background))
                initials[memory] = {k:v for k,v in z.items() if k.endswith('_initial')}
                ids = list(z['neuron_ids']); oi = base.FIELDS.index('O')
                roles = {role:z['cells'][:,[ids.index(n) for n in g[role]],oi].copy()
                         for role in ('vision','audio','mixed_0','mixed_1','prediction','muscle','cpg')}
                current = {k:z[k].copy() for k in ('body','organs','organ_drive','drive','weights','errors')}
                current.update(roles); selected[memory] = current
                if background == 'familiar': familiar[memory] = current
                changes = {k:first(familiar[memory][k],v) for k,v in current.items()}
                bounds = np.abs(z['cells'][:,:,base.FIELDS.index('S')]) >= 1000
                key = f's{seed}_{background}_{memory}'
                for k in ('body','organs','prediction','muscle','errors'):
                    arrays[key+'_'+k] = current[k]
                results.append(dict(background=background,memory=memory,residual=residual,
                    first_change_from_familiar=changes,
                    first_debt={label:(int(a[0]) if len(a) else None)
                        for label,j in (('oxygen',4),('energy',10))
                        for a in [np.flatnonzero(z['organs'][:,j]>1e-12)]},
                    debt_increment_ticks={label:np.flatnonzero(np.diff(np.r_[z['organ_initial'][j],z['organs'][:,j]])>1e-12).tolist()
                        for label,j in (('oxygen',4),('energy',10))},
                    minimum_reserves=z['organs'][:,[0,6]].min(axis=0),final_organs=z['organs'][-1],
                    bound_events=[dict(neuron=int(z['neuron_ids'][j]),ticks=np.flatnonzero(bounds[:,j]).tolist())
                                  for j in np.flatnonzero(bounds.any(axis=0))],
                    selected_weight_change_l1=float(abs(z['weights'][-1]-z['weights_initial']).sum())))
                pred = roles['prediction'][:,0]-roles['prediction'][:,1]
                values = (z['body'][:,1],pred,z['organs'][:,0],z['organs'][:,6])
                for i,v in enumerate(values):
                    axes[i,column].plot((np.arange(TICKS)+1)*.004,v,color=color,ls=style,lw=1.,label=memory)
                del z
            for k,v in initials['retained'].items():
                if k != 'weights_initial': np.testing.assert_array_equal(v,initials['reset'][k],err_msg=k)
            if np.any(initials['reset']['weights_initial']): raise ValueError('Missing selected reset')
            a,b = selected['retained'],selected['reset']
            key = f's{seed}_{background}_retained_minus_reset'
            arrays[key+'_reserves'] = a['organs'][:,[0,6]]-b['organs'][:,[0,6]]
            arrays[key+'_body'] = a['body']-b['body']
            results.append(dict(background=background,comparison='retained-minus-reset',
                first_difference={k:first(a[k],b[k]) for k in a}))
            axes[0,column].set_title(f'Seed {seed}'); axes[-1,column].set_xlabel('Time after branching (s)')
        cases.append(dict(seed=seed,groups=g,results=results))
    if seen != {11,23,44,77}: raise ValueError('Need all four seeds')
    output.mkdir(); np.savez_compressed(output/'per-tick.npz',**arrays)
    for name,(fig,axes) in figures.items():
        for i,label in enumerate(('Joint angle (rad)','Signed prediction','Oxygen reserve (mL)','Energy reserve (J)')):
            axes[i,0].set_ylabel(label)
        for ax in axes.flat:
            ax.axhline(0,color='#bbbbbb',lw=.5);ax.spines[['top','right']].set_visible(False)
        fig.suptitle(name+': same mechanics, same acquired brain; selected prediction weights retained or reset\n'
                     'Both branches continue adapting. Media are unrelated to the physical load.',fontsize=12)
        h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2,frameon=False)
        fig.savefig(output/(name+'.png'),dpi=150);plt.close(fig)
    result = dict(cases=cases,checked_ticks=4*8*TICKS,sources=sources,limits=__doc__,
        analyzer_sha256=base.digest(__file__),files={p.name:base.digest(p) for p in output.iterdir()})
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(checked_ticks=result['checked_ticks'],cases=[dict(seed=c['seed'],results=[
        {k:r[k] for k in ('background','memory','first_debt','minimum_reserves')}
        for r in c['results'] if 'memory' in r]) for c in cases])),flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('paths',nargs='+');p.add_argument('--analyze',action='store_true');p.add_argument('--output')
    a = p.parse_args()
    if a.analyze and a.output: analyze(a.paths,a.output)
    elif not a.analyze and len(a.paths)==3 and a.output is None: run(*a.paths)
    else: p.error('Run: PARENT MEDIA_ROOT OUTPUT; analyze: ROOTS --analyze --output OUTPUT')
