"""Full-trajectory challenge/recovery comparison, including adverse intervals."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import context_organization as base
from .ventilation_changing_air import TICKS, SCHEDULE, CONDITIONS, audit
from .ventilation_clamp_analysis import first


def analyze(roots,output):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    seen = set(); sources = {}; cases = []; arrays = {}; checked = 0
    fig,axes = plt.subplots(6,4,figsize=(14,12),sharex=True,sharey='row',layout='constrained')
    for column,root in enumerate(map(lambda p:Path(p).resolve(),roots)):
        if column >= 4: raise ValueError('Expected four seeds')
        m = json.loads((root/'manifest.json').read_text()); seed = m['seed']
        if seed in seen or m['ticks']!=TICKS or m['schedule']!=[list(s) for s in SCHEDULE]:
            raise ValueError('Unexpected or repeated course')
        seen.add(seed)
        if len(m['rows'])!=2 or {r['name'] for r in m['rows']}!=set(CONDITIONS):
            raise ValueError('Incomplete pair')
        for p,h in m['sources'].items():
            if base.digest(p)!=h: raise ValueError('Changed source: '+p)
            sources[p] = h
        sources[str(root/'manifest.json')] = base.digest(root/'manifest.json')
        old_root = Path(m['parent']); old_m = json.loads((old_root/'manifest.json').read_text())
        verified = Path(old_m['parent']); vm = json.loads((verified/'manifest.json').read_text())
        cfg = json.loads((verified/'config.json').read_text())
        pm = json.loads((Path(vm['parent'])/'manifest.json').read_text())
        with np.load(pm['media']) as f: features = {k:f[k] for k in ('visual','auditory')}
        rows = []; initial = None
        for r in m['rows']:
            name = r['name']; path = root/r['file']
            if base.digest(path)!=r['sha256'] or r['exact_opening_ticks']!=512:
                raise ValueError('Changed or unmatched recording')
            sources[str(path)] = r['sha256']
            with np.load(path) as f: z = {k:f[k] for k in f.files}
            if len(z['body'])!=TICKS: raise ValueError('Incomplete trajectory')
            audit(z,cfg,m['groups'],features,CONDITIONS[name]); checked += TICKS
            current = {k:v for k,v in z.items() if k.endswith('_initial')}
            if initial is None: initial = current
            else:
                for k,v in current.items(): np.testing.assert_array_equal(initial[k],v,err_msg=k)
            with np.load(old_root/(name+'.npz')) as f:
                onset = {k:first(f[k],z[k][:1024]) for k in
                         ('organs','organ_raw','organ_drive','muscles','body','cells','weights','terminal_info')}
            ids = list(z['neuron_ids']); oi = base.FIELDS.index('O'); meta = m['meta']
            roles = {'oxygen_comparator':meta['deficit'],'energy_alarm':meta['energy']['alarm']}
            outputs = {k:z['cells'][:,ids.index(n),oi].copy() for k,n in roles.items()}
            reg_ids = list(z['reg_ids']); relay_columns = [reg_ids.index(n) for n in meta['relays']]
            relay_inputs = z['reg_inputs'][:,relay_columns][:,:,:,0]
            relay_outputs = z['cells'][:,[ids.index(n) for n in meta['relays']],oi]
            # The relays have one-tick dendritic delay. Check the actually
            # contributing previous input, not merely the current rhythm.
            nodes = {n['id']:n for n in cfg['neurons']}
            points = {(p['neuron_id'],p['synapse_id']):p for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
            for nid in meta['relays']:
                node = nodes[nid]
                if (node['params']['lambda_param']!=1 or node['params']['delta_decay']!=.99
                        or node['metadata']['graded_gain']!=3 or node['metadata']['graded_S0']!=1
                        or any(points[nid,s]['distance_to_hillock']!=1 for s in (0,1))):
                    raise ValueError('Unexpected phase-relay dynamics')
            offphase = (relay_inputs[:-1,:,0]==0).all(axis=1) & (relay_outputs[1:]>0).all(axis=1)
            offphase_ticks = 1+np.flatnonzero(offphase)
            potential = (z['reg_scheduled'][:-1,relay_columns]*.99).sum(axis=2)
            expected = 3*np.maximum(potential-1.,0.)
            np.testing.assert_allclose(relay_outputs[1:],expected,rtol=0,atol=3e-6)
            saturation = np.abs(z['cells'][:,:,base.FIELDS.index('S')])>=1000
            bound_rows = [dict(neuron_id=int(z['neuron_ids'][j]),ticks=np.flatnonzero(saturation[:,j]).tolist())
                          for j in np.flatnonzero(saturation.any(axis=0))]
            stage_rows = []
            for a,b,fraction in SCHEDULE:
                before = z['organ_initial'] if a==0 else z['organs'][a-1]
                organ = z['organs'][a:b]
                stage_rows.append(dict(start=a,end=b,air_fraction=fraction,
                    minimum_oxygen=float(organ[:,0].min()),minimum_energy=float(organ[:,6].min()),
                    resource_ledger_increment=z['organs'][b-1]-before,
                    oxygen_debt_ticks=(a+np.flatnonzero(np.diff(np.r_[before[4],organ[:,4]])>1e-12)).tolist(),
                    energy_debt_ticks=(a+np.flatnonzero(np.diff(np.r_[before[10],organ[:,10]])>1e-12)).tolist()))
            key = f's{seed}_{name}'
            for k in ('body','organs','organ_raw','organ_drive','air_fraction','muscles'):
                arrays[key+'_'+k] = z[k]
            for k,v in outputs.items(): arrays[key+'_'+k] = v
            arrays[key+'_phase_relay_inputs'] = relay_inputs
            arrays[key+'_phase_relay_outputs'] = relay_outputs
            color,style = ('#126ca4','-') if name=='intact' else ('#a65324','--')
            label = 'Actual delayed oxygen' if name=='intact' else 'Oxygen reading held at .5'
            values = (z['organs'][:,0],z['organs'][:,4],z['organs'][:,6],z['body'][:,1],
                      outputs['oxygen_comparator'],outputs['energy_alarm'])
            for i,v in enumerate(values):
                axes[i,column].plot(np.arange(1,TICKS+1)*.004,v,color=color,ls=style,lw=1.1,label=label)
            rows.append(dict(name=name,first_change_vs_fixed_world=onset,stages=stage_rows,
                             final_organs=z['organs'][-1],neural_roles=roles,
                             both_relays_without_phase_ticks=offphase_ticks.tolist(),
                             membrane_bound_events=bound_rows))
            del z
        axes[0,column].set_title(f'Seed {seed}')
        axes[-1,column].set_xlabel('Time after branching (s)')
        cases.append(dict(seed=seed,conditions=rows))
    if seen!={11,23,44,77}: raise ValueError('Need all four seeds')
    for i,label in enumerate(('Oxygen reserve (mL)','Accrued oxygen debt (mL)','Energy reserve (J)',
                              'Hinge angle (rad)','Oxygen-comparator output','Energy-alarm output')):
        axes[i,0].set_ylabel(label)
    for ax in axes.flat:
        ax.axvspan(512*.004,1280*.004,color='#dddddd',alpha=.5,zorder=-10)
        ax.axhline(0,color='#bbbbbb',lw=.5); ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Same continuing brain, changing physical oxygen supply\n'
                 'Shaded interval: oxygen fraction .105; outside: .21. No change flag enters the brain.',fontsize=12)
    h,l = axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc='outside lower center',ncol=2,frameon=False)
    output.mkdir(); np.savez_compressed(output/'per-tick.npz',**arrays)
    fig.savefig(output/'challenge-recovery.png',dpi=150); plt.close(fig)
    result = dict(cases=cases,checked_ticks=checked,sources=sources,
                  limits='First divergence is a course-specific observation, not path mediation proof. '
                         'All raw samples retained; stages report actual increments including recovery failures.',
                  source_sha256=base.digest(__file__),sha256=base.digest(output/'per-tick.npz'),
                  figure_sha256=base.digest(output/'challenge-recovery.png'))
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(checked_ticks=checked,cases=cases)),flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('roots',nargs='+'); p.add_argument('--output',required=True)
    a = p.parse_args(); analyze(a.roots,a.output)
