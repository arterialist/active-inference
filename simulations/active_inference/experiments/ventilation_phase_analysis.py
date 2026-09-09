"""Per-tick factorial comparison of phase routing and predictive motor influence."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import context_organization as base
from .ventilation_phase_composition import CONDITIONS,audit_gate
from .ventilation_changing_air import TICKS,SCHEDULE,audit as audit_body
from .ventilation_clamp_analysis import first


def onset(values):
    i = np.flatnonzero(values)
    return int(i[0]) if len(i) else None


def analyze(roots,output):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    sources = {}; arrays = {}; cases = []; seen = set(); checked = 0
    fig,axes = plt.subplots(5,4,figsize=(14,10),sharex=True,sharey='row',layout='constrained')
    styles = {'legacy':('#a65324','-'),'authorized':('#126ca4','-'),
              'legacy-predictor-cut':('#a65324','--'),'authorized-predictor-cut':('#126ca4','--')}
    for column,root in enumerate(map(lambda p:Path(p).resolve(),roots)):
        if column>=4: raise ValueError('Expected four seeds')
        m = json.loads((root/'manifest.json').read_text());seed=m['seed'];g=m['groups']
        if seed in seen or m['ticks']!=TICKS or len(m['rows'])!=4 or {r['condition'] for r in m['rows']}!=set(CONDITIONS):
            raise ValueError('Incomplete or repeated factorial course')
        seen.add(seed)
        for p,h in m['sources'].items():
            if base.digest(p)!=h: raise ValueError('Changed source: '+p)
            sources[p]=h
        sources[str(root/'manifest.json')] = base.digest(root/'manifest.json')
        previous = Path(m['parent']); prev = json.loads((previous/'manifest.json').read_text())
        clamp = json.loads((Path(prev['parent'])/'manifest.json').read_text())
        verified = json.loads((Path(clamp['parent'])/'manifest.json').read_text())
        pm = json.loads((Path(verified['parent'])/'manifest.json').read_text())
        with np.load(pm['media']) as f:features={k:f[k] for k in ('visual','auditory')}
        reference = {}; initial = None; rows = []
        for r in m['rows']:
            name = r['condition']; folder = root/name
            for f,h in r['files'].items():
                if base.digest(folder/f)!=h: raise ValueError('Changed condition artifact: '+str(folder/f))
                sources[str(folder/f)] = h
            cfg = json.loads((folder/'config.json').read_text()); meta = r['meta']; phase = meta['phase_authorization']
            with np.load(folder/'ticks.npz') as f:z={k:f[k] for k in f.files}
            if len(z['body'])!=TICKS: raise ValueError('Truncated course')
            audit_body(z,cfg,g,features,None); residual=audit_gate(z,cfg,meta); checked+=TICKS
            current = {k:v for k,v in z.items() if k.endswith('_initial')}
            if initial is None:initial=current
            else:
                for k,v in current.items():np.testing.assert_array_equal(v,initial[k],err_msg=k)
            ids = list(z['neuron_ids']); reg = list(z['reg_ids']); oi=base.FIELDS.index('O')
            outputs = {k:z['cells'][:,[ids.index(n) for n in ns],oi].copy() for k,ns in
                dict(prediction=g['prediction'],rhythm=g['cpg'],legacy=meta['relays'],gate=phase['outputs'],
                     alarm=[meta['energy']['alarm']]).items()}
            # Q input -> Q dendrite -> G input -> G dendrite is three ticks.
            phase_inputs = z['reg_inputs'][:,[reg.index(n) for n in phase['inhibitors']]][:,:,1,0]
            permitted = np.zeros((TICKS,2),bool);permitted[3:]=phase_inputs[:-3]>0
            gate_off = np.where(permitted,0.,outputs['gate'])
            stages=[]
            for a,b,gas in SCHEDULE:
                before=z['organ_initial'] if a==0 else z['organs'][a-1]
                stages.append(dict(start=a,end=b,gas=gas,oxygen_min=float(z['organs'][a:b,0].min()),
                    energy_min=float(z['organs'][a:b,6].min()),ledger=z['organs'][b-1]-before,
                    oxygen_debt_ticks=(a+np.flatnonzero(np.diff(np.r_[before[4],z['organs'][a:b,4]])>1e-12)).tolist(),
                    energy_debt_ticks=(a+np.flatnonzero(np.diff(np.r_[before[10],z['organs'][a:b,10]])>1e-12)).tolist()))
            selected = {k:z[k] for k in ('body','organs','organ_drive','weights','muscles')}
            selected.update(outputs)
            if name=='legacy':reference={k:v.copy() for k,v in selected.items()}
            changes = {k:first(reference[k],v) for k,v in selected.items()}
            bound=np.abs(z['cells'][:,:,base.FIELDS.index('S')])>=1000
            bounds=[dict(neuron=int(z['neuron_ids'][j]),ticks=np.flatnonzero(bound[:,j]).tolist())
                    for j in np.flatnonzero(bound.any(axis=0))]
            legacy_vs_original=None
            if name=='legacy':
                row=next(row for row in prev['rows'] if row['name']=='intact')
                if base.digest(previous/row['file'])!=row['sha256']:raise ValueError('Changed original baseline')
                sources[str(previous/row['file'])]=row['sha256']
                with np.load(previous/row['file']) as f:
                    legacy_vs_original={k:first(f[k],z[k]) for k in ('body','organs','organ_drive','muscles','weights')}
                    terms=list(map(tuple,z['terminal_ids']))
                    columns=[terms.index(tuple(row)) for row in f['terminal_ids']]
                    legacy_vs_original['terminal_info']=first(f['terminal_info'],z['terminal_info'][:,columns])
            key=f's{seed}_{name}'
            for k,v in selected.items():
                if k!='weights': arrays[key+'_'+k]=v.copy()
            arrays[key+'_gate_off_phase']=gate_off
            rows.append(dict(condition=name,first_difference_from_legacy=changes,stages=stages,
                first_oxygen_debt=onset(z['organs'][:,4]>1e-12),first_energy_debt=onset(z['organs'][:,10]>1e-12),
                first_alarm=onset(outputs['alarm'][:,0]>0),membrane_bounds=bounds,
                peak_gate_off_phase=gate_off.max(axis=0),peak_gate=outputs['gate'].max(axis=0),
                first_change_from_original=legacy_vs_original,gate_audit_residual=residual,
                final_organs=z['organs'][-1],predictive_weight_range=[float(z['weights'].min()),float(z['weights'].max())]))
            color,style=styles[name]
            common_power=.2*z['muscles'][:,0]*z['muscles'][:,1]
            values=(z['organs'][:,0],z['organs'][:,6],z['organs'][:,4],common_power,outputs['prediction'].max(axis=1))
            for i,v in enumerate(values):axes[i,column].plot(np.arange(1,TICKS+1)*.004,v,color=color,ls=style,lw=1.05,label=name)
            del z
        axes[0,column].set_title(f'Seed {seed}');axes[-1,column].set_xlabel('Time after branching (s)')
        cases.append(dict(seed=seed,groups=g,conditions=rows))
    if seen!={11,23,44,77}:raise ValueError('Need all four seeds')
    labels=('Oxygen reserve (mL)','Energy reserve (J)','Accrued oxygen debt (mL)',
            'Common activation\npower (W; symlog)','Largest predictor\noutput (symlog)')
    for i,label in enumerate(labels):axes[i,0].set_ylabel(label)
    for ax in axes[3]:ax.set_yscale('symlog',linthresh=.01)
    for ax in axes[4]:ax.set_yscale('symlog',linthresh=1.)
    for ax in axes.flat:
        ax.axvspan(512*.004,1280*.004,color='#dddddd',alpha=.5,zorder=-10)
        ax.axhline(0,color='#bbbbbb',lw=.5);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Phase routing and predictive motor influence in the same expanded brain\n'
                 'Grey: reduced physical oxygen. Dashed: diagnostic predictor-to-muscle cut; learning continues.',fontsize=12)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2,frameon=False)
    output.mkdir();np.savez_compressed(output/'per-tick.npz',**arrays)
    fig.savefig(output/'phase-composition.png',dpi=150);plt.close(fig)
    result=dict(cases=cases,checked_ticks=checked,sources=sources,source_sha256=base.digest(__file__),
        sha256=base.digest(output/'per-tick.npz'),figure_sha256=base.digest(output/'phase-composition.png'),
        limits='All raw ticks retained. Same expanded graph, different declared motor weights. '
               'First divergence is not proof of mediation; off-phase leakage is measured, not rounded to silence.')
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(checked_ticks=checked,cases=[dict(seed=c['seed'],conditions=[{k:r[k] for k in
        ('condition','first_oxygen_debt','first_energy_debt','peak_gate_off_phase')} for r in c['conditions']]) for c in cases])),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',nargs='+');p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.roots,a.output)
