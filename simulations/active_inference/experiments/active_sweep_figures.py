"""Full-course physical motion and load/prediction traces for every graph seed."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import context_organization as base


def render(analysis,output):
    analysis,output=(Path(p).resolve() for p in (analysis,output))
    if output.exists():raise FileExistsError(output)
    summary=json.loads((analysis/'summary.json').read_text())
    for p,h in summary['sources'].items():
        if base.digest(p)!=h:raise ValueError('Audited source changed')
    with np.load(analysis/'per-tick.npz') as z:
        traces={c['trace']:z[c['trace']] for c in summary['cases']}
    seeds=summary['seeds']
    if len(seeds)!=4:raise ValueError('Need all four audited graph seeds')
    output.mkdir()
    fig,axes=plt.subplots(4,2,figsize=(14,10),sharex=True,layout='constrained')
    style={'free_fused':('#777777','-'),'loaded_fused':('#005ca9','-'),
           'loaded_sensory':('#c55b00','--'),'actuator_cut':('#111111',':')}
    for row,seed in enumerate(seeds):
        for condition,(color,ls) in style.items():
            tr=traces[f's{seed}_{condition}'];t=np.arange(len(tr))
            axes[row,0].plot(t,tr[:,1],color=color,ls=ls,lw=1.3,label=condition.replace('_',' / '))
        for gate in (-.008,.008):
            axes[row,0].axhline(gate,color='#999999',ls=':',lw=1.)
        tr=traces[f's{seed}_loaded_fused'];t=np.arange(len(tr))
        axes[row,1].plot(t,tr[:,4]/.2,color='#777777',lw=1.,label='Physical environmental torque / 0.2 Nm')
        axes[row,1].plot(t,tr[:,5],color='#005ca9',lw=1.3,label='Current signed neural prediction')
        axes[row,0].set_ylabel(f'Seed {seed}\nJoint angle [rad]')
        axes[row,1].set_ylabel('Normalized load / prediction\n[model units]')
        for ax in axes[row]:
            ax.set_xlim(0,len(tr)-1);ax.grid(alpha=.15)
            ax.set_xlabel('Recorded tick, 4 ms per physical step')
    # Shared y scales allow seed-to-seed comparison without hiding excursions.
    for column in (0,1):
        low=min(ax.get_ylim()[0] for ax in axes[:,column]);high=max(ax.get_ylim()[1] for ax in axes[:,column])
        for ax in axes[:,column]:ax.set_ylim(low,high)
    axes[0,0].set_title('Actual movement through the two gates')
    axes[0,1].set_title('Loaded / fused: force present now versus prediction')
    axes[0,0].legend(fontsize=8,ncol=2)
    axes[0,1].legend(fontsize=8)
    fig.suptitle('The resistance breaks the sweep; the current learner does not restore it\n'
                 'Every recorded tick. Plasticity active. Each condition starts from its matched birth state.',fontsize=13)
    fig.savefig(output/'active-sweep-all-seeds.png',dpi=150);plt.close(fig)
    m=dict(analysis_sha256=base.digest(analysis/'summary.json'),data_sha256=base.digest(analysis/'per-tick.npz'),
           producer_sha256=base.digest(__file__),file='active-sweep-all-seeds.png',
           limits='Unaveraged acquisition trajectories, not proof of retained learning. Two gates are at +/-0.008 rad. '
           'Physical load is normalized at the receptor scale; prediction is neural output, not directly measured force.')
    (output/'manifest.json').write_text(base.encode(m)+'\n')
    return m


def render_memory(analysis,output):
    """Full recorded continuation; stroke diagnostics are not task success."""
    analysis,output=(Path(p).resolve() for p in (analysis,output))
    if output.exists():raise FileExistsError(output)
    summary=json.loads((analysis/'summary.json').read_text())
    for p,h in summary['sources'].items():
        if base.digest(p)!=h:raise ValueError('Audited source changed')
    if len(summary['seeds'])!=4:raise ValueError('Need all four graph seeds')
    with np.load(analysis/'per-tick.npz') as z:data={k:z[k] for k in z.files}
    fig,axes=plt.subplots(4,3,figsize=(16,10),sharex=True,layout='constrained')
    for row,case in enumerate(summary['cases']):
        seed=case['seed'];a=data[f's{seed}_intact'];b=data[f's{seed}_reset'];t=np.arange(len(a))
        for trace,label,color,ls in ((a,'Retained weights','#005ca9','-'),(b,'Reset selected weights','#b45400','--')):
            axes[row,0].plot(t,trace[:,1],label=label,color=color,ls=ls,lw=1.4)
            axes[row,2].plot(t,trace[:,5],label=label,color=color,ls=ls,lw=1.4)
        for gate in (-.008,.008):axes[row,0].axhline(gate,color='#777777',ls=':',lw=.8)
        effect=data[f's{seed}_effect_stroke']
        ax=axes[row,1];ax.axhline(0,color='#777777',lw=.8)
        ax.plot(t,effect,color='#333333',lw=1.)
        ax.fill_between(t,0,effect,where=effect>0,color='#005ca9',alpha=.3)
        ax.fill_between(t,0,effect,where=effect<0,color='#b45400',alpha=.3)
        for stroke in case['strokes']:
            ax.axvline(stroke['start'],color='#999999',ls=':',lw=.6)
        axes[row,0].set_ylabel(f'Seed {seed}\nJoint angle [rad]')
        axes[row,1].set_ylabel('Stroke-advance difference [rad]\nRetained minus reset')
        axes[row,2].set_ylabel('Signed prediction [model units]')
        for ax in axes[row]:
            ax.grid(alpha=.15);ax.set_xlim(0,len(a)-1);ax.set_xlabel('Ticks since intervention, 4 ms per step')
    for col in range(3):
        lo=min(ax.get_ylim()[0] for ax in axes[:,col]);hi=max(ax.get_ylim()[1] for ax in axes[:,col])
        for ax in axes[:,col]:ax.set_ylim(lo,hi)
    axes[0,0].set_title('Actual movement; dotted lines are task gates')
    axes[0,1].set_title('Blue: more stroke advance; orange: less\nReference resets at each neural half-cycle')
    axes[0,2].set_title('Retained weights change neural prediction')
    axes[0,0].legend(fontsize=9)
    fig.suptitle('Memory is expressed, but does not restore the loaded sweep\n'
                 'Same acquired brain and body. Selected weights alone reset. Adaptation remains active.',fontsize=13)
    output.mkdir();path=output/'active-sweep-memory.png';fig.savefig(path,dpi=150);plt.close(fig)
    m=dict(analysis_sha256=base.digest(analysis/'summary.json'),data_sha256=base.digest(analysis/'per-tick.npz'),
           producer_sha256=base.digest(__file__),file=path.name,
           limits='Every recorded tick, common seed scales, no smoothing. Stroke-advance differences use each '
           'branch angle immediately before a CPG-driven half-cycle; they are not gate completion or energy.')
    (output/'manifest.json').write_text(base.encode(m)+'\n')
    return m


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('analysis',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--memory',action='store_true')
    a=p.parse_args();print((render_memory if a.memory else render)(a.analysis,a.output))
