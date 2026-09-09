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


def render_credit(analysis,output):
    """Show every seed and timing condition alongside acquired-weight effects."""
    analysis,output=(Path(p).resolve() for p in (analysis,output))
    if output.exists():raise FileExistsError(output)
    summary=json.loads((analysis/'summary.json').read_text())
    for p,h in summary['sources'].items():
        if base.digest(p)!=h:raise ValueError('Audited source changed')
    if len(summary['seeds'])!=4:raise ValueError('Need all four graph seeds')
    with np.load(analysis/'per-tick.npz') as z:data={k:z[k] for k in z.files}
    styles={'old':('#777777','Original exponential'), 'mean_only':('#c66b00','Longer exponential'),
            'shape_only':('#005ca9','Cascade, original mean'), 'matched_cascade':('#813aa0','Cascade, verification mean')}
    fig,axes=plt.subplots(4,3,figsize=(16,10),sharex='col',layout='constrained')
    for row,seed in enumerate(summary['seeds']):
        for condition,(color,label) in styles.items():
            trace=np.concatenate([data[f's{seed}_{condition}_train'],data[f's{seed}_{condition}_intact']])
            t=np.arange(len(trace))
            axes[row,0].plot(t,trace[:,1],color=color,lw=1.15,label=label)
            axes[row,1].plot(t,trace[:,5]-trace[:,4]/.2,color=color,lw=1.15)
            effect=data[f's{seed}_{condition}_stroke_effect']
            axes[row,2].plot(np.arange(len(effect)),effect,color=color,lw=1.15)
        for gate in (-.008,.008):axes[row,0].axhline(gate,color='#999999',ls=':',lw=.8)
        axes[row,0].axvline(1024,color='#999999',ls=':',lw=.8)
        axes[row,1].axvline(1024,color='#999999',ls=':',lw=.8)
        axes[row,1].axhline(0,color='#999999',lw=.7)
        axes[row,2].axhline(0,color='#999999',lw=.7)
        for t in range(0,512,82):axes[row,2].axvline(t,color='#999999',ls=':',lw=.6)
        axes[row,0].set_ylabel(f'Seed {seed}\nAngle [rad]')
        axes[row,1].set_ylabel('Prediction minus current force\n[receptor-scale model units]')
        axes[row,2].set_ylabel('Retained minus reset\nstroke advance [rad]')
        for col,ax in enumerate(axes[row]):
            ax.grid(alpha=.12);ax.set_xlim(0,511 if col==2 else 1535)
            ax.set_xlabel('Ticks since weight intervention' if col==2 else 'Continuous physical tick, 4 ms per tick')
    for col in range(3):
        lo=min(ax.get_ylim()[0] for ax in axes[:,col]);hi=max(ax.get_ylim()[1] for ax in axes[:,col])
        for ax in axes[:,col]:ax.set_ylim(lo,hi)
    axes[0,0].set_title('Actual movement against resistance')
    axes[0,1].set_title('Current-force prediction, not delayed teaching error')
    axes[0,2].set_title('Stored-weight effect in each motor half-cycle\nPositive: further advance; negative: less')
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('Local learning windows in one unchanged sensorimotor circuit\n'
                 'All ticks and seeds. Same forward wiring. Continuous adaptation. Gate lines at +/-0.008 rad.',fontsize=13)
    output.mkdir();path=output/'active-sweep-credit.png';fig.savefig(path,dpi=150);plt.close(fig)
    m=dict(analysis_sha256=base.digest(analysis/'summary.json'),data_sha256=base.digest(analysis/'per-tick.npz'),
           producer_sha256=base.digest(__file__),file=path.name,
           limits='No temporal averaging. Acquisition and intact continuation join at 1024; right column '
           'compares acquired-state branches and resets its physical reference at each neural half-cycle. '
           'Current-force residual and stroke advance alone do not establish task success.')
    (output/'manifest.json').write_text(base.encode(m)+'\n')
    return m


def render_acquisition(analysis,output,earlier):
    """Compare complete physical history and two ages of retained-weight use."""
    analysis,output,earlier=(Path(p).resolve() for p in (analysis,output,earlier))
    if output.exists():raise FileExistsError(output)
    new=json.loads((analysis/'summary.json').read_text())
    old=json.loads((earlier/'summary.json').read_text())
    for report in (new,old):
        for p,h in report['sources'].items():
            if base.digest(p)!=h:raise ValueError('Audited source changed: '+p)
        if report['seeds']!=[11,23,44,77]:raise ValueError('Need the four declared graph seeds')
    with np.load(analysis/'per-tick.npz') as z:now={k:z[k] for k in z.files}
    with np.load(earlier/'per-tick.npz') as z:before={k:z[k] for k in z.files}
    fig,axes=plt.subplots(4,3,figsize=(17,11),sharex='col',layout='constrained')
    styles={'old':('#777777','Original exponential'),'matched_cascade':('#813aa0','Matched cascade')}
    for row,seed in enumerate(new['seeds']):
        for condition,(color,label) in styles.items():
            key=f's{seed}_{condition}'
            course=np.concatenate([before[key+'_train'],before[key+'_intact'],
                                   now[key+'_course'],now[key+'_intact']])
            if len(course)!=4816 or np.max(abs(np.diff(course[:,0])-.004))>1e-10:
                raise ValueError('Broken continuous physical course')
            axes[row,0].plot(np.arange(len(course)),course[:,1],color=color,lw=1.,label=label)
            for data,age,ls in ((before,1024,':'),(now,4304,'-')):
                effect=data[key+'_stroke_effect']
                axes[row,1].plot(np.arange(len(effect)),effect*1e6,color=color,ls=ls,lw=1.2,
                                 label=f'{label}, age {age}')
            trace=now[key+'_intact']
            axes[row,2].plot(np.arange(len(trace)),trace[:,5]-trace[:,4]/.2,color=color,lw=1.2)
        for gate in (-.008,.008):axes[row,0].axhline(gate,color='#333333',ls=':',lw=.8)
        for tick in (1024,4304):axes[row,0].axvline(tick,color='#999999',ls=':',lw=.8)
        for col in (1,2):axes[row,col].axhline(0,color='#333333',lw=.7)
        for tick in range(0,512,82):axes[row,1].axvline(tick,color='#999999',ls=':',lw=.5)
        axes[row,0].set_ylabel(f'Seed {seed}\nJoint angle [rad]')
        axes[row,1].set_ylabel('Retained minus reset\nstroke advance [microrad]')
        axes[row,2].set_ylabel('Prediction minus current force\n[receptor-scale units]')
        for col,ax in enumerate(axes[row]):
            ax.grid(alpha=.12);ax.set_xlim(0,4815 if col==0 else 511)
            ax.set_xlabel('Continuous tick, 4 ms per step' if col==0 else 'Ticks since selected-weight intervention')
    for col in range(3):
        lo=min(ax.get_ylim()[0] for ax in axes[:,col]);hi=max(ax.get_ylim()[1] for ax in axes[:,col])
        for ax in axes[:,col]:ax.set_ylim(lo,hi)
    axes[0,0].set_title('Full physical course; dotted horizontal lines are gates')
    axes[0,1].set_title('Early versus late retained-weight contribution\nPositive: further advance. Negative: less.')
    axes[0,2].set_title('Late intact prediction versus force acting now')
    axes[0,0].legend(fontsize=8)
    axes[0,1].legend(fontsize=8,ncol=2)
    fig.suptitle('Does continued experience make acquired prediction more useful?\n'
                 'Unchanged brain and world. Every tick and seed. Learning remains active in the reset probes.',fontsize=13)
    output.mkdir();path=output/'active-sweep-acquisition.png';fig.savefig(path,dpi=160);plt.close(fig)
    manifest=dict(analysis_sha256=base.digest(analysis/'summary.json'),data_sha256=base.digest(analysis/'per-tick.npz'),
                  earlier_analysis_sha256=base.digest(earlier/'summary.json'),earlier_data_sha256=base.digest(earlier/'per-tick.npz'),
                  producer_sha256=base.digest(__file__),file=path.name,
                  limits='No smoothing or seed averaging. Early and late brains differ in experience and state; '
                  'motor phase is matched, audiovisual phase is not. Stroke references reset at each half-cycle. '
                  'A positive learned contribution does not imply gate completion or transfer.')
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('analysis',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--memory',action='store_true')
    p.add_argument('--credit',action='store_true')
    p.add_argument('--acquisition',type=Path,help='Earlier kernel analysis for the continuing-acquisition comparison')
    a=p.parse_args()
    print(render_acquisition(a.analysis,a.output,a.acquisition) if a.acquisition else
          (render_credit if a.credit else render_memory if a.memory else render)(a.analysis,a.output))
