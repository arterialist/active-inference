"""Full recorded trajectories, including failed resource intervals and re-seeding."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import context_organization as base


def render(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    inputs={};seeds=(11,23,44,77)
    styles={'feedback':('Organ feedback','#126ca4','-'),
            'oxygen-cut':('Oxygen output cut','#b34f13','-'),
            'energy-cut':('Energy output cut','#577454',':'),
            'tonic':('Tonic recruitment','#777777','--')}
    fig,axes=plt.subplots(4,4,figsize=(14,9),sharex=True,sharey='row',layout='constrained')
    for j,seed in enumerate(seeds):
        for mode,(label,color,style) in styles.items():
            p=root/f'20260909_ventilation_verified_{mode}_seed{seed}'
            m=json.loads((p/'manifest.json').read_text())
            if m['seed']!=seed or m['mode']!=mode or m['audited_ticks']!=1024:
                raise ValueError('Wrong course')
            if base.digest(p/'ticks.npz')!=m['sha256']: raise ValueError('Changed evidence')
            inputs[str(p/'manifest.json')]=base.digest(p/'manifest.json')
            inputs[str(p/'ticks.npz')]=m['sha256']
            with np.load(p/'ticks.npz') as z:
                time=np.arange(1,1025)*.004
                for i,col in enumerate((0,4,6,10)):
                    axes[i,j].plot(time,z['organs'][:,col],color=color,ls=style,lw=1.15,label=label)
        axes[0,j].set_title(f'Seed {seed}',fontsize=11)
        axes[-1,j].set_xlabel('Time after composition (s)')
    for i,label in enumerate(('Oxygen reserve (mL)','Oxygen debt accrued (mL)',
                              'Energy reserve (J)','Energy debt accrued (J)')):
        axes[i,0].set_ylabel(label)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False);ax.axhline(0,color='#bbbbbb',lw=.5)
    fig.suptitle('Acquired PAULA brain with organ feedback, after removing the unintended startup pulse\n'
                 'Every recorded tick; four-second course; no claim of lifelong maintenance',fontsize=12)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    output.mkdir();path=output/'resources.png';fig.savefig(path,dpi=150);plt.close(fig)
    figures={path.name:base.digest(path)}
    fig,axes=plt.subplots(2,4,figsize=(14,5),sharex=True,sharey='row',layout='constrained')
    for j,seed in enumerate(seeds):
        p=root/f'20260909_ventilation_reseed_verified_seed{seed}'
        m=json.loads((p/'manifest.json').read_text());inputs[str(p/'manifest.json')]=base.digest(p/'manifest.json')
        for mode,offset,color in (('unchanged',0.,'#126ca4'),('stale-rebuild',.3,'#b34f13')):
            row=next(r for r in m['rows'] if r['mode']==mode);f=p/row['path']
            if base.digest(f)!=row['sha256']:raise ValueError('Changed reseed evidence')
            inputs[str(f)]=row['sha256']
            with np.load(f) as z:
                ids=list(z['neuron_ids']);oi=base.FIELDS.index('O')
                for phase,nid in enumerate(z['cpg_ids']):
                    t=np.flatnonzero(z['cells'][:,ids.index(nid),oi]>0)*.004
                    axes[0,j].scatter(t,np.full(len(t),phase+offset),s=12,color=color,marker='|',label=mode if phase==0 else None)
                axes[1,j].plot(np.arange(1,len(z['body'])+1)*.004,z['body'][:,1],color=color,lw=1.1)
        axes[0,j].set_title(f'Seed {seed}');axes[1,j].set_xlabel('Time after branching (s)')
        axes[0,j].axvline(.16,color='#777777',lw=.7,ls=':')
    axes[0,0].set_ylabel('Rhythm cell / phase');axes[1,0].set_ylabel('Hinge angle (rad)')
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('A cache rebuild injected another neural pulse: extra wave appears after 40 ticks / 160 ms\n'
                 'Stale rebuild = explicit pulse; safe rebuild = unchanged, in every recorded field',fontsize=12)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    path=output/'reseed.png';fig.savefig(path,dpi=150);plt.close(fig);figures[path.name]=base.digest(path)
    (output/'manifest.json').write_text(base.encode(dict(inputs=inputs,figures=figures,
        source_sha256=base.digest(__file__),all_samples=True))+'\n')
    print(output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('output')
    a=p.parse_args();render(a.root,a.output)
