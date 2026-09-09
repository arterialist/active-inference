"""Static scientific views of recorded arm movement and neural memory effects.

The arm drawing is a planar kinematic projection of the MuJoCo hinge, not a
camera render. Every sample is plotted; no interpolated neural state or fitted
trajectory is presented. Generate all seeds for the declared pair, not only the
most favorable seed. Source digests accompany the image collection.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import context_organization as base


def render(audit,output,video=0,audio=1):
    audit=Path(audit).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    s=json.loads((audit/'summary.json').read_text())
    for p,h in s['sources'].items():
        if base.digest(p)!=h:raise ValueError('Source record changed')
    output.mkdir();images=[]
    with np.load(audit/'per-tick.npz') as data:
        for row in s['cases']:
            if (row['video'],row['audio'])!=(video,audio):continue
            seed=row['seed'];fig,axs=plt.subplots(4,2,figsize=(10,12),layout='constrained')
            for col,physical in enumerate(('acquired','rest')):
                prefix=f's{seed}_{physical}'
                learned=data[f'{prefix}-learned-v{video}-a{audio}']
                reset=data[f'{prefix}-reset-v{video}-a{audio}']
                # Recorded rows contain the body AFTER each physical step.
                t=(np.arange(len(learned))+1)*base.DT
                ax=axs[0,col]
                for trace,label,color,style in ((learned,'Learned','#14647a','-'),(reset,'Weights reset','#555555','--')):
                    q=trace[:,1];x=.25*np.cos(q);y=.25*np.sin(q)
                    ax.plot(x,y,style,color=color,label=label,lw=2)
                    ax.plot([0,x[-1]],[0,y[-1]],style,color=color,lw=2)
                    ax.scatter(x[[0,63,-1]],y[[0,63,-1]],color=color,s=16)
                ax.plot([0,.25],[0,0],':',color='#aaaaaa',lw=1)
                ax.set(xlim=(-.28,.28),ylim=(-.28,.28),xlabel='x (m)',ylabel='y (m)',
                       title='Acquired body' if physical=='acquired' else 'Body moved to rest')
                ax.set_aspect('equal');ax.legend(fontsize=9,loc='lower left')
                for r,column,label in ((1,1,'Joint angle (rad)'),(2,6,'Prediction along load'),(3,None,'Memory effect on |angle| (rad)')):
                    ax=axs[r,col]
                    if column is None:
                        delta=abs(learned[:,1])-abs(reset[:,1])
                        ax.plot(t,delta,color='#14647a',lw=1.8)
                        ax.axhline(0,color='#777777',lw=.7)
                    else:
                        ax.plot(t,learned[:,column],color='#14647a',lw=1.8)
                        ax.plot(t,reset[:,column],'--',color='#555555',lw=1.4)
                    ax.axvline(64*base.DT,color='#999999',linestyle=':',lw=1)
                    ax.set(xlabel='Time since intervention (s)',ylabel=label)
                    ax.spines[['top','right']].set_visible(False)
            for r in (1,2,3):
                limits=[ax.get_ylim() for ax in axs[r]]
                lo=min(x[0] for x in limits);hi=max(x[1] for x in limits)
                for ax in axs[r]:ax.set_ylim(lo,hi)
            fig.suptitle(f'Seed {seed}, audiovisual pair {video}{audio}: acquired neural state in both columns\n'
                         'Planar arm projection and every recorded tick; vertical line: 64-tick sensory delay',fontsize=12)
            fig.supxlabel('Bottom row: positive = learned weights leave greater displacement than reset.\n'
                          'Physical work and pose are separate outcomes; this is not an overall success score.',fontsize=10)
            name=f'body-memory-seed{seed}.png';fig.savefig(output/name,dpi=140);plt.close(fig)
            images.append(dict(seed=seed,file=name,sha256=base.digest(output/name)))
    if not images:raise ValueError('No matching pair')
    result=dict(images=images,video=video,audio=audio,
        summary_sha256=base.digest(audit/'summary.json'),data_sha256=base.digest(audit/'per-tick.npz'),
        producer_sha256=base.digest(__file__),limits=__doc__)
    (output/'manifest.json').write_text(base.encode(result)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('audit',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();render(a.audit,a.output)
