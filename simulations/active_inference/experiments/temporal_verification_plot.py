"""Plot every weight-transfer trajectory, including wrong-direction intervals."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def plot(root):
    root=Path(root)
    summary=json.loads((root/'summary.json').read_text())
    colors={11:'#28618c',23:'#b77820',44:'#39846d',77:'#b33548'}
    with np.load(root/'per-tick.npz') as data:
        for aligned in (True,False):
            fig,axes=plt.subplots(2,4,figsize=(14,7),sharex=True,sharey='row',
                facecolor='#f4f0e6')
            for ax in axes.flat:
                ax.set_facecolor('#f4f0e6')
                ax.axvspan(0,256,color='black',alpha=.045)
                ax.axvline(256,color='#777777',ls=':',lw=1)
                ax.axhline(0,color='#777777',lw=.6)
                ax.spines[['top','right']].set_visible(False)
            for context in (0,1):
                for clip in (0,1):
                    col=2*context+clip
                    for row in summary['cases']:
                        if (row['aligned'],row['context'],row['clip'])!=(aligned,context,clip):continue
                        z=data[row['key']]
                        # Neural outputs are post-tick states; body samples follow each 4 ms step.
                        t=z[:,0]*1000
                        style='-' if row['weights']=='learned' else ':'
                        for i,column in enumerate((8,9)):
                            axes[i,col].plot(t,z[:,column],color=colors[row['seed']],ls=style,lw=1.35)
                    axes[0,col].set_title(f'Context {context}, recording {clip}')
                    axes[1,col].set_xlabel('Time after onset [ms]')
            axes[0,0].set_ylabel('Prediction along imposed load\n[neural release units]')
            axes[1,0].set_ylabel('Displacement along imposed load\n[rad]')
            handles=[Line2D([],[],color=c,label=f'Seed {seed}') for seed,c in colors.items()]
            handles += [Line2D([],[],color='black',label='Learned weights',ls='-'),
                        Line2D([],[],color='black',label='Birth weights',ls=':')]
            fig.legend(handles=handles,loc='lower center',ncol=6,frameon=False,bbox_to_anchor=(.5,.06))
            label='Delayed verification' if aligned else 'Unmatched verification'
            fig.suptitle(f'{label}: memory survives state reset, with a seed-dependent failure',fontsize=15)
            fig.text(.5,.017,'Same birth brain and resting body; only selected weights transferred. '
                     'All 96 ticks shown; plasticity stays active.\n'
                     'Shading ends before the first fresh somatic input. Negative prediction means wrong load direction. '
                     'Both variants share gain and credit-trace changes.',ha='center',fontsize=9)
            fig.subplots_adjust(left=.09,right=.98,top=.87,bottom=.21,hspace=.22,wspace=.17)
            fig.savefig(root/f'{"aligned" if aligned else "unmatched"}-memory.png',dpi=160,
                        facecolor=fig.get_facecolor())
            plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    plot(p.parse_args().root)
