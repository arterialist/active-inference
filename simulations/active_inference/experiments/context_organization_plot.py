"""Shared-scale physical trajectories, every seed and adverse interval visible."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(analyses,output,expressions=()):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    loaded=[]
    for root in map(Path,analyses):
        meta=json.loads((root/'summary.json').read_text())
        with np.load(root/'per-tick.npz') as z:
            loaded.append((meta,{k:z[k] for k in z.files if k.startswith('c')}))
    if len({m['seed'] for m,_ in loaded})!=len(loaded):raise ValueError('Duplicate graph seed')
    if len({m['contextual'] for m,_ in loaded})!=1:raise ValueError('Use one architecture per figure')
    expression_data={}
    for root in map(Path,expressions):
        meta=json.loads((root/'manifest.json').read_text())
        expression_data[meta['seed']]={}
        for context in (0,1):
            for clip in (0,1):
                with np.load(root/f'lesion-c{context}-v{clip}.npz') as z:
                    body=z['body']
                    expression_data[meta['seed']][f'c{context}_v{clip}_teaching_cut']=np.column_stack(
                        (body,body[:,4]+.2*body[:,3]))
    if expressions and set(expression_data)!={m['seed'] for m,_ in loaded}:
        raise ValueError('Expression seeds must match the main comparison')
    fig,axes=plt.subplots(2,4,figsize=(14,6.6),sharex=True,sharey='row',layout='constrained')
    fig.patch.set_facecolor('#f6f3eb')
    for ax in axes.flat:
        ax.set_facecolor('#f6f3eb');ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.16)
    for column,(context,clip) in enumerate(((0,0),(0,1),(1,0),(1,1))):
        for meta,original in loaded:
            data={**original,**expression_data.get(meta['seed'],{})}
            styles=[('intact','#174a63','-'),('reset','#ad5a34','--')]
            if expressions:styles.append(('teaching_cut','#343a30',':'))
            for kind,color,style in styles:
                z=data[f'c{context}_v{clip}_{kind}']
                t=np.arange(len(z))*.004
                force_sign=np.sign(z[0,4])
                axes[0,column].plot(t,z[:,5]*force_sign,color=color,ls=style,lw=1,alpha=.7)
                axes[1,column].plot(t,z[:,1],color=color,ls=style,lw=1,alpha=.7)
        axes[0,column].set_title(f'Context {context} · recording {clip}',fontsize=11)
        axes[0,column].axhline(0,color='#444',lw=.7)
        for ax in axes[:,column]:ax.axvline(1.2,color='#777',lw=.7,ls=':')
        axes[1,column].set_xlabel('Time from cue onset (s)')
    axes[0,0].set_ylabel('Residual torque along push (Nm)')
    axes[1,0].set_ylabel('Joint angle (rad)')
    handles=[plt.Line2D([],[],color='#174a63',label='Learned weights'),
             plt.Line2D([],[],color='#ad5a34',ls='--',label='Selected weights reset; learning continues')]
    if expressions:handles.append(plt.Line2D([],[],color='#343a30',ls=':',label='New neural teaching signals interrupted'))
    fig.legend(handles=handles,loc='outside lower center',ncol=1 if expressions else 2,frameon=False)
    fig.suptitle(('Stored responses versus online correction' if expressions else
                 'Does the learned context still control useful movement?')+'\n'
                 'Every line is one graph seed. Media and push end at 1.2 s.',fontsize=14)
    fig.savefig(output,dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analyses',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expressions',nargs='*',type=Path,default=[])
    a=p.parse_args();plot(a.analyses,a.output,a.expressions)
