"""Measured neural/body phase portraits, not a fitted dynamical reconstruction."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import context_organization as base
from .opponent_context_analysis import read_record


def plot(root,output,seed=11):
    root=Path(root);output=Path(output)
    sets={}
    for order in (0,1):
        for kind in ('baseline','opponent','half'):
            if kind=='half':
                source=root/f'20260909_opponent_feedback_half_order{order}_seed{seed}'
                m=json.loads((source/'manifest.json').read_text())
                summary=json.loads((source/'summary.json').read_text())
                name='half-d64-c0-v1.npz';rows=summary['results']
            else:
                source=root/f'20260909_opponent_context_{kind}_order{order}_seed{seed}'
                m=json.loads((source/'manifest.json').read_text())
                summary=json.loads((source/'summary.json').read_text())
                name='probe-d64-r0-c0-v1.npz';rows=summary['probes']
            z=read_record(source,next(r for r in rows if r['file']==name),m)
            ids=list(z['neuron_ids']);pred=m['groups']['prediction']
            p=z['cells'][:,ids.index(pred[0]),1]-z['cells'][:,ids.index(pred[1]),1]
            sets[order,kind]=(p,z['errors'][:,0,1],z['body'])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(12,8),facecolor='#f4f0e6')
    axes=[];phase=[];colors={'baseline':'#323232','opponent':'#b34532','half':'#23749b'}
    labels={'baseline':'Separate errors','opponent':'Coupled errors','half':'Same acquired coupled state, half feedback'}
    bounds=[]
    for axis in range(3):
        values=np.concatenate([s[axis] if axis<2 else s[2][:,2] for s in sets.values()])
        low,high=float(values.min()),float(values.max());pad=max((high-low)*.08,.01)
        bounds.append((low-pad,high+pad))
    for order in (0,1):
        ax=fig.add_subplot(2,2,order+1,projection='3d');phase.append(ax)
        ax.set_facecolor('#f4f0e6')
        bx=fig.add_subplot(2,2,order+3);axes.append(bx);bx.set_facecolor('#f4f0e6')
        for kind in colors:
            p,e,b=sets[order,kind];color=colors[kind]
            ax.plot(p,e,b[:,2],color=color,lw=1.35,label=labels[kind])
            ax.scatter(p[0],e[0],b[0,2],color=color,s=18,marker='o')
            ax.scatter(p[-1],e[-1],b[-1,2],color=color,s=25,marker='^')
            bx.plot(np.arange(len(b))*4,b[:,4]+.2*b[:,3],color=color,lw=1.4)
        ax.set(xlabel='Signed prediction [release]',ylabel='Teaching input [release]',
               zlabel='Joint velocity [rad/s]',xlim=bounds[0],ylim=bounds[1],zlim=bounds[2])
        ax.set_title(f'Order {order}: last trained recording {order}',pad=12)
        ax.view_init(elev=23,azim=-61)
        bx.axhline(0,color='#777777',lw=.7)
        bx.axvspan(0,256,color='#777777',alpha=.09)
        bx.axvline(256,color='#777777',lw=.8,ls=':')
        bx.set(xlabel='Time from probe onset [ms]',ylabel='Net physical torque [Nm]')
        bx.grid(alpha=.16)
    low=min(ax.get_ylim()[0] for ax in axes);high=max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:ax.set_ylim(low,high)
    fig.suptitle('A learned correction loop can oscillate and drive the body',fontsize=16,y=.985)
    fig.legend(*phase[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.055))
    fig.text(.5,.014,'Seed 11, context 0, recording 1. All 192 ticks. Shading: no new bodily afferents yet.\n'
             'Circles mark onset; triangles mark end. Half-feedback branch matches the coupled initial state; baseline does not.',
             ha='center',fontsize=9)
    fig.subplots_adjust(left=.08,right=.94,top=.9,bottom=.16,hspace=.36,wspace=.28)
    fig.savefig(output,dpi=160,facecolor=fig.get_facecolor());plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--seed',type=int,default=11)
    a=p.parse_args();plot(a.root,a.output,a.seed)
