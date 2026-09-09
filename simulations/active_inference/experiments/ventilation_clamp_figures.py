"""Plot recorded neural signals and actual resources without dropping ticks."""
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
    if output.exists():raise FileExistsError(output)
    summary=json.loads((root/'summary.json').read_text())
    if base.digest(root/'per-tick.npz')!=summary['sha256']:raise ValueError('Changed plot evidence')
    for p,h in summary['sources'].items():
        if base.digest(p)!=h:raise ValueError('Changed underlying evidence: '+p)
    data=np.load(root/'per-tick.npz');t=np.arange(1,1025)*.004;output.mkdir();figures={}
    fig,axes=plt.subplots(4,4,figsize=(14,8),sharex=True,sharey='row',layout='constrained')
    for j,c in enumerate(summary['cases']):
        seed=c['seed'];alarm=c['roles']['energy_alarm'][0]
        for name,color,style,label in (('held-low','#126ca4','-','Energy feedback intact'),
                                       ('energy-cut','#b34f13','--','Energy muscle projections cut')):
            key=f's{seed}_{name}';organs=data[key+'_organs'];ids=list(data[key+'_neuron_ids'])
            for i,v in enumerate((organs[:,6],organs[:,10],data[key+'_outputs'][:,ids.index(alarm)],organs[:,0])):
                axes[i,j].plot(t,v,color=color,ls=style,lw=1.25,label=label)
        axes[0,j].set_title(f'Seed {seed}');axes[-1,j].set_xlabel('Time after branching (s)')
    for i,label in enumerate(('Actual energy (J)','Accrued energy debt (J)',
                              'Energy-alarm neural output','Actual oxygen (mL)')):
        axes[i,0].set_ylabel(label)
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False);ax.axhline(0,color='#bbbbbb',lw=.5)
    fig.suptitle('Energy feedback restrains expenditure driven by a fixed low-oxygen reading\n'
                 'Both brains still sense actual energy; only two incoming muscle weights differ',fontsize=12)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2,frameon=False)
    p=output/'energy-conflict.png';fig.savefig(p,dpi=150);plt.close(fig);figures[p.name]=base.digest(p)
    fig,axes=plt.subplots(4,4,figsize=(14,8),sharex=True,sharey='row',layout='constrained')
    styles=(('intact','#126ca4','-','Actual delayed oxygen'),('held-initial','#555555','--','Held at initial reading'),
            ('held-low','#b34f13','-','Held low'),('held-high','#577454',':','Held high'))
    for j,c in enumerate(summary['cases']):
        seed=c['seed'];comp=c['roles']['oxygen_comparator'][0]
        for name,color,style,label in styles:
            key=f's{seed}_{name}';organs=data[key+'_organs'];ids=list(data[key+'_neuron_ids'])
            for i,v in enumerate((data[key+'_organ_drive'][:,2],data[key+'_outputs'][:,ids.index(comp)],organs[:,0],organs[:,6])):
                axes[i,j].plot(t,v,color=color,ls=style,lw=1.1,label=label)
        axes[0,j].set_title(f'Seed {seed}');axes[-1,j].set_xlabel('Time after branching (s)')
    for i,label in enumerate(('Oxygen signal to brain','Oxygen-comparator output','Actual oxygen (mL)','Actual energy (J)')):
        axes[i,0].set_ylabel(label)
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False);ax.axhline(0,color='#bbbbbb',lw=.5)
    fig.suptitle('Sensory substitution changes recruitment, but varying feedback is not best in this fixed world\n'
                 'Raw bodily resources continue evolving in every condition',fontsize=12)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=4,frameon=False)
    p=output/'oxygen-clamps.png';fig.savefig(p,dpi=150);plt.close(fig);figures[p.name]=base.digest(p);data.close()
    (output/'manifest.json').write_text(base.encode(dict(figures=figures,all_samples=True,
        inputs={str(root/'summary.json'):base.digest(root/'summary.json'),str(root/'per-tick.npz'):summary['sha256']},
        source_sha256=base.digest(__file__)))+'\n')
    print(output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('output')
    a=p.parse_args();render(a.root,a.output)
