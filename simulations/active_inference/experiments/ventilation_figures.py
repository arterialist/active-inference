"""Render every recorded resource sample from the physical-necessity screen."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .ventilation_screen import digest, audit, SEEDS


def render(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    manifest=json.loads((root/'manifest.json').read_text())
    rows={r['key']:r for r in manifest['cases']}
    styles={'natural':('Recorded muscles','#252525','-'),
            'double':('2x replay control','#126ca4','-'),
            'quadruple':('4x replay control','#b34f13','-'),
            'silent':('No activation','#777777',':')}
    fields=(('Oxygen reserve (mL)',0),('Oxygen deficit accrued (mL)',4),
            ('Energy reserve (J)',6),('Energy deficit accrued (J)',10))
    fig,axes=plt.subplots(4,4,figsize=(14,9),sharex=True,sharey='col',layout='constrained')
    inputs={str(root/'manifest.json'):digest(root/'manifest.json')}
    for i,seed in enumerate(SEEDS):
        for kind,(label,color,style) in styles.items():
            row=rows[f's{seed}-return-current-{kind}']; p=root/row['file']
            if digest(p)!=row['sha256']: raise ValueError('Changed trace: '+str(p))
            with np.load(p) as f: z={k:f[k] for k in f.files}
            audit(z); inputs[str(p)]=row['sha256']
            t=np.arange(1,len(z['organs'])+1)*.004
            for j,(_,col) in enumerate(fields):
                axes[i,j].plot(t,z['organs'][:,col],color=color,ls=style,lw=1.2,label=label)
        for j,(title,_) in enumerate(fields):
            ax=axes[i,j]; ax.axhline(0,color='#bbbbbb',lw=.5)
            ax.spines[['top','right']].set_visible(False)
            if i==0: ax.set_title(title,fontsize=10)
            if i==3: ax.set_xlabel('Time since replay start (s)')
            if j==0: ax.set_ylabel(f'Seed {seed}')
    fig.suptitle('Physical replay: too little movement loses oxygen; excess activation spends energy\n'
                 'Organs receive motion only. No new neural feedback was tested.',fontsize=13)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    output.mkdir(); path=output/'ventilation-resources.png'; fig.savefig(path,dpi=160); plt.close(fig)
    (output/'manifest.json').write_text(json.dumps(dict(inputs=inputs,source_sha256=digest(__file__),
        figure=path.name,sha256=digest(path),all_samples=True),indent=2)+'\n')
    print(path)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();render(a.root,a.output)
