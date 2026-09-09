"""All-checkpoint recall raster, with no temporal averaging or selected seeds.

Tiles are separate weight-transfer diagnostic replays, not continuous body
time. This figure displays signed neural prediction, not EEG or correctness
of embodied action. Complete acquired-body controls remain in the input audit.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from . import context_organization as base


def render(comparison, output):
    comparison, output = map(lambda p: Path(p).resolve(), (comparison, output))
    if output.exists(): raise FileExistsError(output)
    summary = json.loads((comparison/'summary.json').read_text())
    for p,h in summary['sources'].items():
        if base.digest(p) != h: raise ValueError('Comparison source changed')
    seeds = sorted({r['identity'][0] for r in summary['groups']})
    labels = ('reference','competition'); rasters = {}
    with np.load(comparison/'per-tick.npz') as z:
        for seed in seeds:
            for label in labels:
                parts = []
                for b in range(1,17):
                    x = z[f'{seed}_{b}_resting_learned_{label}_pairs']
                    if x.shape != (96,4) or not np.isfinite(x).all():
                        raise ValueError('Incomplete or invalid raster')
                    parts.append((x*np.array([1,-1,-1,1])).T)
                rasters[seed,label] = np.concatenate(parts,axis=1)
    vmax = max(float(abs(x).max()) for x in rasters.values()) or 1.
    cmap = LinearSegmentedColormap.from_list('signed_recall',['#9f3f20','#fffdf5','#14647a'])
    fig, axs = plt.subplots(len(seeds),2,figsize=(18,2.0*len(seeds)+1.6),layout='constrained',squeeze=False)
    for row,seed in enumerate(seeds):
        for col,label in enumerate(labels):
            ax = axs[row,col]
            im = ax.imshow(rasters[seed,label],vmin=-vmax,vmax=vmax,cmap=cmap,
                           interpolation='nearest',aspect='auto',extent=(0,16,3.5,-.5))
            for block in range(16):
                ax.axvline(block,color='#222222',lw=.45)
                ax.axvline(block+64/96,color='#333333',ls=':',lw=.6)
            ax.set_xticks(np.arange(16)+.5,labels=np.arange(1,17))
            ax.set_yticks(range(4),labels=['00','01','10','11'])
            ax.set(ylabel=f'Seed {seed}\nVideo/audio pair',xlabel='Acquisition block, followed by a separate 96-tick replay')
            if row == 0: ax.set_title('Without local competition' if col == 0 else 'With local competition')
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.7,pad=.015,label='Prediction aligned with actual load: negative = wrong direction, positive = correct')
    fig.suptitle('Does local competition preserve joint audiovisual recall?',fontsize=16)
    fig.supxlabel('Every tick 0..95 in every tile. Dotted line at tick 64: new body afferents become available.\n'
                  'Rust = wrong direction; paper = zero; teal = correct direction. One shared magnitude scale.\n'
                  'These are neural predictions, not a score of physical control or independent time-continuous episodes.',fontsize=10)
    output.mkdir(); path=output/'competition-recall-course.png'
    fig.savefig(path,dpi=250);plt.close(fig)
    result=dict(image=str(path),image_sha256=base.digest(path),seeds=seeds,
        comparison_sha256=base.digest(comparison/'summary.json'),data_sha256=base.digest(comparison/'per-tick.npz'),
        producer_sha256=base.digest(__file__),limits=__doc__)
    (output/'manifest.json').write_text(base.encode(result)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('comparison',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();render(a.comparison,a.output)
