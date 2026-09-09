"""Plot every acquired-state yoke case, with raw motion and unaveraged effects."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import context_organization as base
from .eligibility_reference_yoke import load


def render(analysis, output):
    analysis, output = (Path(p).resolve() for p in (analysis, output))
    if output.exists():
        raise FileExistsError(output)
    summary = json.loads((analysis/'summary.json').read_text())
    roots = []
    for path, digest in summary['sources'].items():
        if base.digest(path) != digest:
            raise ValueError('Audited source changed')
        if Path(path).name == 'manifest.json':
            roots.append(Path(path).parent)
    if len(roots) != 4:
        raise ValueError('Need the four-seed audited case family')
    evidence = {}; hashes = {}
    for root in roots:
        m = json.loads((root/'manifest.json').read_text())
        s = json.loads((root/'summary.json').read_text()); parent = Path(m['parent'])
        for row in s['rows']:
            key = (m['seed'], row['video'], row['audio'])
            path = root/row['file']; hashes[str(path)] = row['sha256']
            evidence.setdefault(key, {})[row['condition']] = load(path, row['sha256'])
            ref = row['references']['intact']; path = parent/ref['file']
            hashes[str(path)] = ref['sha256']
            evidence[key]['intact'] = load(path, ref['sha256'])
    # Common vertical scales within each measurement across all 16 cases.
    angle = np.concatenate([z['body'][:, 1] for b in evidence.values() for z in b.values()])
    bodily = np.concatenate([1e6*(b['closed']['body'][:, 1]-b['yoked']['body'][:, 1]) for b in evidence.values()])
    error = np.concatenate([b['yoked']['errors'][:, 0, 0]-b['intact']['errors'][:, 0, 0] for b in evidence.values()])
    def limits(values):
        low, high = float(values.min()), float(values.max())
        pad = max(high-low, 1e-12)*.08
        return low-pad, high+pad
    scales = [limits(angle), limits(bodily), limits(np.r_[error, 0.])]
    output.mkdir(); files = []
    for seed in sorted({k[0] for k in evidence}):
        fig, axes = plt.subplots(4, 3, figsize=(13, 10), sharex=True, layout='constrained')
        for i, (v, a) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            b = evidence[seed, v, a]; t = np.arange(96)
            for c, color, style in (('intact', '#555555', '-'), ('closed', '#005ca9', '--'), ('yoked', '#bf4d00', ':')):
                axes[i, 0].plot(t, b[c]['body'][:, 1], color=color, ls=style, label=c, lw=1.4)
            axes[i, 1].plot(t, 1e6*(b['closed']['body'][:, 1]-b['yoked']['body'][:, 1]), color='#005ca9')
            axes[i, 2].plot(t, b['yoked']['errors'][:, 0, 0]-b['intact']['errors'][:, 0, 0],
                            color='#bf4d00', label='Internal learning intervention')
            axes[i, 2].plot(t, b['closed']['errors'][:, 0, 0]-b['yoked']['errors'][:, 0, 0],
                            color='#005ca9', ls='--', label='Changed bodily feedback')
            for j, ax in enumerate(axes[i]):
                ax.set_ylim(*scales[j]); ax.set_xlim(0, 95)
                ax.axhline(0, color='#aaaaaa', lw=.5); ax.grid(alpha=.15)
                ax.set_xlabel('Recorded tick, 4 ms per physical step')
            axes[i, 0].set_ylabel(f'Video {v} / audio {a}\nJoint angle [rad]')
            axes[i, 1].set_ylabel('Angle difference [µrad]')
            axes[i, 2].set_ylabel('Error difference\n[model units]')
        axes[0, 0].set_title('Actual freely moving body')
        axes[0, 1].set_title('Motion caused by changed bodily feedback')
        axes[0, 2].set_title('Predictor 0 error used for learning')
        axes[0, 0].legend(fontsize=8)
        axes[0, 2].legend(fontsize=8)
        fig.suptitle(f'Acquired-state sensory substitution • graph seed {seed}\n'
                     'Same starting brain and body. Plasticity active. Each row is a separate replay.', fontsize=13)
        name = f'body-learning-yoke-seed{seed}.png'; fig.savefig(output/name, dpi=150); plt.close(fig)
        files.append(name)
    manifest = dict(files=files, sources=hashes, analysis_sha256=base.digest(analysis/'summary.json'),
                    producer_sha256=base.digest(__file__), scales=scales,
                    limits='All 96 recorded ticks, no temporal smoothing or averaging. Common measurement scales across seeds. '
                    'Raw angle curves nearly overlap; the middle column magnifies their physical difference. '
                    'Third column shows predictor 0; the audit includes both predictors and every neuron.')
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')
    return files


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('analysis', type=Path); p.add_argument('output', type=Path)
    a = p.parse_args(); print(render(a.analysis, a.output))
