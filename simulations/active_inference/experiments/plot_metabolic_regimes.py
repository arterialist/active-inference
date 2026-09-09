"""Static scientific figure from the completed metabolic factorial analysis."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def run(source):
    source = Path(source)
    with np.load(source/'regimes-per-tick.npz') as z:
        fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True, sharey='row')
        paper = '#f7f4ed'; fig.set_facecolor(paper)
        for col, connected in enumerate((True, False)):
            for fed, style, color in ((False, '--', '#777777'), (True, '-', '#153e55')):
                prefix = str((connected, fed))+'/'
                e = z[prefix+'energy_after']; ctrl = z[prefix+'actuator_ctrl']
                time = (np.arange(len(e))+1)*.004
                label = 'Refed at 4.096 s' if fed else 'No meal'
                axes[0, col].plot(time, e[:, 0], style, color=color, lw=1.5, label=label)
                # Antagonistic actuator gears are +25/-25 in the retained body.
                axes[1, col].plot(time, 25*(ctrl[:, 0]-ctrl[:, 1]), style, color=color, lw=1.)
                axes[2, col].plot(time, 1000*z[prefix+'forward_progress'], style, color=color, lw=1.5)
            cpg = z[str((connected, True))+'/cpg']
            events = (np.flatnonzero(cpg[:, 0] > 0)+1)*.004
            for row in range(3):
                ax = axes[row, col]; ax.set_facecolor(paper)
                ax.axvline(4.096, color='#777777', lw=.8)
                ax.axhline(0, color='#aaaaaa', lw=.5)
                ax.spines[['right', 'top']].set_visible(False)
                ax.set_xlim(0, time[-1])
            axes[1, col].plot(events, np.full(len(events), -.08), '|', color='black',
                              transform=axes[1, col].get_xaxis_transform(), clip_on=False)
            axes[0, col].set_title('Neural energy feedback intact' if connected else 'Motor influence disconnected', loc='left')
            axes[0, col].legend(frameon=False, fontsize=9, loc='upper right')
            axes[2, col].set_xlabel('Time from acquired-state continuation, s')
            if not connected:
                axes[0, col].annotate('Budget failure begins at 0.748 s', xy=(.752, 0),
                    xytext=(1.45, 5.5), fontsize=9, arrowprops=dict(arrowstyle='->', lw=.8))
        for row, label in enumerate(('Usable energy, J', 'Left paddle net torque, Nm', 'Forward displacement, mm')):
            for col in range(2):
                axes[row, col].set_ylabel(label)
        fig.suptitle('Activity returns after digestion. Forward travel does not.', x=.07, ha='left', fontsize=16)
        fig.text(.07, .017, 'Each curve uses every recorded tick. Black marks under torque: internal phase-0 events.\n'
                 'One acquired PAULA graph. Illustrative energy accounting; movement after budget failure is not viable behavior.', fontsize=9)
        fig.tight_layout(rect=(0, .06, 1, .94))
        output = source/'regimes-shared-scale.png'
        if output.exists():
            raise FileExistsError(output)
        fig.savefig(output, dpi=150, facecolor=paper); plt.close(fig)
        print(output.resolve())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    run(**vars(p.parse_args()))
