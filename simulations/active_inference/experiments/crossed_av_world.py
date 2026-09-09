"""Cross actual audiovisual recordings with a physical, conjunctive load rule.

The same video occurs with both sounds and opposing loads, and vice versa.
Pair identity controls the experimental load only. No identity, correct action,
pairing flag or host-computed error is delivered to a neuron. Context lamp 0
stays constant, preserving the previous brain configuration for this world test.
"""
import numpy as np

from . import context_organization as base
from .opponent_context import course as recorded_course


PRESENTATIONS=('both','vision','audio')


def crossed_features(features, video, audio, presentation='both', offset=0):
    if video not in (0,1) or audio not in (0,1) or presentation not in PRESENTATIONS:
        raise ValueError('Invalid crossed sensory presentation')
    if type(offset) is not int or not 0<=offset<300:
        raise ValueError('Offset must be an integer in 0..299')
    visual=np.asarray(features[video]['visual'])
    auditory=np.asarray(features[audio]['auditory'])
    if any(a.shape!=(300,96) or not np.isfinite(a).all() or np.any(a<0)
           for a in (visual,auditory)):
        raise ValueError('Need nonnegative finite 300-by-96 sensory recordings')
    visual=np.roll(visual,-offset,axis=0)
    auditory=np.roll(auditory,-offset,axis=0)
    return dict(visual=visual if presentation!='audio' else np.zeros_like(visual),
                auditory=auditory if presentation!='vision' else np.zeros_like(auditory))


def world_drive(features,video,audio,tick,*,reverse=False,presentation='both',offset=0):
    pair=crossed_features(features,video,audio,presentation,offset)
    # Reuse the verified physical transducer. Both entries contain the SAME
    # sensory arrays: the selected index changes only the imposed load sign.
    return base.physical_drive([pair,pair],0,video^audio^int(reverse),tick)


def course(net,arm,features,groups,selected,video,audio,*,reverse=False,
           presentation='both',offset=0,ticks=364,delay=None):
    pair=crossed_features(features,video,audio,presentation,offset)
    return recorded_course(net,arm,[pair,pair],groups,selected,0,
        video^audio^int(reverse),ticks=ticks,delay=delay)


def schedule(repeats=4):
    """Balanced blocks, independent of graph seed and physical assignment."""
    if type(repeats) is not int or not 1<=repeats<=16:
        raise ValueError('Need 1..16 balanced acquisition blocks')
    rng=np.random.default_rng(91473)
    return [[(int(p)//2,int(p)%2) for p in rng.permutation(4)] for _ in range(repeats)]
