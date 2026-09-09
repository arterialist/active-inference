"""Explicit receptor stimulation schedules for timing-controlled experiments.

This is environmental input instrumentation, never a neural controller. It
only uses the ordinary external receptor ports. Schedules are saved verbatim.
"""
import numpy as np

TIMINGS=('synchronous','shared_delay','fixed_spread','resampled_spread','minority_first','majority_first')


def receptor_schedule(case,masks,seed,timing,ticks=64):
    if timing not in TIMINGS or ticks<32:raise ValueError('Invalid timing protocol')
    rng=np.random.default_rng(seed+31817+case['cue']*101+case['sample']*19)
    receptors=case['selected_receptors'];selected=set(receptors)
    if len(selected)!=len(receptors) or not selected<=set(range(1,33)):raise ValueError('Invalid visual receptor mask')
    own=set(masks['vision'][case['cue']]);out=np.zeros((ticks,32),np.uint8)
    offsets=rng.integers(0,4,len(receptors))
    for pulse in (0,8,16,24):
        if timing=='resampled_spread':offsets=rng.integers(0,4,len(receptors))
        for j,nid in enumerate(receptors):
            if timing=='synchronous':offset=0
            elif timing=='shared_delay':offset=3
            elif timing in ('fixed_spread','resampled_spread'):offset=int(offsets[j])
            elif timing=='minority_first':offset=3 if nid in own else 0
            else:offset=0 if nid in own else 3
            out[pulse+offset,nid-1]=1
    return out


class ScheduledReceptors:
    def __init__(self,net,schedule):
        self.net=net;self.network=net.network;self.start=net.current_tick
        self.schedule=np.asarray(schedule)
        if self.schedule.ndim!=2 or self.schedule.shape[1]!=32 or not np.isin(self.schedule,[0,1]).all():
            raise ValueError('Expected binary visual receptor schedule')

    def run_tick(self):
        relative=self.net.current_tick-self.start
        if not 0<=relative<len(self.schedule):raise ValueError('Schedule exhausted')
        for index in np.flatnonzero(self.schedule[relative]):self.net.set_external_input(int(index)+1,0,1.)
        return self.net.run_tick()
