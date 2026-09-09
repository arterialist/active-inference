"""Causal learned-weight branches of a moving full-population brain.

Every branch starts from the same executable neural and physical state. Only
the selected motor-context prediction weights change. Adaptation continues.
No measured error or action label is fed into the network by this script.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode
from .proprioceptive_loop_probe import record_loop
from .proprioceptive_body_comparison import audit_body_replay
from .predictive_bridge_probe import audit_record
from .predictive_weight_transplant import arrange_weights, install_weights
from ..components.body.radian_research_rower import RadianResearchRower
from ..core.runtime_checkpoint import load_checkpoint, save_checkpoint
from neuron.neuron import setup_neuron_logger


class BranchNetwork:
    """Use the checkpoint's own RNG streams through the recording interface."""
    def __init__(self, branch):
        self.branch=branch

    @property
    def network(self):
        return self.branch.network.network

    @property
    def current_tick(self):
        return self.branch.network.current_tick

    def set_external_input(self,*args):
        self.branch.network.set_external_input(*args)

    def run_tick(self):
        return self.branch.step()


def run(source,output,*,condition='learned',ticks=512):
    if condition not in ('learned','birth','shuffled') or not 64<=ticks<=1200:
        raise ValueError('Invalid bounded weight intervention')
    source,output=Path(source).resolve(),Path(output).resolve()
    m=json.loads((source/'manifest.json').read_text())
    parent=Path(m['parent'])
    cfg=json.loads((parent/'config.json').read_text())
    if shutil.disk_usage(output.parent).free<2*1024**3:
        raise OSError('Keep 2 GiB free')
    setup_neuron_logger('CRITICAL')
    branch=load_checkpoint(source/'final.neural-checkpoint',trusted=True)
    bridge=m['bridge']; net=branch.network
    with np.load(source/'closed-loop.npz') as z:
        body_state=z['physical_after'][-1].copy()
        learned=z['weights'][-1].copy()
        birth=z['start_weights'].copy()
    actual=np.array([[net.network.neurons[n].postsynaptic_points[s].u_i.info
        for s in net.network.neurons[n].prediction_ports] for n in bridge['prediction']])
    np.testing.assert_array_equal(actual,learned)
    replacement=learned if condition=='learned' else birth if condition=='birth' else arrange_weights(learned,shuffled=True)
    install_weights(net,bridge,replacement)
    body=RadianResearchRower();body.restore(body_state)
    output.mkdir(exist_ok=False)
    manifest=dict(source=str(source),parent=str(parent),condition=condition,ticks=ticks,
        source_checkpoint_sha256=digest(source/'final.neural-checkpoint'),
        source_record_sha256=digest(source/'closed-loop.npz'),
        source_manifest_sha256=digest(source/'manifest.json'),
        source_code_sha256=digest(__file__),motor=m['motor'],bridge=bridge,gain=m['gain'],
        fields=m['fields'],neuron_ids=m['neuron_ids'],
        limitation='One graph, three complete-state branches. Selected acquired contextual weights '
        'are reset or permuted; ongoing learning, return paths and physical feedback remain. '
        'Future sensory input can therefore diverge. This is not a matched-input open-loop assay '
        'or an autonomous action-selection result.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    save_checkpoint(branch,output/'initial.neural-checkpoint',sources=(__file__,))
    started=time.perf_counter()
    data=record_loop(BranchNetwork(branch),body,m['motor'],bridge,ticks,gain=m['gain'])
    np.savez_compressed(output/'closed-loop.npz',**data)
    neural=audit_record(data,cfg,bridge)
    physics=audit_body_replay(data,RadianResearchRower(),gain=m['gain'])
    save_checkpoint(branch,output/'final.neural-checkpoint',sources=(__file__,))
    result=dict(condition=condition,ticks=ticks,neural_residual=neural,
        physical_residual=physics,seconds=time.perf_counter()-started,
        raw_bytes=(output/'closed-loop.npz').stat().st_size,
        raw_sha256=digest(output/'closed-loop.npz'))
    (output/'summary.json').write_text(encode(result)+'\n');print(encode(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--condition',choices=('learned','birth','shuffled'),default='learned')
    p.add_argument('--ticks',type=int,default=512)
    run(**vars(p.parse_args()))
