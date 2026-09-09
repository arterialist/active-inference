"""Finish a recorded branch after the old float32 post-audit failed.

Do not change the original producer or raw evidence. Reload its complete
initial checkpoint, restore its physical state, reproduce every recorded array
exactly with branch-local RNG, run the corrected independent audit, then save
the missing final runtime. No duplicate raw recording is written.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode
from .proprioceptive_learning_probe import BranchNetwork
from .proprioceptive_loop_probe import record_loop
from .proprioceptive_learning_audit import verify_physics
from .predictive_bridge_probe import audit_record
from ..components.body.radian_research_rower import RadianResearchRower
from ..core.runtime_checkpoint import load_checkpoint,save_checkpoint
from neuron.neuron import setup_neuron_logger


def run(source):
    source=Path(source).resolve()
    if (source/'final.neural-checkpoint').exists() or (source/'recovery.json').exists():
        raise FileExistsError('Do not overwrite a completed recovery')
    m=json.loads((source/'manifest.json').read_text())
    cfg=json.loads((Path(m['parent'])/'config.json').read_text())
    setup_neuron_logger('CRITICAL')
    branch=load_checkpoint(source/'initial.neural-checkpoint',trusted=True)
    with np.load(source/'closed-loop.npz') as z:old={k:z[k] for k in z.files}
    body=RadianResearchRower();body.restore(old['physical_before'][0])
    started=time.perf_counter()
    data=record_loop(BranchNetwork(branch),body,m['motor'],m['bridge'],m['ticks'],gain=m['gain'])
    if set(data)!=set(old):raise ValueError('Recorded schema differs')
    for key in data:np.testing.assert_array_equal(data[key],old[key],err_msg=key)
    neural=audit_record(data,cfg,m['bridge']);physical=verify_physics(data,m['gain'])
    save_checkpoint(branch,source/'final.neural-checkpoint',sources=(__file__,))
    result=dict(neural_residual=neural,physical_residual=physical,exact_replayed_ticks=m['ticks'],
        exact_replayed_arrays=sorted(data),raw_sha256=digest(source/'closed-loop.npz'),
        final_checkpoint_sha256=digest(source/'final.neural-checkpoint'),seconds=time.perf_counter()-started,
        cause='Original post-run actuator audit multiplied float32 before conversion. '
              'Physical bridge converts first to float64. Independent audit mirrors that boundary. '
              'All raw data retained unchanged; final runtime recovered by exact full-field replay.')
    (source/'recovery.json').write_text(encode(result)+'\n');print(encode(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    run(**vars(p.parse_args()))
