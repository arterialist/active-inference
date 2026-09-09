"""Construction boundary for PAULA's authoritative vectorized input state.

The current shared runtime can retain stale bootstrap values in its dictionary
after consuming and clearing the vectorized values. Before an additive graph
builder invalidates that vector, synchronize the backing representation. This
does not change neural equations or clear real pending sensory input.
"""
from copy import deepcopy

import numpy as np


def synchronize_quiescent_external_inputs(topology):
    """Reject pending drives, then make a cache rebuild behavior-preserving.

Call only at a construction boundary, before changing ports or cache keys.
Validation completes before mutation. A missing/stale index is an error rather
than permission to guess which representation contains the current drive.
"""
    ext = topology.external_inputs
    vec = getattr(topology, '_ext_vec', None)
    if vec is None:
        values = list(ext.values())
    else:
        keys = vec['keys']
        if (len(keys) != len(ext) or set(keys) != set(ext)
                or vec['nkeys'] != len(keys)
                or vec['row_of'] != {key: i for i, key in enumerate(keys)}
                or vec['info'].shape != (len(keys),)
                or vec['plast'].shape != (len(keys),)
                or vec['mod'].shape != (len(keys), 2)):
            raise ValueError('External input cache does not match its interface')
        values = [dict(info=vec['info'][i], plast=vec['plast'][i], mod=vec['mod'][i])
                  for i in range(len(keys))]
    if any(np.any(np.asarray(value.get(field, 0.)) != 0.)
           for value in values for field in ('info', 'plast', 'mod')):
        raise ValueError('Cannot discard pending external input')
    if vec is not None:
        for key, value in zip(keys, values):
            ext[key].update(deepcopy(value))
