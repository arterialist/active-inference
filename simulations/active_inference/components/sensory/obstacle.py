"""PAULA bilateral obstacle-proximity and temporal-onset populations."""

from __future__ import annotations

import numpy as np

from paula_agent import ckit as k


N_OBSTACLE = 6
OBSTACLE_THRESHOLDS = np.linspace(0.20, 1.50, N_OBSTACLE)
OBL = [88400 + i for i in range(N_OBSTACLE)]
OBR = [88410 + i for i in range(N_OBSTACLE)]
OBDL = [88420 + i for i in range(N_OBSTACLE)]
OBDR = [88430 + i for i in range(N_OBSTACLE)]


def _term(nid: int) -> int:
    return 20000 + int(nid)


def parts(ne, sy, conns, ex, *, onset_delay: int = 6,
          proximity_gain: float = 1.0, onset_gain: float = 1.5):
    """Append ordinary PAULA sensory cells.

    ``OBL``/``OBR`` are the raw bilateral range populations.  ``OBDL``/``OBDR``
    compare the present afferent spikes against a delayed copy of the same
    physical signal, making obstacle onset inspectable without a Python edge
    detector.  The only external ports are the raw left/right body currents.
    """
    for population in (OBL, OBR):
        for i, nid in enumerate(population):
            ne.append(k.neuron(nid, r=float(OBSTACLE_THRESHOLDS[i]), c=2, lam=3))
            sy.extend((k.syn(nid, 0, float(proximity_gain), 1), k.term(nid, _term(nid))))
            ex.append(k.ext(nid, 0))

    # A delayed copy is a normal PAULA synapse from each range cell.  It is not
    # a host-side derivative, and remains visible in the topology/replay.
    for raw, derivative in ((OBL, OBDL), (OBR, OBDR)):
        for i, nid in enumerate(derivative):
            ne.append(k.neuron(nid, r=float(OBSTACLE_THRESHOLDS[i]) * 0.65, c=2, lam=2))
            sy.extend((k.syn(nid, 0, float(onset_gain), 1),
                       k.syn(nid, 1, -float(onset_gain), int(onset_delay)),
                       k.term(nid, _term(nid))))
            conns.append(k.conn(raw[i], nid, 0, _term(raw[i])))
            conns.append(k.conn(raw[i], nid, 1, _term(raw[i])))
    return ne, sy, conns, ex
