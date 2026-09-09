"""PAULA fragment for V3 metabolic interoception and sleep arbitration."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from paula_agent import ckit as k

# ``aif_agent3d.py`` loads its moved arbiter as the short module name ``ar``
# before loading this fragment.  The fallback keeps this file importable on its
# own and from path-based legacy tests where the repository package is absent.
if "ar" in sys.modules:
    ar = sys.modules["ar"]
else:
    try:
        from simulations.active_inference.components.arbitration import paula as ar
    except ModuleNotFoundError:
        _source = Path(__file__).resolve().parent / "paula.py"
        _spec = spec_from_file_location("_aif_paula_component", _source)
        if _spec is None or _spec.loader is None:  # pragma: no cover - import failure guard
            raise ImportError(f"Cannot load moved arbiter implementation: {_source}")
        ar = module_from_spec(_spec)
        _spec.loader.exec_module(ar)
        del _source, _spec


GUT_AFFERENTS = [88300 + i for i in range(4)]
ENERGY_AFFERENTS = [88310 + i for i in range(4)]
LOW_ENERGY_AFFERENTS = [88320 + i for i in range(4)]
DIGESTION_AFFERENTS = [88330 + i for i in range(4)]


def _count(sy, nid):
    return sum(1 for item in sy if item.get("neuron_id") == nid and item.get("type") == "postsynaptic")


def _afferent(ne, sy, conns, ex, ids, *, threshold=0.45, lam=4.0):
    for i, nid in enumerate(ids):
        ne.append(k.neuron(nid, r=threshold + 0.10 * i, c=2, lam=lam))
        sy.append(k.syn(nid, 0, 1.0, 1)); ex.append(k.ext(nid, 0))
        sy.append(k.term(nid, 20000 + nid))


def parts(ne, sy, conns, ex, *, w_sleep_gut=2.5, w_sleep_energy=0.5,
          w_sleep_low=0.0, w_forage_low=0.5, w_explore_energy=2.0,
          w_sleep_hazard=-4.0):
    """Append body afferents and connect them to the existing PAULA WTA.

    ``aif_arbiter.build(sleep=True)`` must have been called first.  The body
    values arrive through ordinary external sensory ports; all competition and
    inhibition below is neural.
    """
    _afferent(ne, sy, conns, ex, GUT_AFFERENTS)
    _afferent(ne, sy, conns, ex, ENERGY_AFFERENTS)
    _afferent(ne, sy, conns, ex, LOW_ENERGY_AFFERENTS)
    _afferent(ne, sy, conns, ex, DIGESTION_AFFERENTS)

    for target in ar.SLEEP_MODE:
        for source in GUT_AFFERENTS:
            sid = _count(sy, target); sy.append(k.syn(target, sid, w_sleep_gut, 1))
            conns.append(k.conn(source, target, sid, 20000 + source))
        for source in ENERGY_AFFERENTS:
            sid = _count(sy, target); sy.append(k.syn(target, sid, w_sleep_energy, 1))
            conns.append(k.conn(source, target, sid, 20000 + source))
        for source in LOW_ENERGY_AFFERENTS:
            sid = _count(sy, target); sy.append(k.syn(target, sid, w_sleep_low, 1))
            conns.append(k.conn(source, target, sid, 20000 + source))
        # Aversive contact wakes the animal so a sleeping body can escape.
        sid = _count(sy, target); sy.append(k.syn(target, sid, w_sleep_hazard, 1))
        conns.append(k.conn(84600, target, sid, 20000 + 84600))

    for target in ar.MODE[0]:
        for source in LOW_ENERGY_AFFERENTS:
            sid = _count(sy, target); sy.append(k.syn(target, sid, w_forage_low, 1))
            conns.append(k.conn(source, target, sid, 20000 + source))

    # Strict V3 intentionally omits the compass/visual uncertainty ladder.
    # Without a replacement, EXPLORE is a label with no causal input and the
    # animal can remain in FORAGE forever.  Usable energy is the minimal body
    # transducer for exploratory readiness: energy availability recruits the
    # existing EXPLORE WTA cells, while low energy recruits FORAGE above.  The
    # WTA and all motor consequences remain ordinary PAULA synapses.
    for target in ar.MODE[2]:
        for source in ENERGY_AFFERENTS:
            sid = _count(sy, target); sy.append(k.syn(target, sid, w_explore_energy, 1))
            conns.append(k.conn(source, target, sid, 20000 + source))
