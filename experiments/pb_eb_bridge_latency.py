"""Audit PAULA timing before comparing a direct P-EN edge with a PB relay.

A dendritic delay is not the entire end-to-end latency: an ordinary graded
interneuron receives on one tick and releases on the next.  This tiny,
no-body harness sends the same impulse down a direct edge and a two-edge PB
path, records every tick, and asserts matched arrival.  It prevents a bridge
topology experiment from being mistaken for a one-tick delay sweep.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "neuron-model"))

from paula_agent import ckit as k


def _trace(*, bridge: bool, direct_delay: int = 4, upstream_delay: int = 1) -> list[float]:
    """Return target release after one source impulse in an ordinary PAULA circuit."""
    # IDs and terminal IDs deliberately differ: PAULA terminals are explicit
    # routing endpoints, so this is the same mechanism used by the compass.
    neurons = [
        k.neuron(1, r=1e9, c=2, lam=1, meta={"graded_gain": 1.0}),
        k.neuron(2, r=1e9, c=2, lam=1, meta={"graded_gain": 1.0}),
    ]
    synapses = [
        k.syn(1, 0, 1.0, 1), k.term(1, 5001),
        k.syn(2, 0, 1.0, direct_delay), k.term(2, 5002),
    ]
    connections = []
    if bridge:
        # The relay's one update/release tick means these distances sum to
        # direct_delay - 1, not direct_delay.
        downstream_delay = direct_delay - upstream_delay - 1
        neurons.append(k.neuron(3, r=1e9, c=2, lam=1, meta={"graded_gain": 1.0}))
        synapses.extend([
            k.syn(3, 0, 1.0, upstream_delay), k.term(3, 5003),
            k.syn(2, 1, 1.0, downstream_delay),
        ])
        connections.extend([k.conn(1, 3, 0, 5001), k.conn(3, 2, 1, 5003)])
    else:
        connections.append(k.conn(1, 2, 0, 5001))
    path = k.build(neurons, synapses, connections, [k.ext(1, 0)])
    network, core = k.load(path)
    cells = network.network.neurons
    output: list[float] = []
    for tick in range(12):
        network.set_external_input(1, 0, 1.0 if tick == 0 else 0.0)
        core.do_tick()
        output.append(float(cells[2].O))
    return output


def first_release(trace: list[float]) -> int:
    return next(tick for tick, value in enumerate(trace) if value > 0.0)


def main() -> int:
    direct = _trace(bridge=False)
    bridged = _trace(bridge=True)
    direct_tick = first_release(direct)
    bridge_tick = first_release(bridged)
    print(f"direct target release tick={direct_tick}; PB relay target release tick={bridge_tick}")
    assert direct_tick == bridge_tick, (direct, bridged)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
