"""PAULA obstacle avoidance and wall-following motor fragment."""

from __future__ import annotations

from paula_agent import ckit as k

# Keep the numeric contract local as well as in the sensory module.  The
# legacy agent is intentionally loadable by file path (without a Python
# package), so a relative import here would make the modular fragment less
# reusable in isolated harnesses.
OBL = [88400 + i for i in range(6)]
OBR = [88410 + i for i in range(6)]
OBDL = [88420 + i for i in range(6)]
OBDR = [88430 + i for i in range(6)]


OBS_LEFT = 88440
OBS_RIGHT = 88441
OBS_BRAKE = 88442
OBS_WALL = 88443


def _next_syn(sy, nid: int) -> int:
    return sum(1 for item in sy if item.get("neuron_id") == nid and item.get("type") == "postsynaptic")


def _wire(sy, conns, source: int, target: int, weight: float) -> None:
    sid = _next_syn(sy, target)
    sy.append(k.syn(target, sid, float(weight), 1))
    conns.append(k.conn(source, target, sid, 20000 + int(source)))


def parts(ne, sy, conns, ex, *, turn_gain: float = 3.4,
          onset_turn_gain: float = 2.2, brake_gain: float = -1.15,
          wall_gain: float = 0.8, turn_left: int | None = None,
          turn_right: int | None = None, relays: dict[int, list[int]] | None = None):
    """Append bilateral obstacle command cells and converge them on the motor core.

    The left afferent crosses to the right turn command (and vice versa), as in
    the established toxin-avoidance route.  A shared brake reduces forward
    drive while an obstacle is close; it does not choose a pose or a heading.
    ``OBS_WALL`` is a low-gain persistent gate: its bilateral input keeps the
    chosen side biased while a body is alongside a wall, instead of requiring
    a one-tick contact pulse.
    """
    if turn_left is None:
        turn_left = 83382  # aif_agent3d.TL
    if turn_right is None:
        turn_right = 83383  # aif_agent3d.TR
    if relays is None:
        relays = {}

    for nid, raw, onset in (
        (OBS_LEFT, OBL, OBDL),
        (OBS_RIGHT, OBR, OBDR),
    ):
        ne.append(k.neuron(nid, r=0.55, c=2, lam=3))
        j = 0
        for source in raw:
            sy.append(k.syn(nid, j, 1.05, 1)); conns.append(k.conn(source, nid, j, 20000 + source)); j += 1
        for source in onset:
            sy.append(k.syn(nid, j, float(onset_turn_gain), 1)); conns.append(k.conn(source, nid, j, 20000 + source)); j += 1
        sy.append(k.term(nid, 20000 + nid))

    ne.append(k.neuron(OBS_BRAKE, r=0.50, c=1, lam=3))
    j = 0
    for source in OBL + OBR:
        sy.append(k.syn(OBS_BRAKE, j, 0.90, 1)); conns.append(k.conn(source, OBS_BRAKE, j, 20000 + source)); j += 1
    sy.append(k.term(OBS_BRAKE, 20000 + OBS_BRAKE))

    ne.append(k.neuron(OBS_WALL, r=0.40, c=1, lam=4))
    j = 0
    for source in OBL + OBR:
        sy.append(k.syn(OBS_WALL, j, 0.45, 1)); conns.append(k.conn(source, OBS_WALL, j, 20000 + source)); j += 1
    sy.append(k.term(OBS_WALL, 20000 + OBS_WALL))

    # Crossed avoidance route.  The low-gain wall signal goes to the same
    # command so it can maintain a detour when raw onset has decayed.
    _wire(sy, conns, OBS_LEFT, turn_right, turn_gain)
    _wire(sy, conns, OBS_RIGHT, turn_left, turn_gain)
    _wire(sy, conns, OBS_LEFT, turn_right, wall_gain)
    _wire(sy, conns, OBS_RIGHT, turn_left, wall_gain)
    _wire(sy, conns, OBS_BRAKE, turn_left, -0.20)
    _wire(sy, conns, OBS_BRAKE, turn_right, -0.20)
    for muscles in relays.values():
        for relay in muscles:
            _wire(sy, conns, OBS_BRAKE, relay, brake_gain)
            # A slower, bilateral wall-presence brake keeps the body in the
            # detour regime after the onset cells have adapted.  Steering is
            # still selected by the crossed side-specific outputs above.
            _wire(sy, conns, OBS_WALL, relay, -0.05 * abs(wall_gain))
    return ne, sy, conns, ex
