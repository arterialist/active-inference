# Obstacle-detour challenge (V4)

This module is the environment-first authorization record for V4. The prior
V3 composition is frozen and placed in deterministic MuJoCo geometry worlds.
The suite has a full-width wall, an L-corner, an alternating chicane, and a
compact maze; all fixtures contain no food target.

The minimum body delta is `body.obstacle_geometry`: two forward-biased physical
whisker/range measurements and contact from the barrier geometry. The minimum
neural delta is `sensory.obstacle_proximity` plus `motor.obstacle_reflex`.
The latter crosses left obstacle input to the right turn command, mirrors the
other side, and adds a shared brake/wall-presence gate to the existing PAULA
relay/muscle route.

Acceptance is full-tick and causal:

1. unchanged V3 approaches the geometry and contacts/stalls;
2. intact V4 detects it, deflects laterally, and remains contact-free;
3. zeroing only the body afferents restores the collision/stall;
4. zeroing only the reflex motor weights restores the collision/stall;
5. the chicane and maze report how many obstacle segments intact V4 reaches,
   exposing route-depth limitations for the next environmental upgrade.

The maintained harness is
`experiments/embodied_obstacle_detour_causal.py`; its JSON artifacts include
the sensor → PAULA → TL/TR → relay → muscle → MuJoCo trace for every neural
tick.
