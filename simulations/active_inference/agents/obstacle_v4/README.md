# Tactile-detour V4

V4 is the first agent whose added complexity is forced by a changed physical
rule: deterministic MuJoCo geometry blocks the established V3 route. The V4
world suite contains no food source, so random placement or respawn cannot turn
a collision test into an accidental foraging success.

The maintained geometries are:

- `head_on_wall`: required primitive, a full-width wall;
- `corner`: required compound case, an asymmetric L-corner;
- `chicane`: diagnostic stress case, three alternating bars;
- `maze`: diagnostic stress case, four alternating bars plus a cross-cap.

The body adds only the transducer needed by that rule. `World3D` exposes two
forward-biased whisker/range samples and a contact bit from the actual barrier
geometry. The neural delta is entirely PAULA:

- `sensory.obstacle_proximity`: six-cell left/right range populations and
  delayed onset populations;
- `motor.obstacle_reflex`: crossed opponent turn commands, a shared brake, and
  a persistent wall-follow contribution converging on the existing `TL/TR`
  and graded motor relays.

There is no visual cortex, compass, path integrator, or Python turn policy in
strict V4. The causal harness compares unchanged V3 against V4 and ablates the
body channel and reflex independently. Acceptance is trajectory-based:
unchanged V3 must contact/stall, while intact V4 must approach, fire the
tactile/reflex pathway, deflect laterally, and remain contact-free. The
chicane and maze additionally report route depth; they are deliberate probes
of the limit of this local reflex, not silently counted as solved navigation.
