# Live brain control plane

The live brain is one PAULA graph running in a MuJoCo worker. The browser is a
protocol client, not a generated copy of the graph.

```bash
uv run aif-live versions
uv run aif-live start --version v1 --port 8770 --run
uv run aif-live start --version v2 --port 8780 --run
uv run aif-live start --version v3 --port 8790 --run
uv run aif-live start --version v4 --port 8800 --world obstacle_detour --run
uv run aif-live status
uv run aif-live send v3 pause
uv run aif-live stop v3
uv run aif-live lab --port 8850
```

Each live server exposes:

- `/api/session`, `/api/schema`, `/api/health` — negotiated version and wire contracts;
- `/api/topology` — packed topology generated from the actual selected version;
- `/api/state` — decoded read-only state and camera frames;
- `/api/neuron/{id}` — bounded intracellular state plus incoming/outgoing synapse details;
- `/api/introspection` — static neuron/edge manifest plus the bounded trace range;
- `/api/trace` — seekable body/firing timeline;
- `/api/tick/{tick}?detail=neurons|synapses|all` — exact post-tick state;
- `/api/neuron/{id}/history` and `/api/synapse?...&tick=...` — historical intracellular and synaptic-point traces;
- `/api/command` — JSON control commands;
- WebSocket `ws://127.0.0.1:<http-port+1>/ws` — one binary frame per PAULA neural tick.

V4 state frames additionally expose the physical barrier rectangles and the
latest left/right obstacle, onset, and contact transducer values.  The same
fields are declared by `/api/schema`; they are measurements, not browser-side
control commands.

The binary frame is `uint32 tick | ceil(N/8) MSB-first spike bytes | four
little-endian float32 graded-muscle values`. A client must use the `hello`
message to obtain `N` and the IDs; no client-side neuron count or muscle ID is
authoritative.

The version profiles are strict topology selections:

- V1 (`sensory.olfactory_valence` + `motor.cpg_muscle`): 95 neurons;
- V2 (V1 + `learning.mushroom_body`): 299 neurons;
- V3 (V2 + FORAGE/EXPLORE/SLEEP arbiter and metabolic body): 345 neurons.
- V4 (V3 + physical barrier body transducer, bilateral obstacle onset, and
  PAULA obstacle reflex): 373 neurons. Its default live world is
`obstacle_detour`, whose default is a deterministic full-width MuJoCo wall
  with no food source; the live view therefore exposes collision-free
  deflection rather than lucky target discovery. The lab harness can also run
  the `obstacle_corner`, `obstacle_chicane`, and `obstacle_maze` fixtures and
  reports route depth.

Visual cortex, compass/path integration, belief core, and uncertainty are not
silently inherited by these profiles. Experimental navigation circuits remain
available only through an explicit `PrototypeSelection`/legacy build.
