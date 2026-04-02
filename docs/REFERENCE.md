# Active Inference Sensorimotor Simulations — Technical Reference

A framework for simulating artificial life through active inference. Organisms are assembled from PAULA spiking neurons wired by biological connectomes and embodied in MuJoCo physics environments. The first organism implemented is the nematode *C. elegans* (302 neurons, Cook et al. 2019 connectome).

**Upstream references**

| Resource | URL |
|----------|-----|
| PAULA paper | <https://al.arteriali.st/blog/paula-paper> |
| ALERM framework | <https://al.arteriali.st/blog/alerm-framework> |
| Neuron model source | <https://github.com/arterialist/neuron-model> |

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [Core Simulation Engine](#3-core-simulation-engine)
4. [PAULA Neuron Model](#4-paula-neuron-model)
5. [Connectome Pipeline](#5-connectome-pipeline)
6. [C. elegans Implementation](#6-c-elegans-implementation)
7. [Neuromodulation System](#7-neuromodulation-system)
8. [Motor Decoding](#8-motor-decoding)
9. [Evolutionary Optimisation](#9-evolutionary-optimisation)
10. [Scripts and Entry Points](#10-scripts-and-entry-points)
11. [Data, Logging, and Analysis](#11-data-logging-and-analysis)
12. [Configuration Reference](#12-configuration-reference)

---

## 1. Architecture Overview

### 1.1 High-Level Loop

```
Environment ──sensory stimuli──► Sensor Encoder
                                       │
                                  dict[str, float]
                                       │
                          ┌─────── Nervous System (PAULA) ───────┐
                          │  Sensory Neurons                     │
                          │       │                              │
                          │  Interneurons                        │
                          │       │                              │
                          │  Motor Neurons                       │
                          └──────────────────────────────────────┘
                                       │
                                 dict[str, float]
                                       │
                              Neuromuscular Junction
                                       │
                              MuJoCo Body Physics
                                       │
                                    BodyState
                                       │
                          ◄────── Environment Update
```

Each physics step:

1. The **environment** produces an `EnvironmentObservation` (chemical concentrations, contact forces, proprioception).
2. A **sensor encoder** maps the observation to a flat `dict[str, float]` keyed by neuron name.
3. The **nervous system** runs one or more PAULA ticks, injecting sensory inputs and reading motor outputs.
4. The **neuromuscular junction** translates motor activations to MuJoCo actuator commands.
5. The **body** advances one physics step and returns a new `BodyState`.
6. The **environment** updates based on the new body state (e.g. food consumption).

### 1.2 Module Layout

```
active-inference/
├── simulations/                    Core engine (organism-agnostic)
│   ├── engine.py                   SimulationEngine
│   ├── sensorimotor_loop.py        SensorimotorLoop + FreeEnergyTrace
│   ├── base_body.py                BaseBody, BodyState
│   ├── base_environment.py         BaseEnvironment, EnvironmentObservation
│   ├── base_nervous_system.py      BaseNervousSystem
│   ├── types.py                    Type aliases and protocols
│   ├── connectome_loader.py        ConnectomeData → PAULA NeuronNetwork
│   ├── paula_loader.py             Ensures neuron-model on sys.path
│   ├── run_log.py                  Run logging and post-hoc loading
│   ├── interactive/                Base interactive viewer
│   │   └── base.py                 BaseInteractiveViewer (ABC)
│   │
│   └── c_elegans/                  C. elegans organism
│       ├── simulation.py           CElegansEngine factory
│       ├── connectome.py           Cook 2019 connectome loader + cache
│       ├── neuron_mapping.py       CElegansNervousSystem (302 neurons)
│       ├── body.py                 CElegansBody (MuJoCo wrapper)
│       ├── body_model.xml          MJCF model (13 segments, 48 actuators)
│       ├── environment.py          AgarPlateEnvironment
│       ├── sensors.py              SensorEncoder
│       ├── muscles.py              NeuromuscularJunction
│       ├── config.py               Biological constants
│       └── interactive_viewer.py   2-D real-time viewer
│
├── scripts/
│   ├── run_c_elegans.py            Main simulation runner
│   ├── evolve_food_seeking.py      Differential evolution of neuromod gains
│   ├── analyze_neuromod_food_seeking.py  Post-hoc analysis
│   ├── apply_evolved_config.py     Write evolved params back to source
│   └── download_connectome.py      One-shot connectome cache
│
├── data/c_elegans/                 Cached connectome JSON
├── logs/                           Per-run output directories
└── evolved_food_seeking_config.json  Best evolved parameters
```

---

## 2. Theoretical Foundations

### 2.1 Active Inference

The simulation frames each organism as an active inference agent. At every tick, two processes run simultaneously:

- **Perception (plasticity):** Each PAULA synapse computes a prediction error `E_dir = received_input − u_i.info`. The postsynaptic learning rule `Δu_i = η × direction × |E_dir|` adjusts synaptic weights to minimise future surprise. Over time, the network's weights encode a generative model of the sensory statistics the organism encounters.

- **Action (motor output):** Motor neuron spikes drive muscle activations, physically moving the body. Movement changes the sensory stream, closing the perception–action loop. The organism's behaviour implicitly minimises prediction error (free energy) by seeking out predictable sensory states.

### 2.2 PAULA

PAULA (Predictive Adaptive Unsupervised Learning Agent, see §4) is a spiking neuron model that unifies:

- **Leaky integration** with configurable time constant.
- **Spike-timing-dependent plasticity (STDP)** via a directional learning window.
- **Retrograde signaling** from post- to presynaptic terminals.
- **Neuromodulation** of threshold, post-cooldown threshold, and learning window duration.
- **Dendritic cable propagation** with exponential decay and configurable delay.

### 2.3 ALERM

The ALERM (Architecture, Learning, Energy, Recall, Memory) framework proposes that biological constraints — metabolic cost, localised plasticity, temporal delays — are not limitations but indispensable generative regularisers required for robust intelligence. It formalises five continuously interacting system components:

| Component | Role |
|-----------|------|
| **A (Architecture)** | Physical capacity of the spatiotemporal graph G(V,E,d); defines the Markov Blanket boundary |
| **L (Learning)** | Continuous local adaptation of topology; no global gradients |
| **E (Energy)** | Metabolic budget that selects for sparse coding and temporal efficiency |
| **R (Recall)** | Temporal inference — evidence accumulation until an attractor threshold is crossed |
| **M (Memory)** | Emergent property of architecture: crystallised limit-cycle pathways (M ⊆ A) |

PAULA is the concrete implementation of ALERM. In the *C. elegans* simulation, the A↔L coupling is driven by two neuromodulators derived from the chemosensory gradient:

| Modulator | Trigger | Biological analogue | Effect on PAULA |
|-----------|---------|---------------------|-----------------|
| **M0** (stress) | dC/dt < 0 (concentration decreasing) | Octopamine, tyramine | Broadens learning window (t_ref ↑), lowers threshold (r ↓) → pirouette, explore |
| **M1** (reward) | dC/dt > 0 (concentration increasing) | Dopamine, serotonin | Narrows learning window (t_ref ↓), raises threshold slightly → crystallise current run |

These global signals are broadcast to all neurons via **volume transmission** after each network tick, modelling the diffuse monoamine signaling in *C. elegans*.

---

## 3. Core Simulation Engine

### 3.1 SimulationEngine

**File:** `simulations/engine.py`

Orchestrates one physics body, one environment, and one nervous system in a tick-based loop.

```python
class SimulationEngine:
    def __init__(
        body:                        BaseBody,
        environment:                 BaseEnvironment,
        nervous_system:              BaseNervousSystem,
        neural_ticks_per_physics_step: int = 2,
        on_step:                     StepCallback | None = None,
        record_neural_states:        bool = True,
        max_history:                 int = 200,
    )
```

**Key methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `reset()` | `SimulationStep` | Reset all three subsystems, return initial state |
| `step()` | `SimulationStep` | Advance one physics step (may run multiple neural ticks) |
| `run(n_steps, ...)` | `list[SimulationStep]` | Run *n* steps with optional progress bar |

Each `step()` call:

1. Reads body state and environment observation.
2. Flattens observation to `dict[str, float]` via `_observation_to_sensory_inputs`.
3. Runs `neural_ticks_per_physics_step` neural ticks, collecting the last motor output.
4. Steps the body with the motor output.
5. Steps the environment with the new body state.
6. Records timing and neural states into a `SimulationStep` dataclass.

**SimulationStep** (dataclass):

| Field | Type | Description |
|-------|------|-------------|
| `tick` | `int` | Global tick counter |
| `body_state` | `BodyState` | Position, orientation, joint angles, contacts |
| `observation` | `EnvironmentObservation` | Chemicals, touch forces, proprioception |
| `motor_outputs` | `dict[str, float]` | Per-muscle activation |
| `neural_states` | `dict[str, Any]` | Membrane potentials, firing flags |
| `elapsed_ms` | `float` | Wall-clock time for this step |

### 3.2 SensorimotorLoop

**File:** `simulations/sensorimotor_loop.py`

Wraps `SimulationEngine` with active-inference instrumentation.

```python
class SensorimotorLoop:
    def __init__(
        engine:           SimulationEngine,
        log_free_energy:  bool = True,
        on_step:          StepCallbackWithLoop | None = None,
    )
```

**FreeEnergyTrace** records per-tick:

- **Prediction error:** Mean absolute `E_dir` across all neurons (from `neural_states`). This is the raw surprise signal.
- **Motor entropy:** Shannon entropy of the motor output distribution. Low entropy = stereotyped behaviour; high entropy = disorganised.

The `run()` method supports optional **convergence monitoring**: if the mean prediction error over a rolling window drops below a threshold, the simulation stops early.

### 3.3 Abstract Interfaces

**BaseBody** (`base_body.py`):

| Abstract member | Type | Description |
|-----------------|------|-------------|
| `reset()` | method → `BodyState` | Reset to neutral pose |
| `step(muscle_activations)` | method → `BodyState` | Apply muscles, advance physics |
| `get_state()` | method → `BodyState` | Read state without stepping |
| `render(camera)` | method → `ndarray \| None` | RGB frame or None |
| `dt` | property → `float` | Physics timestep (seconds) |
| `joint_names` | property → `list[str]` | Controllable joints |
| `muscle_names` | property → `list[str]` | Actuated muscles |

**BodyState** (dataclass):

| Field | Type | Description |
|-------|------|-------------|
| `position` | `ndarray(3,)` | Centre-of-mass [x, y, z] |
| `orientation` | `ndarray(4,)` | Unit quaternion [w, x, y, z] |
| `joint_angles` | `dict[str, float]` | Named joint angles (radians) |
| `joint_velocities` | `dict[str, float]` | Angular velocities |
| `contact_forces` | `dict[str, ndarray]` | Named contact force 3-vectors |
| `head_position` | `ndarray(3,)` | Nose / head tip position |
| `extra` | `dict[str, Any]` | Organism-specific quantities |

**BaseEnvironment** (`base_environment.py`):

| Abstract member | Type | Description |
|-----------------|------|-------------|
| `reset()` | method → `EnvironmentObservation` | Initial observation |
| `step(body_state)` | method → `EnvironmentObservation` | Advance, return new observation |
| `render()` | method → `ndarray \| None` | Top-down RGB |

**EnvironmentObservation** (dataclass):

| Field | Type | Description |
|-------|------|-------------|
| `chemicals` | `dict[str, float]` | Scalar concentrations by molecule |
| `contact_forces` | `dict[str, ndarray]` | Touch forces at body sites |
| `proprioception` | `dict[str, float]` | Joint angles and velocities |
| `extra` | `dict[str, Any]` | Additional modalities |

**BaseNervousSystem** (`base_nervous_system.py`):

| Abstract member | Type | Description |
|-----------------|------|-------------|
| `reset()` | method → `None` | Reset all neurons |
| `tick(sensory_inputs, current_tick)` | method → `dict[str, float]` | One neural tick → motor outputs |
| `get_neuron_states()` | method → `dict[str, Any]` | Snapshot of membrane potentials + firing |
| `n_neurons` | property → `int` | Total neuron count |

---

## 4. PAULA Neuron Model

**Source:** `neuron-model/neuron/neuron.py`, `neuron-model/neuron/network.py`

### 4.1 NeuronParameters

Every PAULA neuron is configured with a `NeuronParameters` dataclass. Arrays are auto-resized to match `num_neuromodulators` in `__post_init__`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r_base` | `float` | 1.0 | Resting primary threshold |
| `b_base` | `float` | 1.2 | Post-cooldown threshold (higher than `r`, prevents immediate re-spiking) |
| `c` | `int` | 10 | Refractory period (ticks) |
| `lambda_param` | `float` | 20.0 | Membrane time constant (leaky integrator decay) |
| `p` | `float` | 1.0 | Spike output amplitude |
| `delta_decay` | `float` | 0.95 | Cable propagation decay per compartment |
| `eta_post` | `float` | 0.01 | Postsynaptic learning rate |
| `eta_retro` | `float` | 0.01 | Retrograde (presynaptic) learning rate |
| `beta_avg` | `float` | 0.999 | Firing rate EMA decay |
| `gamma` | `ndarray` | [0.99, 0.995] | M_vector EMA decay per modulator |
| `w_r` | `ndarray` | [-0.2, 0.05] | M_vector → primary threshold sensitivity |
| `w_b` | `ndarray` | [-0.2, 0.05] | M_vector → post-cooldown threshold sensitivity |
| `w_tref` | `ndarray` | [-20.0, 10.0] | M_vector → learning window sensitivity |
| `num_neuromodulators` | `int` | 2 | Dimensionality of M_vector |
| `num_inputs` | `int` | 10 | Number of postsynaptic points (synapses) |

**Global bounds:**

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_SYNAPTIC_WEIGHT` | 2.0 | Excitatory ceiling for `u_i.info` and `u_o.info` |
| `MIN_SYNAPTIC_WEIGHT` | 0.01 | Floor (prevents zero-weight deadlock) |
| `MAX_MEMBRANE_POTENTIAL` | 20.0 | Clamp for S |
| `MIN_MEMBRANE_POTENTIAL` | -20.0 | Clamp for S |

### 4.2 Synaptic Data Structures

**PostsynapticInputVector** (`u_i`): The receiving end of a synapse on the dendrite.

| Field | Type | Init | Description |
|-------|------|------|-------------|
| `info` | `float` | U(0.5, 1.5) | Synaptic efficacy for information |
| `plast` | `float` | U(0.5, 1.5) | Synaptic efficacy for plasticity signal |
| `adapt` | `ndarray` | U(0.1, 0.5) | Receptor sensitivity to each neuromodulator |

**PresynapticOutputVector** (`u_o`): The releasing end at the axon terminal.

| Field | Type | Init | Description |
|-------|------|------|-------------|
| `info` | `float` | p (1.0) | Spike amplitude transmitted |
| `mod` | `ndarray` | U(0.1, 0.5) | Neuromodulator release profile |

**PostsynapticPoint:** Wraps `u_i` + a local potential `float`. One per input synapse.

**PresynapticPoint:** Wraps `u_o` + retrograde input susceptibility `u_i_retro`. One per axon terminal.

### 4.3 Neuron State Variables

| Variable | Type | Initial | Description |
|----------|------|---------|-------------|
| `S` | `float` | 0.0 | Membrane potential at axon hillock |
| `O` | `float` | 0.0 | Output (p on spike, 0 otherwise) |
| `t_last_fire` | `float` | -∞ | Tick of last spike |
| `F_avg` | `float` | 0.0 | Long-term firing rate (EMA) |
| `M_vector` | `ndarray` | zeros | Neuromodulatory state (EMA) |
| `r` | `float` | r_base | Dynamic primary threshold |
| `b` | `float` | b_base | Dynamic post-cooldown threshold |
| `t_ref` | `float` | c × num_inputs | Dynamic learning window |

**Bounds for t_ref:**

- `upper_t_ref_bound = c × num_inputs` (widest learning window)
- `lower_t_ref_bound = 2 × c` (narrowest)

### 4.4 The tick() Method — Full State Transition

A single `tick(external_inputs, current_tick, dt=1.0)` call executes five phases:

#### Phase A — Neuromodulation and Dynamic Parameters

**A.1 Aggregate modulation input:**

```
total_adapt_signal = Σ_synapses( O_ext["mod"] ⊙ u_i.adapt )
```

Where `O_ext["mod"]` is the incoming neuromodulator concentration vector at each active synapse and `u_i.adapt` is that synapse's receptor sensitivity. Element-wise product, summed across all active synapses.

**A.2 Update M_vector (EMA):**

```
M(t+1) = γ ⊙ M(t) + (1 − γ) ⊙ total_adapt_signal
```

Per-element decay `γ` is typically [0.99, 0.995], so 1% and 0.5% of new signal mixes in per tick.

**A.3 Update firing rate (EMA):**

```
F_avg(t+1) = β_avg × F_avg(t) + (1 − β_avg) × O(t)
```

Where `O(t)` is the output from the *previous* tick.

**A.4 Dynamic thresholds:**

```
r(t) = r_base + w_r · M(t)
b(t) = b_base + w_b · M(t)
```

**A.5 Dynamic learning window:**

```
normalised_F = clip(F_avg × c, 0, 1)
t_ref_homeo  = upper_bound − (upper_bound − lower_bound) × normalised_F
t_ref(t)     = clip(t_ref_homeo + w_tref · M(t), lower_bound, upper_bound)
```

High average firing rate → homeostatic narrowing. Neuromodulators shift on top of that.

#### Phase B — Input Processing

For each synapse with non-zero input in `input_buffer`:

```
V_local = info_val × (u_i.info + u_i.plast)
```

The local potential is pushed onto a min-heap with arrival time:

```
arrival = current_tick + cable_distance[synapse_id]
```

#### Phase C — Signal Integration at Axon Hillock

Pop all signals that have arrived by `current_tick`:

```
I_t = Σ( V_initial × δ^distance )
```

Where `δ = delta_decay = 0.95`. Cable propagation attenuates signals exponentially with distance.

#### Phase D — Spike Generation

Leaky integrator update:

```
S(t+1) = S(t) + (dt / λ) × (−S(t) + I_t)
```

Threshold selection:

```
threshold = b   if (current_tick − t_last_fire) ≤ c   (refractory period)
            r   otherwise
```

Spike condition:

```
if S ≥ threshold  and  (current_tick − t_last_fire) ≥ c:
    O = p        # emit spike
    S = 0        # reset
    t_last_fire = current_tick
    → generate PresynapticReleaseEvent for each axon terminal
```

#### Phase E — Plasticity

For each synapse with input this tick:

**E.1 Prediction error:**

```
E_dir = [ info_received − u_i.info,
          plast_received − u_i.plast,
          mod_0_received, mod_1_received, ... ]
```

**E.2 Temporal direction:**

```
direction = +1   if (current_tick − t_last_fire) ≤ t_ref    (causal window)
            −1   otherwise                                    (anti-causal)
```

**E.3 Postsynaptic weight update (Hebbian STDP):**

```
Δu_i.info = η_post × direction × ‖E_dir‖ × u_i.info
u_i.info  = clip(u_i.info + Δu_i.info, 0.01, 2.0)
```

**E.4 Retrograde signaling:**

The error vector, scaled by direction, is sent back to the presynaptic neuron's axon terminal:

```
terminal.u_o.info += η_retro × E_dir[0]
terminal.u_o.mod  += η_retro × E_dir[2:]
```

Both clipped to bounds. This creates a bi-directional learning loop: the postsynaptic cell refines its receptive field, then tells the presynaptic cell to adjust its output accordingly.

### 4.5 NeuronNetwork

**File:** `neuron-model/neuron/network.py`

Manages a collection of `Neuron` objects with inter-neuron signal propagation.

**Event scheduling** uses a pair of calendar-queue wheels (circular buffers of size `max_delay + 1`). Events are placed in slots by `arrival_tick % wheel_size`, giving O(1) insertion and retrieval.

**run_tick()** proceeds as:

1. Extract events from current wheel slot.
2. Write external inputs directly to neuron `input_buffer` arrays.
3. Deliver presynaptic events via `fast_connection_cache` — a dict mapping `(source_id, terminal_id)` to a list of `(target_buffer_ref, synapse_index)` pairs. This avoids dictionary lookups in the hot loop.
4. Deliver retrograde events to target neurons.
5. Call `tick()` on every neuron, collecting output events.
6. Schedule new events into future wheel slots.
7. Record history (membrane potentials, firing flags, activity).

**Key performance optimisations:**

- Pre-allocated `input_buffer` numpy arrays (one per neuron, shape `(num_inputs, 2 + num_neuromodulators)`).
- `fast_connection_cache` stores direct numpy array references — signal delivery is a single indexed addition.
- Tuple events `(source_id, terminal_id, info_value)` for presynaptic release (avoids object allocation).
- Calendar-queue wheels replace priority queues for event scheduling.

### 4.6 Ablation Flags

The `ablation` parameter on `Neuron.__init__` accepts a comma-separated string:

| Flag | Effect |
|------|--------|
| `thresholds_frozen` | r and b do not update with M_vector |
| `tref_frozen` | t_ref stays at initial value |
| `directional_error_disabled` | Direction always +1 (no LTD) |
| `weight_update_disabled` | Postsynaptic weights frozen |
| `retrograde_disabled` | No retrograde signal generation |

---

## 5. Connectome Pipeline

### 5.1 ConnectomeData

**File:** `simulations/connectome_loader.py`

A connectome is represented as:

```python
@dataclass
class ConnectomeData:
    neurons:             list[NeuronInfo]
    chemical_edges:      list[SynapticEdge]
    gap_junction_edges:  list[SynapticEdge]
    name_to_info:        dict[str, NeuronInfo]   # built in __post_init__
```

**NeuronInfo:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | e.g. "ASEL", "DB3" |
| `neuron_type` | `str` | "sensory", "interneuron", "motor", "unknown" |
| `paula_id` | `int` | Stable integer ID for PAULA |
| `in_degree_chem` | `int` | Incoming chemical synapses |
| `in_degree_gap` | `int` | Incoming gap junctions |
| `out_degree_chem` | `int` | Outgoing chemical synapses |
| `out_degree_gap` | `int` | Outgoing gap junctions |

**SynapticEdge:**

| Field | Type | Description |
|-------|------|-------------|
| `pre_name` | `str` | Presynaptic neuron |
| `post_name` | `str` | Postsynaptic neuron |
| `synapse_type` | `str` | "chemical" or "gap_junction" |
| `weight` | `float` | Raw synapse count from the connectome |

### 5.2 build_paula_network()

Converts a `ConnectomeData` into a PAULA `NeuronNetwork` + a `name_to_paula_id` mapping.

```python
def build_paula_network(
    connectome:        ConnectomeData,
    base_params:       NeuronParameters | None = None,
    sensory_params:    NeuronParameters | None = None,
    motor_params:      NeuronParameters | None = None,
    interneuron_params: NeuronParameters | None = None,
    weight_max:        float = 5.0,
    log_level:         str = "WARNING",
    param_overrides:   dict[str, NeuronParameters] | None = None,
) -> tuple[NeuronNetwork, dict[str, int]]
```

Steps:

1. **Assign degrees** — computes in/out degrees and stable PAULA IDs.
2. **Create neurons** — selects parameter preset by type (sensory / motor / interneuron / base), with optional per-name overrides. `num_inputs` is set to the neuron's actual in-degree.
3. **Create connections** — for each edge, assigns a terminal ID on the presynaptic neuron and a synapse ID on the postsynaptic neuron. Chemical synapses get `distance = 2`, gap junctions get `distance = 1`.
4. **Assemble network** — builds an `_EmptyTopology` (duck-type for `NetworkTopology`) and constructs the `NeuronNetwork` around it.

Constants:

| Name | Value | Description |
|------|-------|-------------|
| `PAULA_SYNAPSE_LIMIT` | 4095 | 12-bit terminal/synapse ID ceiling |
| `CHEMICAL_SYNAPSE_DISTANCE` | 2 | Cable distance for chemical synapses |
| `GAP_JUNCTION_DISTANCE` | 1 | Cable distance for gap junctions |

### 5.3 C. elegans Connectome Loader

**File:** `simulations/c_elegans/connectome.py`

```python
def load_connectome(use_cache: bool = True) -> ConnectomeData
```

- **First call:** Parses Cook 2019 hermaphrodite data via the `cect` library (OpenWorm ConnectomeToolbox). Classifies each of the 302 neurons as sensory, motor, interneuron, or unknown. Extracts chemical synapses and gap junctions. Saves to `data/c_elegans/connectome_cache.json`.
- **Subsequent calls:** Loads from JSON cache (fast). An in-memory cache (`_connectome_memory_cache`) persists across calls within a process (important during evolution).

Result: **302 neurons**, **~3,709 chemical synapses**, **~1,092 gap junctions**.

---

## 6. C. elegans Implementation

### 6.1 Biological Constants

**File:** `simulations/c_elegans/config.py`

**Body geometry:**

| Constant | Value | Description |
|----------|-------|-------------|
| `N_BODY_SEGMENTS` | 13 | Rigid segments in body chain |
| `BODY_LENGTH_M` | 1e-3 | Total length (1 mm) |
| `BODY_RADIUS_M` | 4e-5 | Body radius (40 µm) |
| `BODY_MASS_KG` | 3e-10 | Approximate mass |
| `MUSCLE_QUADRANTS` | ("DL","DR","VL","VR") | Per-segment muscle groups |
| `N_MUSCLES` | 52 | 13 segments × 4 quadrants |

**Physics:**

| Constant | Value | Description |
|----------|-------|-------------|
| `PHYSICS_TIMESTEP_S` | 0.002 | MuJoCo timestep (2 ms) |
| `CONTROL_DECIMATION` | 5 | Physics steps per control step |
| `NEURAL_TICKS_PER_PHYSICS_STEP` | 1 | PAULA ticks per physics step |

**Environment:**

| Constant | Value | Description |
|----------|-------|-------------|
| `ENV_PLATE_RADIUS_M` | 0.05 | 5 cm agar plate |
| `FOOD_SOURCE_POSITION` | (0.0005, 0, 0) | Default food at x = 0.5 mm |
| `FOOD_GRADIENT_DECAY` | 1800.0 | Exponential decay constant |
| `FOOD_CONSUMPTION_RADIUS_M` | 0.0001 | Head within 0.1 mm = consumed |

**Sensorimotor interface:**

| Constant | Value | Description |
|----------|-------|-------------|
| `CHEM_CONCENTRATION_MAX` | 1.0 | Normalisation ceiling |
| `JOINT_ANGLE_MAX_RAD` | 1.2 | ~70° max bend |
| `MUSCLE_FILTER_ALPHA` | 0.3 | Low-pass filter for muscle activation |

**Neuron name lists:**

- `CHEMOSENSORY_NEURONS` — 16 neurons (ASEL/R, AWCL/R, AWBL/R, AFDL/R, ASHL/R, ASJL/R, AIZL/R)
- `TOUCH_NEURONS` — 6 neurons (PLML/R, ALML/R, AVM, PVM)
- `VENTRAL_CORD_MOTOR_NEURONS` — 57 neurons (DB, VB, DA, VA, DD, VD, AS classes)
- `MOTOR_NEURON_POSITIONS` — dict mapping each motor neuron name to a fractional body position `[0=head, 1=tail]`
- `COMMAND_INTERNEURONS_FORWARD` — AVBL, AVBR, PVCL, PVCR
- `COMMAND_INTERNEURONS_BACKWARD` — AVAL, AVAR, AVDL, AVDR
- `LOCOMOTION_INTERNEURONS` — 12 interneurons (RIML/R, RMEL/R, SMDVL/R, SMDDL/R, RIVL/R, RIS, DVA)

### 6.2 MuJoCo Body

**File:** `simulations/c_elegans/body.py`

`CElegansBody` wraps a MuJoCo model defined in `body_model.xml`.

**Scaling:** All lengths in the MJCF are 1000× biological. Biological 1 mm → model 1.0 m. This keeps MuJoCo well above its internal precision floor (`mjMINVAL ≈ 1e-15`). The dimensionless locomotion dynamics are unaffected.

**MJCF model structure (`body_model.xml`):**

- 13 capsule segments (`seg0` = head at origin, `seg12` = tail at ~+1 m in model space).
- Each inter-segment joint has two DOF: pitch (y-axis, dorsal-ventral) and yaw (z-axis, left-right).
- 48 actuators: 4 muscle quadrants × 12 inter-segment joints. Named `muscle_seg{N}_{QUAD}`.
- Contact dynamics: low friction floor (`0.005`) models agar surface crawling.
- Physics: `implicitfast` integrator, gravity -5 m/s² (scaled), fluid density 4000, viscosity 0.1.

**Sensor sites:**

| Site | Position | Purpose |
|------|----------|---------|
| `nose` | Front of seg0 | Head position tracking |
| `touch_anterior` | Near nose | Anterior touch detection |
| `touch_posterior` | Rear of seg12 | Posterior touch detection |
| `touch_ant_sensor` | Gyro at anterior | Contact force sensor |
| `touch_post_sensor` | Gyro at posterior | Contact force sensor |

**Key methods:**

| Method | Description |
|--------|-------------|
| `reset()` | Resets to default pose, runs 2000 settle steps under gravity |
| `step(muscle_activations)` | Applies actuator controls, steps physics by `CONTROL_DECIMATION` substeps |
| `get_state()` | Reads position (biological metres via `_SCALE_MODEL_TO_BIO`), orientation, joint angles/velocities, contact forces, head position |
| `get_body_shape()` | Returns (13, 3) array of segment centres in biological metres |
| `render(camera)` | Returns RGB frame from named camera |

### 6.3 Agar Plate Environment

**File:** `simulations/c_elegans/environment.py`

`AgarPlateEnvironment` models a circular agar plate with chemical point sources.

**ChemSource** (dataclass):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `molecule` | `str` | — | Chemical name |
| `position` | `ndarray` | — | 3-D position in metres |
| `max_concentration` | `float` | 1.0 | Peak concentration |
| `decay_constant` | `float` | 1800.0 | Exponential decay rate |
| `valence` | `str` | "attractive" | "attractive" or "aversive" |

Concentration at position `p`:

```
C(p) = max_concentration × exp(−decay_constant × ‖p − source‖)
```

**Food items:** Each food source generates two chemical gradients: NaCl (sensed by ASE neurons) and butanone (sensed by AWC neurons).

**Additional sources:**

- 2-nonanone aversive odour (sensed by AWB neurons), placed at a fixed offset.
- Nociceptive zone: circular region where `is_nociceptive()` returns true (sensed by ASH neurons).

**Food dynamics:**

- `step()` checks if the worm's head is within `FOOD_CONSUMPTION_RADIUS_M` of any food item. If so, the food is removed.
- `add_food(position)` creates a new food source at runtime (used by interactive viewer).
- `remove_food_near(position, radius)` removes the nearest food within a radius.

**Observation construction** (`_build_observation`):

- Chemicals: sum of all source concentrations at the head position, keyed by molecule name.
- Touch: reads body contact force sensors.
- Proprioception: joint angles from the body.

### 6.4 Sensor Encoder

**File:** `simulations/c_elegans/sensors.py`

`SensorEncoder` maps `EnvironmentObservation + BodyState` to `dict[str, float]` keyed by neuron name. All outputs normalised to `[0, 1]`.

**Chemosensory encoding** — maps molecule concentrations to neuron names:

| Neuron pair | Molecule |
|-------------|----------|
| ASEL, ASER | NaCl |
| AWCL, AWCR | butanone |
| AWBL, AWBR | 2-nonanone |
| AFDL, AFDR | temperature |
| ASHL, ASHR | nociceptive |
| ASJL, ASJR | ascaroside |
| AIZL, AIZR | NaCl |

**Mechanosensory encoding** — maps contact forces to touch neurons:

| Neuron | Body site | Encoding |
|--------|-----------|----------|
| PLML, PLMR | touch_post_sensor | ‖force‖ / 10 |
| ALML, ALMR | touch_ant_sensor | ‖force‖ / 10 |
| AVM | touch_ant_sensor | ‖force‖ / 10 |
| PVM | touch_post_sensor | ‖force‖ / 10 |

**Proprioceptive encoding** — two sources:

1. **Stretch receptors** (PVDL, PVDR at segment 10; DVA at segment 6): `|angle| / JOINT_ANGLE_MAX_RAD`, clipped to [0, 1].
2. **Motor proprioception** (B-type motor neurons): segment curvature at the neuron's anatomical position, keyed as `_mpr_{neuron_name}`. Based on Wen et al. 2012.

### 6.5 Neuromuscular Junction

**File:** `simulations/c_elegans/muscles.py`

`NeuromuscularJunction` is a stateless translator with three static methods:

| Method | Description |
|--------|-------------|
| `to_ctrl(activations)` | Converts 0-indexed internal muscle names (`seg{N}_{QUAD}`) to 1-indexed MuJoCo actuator names (`muscle_seg{N+1}_{QUAD}`) |
| `dorsal_minus_ventral(activations)` | Returns `(N_BODY_SEGMENTS,)` array of (dorsal − ventral) for each segment. Positive = dorsal contraction |
| `mean_activation(activations)` | Average activation across all muscles |

### 6.6 CElegansEngine and Factory

**File:** `simulations/c_elegans/simulation.py`

`CElegansEngine` subclasses `SimulationEngine` to:

1. Override `_observation_to_sensory_inputs` to use `SensorEncoder.encode()`.
2. Override `step()` to translate motor outputs through `NeuromuscularJunction.to_ctrl()`.

**Factory function:**

```python
def build_c_elegans_simulation(
    use_connectome_cache: bool = True,
    food_position:  tuple[float, float, float] = (0.03, 0.0, 0.0),
    food_positions: list[tuple[float, float, float]] | None = None,
    log_level:      str = "WARNING",
    enable_m0:      bool = True,
    enable_m1:      bool = True,
    evol_config:    dict[str, Any] | None = None,
    max_history:    int = 200,
    ...
) -> tuple[CElegansEngine, SensorimotorLoop]
```

Assembly sequence:

1. Load connectome (cached).
2. Build `CElegansNervousSystem` with 302 PAULA neurons.
3. Initialise `CElegansBody` (MuJoCo).
4. Initialise `AgarPlateEnvironment` with food sources.
5. Create `SensorEncoder`.
6. Assemble `CElegansEngine`.
7. Wrap in `SensorimotorLoop`.

### 6.7 Interactive Viewer

**File:** `simulations/c_elegans/interactive_viewer.py`

`CElegansInteractiveViewer` provides a real-time 2-D matplotlib display with two panels:

- **Left panel:** Worm trajectory on the agar plate with food markers (magenta stars). Left-click adds food; right-click removes nearest food.
- **Right panel:** Rolling metrics — prediction error, motor entropy, M0 (red), and M1 (green).

The main loop calls `engine.step()` repeatedly and updates the artists each frame.

---

## 7. Neuromodulation System

**File:** `simulations/c_elegans/neuron_mapping.py`, class `CElegansNervousSystem`

### 7.1 Overview

The neuromodulation system implements ALERM Eq. 4–5. Two modulators (M0 and M1) are computed from the temporal derivative of chemosensory concentration and delivered through two pathways:

1. **Synaptic injection** — per-sensory-neuron m0/m1 values are passed as the `mod` array to each neuron's input via `set_external_input`. These pass through the standard PAULA M_vector EMA pathway.
2. **Volume transmission** — a global average M0/M1 is added directly to every neuron's `M_vector` after the network tick, modelling diffuse monoamine signaling.

### 7.2 Chemosensory Filtering — Dual-EMA Bandpass

The four primary chemosensory neurons (ASEL, ASER, AWCL, AWCR) use a dual-EMA bandpass filter to extract the true navigational gradient while rejecting head-sweep oscillation noise.

For each chemosensory neuron on each tick:

```
fast(t) = α_fast × clamped + (1 − α_fast) × fast(t−1)
slow(t) = α_slow × clamped + (1 − α_slow) × slow(t−1)
delta_c  = fast − slow
```

| Parameter | Default | Evolved | Description |
|-----------|---------|---------|-------------|
| `_CHEM_EMA_ALPHA_FAST` | 0.2 | 0.2 | Fast EMA: tracks head-sweep (~10 tick window) |
| `_CHEM_EMA_ALPHA_SLOW` | 0.01 | 0.0255 | Slow EMA: tracks environmental gradient (~40+ tick window) |

The fast EMA responds to rapid oscillations (head sweep); the slow EMA tracks the underlying gradient. Their difference cancels the head-sweep frequency, leaving only the navigational signal.

### 7.3 Per-Synapse Modulation

Given `delta_c`, the system computes synapse-level neuromodulatory signals:

**Regular sensory neurons:**

```
if delta_c < −STRESS_DEADZONE:
    excess = |delta_c| − STRESS_DEADZONE
    m0 = tanh(excess × K_STRESS_SYN / 5.0) × 5.0
    m1 = 0
elif delta_c > 0:
    m0 = 0
    m1 = tanh(delta_c × K_REWARD_SYN / 5.0) × 5.0
else:
    m0 = 0, m1 = 0
```

**OFF-cell neurons (AWCL, AWCR, ASER):** Tonically active, suppressed by stimulus, burst on removal (Chalasani et al. 2007).

```
m1 = tanh(clamped × K_OFF_SUPPRESS / 5.0) × 5.0    # Absolute concentration → suppression
if delta_c < −STRESS_DEADZONE:
    m0 = tanh(excess × K_STRESS_SYN / 5.0) × 5.0    # Decrease → burst facilitation
```

**Non-chemosensory neurons:** Use simple frame-to-frame difference `delta_c = clamped − prev`.

The `mod = [m0, m1]` array is injected via `set_external_input()`. Inside PAULA, it is multiplied by the synapse's `u_i.adapt` receptor sensitivity before updating `M_vector`.

The saturation function `tanh(x / ceiling) × ceiling` provides smooth soft-capping:

- Linear for small inputs (derivative = 1 at origin).
- Smoothly approaches the ceiling for large inputs (no hard discontinuity).
- Ceiling is 5.0 for per-synapse signals.

### 7.4 Global Volume Transmission

After all sensory neurons are processed, the average chemosensory delta drives a global broadcast:

```
avg_delta = Σ(delta_c for chemosensory neurons) / n_chem

if avg_delta < −STRESS_DEADZONE:
    global_m0 = tanh(excess × K_VOL_STRESS / 2.0) × 2.0
    global_m1 = 0
elif avg_delta > 0:
    global_m0 = 0
    global_m1 = tanh(avg_delta × K_VOL_REWARD / 2.0) × 2.0
```

Volume ceiling is 2.0. In `_volume_broadcast()`, for every neuron in the network:

```
M_vector[k] += (1 − gamma[k]) × global_m[k]
```

This additive term, combined with the existing EMA decay `M_vector = gamma × M_vector + ...`, converges `M_vector[k]` toward `global_m[k]` at steady state.

### 7.5 Tonic Drives

Three tonic currents are injected each tick to maintain biologically realistic baseline activity:

**Forward drive** (`_inject_tonic_forward`): Models AVB↔B-motor gap junction coupling.

| Target | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| AVBL, AVBR | `_TONIC_FWD_CMD` | 0.25 | Depolarising current to command interneurons |
| DB*, VB* neurons | `_TONIC_FWD_MOTOR` | 0.0 | Depolarising current to B-type motor neurons |

**OFF-cell tonic** (`_inject_off_cell_tonic`): Models spontaneous firing of AWC/ASER in the absence of stimulus.

| Target | Parameter | Default |
|--------|-----------|---------|
| AWCL, AWCR, ASER | `_TONIC_OFF_CELL` | 0.15 |

**Motor proprioception** (`_inject_motor_proprioception`): B-type motor neurons receive curvature feedback from the sensor encoder (keys prefixed `_mpr_`). Gain: `_PROPRIO_MOTOR_GAIN = 0.08`.

### 7.6 Gain Constants

All gains can be overridden at runtime via `evol_config`:

| Constant | Default | Evolved | Description |
|----------|---------|---------|-------------|
| `_K_STRESS_SYN` | 4000.0 | 2655.7 | Per-synapse stress gain |
| `_K_REWARD_SYN` | 4000.0 | 2678.0 | Per-synapse reward gain |
| `_K_VOL_STRESS` | 2000.0 | 3689.7 | Volume transmission stress gain |
| `_K_VOL_REWARD` | 2000.0 | 1205.3 | Volume transmission reward gain |
| `_STRESS_DEADZONE` | 0.00005 | 0.00228 | Minimum |delta_c| to trigger M0 |
| `_K_OFF_SUPPRESS` | 5.0 | 5.0 | Absolute concentration → M1 for OFF cells |

---

## 8. Motor Decoding

**File:** `simulations/c_elegans/neuron_mapping.py`, method `_decode_motor_outputs`

### 8.1 Segment Map

A lazy-built mapping from body segment to contributing motor neurons, constructed via Gaussian spillover:

```python
@classmethod
def _build_segment_map(cls) -> dict[str, list[tuple[str, float]]]:
    sigma = 0.8  # Gaussian spread
    for motor_name, frac_pos in MOTOR_NEURON_POSITIONS.items():
        for seg in range(N_BODY_SEGMENTS):
            seg_frac = seg / (N_BODY_SEGMENTS - 1)
            dist = abs(frac_pos - seg_frac) * (N_BODY_SEGMENTS - 1)
            if dist > 3.0:
                continue
            w = exp(-0.5 * (dist / sigma) ** 2)
            if w > 0.01:
                seg_map[seg_key].append((motor_name, w))
```

Each motor neuron contributes to nearby segments with a Gaussian weight that falls off with anatomical distance.

### 8.2 Graded Output

Motor neurons transmit via graded potentials (not binary spikes). The membrane potential S is normalised:

```
graded(S) = clip(S / _S_NORM, 0, 1)      # _S_NORM = 0.25
```

### 8.3 Excitation and Inhibition

For each body segment, motor neuron contributions are accumulated:

| Neuron prefix | Type | Target | Weight |
|---------------|------|--------|--------|
| DB | Excitatory forward | Dorsal | `_FWD_WEIGHT = 1.0` |
| DA | Excitatory backward | Dorsal | `_BKW_WEIGHT = 0.3` |
| AS | Excitatory (dorsal sublateral) | Dorsal | 0.5 |
| VB | Excitatory forward | Ventral | `_FWD_WEIGHT = 1.0` |
| VA | Excitatory backward | Ventral | `_BKW_WEIGHT = 0.3` |
| DD | Inhibitory | Dorsal | Gain `_INHIB_WEIGHT = 0.3` |
| VD | Inhibitory | Ventral | Gain `_INHIB_WEIGHT = 0.3` |

Net excitation:

```
d_excit = d_exc − d_inh × _INHIB_WEIGHT
v_excit = v_exc − v_inh × _INHIB_WEIGHT
```

### 8.4 Reciprocal Inhibition

Cross-inhibition between dorsal and ventral channels produces the alternating contraction pattern:

```
d_push = clip(d_excit − v_excit × _RECIP_INHIB, 0, ∞)
v_push = clip(v_excit − d_excit × _RECIP_INHIB, 0, ∞)
```

`_RECIP_INHIB = 0.5`.

### 8.5 Muscle Filtering

Final activations are low-pass filtered to smooth jitter:

```
activation = MUSCLE_FILTER_ALPHA × target + (1 − MUSCLE_FILTER_ALPHA) × prev
```

`MUSCLE_FILTER_ALPHA = 0.3` (30% new, 70% previous). Outputs are stored in `_muscle_activations` and returned as `{seg{N}_{QUAD}: float}`.

---

## 9. Evolutionary Optimisation

**File:** `scripts/evolve_food_seeking.py`

### 9.1 Algorithm

Uses `scipy.optimize.differential_evolution` with strategy `"best1bin"` (best-of-generation, binomial crossover). All parameters are normalised to `[0, 1]` and mapped to biological ranges via `x_to_config()`.

### 9.2 Parameter Space

13-dimensional optimisation vector:

| Index | Parameter | Range | Scale |
|-------|-----------|-------|-------|
| 0 | K_STRESS_SYN | [1000, 8000] | Linear |
| 1 | K_REWARD_SYN | [1000, 8000] | Linear |
| 2 | K_VOL_STRESS | [500, 4000] | Linear |
| 3 | K_VOL_REWARD | [500, 4000] | Linear |
| 4 | STRESS_DEADZONE | [1e-6, 0.01] | Log |
| 5 | CHEM_EMA_ALPHA_SLOW | [0.01, 0.10] | Linear |
| 6 | CHEM_EMA_ALPHA_FAST | [0.05, 0.40] | Linear |
| 7 | TONIC_FWD_CMD | [0.1, 0.5] | Linear |
| 8 | TONIC_FWD_MOTOR | [0.05, 0.2] | Linear |
| 9 | Motor w_tref M0 | [15, 45] | Linear |
| 10 | Motor w_tref M1 | [-25, -5] | Linear |
| 11 | Sensory w_tref M0 | [8, 25] | Linear |
| 12 | Sensory w_tref M1 | [-12, -3] | Linear |

### 9.3 Fitness Function

Each genome is evaluated against 3 distinct food positions to prevent directional overfitting:

```python
TEST_ENVIRONMENTS = [
    (-0.002,  0.002, 0.0),   # Top-left
    ( 0.002, -0.002, 0.0),   # Bottom-right
    ...
]
```

Fitness = mean (or worst with `--robust`) minimum distance to food across environments. Lower is better (DE minimises).

Early exit: if the worm reaches within `EAT_RADIUS_M = 0.0001` (0.1 mm), that environment scores 0.

### 9.4 Robustness Protocol

- Tests multiple food positions → prevents "lucky torpedo" genomes.
- `--robust` flag scores on the *worst* environment instead of the average.
- Checkpointing: atomic JSON save every generation (temp file + rename, crash-safe).
- Signal handling: SIGINT saves checkpoint before exiting.

### 9.5 Memory Management

- `--low-memory` mode: forces GC after each evaluation, sets `max_history=1`, suppresses connectome summary. Designed for Raspberry Pi / 8 GB machines.
- `--measure-memory` prints peak RSS (MB) each generation.

### 9.6 Output

Best genome is saved to `evolved_food_seeking_config.json`:

```json
{
    "K_STRESS_SYN": 2655.68,
    "K_REWARD_SYN": 2678.00,
    "K_VOL_STRESS": 3689.72,
    "K_VOL_REWARD": 1205.26,
    "STRESS_DEADZONE": 0.00228,
    "CHEM_EMA_ALPHA": 0.0255,
    "TONIC_FWD_CMD": 0.262,
    "TONIC_FWD_MOTOR": 0.098,
    "neuron_params": {
        "motor":   { "w_tref": [29.82, -16.58] },
        "sensory": { "w_tref": [14.97, -8.83] }
    }
}
```

Checkpoint format adds metadata: `best_x`, `best_dist_mm`, `generation`, `n_evals`, `elapsed_min`, `evol_args`.

---

## 10. Scripts and Entry Points

### 10.1 run_c_elegans.py

Main simulation runner.

**Usage:**

```bash
# Basic simulation
uv run python scripts/run_c_elegans.py --steps 1000

# With plot and animation
uv run python scripts/run_c_elegans.py --steps 5000 --save-plot --save-animation

# With evolved config
uv run python scripts/run_c_elegans.py --steps 50000 --evol-config evolved_food_seeking_config.json

# Interactive 2-D viewer
uv run python scripts/run_c_elegans.py --interactive --evol-config evolved_food_seeking_config.json

# 3-D MuJoCo viewer (requires mjpython)
uv run mjpython scripts/run_c_elegans.py --viewer --steps 500

# Multiple food sources
uv run python scripts/run_c_elegans.py --food-positions "0.5,0.5 -0.5,-0.5" --food-scale 0.01

# Disable neuromodulators for ablation studies
uv run python scripts/run_c_elegans.py --no-M0 --no-M1
```

**Command-line arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | 500 | Number of physics steps |
| `--food-x` | 0.0005 | Food x-position (metres) |
| `--food-z` | 0.0 | Food z-position (metres) |
| `--food-positions` | — | Multiple foods as "x1,z1 x2,z2 ..." |
| `--food-scale` | 1.0 | Scale factor for food coordinates |
| `--save-plot` | false | Save trajectory + motor wave plot |
| `--plot-output` | c_elegans_run.png | Plot file path |
| `--save-animation` | false | Save animated GIF |
| `--animation-output` | c_elegans_run.gif | Animation file path |
| `--animation-fps` | 15 | Animation frame rate |
| `--animation-frames` | 300 | Max animation frames |
| `--save-log` | false | Save full run log to disk |
| `--log-dir` | auto-timestamped | Log directory path |
| `--viewer` | false | 3-D MuJoCo viewer |
| `--interactive` | false | 2-D matplotlib viewer |
| `--no-cache` | false | Force fresh connectome download |
| `--no-M0` | false | Disable stress modulator |
| `--no-M1` | false | Disable reward modulator |
| `--verbose` | false | PAULA INFO logging |
| `--evol-config` | — | Path to evolved config JSON |

**StreamingCollector:** Pre-allocates numpy arrays and fills them step-by-step to avoid holding full `SimulationStep` dicts in memory. Arrays include: positions, head_positions, elapsed_ms, ticks, prediction_error, motor_entropy, joint_angles, motor_outputs, chemicals, neural_S, neural_fired.

**Plot output (3 panels):**

1. Head trajectory on floor plane (mm scale) with food markers.
2. Motor wave heatmap: dorsal − ventral activation per body segment over time.
3. Prediction error and motor entropy traces.

### 10.2 evolve_food_seeking.py

See §9 above.

```bash
# Default evolution (30 gen × 20 pop × 50k ticks)
uv run python scripts/evolve_food_seeking.py

# Large-scale robust evolution
uv run python scripts/evolve_food_seeking.py --generations 100 --population 40 --robust

# Low-memory mode
uv run python scripts/evolve_food_seeking.py --low-memory --measure-memory

# Resume from checkpoint
uv run python scripts/evolve_food_seeking.py --checkpoint my_checkpoint.json
```

### 10.3 analyze_neuromod_food_seeking.py

Post-hoc analysis of logged simulation runs.

```bash
# Analyse a single run
uv run python scripts/analyze_neuromod_food_seeking.py --run-dir logs/my_run/

# Compare neuromod vs no-neuromod
uv run python scripts/analyze_neuromod_food_seeking.py --compare

# List available log dirs
uv run python scripts/analyze_neuromod_food_seeking.py --list
```

Metrics computed: start/end/min distance to food, fraction of steps moving toward food, chemical concentration change over the run.

### 10.4 apply_evolved_config.py

Writes evolved parameters back to `neuron_mapping.py` class-level defaults via regex substitution.

```bash
uv run python scripts/apply_evolved_config.py --config evolved_food_seeking_config.json
```

### 10.5 download_connectome.py

One-shot connectome download and cache.

```bash
uv run python scripts/download_connectome.py
```

---

## 11. Data, Logging, and Analysis

### 11.1 Run Logs

Each logged run produces a directory under `logs/` containing:

| File | Format | Contents |
|------|--------|----------|
| `config.json` | JSON | RunConfig, RunSummary, FreeEnergyTrace, data_keys |
| `data.npz` | Numpy compressed | All time-series arrays |
| `positions.csv` | CSV | [tick, x, y, z] centre-of-mass |
| `head_position.csv` | CSV | [tick, x, y, z] head position |
| `joint_angle.csv` | CSV | [tick, joint_0, joint_1, ...] |
| `motor_output.csv` | CSV | [tick, muscle_0, muscle_1, ...] |
| `chemical.csv` | CSV | [tick, NaCl, butanone, ...] |
| `neural_S.csv` | CSV | [tick, neuron_0_S, neuron_1_S, ...] |
| `neural_fired.csv` | CSV | [tick, neuron_0_fired, ...] |
| `elapsed_ms.csv` | CSV | [tick, ms] |

### 11.2 Loading Logs

```python
from simulations.run_log import load_run_log
config, data = load_run_log(Path("logs/my_run"))
# config: dict with RunConfig, RunSummary, data_keys
# data: dict[str, ndarray] with tick, position, head_position, etc.
```

### 11.3 Evolved Config Files

| File | Description |
|------|-------------|
| `evolved_food_seeking_config.json` | Best parameters from evolution |
| `evolved_food_seeking_config_2.json` | Alternative run |
| `evolved_food_seeking_checkpoint.json` | Latest checkpoint with metadata |
| `*.bak` | Backup copies |

---

## 12. Configuration Reference

### 12.1 CElegansNervousSystem Class Constants

Complete list of tunable parameters on the `CElegansNervousSystem` class. All can be overridden via `evol_config` dict at construction time.

| Attribute | Type | Default | Evolved | Description |
|-----------|------|---------|---------|-------------|
| `_K_STRESS_SYN` | float | 4000.0 | 2655.7 | Per-synapse stress (M0) gain |
| `_K_REWARD_SYN` | float | 4000.0 | 2678.0 | Per-synapse reward (M1) gain |
| `_K_VOL_STRESS` | float | 2000.0 | 3689.7 | Volume transmission stress gain |
| `_K_VOL_REWARD` | float | 2000.0 | 1205.3 | Volume transmission reward gain |
| `_STRESS_DEADZONE` | float | 0.00005 | 0.00228 | |delta_c| threshold for M0 |
| `_CHEM_EMA_ALPHA_FAST` | float | 0.2 | 0.2 | Fast EMA filter (head-sweep) |
| `_CHEM_EMA_ALPHA_SLOW` | float | 0.01 | 0.0255 | Slow EMA filter (gradient) |
| `_TONIC_FWD_CMD` | float | 0.25 | 0.262 | Tonic current to AVB |
| `_TONIC_FWD_MOTOR` | float | 0.0 | 0.098 | Tonic current to B-motors |
| `_K_OFF_SUPPRESS` | float | 5.0 | 5.0 | Concentration → M1 for OFF cells |
| `_TONIC_OFF_CELL` | float | 0.15 | 0.15 | Tonic baseline for AWC/ASER |
| `_PROPRIO_MOTOR_GAIN` | float | 0.08 | 0.08 | Motor proprioception gain |

### 12.2 Neuron Parameter Presets

Eight preset functions define PAULA parameters for distinct neuron classes:

| Preset | Applies to | r_base | b_base | c | λ | w_tref |
|--------|-----------|--------|--------|---|---|--------|
| `_base_params` | Fallback | 0.9 | 1.1 | 8 | 15.0 | [15, -8] |
| `_sensory_params` | Sensory neurons | 0.5 | 0.7 | 4 | 6.0 | [12, -6] |
| `_off_cell_sensory_params` | AWCL, AWCR, ASER | 0.25 | 0.35 | 4 | 20.0 | [12, 0] |
| `_motor_params` | Excitatory motor (DB, VB, DA, VA, AS) | 0.2 | 0.3 | 4 | 6.0 | [30, -15] |
| `_motor_inhib_params` | Inhibitory motor (DD, VD) | 0.15 | 0.25 | 3 | 4.0 | [20, -10] |
| `_interneuron_params` | Local interneurons | 0.6 | 0.85 | 8 | 10.0 | [15, -8] |
| `_command_fwd_params` | AVBL, AVBR, PVCL, PVCR | 0.4 | 0.6 | 8 | 10.0 | [12, -6] |
| `_command_bkw_params` | AVAL, AVAR, AVDL, AVDR | 0.85 | 1.1 | 12 | 16.0 | [12, -6] |

**Design rationale for parameter choices:**

- **Sensory neurons** have the lowest thresholds (`r_base=0.5`) and fastest time constants (`λ=6`) — they must respond quickly to environmental stimuli.
- **OFF-cell neurons** (AWC, ASER) have extremely low `r_base=0.25` for tonic firing, large `w_r[1]=1.2` for strong M1-mediated suppression, and fast `gamma[1]=0.90` so the threshold drops quickly on stimulus removal (generating the OFF burst).
- **Excitatory motor neurons** have low thresholds (`r_base=0.2`) and strong t_ref modulation (`w_tref=[30, -15]`) — neuromodulators dramatically alter their learning dynamics.
- **Inhibitory motor neurons** (DD, VD) are faster (`λ=4`) than excitatory — cross-inhibition must respond quickly for dorsal-ventral alternation.
- **Forward command interneurons** (AVB, PVC) have lower thresholds than backward (AVA, AVD), modelling the ~80% forward bias of *C. elegans* locomotion.
- **Backward command interneurons** have the highest thresholds (`r_base=0.85`) — only strong aversive stimuli should trigger reversal.

### 12.3 Motor Decoding Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `_S_NORM` | 0.25 | Graded output normalisation |
| `_INHIB_WEIGHT` | 0.3 | Inhibitory synapse gain |
| `_RECIP_INHIB` | 0.5 | Cross-inhibition (D↔V) factor |
| `_FWD_WEIGHT` | 1.0 | Forward motor neuron contribution |
| `_BKW_WEIGHT` | 0.3 | Backward motor neuron contribution |

### 12.4 Dependencies

From `pyproject.toml`:

| Package | Purpose |
|---------|---------|
| numpy | Array computation |
| mujoco | Physics simulation |
| loguru | Structured logging |
| networkx | Graph utilities |
| pandas | Data analysis |
| scipy | Optimisation (differential evolution) |
| matplotlib | Plotting and animation |
| tqdm | Progress bars |
| cect | OpenWorm ConnectomeToolbox (Cook 2019 data) |

Dev dependencies: pytest, pytest-asyncio, ipython.

Build system: Hatchling. Python ≥ 3.11.

### 12.5 Adding a New Organism

1. Create `simulations/<organism>/` with your organism's module.
2. Define or download the organism's connectome as a `ConnectomeData`.
3. Call `build_paula_network(connectome, ...)` to get a `NeuronNetwork`.
4. Subclass `BaseNervousSystem` — implement `reset()`, `tick()`, `get_neuron_states()`.
5. Subclass `BaseBody` — implement MuJoCo wrapper with `reset()`, `step()`, `get_state()`.
6. Subclass `BaseEnvironment` — implement stimulus generation and food dynamics.
7. Create a factory function (like `build_c_elegans_simulation`) that assembles all components.
8. The `SensorimotorLoop` handles active inference framing automatically.
