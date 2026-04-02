# Abstract Interfaces

← [Index](../INDEX.md) | [SimulationEngine](simulation-engine.md)

## BaseBody (`base_body.py`)

| Abstract member | Type | Description |
|-----------------|------|-------------|
| `reset()` | method → `BodyState` | Reset to neutral pose |
| `step(muscle_activations)` | method → `BodyState` | Apply muscles, advance physics |
| `get_state()` | method → `BodyState` | Read state without stepping |
| `render(camera)` | method → `ndarray \| None` | RGB frame or None |
| `dt` | property → `float` | Physics timestep (seconds) |
| `joint_names` | property → `list[str]` | Controllable joints |
| `muscle_names` | property → `list[str]` | Actuated muscles |

### BodyState Dataclass

| Field | Type | Description |
|-------|------|-------------|
| `position` | `ndarray(3,)` | Centre-of-mass [x, y, z] |
| `orientation` | `ndarray(4,)` | Unit quaternion [w, x, y, z] |
| `joint_angles` | `dict[str, float]` | Named joint angles (radians) |
| `joint_velocities` | `dict[str, float]` | Angular velocities |
| `contact_forces` | `dict[str, ndarray]` | Named contact force 3-vectors |
| `head_position` | `ndarray(3,)` | Nose / head tip position |
| `extra` | `dict[str, Any]` | Organism-specific quantities |

---

## BaseEnvironment (`base_environment.py`)

| Abstract member | Type | Description |
|-----------------|------|-------------|
| `reset()` | method → `EnvironmentObservation` | Initial observation |
| `step(body_state)` | method → `EnvironmentObservation` | Advance, return new observation |
| `render()` | method → `ndarray \| None` | Top-down RGB |

### EnvironmentObservation Dataclass

| Field | Type | Description |
|-------|------|-------------|
| `chemicals` | `dict[str, float]` | Scalar concentrations by molecule |
| `contact_forces` | `dict[str, ndarray]` | Touch forces at body sites |
| `proprioception` | `dict[str, float]` | Joint angles and velocities |
| `extra` | `dict[str, Any]` | Additional modalities |

---

## BaseNervousSystem (`base_nervous_system.py`)

| Abstract member | Type | Description |
|-----------------|------|-------------|
| `reset()` | method → `None` | Reset all neurons |
| `tick(sensory_inputs, current_tick)` | method → `dict[str, float]` | One neural tick → motor outputs |
| `get_neuron_states()` | method → `dict[str, Any]` | Snapshot of membrane potentials + firing |
| `n_neurons` | property → `int` | Total neuron count |

## Concrete Implementations

- [CElegansBody](../c-elegans/mujoco-body.md)
- [AgarPlateEnvironment](../c-elegans/agar-plate-environment.md)
- [CElegansNervousSystem](../c-elegans/engine-factory.md)
