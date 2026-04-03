# Biological Constants

← [Index](../INDEX.md) | [C. elegans Overview](overview.md) | **File:** `simulations/c_elegans/config.py`

## Body Geometry

| Constant | Value | Description |
|----------|-------|-------------|
| `N_BODY_SEGMENTS` | 13 | Rigid segments in body chain |
| `BODY_LENGTH_M` | 1e-3 | Total length (1 mm) |
| `BODY_RADIUS_M` | 4e-5 | Body radius (40 µm) |
| `BODY_MASS_KG` | 3e-10 | Approximate mass |
| `MUSCLE_QUADRANTS` | ("DL","DR","VL","VR") | Per-segment muscle groups |
| `N_MUSCLES` | 52 | 13 segments × 4 quadrants |

## Physics

| Constant | Value | Description |
|----------|-------|-------------|
| `PHYSICS_TIMESTEP_S` | 0.002 | MuJoCo timestep (2 ms) |
| `CONTROL_DECIMATION` | 5 | Physics steps per control step |
| `NEURAL_TICKS_PER_PHYSICS_STEP` | 1 | PAULA ticks per physics step |

## Environment

| Constant | Value | Description |
|----------|-------|-------------|
| `ENV_PLATE_RADIUS_M` | 0.05 | 5 cm agar plate |
| `FOOD_SOURCE_POSITION` | (0.0005, 0, 0) | Default food at x = 0.5 mm |
| `FOOD_GRADIENT_DECAY` | 1800.0 | Exponential decay constant |
| `FOOD_CONSUMPTION_RADIUS_M` | 0.0001 | Head within 0.1 mm = consumed |

## Sensorimotor Interface

| Constant | Value | Description |
|----------|-------|-------------|
| `CHEM_CONCENTRATION_MAX` | 1.0 | Normalisation ceiling |
| `JOINT_ANGLE_MAX_RAD` | 1.2 | ~70° max bend |
| `MUSCLE_FILTER_ALPHA` | 0.3 | Low-pass filter for muscle activation |

## Neuron Name Lists

| List | Contents |
|------|---------|
| `CHEMOSENSORY_NEURONS` | 14 neurons (ASEL/R, AWCL/R, AWBL/R, AFDL/R, ASHL/R, ASJL/R, AIZL/R) |
| `TOUCH_NEURONS` | 6 neurons (PLML/R, ALML/R, AVM, PVM) |
| `VENTRAL_CORD_MOTOR_NEURONS` | 69 neurons (DB, VB, DA, VA, DD, VD, AS classes) |
| `MOTOR_NEURON_POSITIONS` | dict mapping each motor neuron → fractional body position [0=head, 1=tail] |
| `COMMAND_INTERNEURONS_FORWARD` | AVBL, AVBR, PVCL, PVCR |
| `COMMAND_INTERNEURONS_BACKWARD` | AVAL, AVAR, AVDL, AVDR |
| `LOCOMOTION_INTERNEURONS` | 12 interneurons (AVAL/R, AVBL/R, AVDL/R, AVEL/R, PVCL/R, AVJL/R) |
