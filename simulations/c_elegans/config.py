"""
C. elegans simulation configuration constants.

Biological values are sourced primarily from:
  - Cook et al. 2019 (Nature) – connectome
  - Zhen & Samuel 2015 (Curr Opin Neurobiol) – locomotion
  - Boyle et al. 2012 (PLoS Comput Biol) – muscle mechanics
"""

from __future__ import annotations

# -----------------------------------------------------------------------
# Body geometry (MuJoCo model)
# -----------------------------------------------------------------------
N_BODY_SEGMENTS = 13       # Number of rigid segments in worm body chain
BODY_LENGTH_M = 1e-3       # Total body length in metres (1 mm)
BODY_RADIUS_M = 4e-5       # Body radius (40 µm)
BODY_MASS_KG = 3e-10       # Approximate total mass

# Per-segment dimensions (derived)
SEGMENT_LENGTH_M = BODY_LENGTH_M / N_BODY_SEGMENTS
SEGMENT_RADIUS_M = BODY_RADIUS_M

# Muscle quadrants per segment
MUSCLE_QUADRANTS = ("DL", "DR", "VL", "VR")   # dorsal-left, dorsal-right, ventral-left, ventral-right
N_MUSCLES = N_BODY_SEGMENTS * len(MUSCLE_QUADRANTS)

# Physics
PHYSICS_TIMESTEP_S = 0.002   # MuJoCo timestep (2 ms)
CONTROL_DECIMATION = 5       # Physics steps per control step

# -----------------------------------------------------------------------
# Neural simulation
# -----------------------------------------------------------------------
NEURAL_TICKS_PER_PHYSICS_STEP = 1

# Neuron names for key sensory cells (chemosensation)
CHEMOSENSORY_NEURONS = [
    "ASEL", "ASER",   # NaCl / primary salt sensing
    "AWCL", "AWCR",   # olfactory (isoamyl alcohol etc.)
    "AWBL", "AWBR",   # avoidance olfactory
    "AFDL", "AFDR",   # temperature
    "ASHL", "ASHR",   # ASH: primary nociceptor / avoidance
    "ASJL", "ASJR",
    "AIZL", "AIZR",
]

# Neuron names for key mechanosensory cells
TOUCH_NEURONS = [
    "PLML", "PLMR",   # posterior touch
    "ALML", "ALMR",   # anterior lateral touch
    "AVM",            # anterior ventral microtubule
    "PVM",            # posterior ventral microtubule
]

# Neuron names for ventral cord motor neurons (body locomotion)
VENTRAL_CORD_MOTOR_NEURONS = [
    "DB1", "DB2", "DB3", "DB4", "DB5", "DB6", "DB7",   # dorsal B-type
    "VB1", "VB2", "VB3", "VB4", "VB5", "VB6", "VB7", "VB8", "VB9", "VB10", "VB11",  # ventral B-type
    "DD1", "DD2", "DD3", "DD4", "DD5", "DD6",           # dorsal D-type (inhibitory)
    "VD1", "VD2", "VD3", "VD4", "VD5", "VD6", "VD7", "VD8", "VD9", "VD10", "VD11", "VD12", "VD13",  # ventral D-type
    "AS1", "AS2", "AS3", "AS4", "AS5", "AS6", "AS7", "AS8", "AS9", "AS10", "AS11",  # A-type
]

# Key interneurons in the locomotion circuit
LOCOMOTION_INTERNEURONS = [
    "AVAL", "AVAR",   # command interneurons (backward)
    "AVBL", "AVBR",   # command interneurons (forward)
    "AVDL", "AVDR",
    "AVEL", "AVER",
    "PVCL", "PVCR",
    "AVJL", "AVJR",
]

# -----------------------------------------------------------------------
# Sensorimotor interface scaling
# -----------------------------------------------------------------------

# Chemical concentration -> PAULA input [0,1]
CHEM_CONCENTRATION_MAX = 1.0      # normalisation maximum

# Proprioception: joint angle range in radians
JOINT_ANGLE_MAX_RAD = 1.2         # ~70° max bend

# Motor neuron spike rate -> muscle activation
# A motor neuron firing at 1 spike/tick -> max muscle activation
SPIKE_RATE_TO_ACTIVATION = 1.0
# Low-pass filter coefficient for muscle activation smoothing
MUSCLE_FILTER_ALPHA = 0.3

# Muscle activation -> MuJoCo torque
# Calibrated so full activation at max torque produces undulatory locomotion
MUSCLE_MAX_TORQUE_NM = 2e-12   # pico-Newton-metres range

# -----------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------
ENV_PLATE_RADIUS_M = 0.05        # 5 cm agar plate
FOOD_SOURCE_POSITION = (0.005, 0.0, 0.0)    # food at x=5mm (ahead of head)
FOOD_GRADIENT_DECAY = 100.0      # exponential decay constant (1/m)
