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
    "DB1", "DB2", "DB3", "DB4", "DB5", "DB6", "DB7",   # dorsal B-type (forward)
    "VB1", "VB2", "VB3", "VB4", "VB5", "VB6", "VB7", "VB8", "VB9", "VB10", "VB11",  # ventral B-type (forward)
    "DA1", "DA2", "DA3", "DA4", "DA5", "DA6", "DA7", "DA8", "DA9",  # dorsal A-type (backward)
    "VA1", "VA2", "VA3", "VA4", "VA5", "VA6", "VA7", "VA8", "VA9", "VA10", "VA11", "VA12",  # ventral A-type (backward)
    "DD1", "DD2", "DD3", "DD4", "DD5", "DD6",           # dorsal D-type (GABAergic inhibitory)
    "VD1", "VD2", "VD3", "VD4", "VD5", "VD6", "VD7", "VD8", "VD9", "VD10", "VD11", "VD12", "VD13",  # ventral D-type
    "AS1", "AS2", "AS3", "AS4", "AS5", "AS6", "AS7", "AS8", "AS9", "AS10", "AS11",  # A-type (dorsal)
]

# -----------------------------------------------------------------------
# Anatomical motor neuron positions (WormAtlas / White et al. 1986)
# -----------------------------------------------------------------------
# Fractional body position [0=head, 1=tail] for each motor neuron.
# Used to map each neuron to its nearest body segment.
MOTOR_NEURON_POSITIONS: dict[str, float] = {
    # Dorsal B-type (forward cholinergic excitatory → dorsal muscles)
    "DB1": 0.08, "DB2": 0.18, "DB3": 0.29, "DB4": 0.40,
    "DB5": 0.53, "DB6": 0.65, "DB7": 0.79,
    # Ventral B-type (forward cholinergic excitatory → ventral muscles)
    "VB1": 0.06, "VB2": 0.13, "VB3": 0.19, "VB4": 0.26,
    "VB5": 0.33, "VB6": 0.41, "VB7": 0.50, "VB8": 0.58,
    "VB9": 0.66, "VB10": 0.74, "VB11": 0.84,
    # Dorsal A-type (backward cholinergic excitatory → dorsal muscles)
    "DA1": 0.06, "DA2": 0.14, "DA3": 0.21, "DA4": 0.29,
    "DA5": 0.37, "DA6": 0.47, "DA7": 0.57, "DA8": 0.69, "DA9": 0.81,
    # Ventral A-type (backward cholinergic excitatory → ventral muscles)
    "VA1": 0.05, "VA2": 0.10, "VA3": 0.16, "VA4": 0.23,
    "VA5": 0.29, "VA6": 0.36, "VA7": 0.43, "VA8": 0.51,
    "VA9": 0.60, "VA10": 0.68, "VA11": 0.76, "VA12": 0.85,
    # Dorsal D-type (GABAergic inhibitory → dorsal muscles)
    "DD1": 0.10, "DD2": 0.26, "DD3": 0.42, "DD4": 0.57,
    "DD5": 0.71, "DD6": 0.84,
    # Ventral D-type (GABAergic inhibitory → ventral muscles)
    "VD1": 0.05, "VD2": 0.09, "VD3": 0.14, "VD4": 0.20,
    "VD5": 0.27, "VD6": 0.33, "VD7": 0.40, "VD8": 0.48,
    "VD9": 0.55, "VD10": 0.63, "VD11": 0.71, "VD12": 0.79, "VD13": 0.87,
    # AS-type (cholinergic excitatory → primarily dorsal muscles)
    "AS1": 0.06, "AS2": 0.14, "AS3": 0.21, "AS4": 0.28,
    "AS5": 0.35, "AS6": 0.43, "AS7": 0.52, "AS8": 0.60,
    "AS9": 0.68, "AS10": 0.76, "AS11": 0.85,
}

# Command interneurons that drive forward/backward motor neuron pools
COMMAND_INTERNEURONS_FORWARD = ["AVBL", "AVBR", "PVCL", "PVCR"]
COMMAND_INTERNEURONS_BACKWARD = ["AVAL", "AVAR", "AVDL", "AVDR"]

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
JOINT_ANGLE_MAX_RAD = 0.45        # tuned with lab gait (see tuning/notes.md)

# Motor neuron spike rate -> muscle activation
# A motor neuron firing at 1 spike/tick -> max muscle activation
SPIKE_RATE_TO_ACTIVATION = 1.0
# Low-pass filter coefficient for muscle activation smoothing
MUSCLE_FILTER_ALPHA = 0.16

# Muscle activation -> MuJoCo torque
# Calibrated so full activation at max torque produces undulatory locomotion
MUSCLE_MAX_TORQUE_NM = 2e-12   # pico-Newton-metres range

# -----------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------
ENV_PLATE_RADIUS_M = 0.05        # 5 cm agar plate
FOOD_SOURCE_POSITION = (0.0005, 0.0, 0.0)   # food at x=0.5mm (ahead of head)
FOOD_GRADIENT_DECAY = 1800.0     # steep gradient at mm scale for stronger chemotaxis
FOOD_CONSUMPTION_RADIUS_M = 0.0001  # head within 0.1mm (≈1.5× head width) = consumed
