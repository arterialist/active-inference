"""
Larval zebrafish simulation constants.

The first implementation targets a 5-7 dpf larva because that stage has the
best public overlap between behavior, optical physiology, atlases, and EM.
"""

from __future__ import annotations

from math import pi

# ---------------------------------------------------------------------------
# Body geometry
# ---------------------------------------------------------------------------

N_BODY_SEGMENTS = 16
BODY_LENGTH_M = 4.2e-3
BODY_RADIUS_M = 1.6e-4
HEAD_LENGTH_FRACTION = 0.22
SEGMENT_LENGTH_M = BODY_LENGTH_M / N_BODY_SEGMENTS
SWIM_DEPTH_M = -1.4e-3
MIN_SWIM_DEPTH_M = -6.8e-3
MAX_SWIM_DEPTH_M = -0.35e-3
DEPTH_HOME_M = -2.2e-3

# ---------------------------------------------------------------------------
# Simulation timing
# ---------------------------------------------------------------------------

PHYSICS_TIMESTEP_S = 0.005
NEURAL_TICKS_PER_PHYSICS_STEP = 1

# ---------------------------------------------------------------------------
# Larval swimming and hydrodynamic approximation
# ---------------------------------------------------------------------------

TAIL_BEAT_FREQ_MIN_HZ = 12.0
TAIL_BEAT_FREQ_MAX_HZ = 38.0
TAIL_BEAT_PHASE_LAG_RAD = 0.62
TAIL_BEAT_MAX_AMPLITUDE_RAD = 0.28
TAIL_SMOOTHING_ALPHA = 0.34
TAIL_YAW_JOINT_DAMPING = 4.8e-8
TAIL_PITCH_JOINT_DAMPING = 6.5e-8
TAIL_YAW_JOINT_STIFFNESS = 1.2e-7
TAIL_PITCH_JOINT_STIFFNESS = 2.4e-7
TAIL_YAW_ACTUATOR_GAIN = 1.44e-6
TAIL_PITCH_ACTUATOR_GAIN = 1.2e-9

MAX_FORWARD_ACCEL_M_S2 = 1.55
LINEAR_DRAG_PER_S = 9.0
ANGULAR_DRAG_PER_S = 22.0
VERTICAL_DRAG_PER_S = 16.0
TURN_ACCEL_RAD_S2 = 52.0
MAX_VERTICAL_ACCEL_M_S2 = 0.18
PITCH_ACCEL_RAD_S2 = 5.5
PITCH_DRAG_PER_S = 18.0
PITCH_RESTORING_PER_S2 = 18.0
PITCH_NONLINEAR_RESTORING_PER_S2 = 0.0
PITCH_NONLINEAR_THRESHOLD_RAD = 0.18
PITCH_LIMIT_RAD = 0.12
DEPTH_HOME_GAIN = 0.34
BOUNDARY_RESTITUTION = 0.22

# Arbitrary selected video is not the same distribution as controlled
# projector/VR stimuli.  Keep vertical behavior available, but prevent upper /
# lower frame salience from acting as a large direct depth command.
VIDEO_DEPTH_CUE_GAIN = 0.18
VIDEO_DEPTH_CUE_DEADBAND = 0.08
VIDEO_DEPTH_PITCH_GAIN = 0.36
VIDEO_VERTICAL_RATE_PITCH_GAIN = 0.10
VIDEO_STARTLE_PITCH_GAIN = 0.08
VIDEO_PITCH_ALPHA = 0.08
VIDEO_PITCH_TARGET_LIMIT = 0.35

# Externally decoded video/calcium tail targets are allowed to override the
# endogenous CPG only when they carry enough confidence and the bout generator
# has non-trivial drive.  This prevents weak low-action video frames from
# accumulating posture error while preserving explicit swimming bouts.
EXTERNAL_TAIL_TARGET_CONFIDENCE_MIN = 0.015
VIDEO_TAIL_TARGET_CONFIDENCE_MIN = 0.040
TAIL_MOTOR_DRIVE_DEADBAND = 0.030

# Bout-and-coast defaults. Larval zebrafish swim in short bouts separated by
# passive coasts; the nervous system modulates this state.
BOUT_MIN_TICKS = 14
BOUT_MAX_TICKS = 54
COAST_MIN_TICKS = 20
COAST_MAX_TICKS = 150
SPONTANEOUS_BOUT_PROB = 0.004

# ZAPBench calcium labels are sampled at ~1 Hz, while larval tail motor output
# is a short bout.  Holding one decoded calcium frame as a continuous command
# for the full inter-frame interval collapses the simulated body into a spin.
CALCIUM_ACTION_PULSE_TICKS = 42
CALCIUM_ACTION_DECAY_TICKS = 22.0
CALCIUM_FORCE_GAIN = 2.2
CALCIUM_TURN_GAIN = 1.6
CALCIUM_TURN_SCORE_LIMIT = 0.65

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

ARENA_RADIUS_M = 0.05
ARENA_DEPTH_M = 0.008
FOOD_CONSUMPTION_RADIUS_M = 0.0011
FOOD_GRADIENT_DECAY = 42.0
VISUAL_RANGE_M = 0.045
WALL_AVOIDANCE_MARGIN_M = 0.008
LIGHT_LEVEL_DEFAULT = 0.7
TEMPERATURE_C_DEFAULT = 28.0
TEMPERATURE_PREFERRED_C = 28.5
WATER_FLOW_M_S = 0.0

DEFAULT_FOOD_POSITIONS: list[tuple[float, float, float]] = [
    (0.018, 0.012, 0.0),
    (-0.017, 0.018, 0.0),
    (0.014, -0.019, 0.0),
]

# ---------------------------------------------------------------------------
# Reduced zebrafish neural circuit
# ---------------------------------------------------------------------------

SENSORY_NEURONS = [
    "RETINA_L",
    "RETINA_R",
    "OPTIC_FLOW_L",
    "OPTIC_FLOW_R",
    "LATERAL_LINE_L",
    "LATERAL_LINE_R",
    "OLFACTORY_L",
    "OLFACTORY_R",
    "THERMO_HOT",
    "THERMO_COLD",
    "VESTIBULAR_L",
    "VESTIBULAR_R",
    "VESTIBULAR_UP",
    "VESTIBULAR_DOWN",
    "WALL_L",
    "WALL_R",
    "DEPTH_SHALLOW",
    "DEPTH_DEEP",
    "STARTLE",
]

INTERNEURONS = [
    "TECTUM_L",
    "TECTUM_R",
    "PRETECTUM_L",
    "PRETECTUM_R",
    "ARTR_L",
    "ARTR_R",
    "HINDBRAIN_L",
    "HINDBRAIN_R",
    "APPROACH_GATE",
    "AVOID_GATE",
    "SWIM_GATE",
    "DEPTH_HOMEOSTAT",
    "BOUT_CLOCK",
    "MAUTHNER_L",
    "MAUTHNER_R",
]

MOTOR_NEURONS = [
    "RETICULOSPINAL_L",
    "RETICULOSPINAL_R",
    "ASCEND_GATE",
    "DIVE_GATE",
    "SPINAL_CPG_L",
    "SPINAL_CPG_R",
] + [
    f"MOTOR_L_{i:02d}" for i in range(N_BODY_SEGMENTS)
] + [
    f"MOTOR_R_{i:02d}" for i in range(N_BODY_SEGMENTS)
] + [
    f"MOTOR_D_{i:02d}" for i in range(N_BODY_SEGMENTS)
] + [
    f"MOTOR_V_{i:02d}" for i in range(N_BODY_SEGMENTS)
]

ALL_NEURONS = SENSORY_NEURONS + INTERNEURONS + MOTOR_NEURONS

MUSCLE_NAMES = [
    name
    for i in range(N_BODY_SEGMENTS)
    for name in (
        f"tail_{i:02d}_left",
        f"tail_{i:02d}_right",
        f"tail_{i:02d}_dorsal",
        f"tail_{i:02d}_ventral",
    )
]

# Utility constants
TWO_PI = 2.0 * pi
