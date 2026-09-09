"""Persistent diagnostic for sustained physical turn compass tracking.

This is a negative/parameter-calibration harness, not an acceptance test. A
one-time PAULA CPG birth seed starts an empty MuJoCo body after ring settling;
an experimental current then enters its existing TR neuron. The trace compares
raw physical yaw with the recurrent-ring heading and preserves every graded
vestibular intermediate. No heading, body pose, or kinematic turn is supplied
to the network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from simulations.active_inference import aif_agent3d as ag


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NEURON_MODEL = ROOT.parent / "neuron-model"
BURN_IN = 200
TURN_START = 500
TURN_STOP = 900
STEPS = 175
SUBSTEPS = 8
COMMON = {"turn_probe_ports": True, "w_opp": 0.0, "w_cpu1": 0.0, "w_musf_mode": 0.0}
CASES = {
    "graded_direct": {**COMMON, "vestibular_notch_graded": True},
    "graded_opponent_fast": {**COMMON, "vestibular_notch_graded_opponent": True},
    "graded_opponent_slow": {
        **COMMON, "vestibular_notch_graded_opponent": True,
        "vop_notch_graded_opp_lam": 24.0,
    },
    "graded_opponent_dpush16": {
        **COMMON, "vestibular_notch_graded_opponent": True, "d_push": 16,
    },
    "graded_opponent_dpush24": {
        **COMMON, "vestibular_notch_graded_opponent": True, "d_push": 24,
    },
    "graded_opponent_continuous_pen": {
        **COMMON,
        "vestibular_notch_graded_opponent": True,
        "vop_notch_graded_opp_gain": 4.0,
        "vop_notch_graded_opp_pen_gain": 3.0,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.4,
    },
    "graded_opponent_conjunctive_rate_ring": {
        **COMMON,
        "vestibular_notch_graded_opponent": True,
        "vop_notch_graded_opp_gain": 4.0,
        "vop_notch_graded_opp_pen_gain": 3.0,
        "graded_shift": True,
        "graded_shift_gain": 0.25,
        "graded_shift_S0": 0.4,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
    },
    "graded_opponent_conjunctive_rate_ring_high": {
        **COMMON,
        "vestibular_notch_graded_opponent": True,
        "vop_notch_graded_opp_gain": 4.0,
        "vop_notch_graded_opp_pen_gain": 3.0,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.4,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
    },
    "graded_opponent_conjunctive_rate_ring_mid": {
        **COMMON,
        "vestibular_notch_graded_opponent": True,
        "vop_notch_graded_opp_gain": 4.0,
        "vop_notch_graded_opp_pen_gain": 3.0,
        "graded_shift": True,
        "graded_shift_gain": 0.4,
        "graded_shift_S0": 0.4,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
    },
    "graded_opponent_conjunctive_rate_ring_opp8": {
        **COMMON,
        "vestibular_notch_graded_opponent": True,
        "vop_notch_graded_opp_gain": 8.0,
        "vop_notch_graded_opp_pen_gain": 3.0,
        "graded_shift": True,
        "graded_shift_gain": 0.25,
        "graded_shift_S0": 0.4,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
    },
    "graded_opponent_normalized_conjunctive_rate_ring": {
        **COMMON,
        "vestibular_notch_graded_opponent": True,
        "vop_notch_graded_opp_gain": 4.0,
        "vop_notch_graded_opp_pen_gain": 3.0,
        "vop_notch_graded_opp_normalized": True,
        "vop_notch_graded_opp_normalized_gain": 2.0,
        "graded_shift": True,
        "graded_shift_gain": 0.25,
        "graded_shift_S0": 0.4,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
    },
    "graded_opponent_normalized4_conjunctive_rate_ring": {
        **COMMON,
        "vestibular_notch_graded_opponent": True,
        "vop_notch_graded_opp_gain": 4.0,
        "vop_notch_graded_opp_pen_gain": 3.0,
        "vop_notch_graded_opp_normalized": True,
        "vop_notch_graded_opp_normalized_gain": 4.0,
        "graded_shift": True,
        "graded_shift_gain": 0.25,
        "graded_shift_S0": 0.4,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
    },
    "graded_opponent_conjunctive_rate_ring_no_d7": {
        **COMMON,
        "d7": False,
        "vestibular_notch_graded_opponent": True,
        "vop_notch_graded_opp_gain": 4.0,
        "vop_notch_graded_opp_pen_gain": 3.0,
        "graded_shift": True,
        "graded_shift_gain": 0.25,
        "graded_shift_S0": 0.4,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
    },
    "graded_opponent_conjunctive_rate_ring_dpush1": {
        **COMMON,
        "d7": True,
        "d_push": 1,
        "vestibular_notch_graded_opponent": True,
        "vop_notch_graded_opp_gain": 4.0,
        "vop_notch_graded_opp_pen_gain": 3.0,
        "graded_shift": True,
        "graded_shift_gain": 0.25,
        "graded_shift_S0": 0.4,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
    },
    "graded_opponent_conjunctive_trace_rate_ring": {
        **COMMON,
        "vestibular_notch_graded_opponent": True,
        "vop_notch_graded_opp_gain": 4.0,
        "vop_notch_graded_opp_pen_gain": 3.0,
        "graded_shift": True,
        "graded_shift_gain": 0.25,
        "graded_shift_S0": 0.4,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "conjunctive_ring_tau": 4.0,
        "conjunctive_velocity_tau": 4.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
    },
    "graded_opponent_conjunctive_rate_ring_pushpull": {
        **COMMON,
        "vestibular_notch_graded_opponent": True,
        "vop_notch_graded_opp_gain": 4.0,
        "vop_notch_graded_opp_pen_gain": 3.0,
        "graded_shift": True,
        "graded_shift_gain": 0.25,
        "graded_shift_S0": 0.4,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_pull": 0.25,
    },
    "graded_opponent_conjunctive_rate_ring_d7pen": {
        **COMMON,
        "vestibular_notch_graded_opponent": True,
        "vop_notch_graded_opp_gain": 4.0,
        "vop_notch_graded_opp_pen_gain": 3.0,
        "graded_shift": True,
        "graded_shift_gain": 0.25,
        "graded_shift_S0": 0.4,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "d7_to_shift": True,
        "w_d7_shift": -1.0,
    },
    "graded_opponent_conjunctive_rate_ring_balanced": {
        **COMMON,
        "vestibular_notch_graded_opponent": True,
        "vop_notch_graded_opp_gain": 4.0,
        "vop_notch_graded_opp_pen_gain": 3.0,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.4,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
    },
    # Lower-lag PAULA alternative to the full-stroke notch.  The paired
    # graded afferents low-pass opposite raw gyro half-waves, and the
    # conjunctive P-EN sums their signed local dendritic drives before its
    # bump × velocity release.  No host yaw filter participates.
    "lowpass_opponent_conjunctive_rate_ring_tau22": {
        **COMMON,
        "vestibular_opponent": True,
        "vop_tau": 22.0,
        "vop_gain": 1.0,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.4,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
    },
    "lowpass_opponent_conjunctive_rate_ring_tau60": {
        **COMMON,
        "vestibular_opponent": True,
        "vop_tau": 60.0,
        "vop_gain": 1.0,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.4,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
    },
    "opponent_population8_conjunctive_rate_ring": {
        **COMMON,
        "vestibular_opponent_bank": True,
        # Log-spaced PAULA membrane constants deliberately span the 44-tick
        # gait period.  The P-EN sees their normalized signed population sum,
        # never a host-computed average.
        "vop_bank_taus": (8.0, 12.0, 18.0, 28.0, 44.0, 68.0, 104.0, 160.0),
        "vop_bank_gain": 1.0,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.4,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
    },
    "relay_efference_population_conjunctive_rate_ring": {
        **COMMON,
        # No raw gyro representation is built or driven in this condition.
        # The neural input is the differential population of the last
        # spiking paddle relays before the graded muscles.
        "relay_efference_population": True,
        "relay_efference_only": True,
        "relay_efference_taus": (6.0, 12.0, 24.0, 48.0),
        "relay_efference_input_gain": 6.0,
        "relay_efference_pen_gain": 1.0,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.0,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
    },
    "graded_opponent_conjunctive_rate_ring_nr72": {
        **COMMON,
        # Literal resolution expansion: twice as many E-PG/RING columns and
        # twice as many directional P-EN columns.  ``AIFAgent3D`` rebuilds
        # the ring-sized structures before assembling PAULA wiring.
        "nr": 72,
        "vestibular_notch_graded_opponent": True,
        "vop_notch_graded_opp_gain": 4.0,
        "vop_notch_graded_opp_pen_gain": 3.0,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.4,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        # Preserve the independently calibrated conductance asymmetry; only
        # angular column spacing and population count change here.
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
    },
    "experimental_phase_locked_stroke44_conjunctive_rate_ring": {
        **COMMON,
        # Diagnostic only.  This uses an engineered sample/reset/hold
        # subclass, not a biologically faithful cell or central-complex claim.
        "vestibular_phase_locked": True,
        "vop_phase_period": 44,
        "vop_phase_input_gain": 1.0,
        "vop_phase_output_gain": 0.02,
        "vop_phase_pen_gain": 1.0,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.0,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
    },
    "experimental_phase_locked_stroke44_pulse8_conjunctive_rate_ring": {
        **COMMON,
        "vestibular_phase_locked": True,
        "vop_phase_period": 44,
        "vop_phase_input_gain": 1.0,
        # Preserve roughly the per-stroke transmitter area of the hold
        # condition (0.02 * 44) while emitting for only 8 PAULA ticks.
        "vop_phase_output_gain": 0.11,
        "vop_phase_hold_ticks": 8,
        "vop_phase_pen_gain": 1.0,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.0,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
    },
    "experimental_phase_locked_stroke44_pulse24_conjunctive_rate_ring": {
        **COMMON,
        "vestibular_phase_locked": True,
        "vop_phase_period": 44,
        "vop_phase_input_gain": 1.0,
        "vop_phase_output_gain": 0.037,
        "vop_phase_hold_ticks": 24,
        "vop_phase_pen_gain": 1.0,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.0,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
    },
    "experimental_phase_locked_stroke44_pulse24_gain055_conjunctive_rate_ring": {
        **COMMON,
        "vestibular_phase_locked": True,
        "vop_phase_period": 44,
        "vop_phase_input_gain": 1.0,
        "vop_phase_output_gain": 0.055,
        "vop_phase_hold_ticks": 24,
        "vop_phase_pen_gain": 1.0,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.0,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
    },
    "ordinary_stroke_reset44_conjunctive_rate_ring": {
        **COMMON,
        # Strict alternative to the experimental phase sampler: every state
        # transition is an ordinary PAULA membrane/synapse operation.  The
        # equal clock baseline is an opponent common mode, not a held value.
        "vestibular_stroke_reset": True,
        "vop_reset_period": 44,
        "vop_reset_input_gain": 1.0,
        "vop_reset_gain": 40.0,
        "vop_reset_lam": 44.0,
        "vop_reset_output_gain": 1.0,
        "vop_reset_output_S0": -1.5,
        "vop_reset_pen_gain": 1.0,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.0,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
    },
    # A falsifiable timing sweep, not a gain sweep.  The strict reset clock
    # emits every 44 neural ticks, while the historical d_push=8 feedback
    # delay advances by four ticks of phase per stroke.  These short delays
    # divide 44 exactly, so a result can distinguish a feedback-phase-slip
    # mechanism from a generic "movement is jittery" explanation.
    "ordinary_stroke_reset44_dpush4_conjunctive_rate_ring": {
        **COMMON,
        "vestibular_stroke_reset": True,
        "vop_reset_period": 44,
        "vop_reset_input_gain": 1.0,
        "vop_reset_gain": 40.0,
        "vop_reset_lam": 44.0,
        "vop_reset_output_gain": 1.0,
        "vop_reset_output_S0": -1.5,
        "vop_reset_pen_gain": 1.0,
        "d_push": 4,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.0,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
    },
    # Explicit PB/EB update route: the four P-EN threshold ranks first
    # converge in named PB relays, then return at the one-column EB offset.
    # Maintenance is intentionally off in this first body run; the isolated
    # harness must establish an active P-EG loop independently.
    "ordinary_stroke_reset44_dpush4_pb_update_bridge": {
        **COMMON,
        "vestibular_stroke_reset": True,
        "vop_reset_period": 44,
        "vop_reset_input_gain": 1.0,
        "vop_reset_gain": 40.0,
        "vop_reset_lam": 44.0,
        "vop_reset_output_gain": 1.0,
        "vop_reset_output_S0": -1.5,
        "vop_reset_pen_gain": 1.0,
        "d_push": 4,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.0,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
        "pb_eb_bridge": True,
        "d_pb_shift": 1,
        "pb_shift_gain": 1.0,
        "pb_shift_lam": 1.0,
    },
    # Isolated gate winner among the first PB bridge variants: an ordinary
    # 22-tick graded PB membrane retains the signed P-EN residual across the
    # alternating gait half-strokes, while negative opponent dendrites remove
    # common-mode P-EN drive locally.  This remains a hypothesis until its
    # full raw replay and embodied trace agree.
    "ordinary_stroke_reset44_dpush4_pb_opponent_lam22": {
        **COMMON,
        "vestibular_stroke_reset": True,
        "vop_reset_period": 44,
        "vop_reset_input_gain": 1.0,
        "vop_reset_gain": 40.0,
        "vop_reset_lam": 44.0,
        "vop_reset_output_gain": 1.0,
        "vop_reset_output_S0": -1.5,
        "vop_reset_pen_gain": 1.0,
        "d_push": 4,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.0,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
        "pb_eb_bridge": True,
        "d_pb_shift": 1,
        "pb_shift_gain": 1.0,
        "pb_shift_lam": 22.0,
        "w_pb_opponent": 1.0,
    },
    # Same PB opponent bridge, but with the existing PAULA stroke-clock
    # providing an ordinary inhibitory reset to the slow PB membranes.  The
    # no-reset trace showed monotonically growing PB release and a pinned
    # E-PG ring; this condition tests that precise causal repair.
    "ordinary_stroke_reset44_dpush4_pb_opponent_lam22_reset8": {
        **COMMON,
        "vestibular_stroke_reset": True,
        "vop_reset_period": 44,
        "vop_reset_input_gain": 1.0,
        "vop_reset_gain": 40.0,
        "vop_reset_lam": 44.0,
        "vop_reset_output_gain": 1.0,
        "vop_reset_output_S0": -1.5,
        "vop_reset_pen_gain": 1.0,
        "d_push": 4,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.0,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
        "pb_eb_bridge": True,
        "d_pb_shift": 1,
        "pb_shift_gain": 1.0,
        "pb_shift_lam": 22.0,
        "w_pb_opponent": 1.0,
        "pb_stroke_reset": True,
        "pb_reset_gain": 8.0,
    },
    "ordinary_stroke_reset44_dpush4_pb_opponent_lam22_reset16": {
        **COMMON,
        "vestibular_stroke_reset": True,
        "vop_reset_period": 44,
        "vop_reset_input_gain": 1.0,
        "vop_reset_gain": 40.0,
        "vop_reset_lam": 44.0,
        "vop_reset_output_gain": 1.0,
        "vop_reset_output_S0": -1.5,
        "vop_reset_pen_gain": 1.0,
        "d_push": 4,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.0,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
        "pb_eb_bridge": True,
        "d_pb_shift": 1,
        "pb_shift_gain": 1.0,
        "pb_shift_lam": 22.0,
        "w_pb_opponent": 1.0,
        "pb_stroke_reset": True,
        "pb_reset_gain": 16.0,
    },
    # A distinct, all-ordinary PAULA buffering hypothesis motivated by
    # phase-dependent sensory gating during locomotion. A 44-cell CPG phase
    # population selects the stable negative-yaw subphase; each PB update
    # cell requires local E-PG, a bounded fast signed gyro afferent, and that
    # phase spike. It is not the rejected sample/hold subclass.
    "ordinary_stroke_reset44_pb_phase_gated_update": {
        **COMMON,
        "vestibular_stroke_reset": True,
        "vop_reset_period": 44,
        "vop_reset_input_gain": 1.0,
        "vop_reset_gain": 40.0,
        "vop_reset_lam": 44.0,
        "vop_reset_output_gain": 1.0,
        "vop_reset_output_S0": -1.5,
        "vop_reset_pen_gain": 1.0,
        "d_push": 4,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.0,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        # A selected phase recruits about nine local PB gates, versus ~56
        # concurrently active direct P-ENs. This keeps the instantaneous
        # aggregate update conductance comparable for the topology test.
        "w_push_ccw": 2.1,
        "w_push_cw": 2.1,
        "pb_eb_bridge": True,
        "pb_phase_update": True,
        # Reset-clock tick + the PAULA phase relay propagation yields gate
        # firing at physical phase 16, within the measured right-turn yaw
        # lobe (phases 14--20); it is an evidence-derived fixed wiring lag.
        "pb_phase_start": 11,
        "pb_phase_width": 3,
        "r_pb_phase": 2.1,
        "lam_pb_phase": 1.0,
        "w_pb_phase_ring": 0.7,
        "w_pb_phase_sensor": 0.3,
        "w_pb_phase_clock": 0.7,
        "pb_phase_sensor_max": 4.0,
    },
    # Measured spatial-coincidence correction to the preceding case.  In the
    # full-tick body/replay trace the local E-PG term supplied only 0.037
    # membrane units, leaving every gate at 1.98 below r=2.10.  Scaling only
    # that local term to 4.0 and setting r=2.08 preserves a subthreshold
    # inactive column (~1.90) while allowing E-PG-active gates through.
    "ordinary_stroke_reset44_pb_phase_spatial_gated_update": {
        **COMMON,
        "vestibular_stroke_reset": True,
        "vop_reset_period": 44,
        "vop_reset_input_gain": 1.0,
        "vop_reset_gain": 40.0,
        "vop_reset_lam": 44.0,
        "vop_reset_output_gain": 1.0,
        "vop_reset_output_S0": -1.5,
        "vop_reset_pen_gain": 1.0,
        "d_push": 4,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.0,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 2.1,
        "w_push_cw": 2.1,
        "pb_eb_bridge": True,
        "pb_phase_update": True,
        "pb_phase_start": 11,
        "pb_phase_width": 3,
        "r_pb_phase": 2.08,
        "lam_pb_phase": 1.0,
        "w_pb_phase_ring": 4.0,
        "w_pb_phase_sensor": 0.3,
        "w_pb_phase_clock": 0.7,
        "pb_phase_sensor_max": 4.0,
    },
    # P-EG-style zero-offset maintenance is a distinct hypothesis from the
    # velocity update.  It is deliberately evaluated separately from a full
    # PB topology: the present PEG relay has no explicit bridge geometry, so
    # a negative outcome must not be misreported as a failed connectome model.
    "ordinary_stroke_reset44_dpush4_peg_maintenance": {
        **COMMON,
        "vestibular_stroke_reset": True,
        "vop_reset_period": 44,
        "vop_reset_input_gain": 1.0,
        "vop_reset_gain": 40.0,
        "vop_reset_lam": 44.0,
        "vop_reset_output_gain": 1.0,
        "vop_reset_output_S0": -1.5,
        "vop_reset_pen_gain": 1.0,
        "d_push": 4,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.0,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
        "w_peg": 0.6,
        "w_peg_ring": 0.8,
        "d_peg": 1,
    },
    # Same structural path at a deliberately suprathreshold input, following
    # the silent-relay audit above.  This is not parameter fishing: it
    # distinguishes "P-EG had no effect because it was absent" from "an
    # active direct P-EG relay does not cure the update instability".
    "ordinary_stroke_reset44_dpush4_peg_active": {
        **COMMON,
        "vestibular_stroke_reset": True,
        "vop_reset_period": 44,
        "vop_reset_input_gain": 1.0,
        "vop_reset_gain": 40.0,
        "vop_reset_lam": 44.0,
        "vop_reset_output_gain": 1.0,
        "vop_reset_output_S0": -1.5,
        "vop_reset_pen_gain": 1.0,
        "d_push": 4,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.0,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
        "w_peg": 1.6,
        "w_peg_ring": 0.8,
        "d_peg": 1,
    },
    "ordinary_stroke_reset44_dpush11_conjunctive_rate_ring": {
        **COMMON,
        "vestibular_stroke_reset": True,
        "vop_reset_period": 44,
        "vop_reset_input_gain": 1.0,
        "vop_reset_gain": 40.0,
        "vop_reset_lam": 44.0,
        "vop_reset_output_gain": 1.0,
        "vop_reset_output_S0": -1.5,
        "vop_reset_pen_gain": 1.0,
        "d_push": 11,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.0,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
    },
    "ordinary_stroke_reset44_dpush22_conjunctive_rate_ring": {
        **COMMON,
        "vestibular_stroke_reset": True,
        "vop_reset_period": 44,
        "vop_reset_input_gain": 1.0,
        "vop_reset_gain": 40.0,
        "vop_reset_lam": 44.0,
        "vop_reset_output_gain": 1.0,
        "vop_reset_output_S0": -1.5,
        "vop_reset_pen_gain": 1.0,
        "d_push": 22,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.0,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
    },
    "ordinary_stroke_reset44_dpush44_conjunctive_rate_ring": {
        **COMMON,
        # Circuit timing hypothesis only: retain the ordinary CPG/reset
        # path, but let the P-EN->E-PG propagation delay span one measured
        # stroke instead of the legacy eight ticks.
        "vestibular_stroke_reset": True,
        "vop_reset_period": 44,
        "vop_reset_input_gain": 1.0,
        "vop_reset_gain": 40.0,
        "vop_reset_lam": 44.0,
        "vop_reset_output_gain": 1.0,
        "vop_reset_output_S0": -1.5,
        "vop_reset_pen_gain": 1.0,
        "d_push": 44,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.0,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
    },
    "ordinary_stroke_reset44_cross_inhib1_conjunctive_rate_ring": {
        **COMMON,
        # Explicit ordinary reciprocal inhibition between the two stroke
        # integrators.  This tests directional selection before P-EN rather
        # than relying on a special phase output primitive.
        "vestibular_stroke_reset": True,
        "vop_reset_period": 44,
        "vop_reset_input_gain": 1.0,
        "vop_reset_gain": 40.0,
        "vop_reset_lam": 44.0,
        "vop_reset_output_gain": 1.0,
        "vop_reset_output_S0": -1.5,
        "vop_reset_cross_inhib": 1.0,
        "vop_reset_pen_gain": 1.0,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.0,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
    },
    "graded_opponent_conjunctive_rate_ring_lead": {
        **COMMON,
        "vestibular_notch_graded_opponent": True,
        "vop_notch_graded_opp_gain": 4.0,
        "vop_notch_graded_opp_pen_gain": 3.0,
        "vop_notch_graded_opp_lead_gain": 1.0,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.4,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
    },
    "graded_opponent_conjunctive_rate_ring_dendritic_lead5": {
        **COMMON,
        "vestibular_notch_graded_opponent": True,
        "vop_notch_graded_opp_gain": 4.0,
        "vop_notch_graded_opp_pen_gain": 3.0,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.4,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        # P-ENs lead the E-PG compass bump during a turn in the biological
        # circuit.  This is a local, causal dendritic phase lead on the
        # signed PAULA velocity input, distinct from the earlier lead on the
        # rectified upstream opponent release.
        "conjunctive_velocity_lead_gain": 5.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
    },
    "graded_opponent_conjunctive_rate_ring_balanced_dpush1": {
        **COMMON,
        "vestibular_notch_graded_opponent": True,
        "vop_notch_graded_opp_gain": 4.0,
        "vop_notch_graded_opp_pen_gain": 3.0,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.4,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        # The prior d_push=1 negative test used half the P-EN gain and the
        # uncalibrated symmetric projections.  This isolates transmission
        # latency using the currently balanced, live rate-ring construction.
        "d_push": 1,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
    },
    "notch_pulse_rate_ring": {
        **COMMON,
        "vestibular_notch": True,
        "vop_notch_window": 44,
        "vop_notch_input_gain": 1000.0,
        "vop_notch_r": 0.12,
        "vop_notch_gain": 1.0,
        "vop_notch_pen_gain": 1.4,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
    },
    "graded_opponent_conjunctive_rate_ring_relay": {
        **COMMON,
        "vestibular_notch_graded_opponent": True,
        "vop_notch_graded_opp_gain": 4.0,
        "vop_notch_graded_opp_pen_gain": 3.0,
        "graded_shift": True,
        "graded_shift_gain": 0.5,
        "graded_shift_S0": 0.4,
        "conjunctive_graded_shift": True,
        "conjunctive_graded_S0": 0.0,
        "graded_ring": True,
        "graded_ring_gain": 0.08,
        "graded_ring_S0": 0.2,
        "w_push_ccw": 0.34,
        "w_push_cw": 0.67,
        "shift_relay": True,
        "shift_relay_gain": 1.0,
        "shift_relay_lam": 4.0,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _ring_heading(window: deque[np.ndarray]) -> float | None:
    vector = np.sum(np.sum(window, axis=0) * np.exp(1j * ag.cc.PHI))
    if abs(vector) < 1e-12:
        # Silence is itself the result of a diagnostic condition.  Keep
        # recording so a failed embodied candidate leaves an auditable trace
        # rather than an exception with no final state.
        return None
    return float(np.angle(vector))


def run_case(seed: int, name: str, build: dict, turn_neuron: int) -> dict:
    np.random.seed(seed)
    agent = ag.AIFAgent3D(seed=seed, **build)
    agent.world = ag.w3.World3D(seed=seed, n_food=0, n_tox=0, arena=8.0)
    agent.img = agent.world.retina()
    agent.birth()
    # Suppress run_episode's usual immediate kick. The declared birth seed
    # below begins the same PAULA CPG only after the ring has settled.
    agent._kicked = True
    ring_window: deque[np.ndarray] = deque(maxlen=24)
    trace: list[dict] = []

    def stimulus(current) -> None:
        if current.t == BURN_IN:
            current.net.set_external_input(ag.CPGP[0], 0, 5.0)
            if current._vestibular_phase_locked and not current._phase_clock_seeded:
                current.net.set_external_input(ag.STROKE_CLOCK, ag.STROKE_CLOCK_SYN, 5.0)
                current._phase_clock_seeded = True
            if current._vestibular_stroke_reset and not current._stroke_reset_clock_seeded:
                current.net.set_external_input(ag.STROKE_RESET_CLOCK, ag.STROKE_RESET_CLOCK_SYN, 5.0)
                current._stroke_reset_clock_seeded = True
        if TURN_START <= current.t < TURN_STOP:
            current.net.set_external_input(turn_neuron, ag.TURN_PROBE_SYN[turn_neuron], 3.0)

    def capture(current) -> None:
        ring_window.append(np.asarray([current.nb[nid].O > 0 for nid in ag.cc.RING], dtype=float))
        x, y, yaw = current.world.pose()
        ring_heading = _ring_heading(ring_window)
        trace.append({
            "neural_tick": current.t,
            "physical_yaw_radians": float(yaw),
            "physical_yaw_rate": float(current.world.yaw_rate()),
            # Keep the physical trajectory too.  The replay viewer uses this
            # as the right-hand MuJoCo pane; the left pane deliberately
            # rebuilds a no-physics kinematic trajectory from the *same raw
            # gyro stream* so a user can see what is and is not shared.
            "physical_pose": {"x": float(x), "y": float(y), "yaw": float(yaw)},
            "physical_speed": float(current.world.speed()),
            "applied_raw_yaw_rate": float(current.last_gyro_yaw_rate),
            "ring_heading_radians": ring_heading,
            "ring_live": ring_heading is not None,
            # Retain the population raster, not only its 24-tick decoder.
            # This makes a later analysis able to distinguish a genuinely
            # irregular P-EN/E-PG travelling wave from decoder jitter.
            "ring_spikes_by_column": [int(current.nb[nid].O > 0) for nid in ag.cc.RING],
            "turn_stimulus_active": TURN_START <= current.t < TURN_STOP,
            "turn_neuron": turn_neuron,
            "turn_spike": int(current.nb[turn_neuron].O > 0),
            "notch_release": {
                "CCW": (float(current.nb[ag.VEST_NET_CCW].O)
                        if ag.VEST_NET_CCW in current.nb else None),
                "CW": (float(current.nb[ag.VEST_NET_CW].O)
                       if ag.VEST_NET_CW in current.nb else None),
            },
            "opponent_release": {
                "CCW": (
                    float(current.nb[ag.VEST_OPP_CCW].O)
                    if current._vestibular_notch_graded_opponent else
                    float(current.nb[ag.VEST_CCW].O) if current._vestibular_opponent else
                    float(np.mean([current.nb[nid].O for nid in ag.VEST_BANK_CCW]))
                    if current._vestibular_opponent_bank else None
                ),
                "CW": (
                    float(current.nb[ag.VEST_OPP_CW].O)
                    if current._vestibular_notch_graded_opponent else
                    float(current.nb[ag.VEST_CW].O) if current._vestibular_opponent else
                    float(np.mean([current.nb[nid].O for nid in ag.VEST_BANK_CW]))
                    if current._vestibular_opponent_bank else None
                ),
            },
            "opponent_population_release": (
                {
                    "CCW": [float(current.nb[nid].O) for nid in ag.VEST_BANK_CCW],
                    "CW": [float(current.nb[nid].O) for nid in ag.VEST_BANK_CW],
                }
                if current._vestibular_opponent_bank else None
            ),
            "phase_locked_release": (
                {
                    "CCW": float(current.nb[ag.STROKE_CCW].O),
                    "CW": float(current.nb[ag.STROKE_CW].O),
                    "clock": int(current.nb[ag.STROKE_CLOCK].O > 0),
                }
                if current._vestibular_phase_locked else None
            ),
            "stroke_reset_release": (
                {
                    "CCW": float(current.nb[ag.STROKE_RESET_CCW].O),
                    "CW": float(current.nb[ag.STROKE_RESET_CW].O),
                    "CCW_membrane": float(current.nb[ag.STROKE_RESET_CCW].S),
                    "CW_membrane": float(current.nb[ag.STROKE_RESET_CW].S),
                    "clock": int(current.nb[ag.STROKE_RESET_CLOCK].O > 0),
                }
                if current._vestibular_stroke_reset else None
            ),
            "shift_spikes": {
                "CL": sum(int(current.nb[nid].O > 0) for column in ag.cc.CL for nid in column),
                "CR": sum(int(current.nb[nid].O > 0) for column in ag.cc.CR for nid in column),
            },
            # P-EG maintenance is an opt-in topology experiment.  Record its
            # actual activity so an unchanged ring cannot be hand-waved as a
            # subtle anatomical benefit when the relay was silent.
            "peg_spikes": (
                sum(int(current.nb[nid].O > 0) for nid in ag.cc.PEG)
                if build.get("w_peg", 0.0) else None
            ),
            "pb_shift_spikes": (
                {
                    "CL": sum(int(current.nb[nid].O > 0) for nid in (
                        ag.cc.PB_PHASE_CL if build.get("pb_phase_update", False) else ag.cc.PB_CL)),
                    "CR": sum(int(current.nb[nid].O > 0) for nid in (
                        ag.cc.PB_PHASE_CR if build.get("pb_phase_update", False) else ag.cc.PB_CR)),
                }
                if build.get("pb_eb_bridge", False) else None
            ),
            "pb_shift_release": (
                {
                    "CL": float(sum(current.nb[nid].O for nid in (
                        ag.cc.PB_PHASE_CL if build.get("pb_phase_update", False) else ag.cc.PB_CL))),
                    "CR": float(sum(current.nb[nid].O for nid in (
                        ag.cc.PB_PHASE_CR if build.get("pb_phase_update", False) else ag.cc.PB_CR))),
                }
                if build.get("pb_eb_bridge", False) else None
            ),
            "pb_phase_clock": (
                [index for index, nid in enumerate(ag.cc.PB_PHASE_CLOCK) if current.nb[nid].O > 0]
                if build.get("pb_phase_update", False) else None
            ),
            "pb_phase_gate_membrane": (
                {"CL": [float(current.nb[nid].S) for nid in ag.cc.PB_PHASE_CL],
                 "CR": [float(current.nb[nid].S) for nid in ag.cc.PB_PHASE_CR]}
                if build.get("pb_phase_update", False) else None
            ),
            "pb_phase_fast_vestibular": (
                {"CCW": float(current.nb[ag.PHASE_VEST_CCW].O),
                 "CW": float(current.nb[ag.PHASE_VEST_CW].O)}
                if build.get("pb_phase_update", False) else None
            ),
            "pb_maintenance_spikes": (
                sum(int(current.nb[nid].O > 0) for nid in ag.cc.PB_ML + ag.cc.PB_MR)
                if build.get("pb_eb_bridge", False) and build.get("w_peg", 0.0) else None
            ),
            "shift_spikes_by_column": {
                "CL": [sum(int(current.nb[nid].O > 0) for nid in column) for column in ag.cc.CL],
                "CR": [sum(int(current.nb[nid].O > 0) for nid in column) for column in ag.cc.CR],
            },
        })

    ag.run_episode(
        agent,
        steps=STEPS,
        sub=SUBSTEPS,
        vision=False,
        render_every=10**9,
        render_ticks=0,
        log_every=10**9,
        tick_hook=capture,
        neural_input_hook=stimulus,
    )
    return {"condition": name, "seed": seed, "build": build, "turn_neuron": turn_neuron, "tick_trace": trace}


def summary(record: dict) -> dict:
    trace = [row for row in record["tick_trace"] if row["neural_tick"] >= BURN_IN]
    body = np.unwrap(np.asarray([row["physical_yaw_radians"] for row in trace]))
    live_indices = [i for i, row in enumerate(trace) if row["ring_live"]]
    if not live_indices:
        raise RuntimeError("ring never became active after compass burn-in")
    ring_indices = np.asarray(live_indices, dtype=int)
    ring = np.unwrap(np.asarray([trace[i]["ring_heading_radians"] for i in ring_indices]))
    # Compare only across the contiguous live prefix.  A later silent ring
    # cannot carry a heading, so inventing a final angle would conceal the
    # failure rather than quantify it.
    live_prefix_end = next((i for i, row in enumerate(trace) if not row["ring_live"]), len(trace))
    prefix_indices = np.arange(live_prefix_end)
    prefix_ring = np.unwrap(np.asarray([trace[i]["ring_heading_radians"] for i in prefix_indices]))
    error = np.degrees((prefix_ring - prefix_ring[0]) - (body[prefix_indices] - body[0]))
    marks = {}
    ticks = np.asarray([row["neural_tick"] for row in trace])
    for marker in (BURN_IN, TURN_START, TURN_STOP, STEPS * SUBSTEPS):
        idx = int(np.argmin(abs(ticks - marker)))
        row = trace[idx]
        prior_live = [i for i in range(idx + 1) if trace[i]["ring_live"]]
        marks[str(marker)] = {
            "body_delta_degrees": float(np.degrees(body[idx] - body[0])),
            "ring_live": row["ring_live"],
            "ring_delta_degrees": (
                float(np.degrees(trace[prior_live[-1]]["ring_heading_radians"] - trace[live_indices[0]]["ring_heading_radians"]))
                if row["ring_live"] and prior_live else None
            ),
        }
    first_silent = next((row["neural_tick"] for row in trace if not row["ring_live"]), None)
    final_live_index = live_indices[-1]
    return {
        "body_delta_degrees": float(np.degrees(body[-1] - body[0])),
        "ring_delta_degrees_last_live": float(np.degrees(ring[-1] - ring[0])),
        "tracking_error_last_live_degrees": float(error[-1]),
        "tracking_error_range_degrees": [float(error.min()), float(error.max())],
        "ring_live_through_terminal": first_silent is None,
        "first_ring_silent_tick": first_silent,
        "last_ring_live_tick": trace[final_live_index]["neural_tick"],
        "mean_abs_yaw_rate": float(np.mean(np.abs([row["physical_yaw_rate"] for row in trace]))),
        "opponent_release_mean": {
            direction: float(np.mean([row["opponent_release"][direction] or 0.0 for row in trace]))
            for direction in ("CCW", "CW")
        },
        "shift_spikes": {direction: int(sum(row["shift_spikes"][direction] for row in trace)) for direction in ("CL", "CR")},
        "milestones": marks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--turn-neuron", choices=("TL", "TR"), default="TR",
                        help="PAULA turn neuron stimulated during the sustained course")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output and args.resume:
        raise SystemExit("use either --output or --resume, not both")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.resume or args.output or HERE / "results" / f"embodied_compass_longturn_diagnostic_{stamp}"
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest.get("experiment") != "embodied_compass_longturn_diagnostic" or manifest.get("seeds") != args.seeds:
            raise SystemExit("--resume requires this diagnostic's matching manifest and seed set")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "manifest.json", {
            "experiment": "embodied_compass_longturn_diagnostic",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": {"active_inference": _revision(ROOT), "neuron_model": _revision(NEURON_MODEL)},
            "source_fingerprints": {"aif_agent3d.py": _sha256(HERE.parent / "aif_agent3d.py"), "runner": _sha256(Path(__file__).resolve())},
            "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "mujoco": mujoco.__version__},
            "seeds": args.seeds,
            "cases": CASES,
            "course": {"cpg_birth_tick": BURN_IN, "turn_start": TURN_START, "turn_stop": TURN_STOP, "turn_neuron": args.turn_neuron, "steps": STEPS, "substeps": SUBSTEPS},
            "scope": "diagnostic only: sustained PAULA TL/TR turn through muscles/MuJoCo/raw gyro/recurrent ring; no PI or homing claim",
        })
    completed = 0
    for seed in args.seeds:
        for name in args.cases:
            path = output / f"{name}_seed{seed}.json"
            if path.exists():
                continue
            if args.max_cases is not None and completed >= args.max_cases:
                break
            record = run_case(seed, name, CASES[name], ag.TL if args.turn_neuron == "TL" else ag.TR)
            _write_json(path, record)
            row = summary(record)
            ring_text = f"{row['ring_delta_degrees_last_live']:+.1f}" if row["ring_live_through_terminal"] else "SILENT"
            print(f"seed={seed} {name}: body={row['body_delta_degrees']:+.1f} ring={ring_text} error={row['tracking_error_last_live_degrees']:+.1f}", flush=True)
            completed += 1
        if args.max_cases is not None and completed >= args.max_cases:
            break
    summaries = {f"{name}_seed{seed}": summary(json.loads((output / f"{name}_seed{seed}.json").read_text())) for seed in args.seeds for name in args.cases if (output / f"{name}_seed{seed}.json").exists()}
    _write_json(output / "summary.json", summaries)
    print(f"Wrote long-turn diagnostic to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
