"""PAULA-only experiment: a four-level ``cylindrical`` compass.

This is a separate research prototype.  It does not import the agent's
central-complex code, does not modify its defaults, and is not used by either
the demo or the embodied organism.  It supplies isolated pulse and recorded
raw-yaw replay harnesses.  The separate
``experiments/embodied_cylinder_compass.py`` runner couples the same circuit
to PAULA CPG/muscle-driven MuJoCo yaw while keeping the default agent intact.

The hypothesis tested here is deliberately different from a continuous ring
attractor.  Four *quaternary* PAULA state rings form the digits of a circular
counter:

    level 0: 4 fine bins          (1 bin)
    level 1: 4 bins of level-0 wraps       (4 bins)
    level 2: 4 bins of level-1 wraps      (16 bins)
    level 3: 4 bins of level-2 wraps      (64 bins)

Together they represent 4**4 = 256 positions, or 1.40625 degrees per code.
Only the level-0 update gates receive the signed vestibular current.  A
positive wrap (3 -> 0) or negative wrap (0 -> 3) excites an ordinary PAULA
carry cell; its spike is the sensory input to the appropriate update gates of
the next ring.  Consequently a higher digit cannot be incremented directly
by the Python harness.

Every state cell, update gate, and carry cell is a normal PAULA neuron.  The
harness supplies only a signed *sensory current* to level 0 and reads spikes
after each tick.  The base-4 decode and all pass/fail metrics are offline
measurement; they are never fed back to the network.

This is an engineering/neuroscience hypothesis, not a claim that brains
implement a literal four-digit counter or that it is biologically faithful.
It has passed an isolated pulse-counter test and received recorded/live
body-yaw tests, but the currently retained embodied two-direction results
fail the bounded compass-tracking gate because the counter is sensitive to
pre-turn gait history.  It has no visual re-anchoring or embodied navigation
test, and no cylinder state is wired into PI, HOME, or motor control.  An
isolated pass is only a clean component foundation, not an acceptance claim.

Run from ``active-inference/``:

    uv run python experiments/paula_cylinder_compass_isolated.py \
        --output experiments/results/paula_cylinder_isolated.json

The JSON has one row per PAULA tick, including every state, update-gate, and
carry spike.  It is intentionally suitable for a replay viewer rather than a
period summary alone.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import numpy as np

from simulations.paula_loader import ensure_paula_available

ensure_paula_available()
from paula_agent import ckit as k  # noqa: E402


BASE = 4
LEVELS = 4
TOTAL_BINS = BASE**LEVELS
BIN_DEGREES = 360.0 / TOTAL_BINS

# Two current ticks are intentional.  A synaptic state signal has one more
# PAULA transport stage than an external vestibular current; two ticks make
# the local ordinary coincidence gate see both.  This is a declared input
# protocol, not a hidden sampling or a host-side heading update.
PULSE_WIDTH = 2
PULSE_PERIOD = 14
SEED_TICKS = 16
DECODE_WINDOW = 6

# Raw-yaw replay uses an explicit PAULA opponent vestibular path.  The host
# writes the *continuous* signed sample through two linear, biased receptor
# currents (bias +/- raw); it neither thresholds it into events nor updates a
# heading.  The receptors, slow signed accumulators, refractory event cells,
# and counter update gates are all PAULA neurons.
RAW_YAW_RECEPTOR_POS = 914_000
RAW_YAW_RECEPTOR_NEG = 914_001
RAW_YAW_DELAY_POS = 914_002
RAW_YAW_DELAY_NEG = 914_003
RAW_YAW_ACC_CCW = 914_010
RAW_YAW_ACC_CW = 914_011
RAW_YAW_PULSE_CCW_A = 914_020
RAW_YAW_PULSE_CCW_B = 914_021
RAW_YAW_PULSE_CW_A = 914_022
RAW_YAW_PULSE_CW_B = 914_023
RAW_STATE_BUFFER_BASE = 915_000
RAW_YAW_BIAS = 45.0  # Must exceed the absolute range of the declared trace.
DEFAULT_BODY_YAW_TRACE = (
    Path(__file__).resolve().parents[1]
    / "simulations/active_inference/experiments/results/sensorimotor_compass_v2_full_seed11.json"
)


def _state_id(level: int, digit: int) -> int:
    return 910_000 + level * 100 + digit


def _positive_update_id(level: int, digit: int) -> int:
    return 911_000 + level * 100 + digit


def _negative_update_id(level: int, digit: int) -> int:
    return 912_000 + level * 100 + digit


def _positive_carry_id(level: int) -> int:
    return 913_000 + level * 2


def _negative_carry_id(level: int) -> int:
    return 913_000 + level * 2 + 1


def _raw_state_buffer_id(level: int, digit: int) -> int:
    return RAW_STATE_BUFFER_BASE + level * BASE + digit


STATES = [[_state_id(level, digit) for digit in range(BASE)] for level in range(LEVELS)]
UPDATE_POS = [
    [_positive_update_id(level, digit) for digit in range(BASE)] for level in range(LEVELS)
]
UPDATE_NEG = [
    [_negative_update_id(level, digit) for digit in range(BASE)] for level in range(LEVELS)
]
CARRY_POS = [_positive_carry_id(level) for level in range(LEVELS - 1)]
CARRY_NEG = [_negative_carry_id(level) for level in range(LEVELS - 1)]
RAW_STATE_BUFFERS = [[_raw_state_buffer_id(level, digit) for digit in range(BASE)] for level in range(LEVELS)]


@dataclass(frozen=True)
class BuildParameters:
    """Fixed topology parameters, recorded in every result artifact."""

    state_threshold: float = 0.8
    state_self_weight: float = 1.8
    state_seed_weight: float = 4.0
    update_threshold: float = 1.0
    update_local_weight: float = 0.6
    update_sensor_weight: float = 0.6
    update_push_weight: float = 2.4
    update_pull_weight: float = -2.2
    carry_threshold: float = 0.8
    carry_input_weight: float = 1.4
    raw_receptor_gain: float = 0.025
    # An optional ordinary graded PAULA sensory relay supplies one extra
    # causal axonal/synaptic transport stage before the opponent integrator.
    # It is deliberately a circuit connection, never a Python previous-value
    # buffer.  The default retains the original no-extra-relay topology.
    raw_sensor_delay_stages: int = 0
    raw_accumulator_lambda: float = 44.0
    raw_accumulator_graded_gain: float = 20.0
    raw_event_threshold: float = 0.5
    raw_event_refractory_ticks: int = 7
    raw_state_buffer_lambda: float = 2.0
    raw_state_buffer_weight: float = 1.2


# Kept separate from ``BuildParameters()`` so replay results cannot silently
# present trace-specific calibration as the generic mechanism.  This setting
# was selected only after inspecting the declared seed-11 waveform and must
# pass an independent body trace before it is a candidate foundation.
RAW_YAW_TUNED_ON_SEED11 = replace(
    BuildParameters(),
    raw_accumulator_lambda=132.0,
    raw_event_threshold=0.55,
    raw_event_refractory_ticks=2,
)


def _build_network(parameters: BuildParameters, *, raw_yaw_transduction: bool = False) -> tuple[str, dict[int, int]]:
    """Build the entire four-level circuit and return its seed synapse map.

    ``seed_synapse`` is retained per state cell because the later push/pull
    connections add more dendrites.  Inferring the final synapse index would
    silently inject a birth seed into an update dendrite.
    """

    neurons: list[dict] = []
    synapses: list[dict] = []
    connections: list[dict] = []
    external_inputs: list[dict] = []
    next_synapse = {state: 0 for ring in STATES for state in ring}
    seed_synapse: dict[int, int] = {}
    if parameters.raw_sensor_delay_stages not in (0, 1):
        raise ValueError("raw_sensor_delay_stages must be 0 or 1")

    def state_synapse(target: int, weight: float, distance: int = 1) -> int:
        sid = next_synapse[target]
        next_synapse[target] += 1
        synapses.append(k.syn(target, sid, weight, distance))
        return sid

    # Persistent state cells.  The self-recurrent loop represents a digit;
    # an update gate excites its successor and inhibits its old digit, giving
    # a one-hot quaternary phase ring without a Python state assignment.
    for level in range(LEVELS):
        for state in STATES[level]:
            neurons.append(
                k.neuron(
                    state,
                    r=parameters.state_threshold,
                    b=parameters.state_threshold,
                    c=1,
                    lam=2,
                )
            )
            connections.append(k.conn(state, state, state_synapse(state, parameters.state_self_weight)))
            seed_synapse[state] = state_synapse(state, parameters.state_seed_weight)
            external_inputs.append(k.ext(state, seed_synapse[state]))
            synapses.append(k.term(state))

    if raw_yaw_transduction:
        # State cells in the compact counter emit spikes with a brief
        # refractory gap.  This local graded buffer is an ordinary PAULA
        # interneuron that bridges that gap for a sensory event arriving one
        # tick earlier or later.  It is a circuit-side timing tolerance, not
        # a host filter and not the rejected sample/hold subclass.
        for level in range(LEVELS):
            for digit, buffer in enumerate(RAW_STATE_BUFFERS[level]):
                neurons.append(k.neuron(
                    buffer, r=1e9, c=1, lam=parameters.raw_state_buffer_lambda,
                    meta={"graded_gain": 1.0, "graded_S0": 0.0},
                ))
                synapses.append(k.syn(buffer, 0, parameters.raw_state_buffer_weight, 1))
                connections.append(k.conn(STATES[level][digit], buffer, 0))
                synapses.append(k.term(buffer))

    # Direction-specific local state x vestibular/carry coincidence gates.
    # The same synapse-1 port is external only at level 0; at higher levels it
    # is physically driven by the immediately lower level's carry cell.
    for level in range(LEVELS):
        for digit in range(BASE):
            for update, direction in ((UPDATE_POS[level][digit], +1), (UPDATE_NEG[level][digit], -1)):
                neurons.append(k.neuron(update, r=parameters.update_threshold,
                                        b=parameters.update_threshold, c=1, lam=1))
                synapses.append(k.syn(update, 0, parameters.update_local_weight, 1))
                synapses.append(k.syn(update, 1, parameters.update_sensor_weight, 1))
                local_source = (
                    RAW_STATE_BUFFERS[level][digit] if raw_yaw_transduction else STATES[level][digit]
                )
                connections.append(k.conn(local_source, update, 0))
                synapses.append(k.term(update))
                successor = STATES[level][(digit + direction) % BASE]
                connections.append(k.conn(update, successor,
                                          state_synapse(successor, parameters.update_push_weight)))
                connections.append(k.conn(update, STATES[level][digit],
                                          state_synapse(STATES[level][digit], parameters.update_pull_weight)))
                if level == 0 and not raw_yaw_transduction:
                    external_inputs.append(k.ext(update, 1))

    # Carry relays are ordinary refractory PAULA neurons.  Level-0 update
    # gates emit twice for the two-tick sensory pulse, so c=3 explicitly
    # rejects the duplicate and supplies one carry event on a digit wrap.
    for level in range(LEVELS - 1):
        for carry, source, target_updates in (
            (CARRY_POS[level], UPDATE_POS[level][BASE - 1], UPDATE_POS[level + 1]),
            (CARRY_NEG[level], UPDATE_NEG[level][0], UPDATE_NEG[level + 1]),
        ):
            neurons.append(k.neuron(carry, r=parameters.carry_threshold,
                                    b=parameters.carry_threshold, c=3, lam=1))
            synapses.append(k.syn(carry, 0, parameters.carry_input_weight, 1))
            connections.append(k.conn(source, carry, 0))
            synapses.append(k.term(carry))
            for update in target_updates:
                # Do not append another synapse here: update cells already
                # own their designated sensory/carry port at index 1.
                connections.append(k.conn(carry, update, 1))

    if raw_yaw_transduction:
        # The two graded receptor neurons are the continuous physical-sensor
        # interface.  They receive the linear opponent currents `bias+raw`
        # and `bias-raw`, so the host never performs a sign branch or creates
        # an event.  Their downstream PAULA circuit cancels the common bias.
        for receptor in (RAW_YAW_RECEPTOR_POS, RAW_YAW_RECEPTOR_NEG):
            neurons.append(k.neuron(
                receptor, r=1e9, c=1, lam=1,
                meta={"graded_gain": 1.0, "graded_S0": 0.0},
            ))
            synapses.append(k.syn(receptor, 0, parameters.raw_receptor_gain, 1))
            external_inputs.append(k.ext(receptor, 0))
            synapses.append(k.term(receptor))

        # A single optional PAULA relay is a causal sensor-path timing
        # ablation. It has ordinary graded membrane/synaptic dynamics and is
        # intentionally *not* a host previous-sample or hold operation.
        delayed_source = {
            RAW_YAW_RECEPTOR_POS: RAW_YAW_RECEPTOR_POS,
            RAW_YAW_RECEPTOR_NEG: RAW_YAW_RECEPTOR_NEG,
        }
        if parameters.raw_sensor_delay_stages:
            for receptor, delay in (
                (RAW_YAW_RECEPTOR_POS, RAW_YAW_DELAY_POS),
                (RAW_YAW_RECEPTOR_NEG, RAW_YAW_DELAY_NEG),
            ):
                neurons.append(k.neuron(
                    delay, r=1e9, c=1, lam=1,
                    meta={"graded_gain": 1.0, "graded_S0": 0.0},
                ))
                synapses.append(k.syn(delay, 0, 1.0, 1))
                connections.append(k.conn(receptor, delay, 0))
                synapses.append(k.term(delay))
                delayed_source[receptor] = delay

        # Signed leaky integrators implement the temporal part of
        # continuous-yaw-to-event conversion.  They are ordinary graded PAULA
        # neurons, not an external rolling average or a counter in Python.
        for accumulator, positive, negative in (
            (RAW_YAW_ACC_CCW, delayed_source[RAW_YAW_RECEPTOR_POS], delayed_source[RAW_YAW_RECEPTOR_NEG]),
            (RAW_YAW_ACC_CW, delayed_source[RAW_YAW_RECEPTOR_NEG], delayed_source[RAW_YAW_RECEPTOR_POS]),
        ):
            neurons.append(k.neuron(
                accumulator, r=1e9, c=1, lam=parameters.raw_accumulator_lambda,
                meta={"graded_gain": parameters.raw_accumulator_graded_gain, "graded_S0": 0.0},
            ))
            synapses.append(k.syn(accumulator, 0, 1.0, 1))
            synapses.append(k.syn(accumulator, 1, -1.0, 1))
            connections.append(k.conn(positive, accumulator, 0))
            connections.append(k.conn(negative, accumulator, 1))
            synapses.append(k.term(accumulator))

        # An ordinary refractory PAULA cell turns a sustained analog signed
        # estimate into sparse events.  Its one-cell delayed partner makes a
        # two-tick PAULA pulse, matching the declared gate-alignment protocol
        # without host-side pulse stretching.
        for accumulator, first, second, targets in (
            (RAW_YAW_ACC_CCW, RAW_YAW_PULSE_CCW_A, RAW_YAW_PULSE_CCW_B, UPDATE_POS[0]),
            (RAW_YAW_ACC_CW, RAW_YAW_PULSE_CW_A, RAW_YAW_PULSE_CW_B, UPDATE_NEG[0]),
        ):
            neurons.append(k.neuron(
                first, r=parameters.raw_event_threshold, b=parameters.raw_event_threshold,
                c=parameters.raw_event_refractory_ticks, lam=1,
            ))
            synapses.append(k.syn(first, 0, 1.0, 1))
            connections.append(k.conn(accumulator, first, 0))
            synapses.append(k.term(first))
            neurons.append(k.neuron(second, r=parameters.raw_event_threshold,
                                    b=parameters.raw_event_threshold, c=1, lam=1))
            synapses.append(k.syn(second, 0, 1.0, 1))
            connections.append(k.conn(first, second, 0))
            synapses.append(k.term(second))
            for update in targets:
                connections.append(k.conn(first, update, 1))
                connections.append(k.conn(second, update, 1))

    return k.build(neurons, synapses, connections, external_inputs), seed_synapse


def _spike(neuron) -> int:
    return int(float(neuron.O) > 0.0)


def _digits_to_code(digits: Iterable[int]) -> int:
    return int(sum(digit * (BASE**level) for level, digit in enumerate(digits)))


def _signed_circular_delta(start: int, end: int) -> int:
    raw = (end - start) % TOTAL_BINS
    return raw - TOTAL_BINS if raw > TOTAL_BINS // 2 else raw


class CylinderCompass:
    """The PAULA circuit plus offline-only, full-tick instrumentation."""

    def __init__(self, parameters: BuildParameters, *, raw_yaw_transduction: bool = False) -> None:
        self.parameters = parameters
        self.raw_yaw_transduction = raw_yaw_transduction
        path, self.seed_synapse = _build_network(parameters, raw_yaw_transduction=raw_yaw_transduction)
        self.net, self.core = k.load(path)
        self.nb = self.net.network.neurons
        # Every cell has at most a few dendrites.  Setting the permissive
        # upper reference window keeps the counter timing comparable across
        # cells rather than allowing a newly-added carry input to alter it.
        for neuron in self.nb.values():
            neuron.upper_t_ref_bound = 2.0
        self.tick_index = 0
        self._digit_history: list[deque[list[int]]] = [
            deque(maxlen=DECODE_WINDOW) for _ in range(LEVELS)
        ]

    def tick(self, *, phase: str, ccw_current: float = 0.0, cw_current: float = 0.0,
             raw_yaw_rate: float | None = None, seed: bool = False,
             reference_code: int | None = None) -> dict:
        """Advance one PAULA tick; all writes are sanctioned sensory/birth ports."""

        for level in range(LEVELS):
            for digit, state in enumerate(STATES[level]):
                self.net.set_external_input(
                    state,
                    self.seed_synapse[state],
                    3.0 if seed and digit == 0 else 0.0,
                )
        if self.raw_yaw_transduction:
            if raw_yaw_rate is None:
                raw_yaw_rate = 0.0
            if abs(float(raw_yaw_rate)) > RAW_YAW_BIAS:
                raise ValueError(
                    f"raw yaw {raw_yaw_rate} exceeds opponent-receptor bias {RAW_YAW_BIAS}; "
                    "increase the declared bias rather than clipping a physical sample"
                )
            # This is the physical two-polarity receptor encoding, not a
            # discrete event conversion.  Everything after these continuous
            # currents is built from PAULA neurons above.
            self.net.set_external_input(RAW_YAW_RECEPTOR_POS, 0, RAW_YAW_BIAS + float(raw_yaw_rate))
            self.net.set_external_input(RAW_YAW_RECEPTOR_NEG, 0, RAW_YAW_BIAS - float(raw_yaw_rate))
        else:
            for update in UPDATE_POS[0]:
                self.net.set_external_input(update, 1, float(ccw_current))
            for update in UPDATE_NEG[0]:
                self.net.set_external_input(update, 1, float(cw_current))

        self.core.do_tick()
        state_spikes = [[_spike(self.nb[state]) for state in ring] for ring in STATES]
        for level, spikes in enumerate(state_spikes):
            self._digit_history[level].append(spikes)
        decoded_digits, decoded_code, ambiguous = self._decode()
        row = {
            "tick": self.tick_index,
            "phase": phase,
            "vestibular_ccw_current": float(ccw_current),
            "vestibular_cw_current": float(cw_current),
            "raw_yaw_rate": None if raw_yaw_rate is None else float(raw_yaw_rate),
            "reference_code": reference_code,
            "state_spikes": state_spikes,
            "update_spikes": {
                "positive": [[_spike(self.nb[nid]) for nid in ring] for ring in UPDATE_POS],
                "negative": [[_spike(self.nb[nid]) for nid in ring] for ring in UPDATE_NEG],
            },
            "carry_spikes": {
                "positive": [_spike(self.nb[nid]) for nid in CARRY_POS],
                "negative": [_spike(self.nb[nid]) for nid in CARRY_NEG],
            },
            "raw_yaw_transducer": (
                {
                    "state_buffer_release": [
                        [float(self.nb[nid].O) for nid in ring] for ring in RAW_STATE_BUFFERS
                    ],
                    "receptor_release": {
                        "positive": float(self.nb[RAW_YAW_RECEPTOR_POS].O),
                        "negative": float(self.nb[RAW_YAW_RECEPTOR_NEG].O),
                    },
                    "sensor_delay_release": (
                        None if not self.parameters.raw_sensor_delay_stages else {
                            "positive": float(self.nb[RAW_YAW_DELAY_POS].O),
                            "negative": float(self.nb[RAW_YAW_DELAY_NEG].O),
                        }
                    ),
                    "accumulator": {
                        "ccw_membrane": float(self.nb[RAW_YAW_ACC_CCW].S),
                        "cw_membrane": float(self.nb[RAW_YAW_ACC_CW].S),
                        "ccw_release": float(self.nb[RAW_YAW_ACC_CCW].O),
                        "cw_release": float(self.nb[RAW_YAW_ACC_CW].O),
                    },
                    "event_spikes": {
                        "ccw_a": _spike(self.nb[RAW_YAW_PULSE_CCW_A]),
                        "ccw_b": _spike(self.nb[RAW_YAW_PULSE_CCW_B]),
                        "cw_a": _spike(self.nb[RAW_YAW_PULSE_CW_A]),
                        "cw_b": _spike(self.nb[RAW_YAW_PULSE_CW_B]),
                    },
                }
                if self.raw_yaw_transduction else None
            ),
            # These fields are measured from preceding state spikes only.
            # They never enter the PAULA network.
            "decoded_digits_offline": decoded_digits,
            "decoded_code_offline": decoded_code,
            "decoded_heading_degrees_offline": (
                None if decoded_code is None else decoded_code * BIN_DEGREES
            ),
            "decode_ambiguous": ambiguous,
        }
        self.tick_index += 1
        return row

    def _decode(self) -> tuple[list[int] | None, int | None, bool]:
        if any(len(history) < DECODE_WINDOW for history in self._digit_history):
            return None, None, True
        digits: list[int] = []
        ambiguous = False
        for history in self._digit_history:
            counts = np.sum(np.asarray(history, dtype=int), axis=0)
            max_count = int(np.max(counts))
            if max_count == 0 or int(np.sum(counts == max_count)) != 1:
                ambiguous = True
                return None, None, ambiguous
            digits.append(int(np.argmax(counts)))
        return digits, _digits_to_code(digits), ambiguous


def _drive_phase(compass: CylinderCompass, trace: list[dict], *, name: str, pulses: int,
                 direction: int, reference_code: int) -> int:
    """Deliver a signed sequence of level-0 vestibular events.

    The reference count describes the laboratory stimulus only.  It remains
    outside the circuit and is used solely after the run to score its result.
    """

    for pulse in range(pulses):
        for within in range(PULSE_PERIOD):
            active = within < PULSE_WIDTH
            ccw = 1.0 if active and direction > 0 else 0.0
            cw = 1.0 if active and direction < 0 else 0.0
            trace.append(compass.tick(
                phase=name,
                ccw_current=ccw,
                cw_current=cw,
                reference_code=reference_code,
            ))
        reference_code = (reference_code + direction) % TOTAL_BINS
    return reference_code


def _hold_phase(compass: CylinderCompass, trace: list[dict], *, name: str, ticks: int,
                reference_code: int, seed: bool = False) -> None:
    for _ in range(ticks):
        trace.append(compass.tick(phase=name, seed=seed, reference_code=reference_code))


def _last_code(rows: list[dict]) -> int | None:
    for row in reversed(rows):
        code = row["decoded_code_offline"]
        if code is not None:
            return int(code)
    return None


def _max_hold_deviation(rows: list[dict]) -> int | None:
    codes = [int(row["decoded_code_offline"]) for row in rows if row["decoded_code_offline"] is not None]
    if not codes:
        return None
    reference = codes[0]
    return max(abs(_signed_circular_delta(reference, code)) for code in codes)


def _carry_count(rows: list[dict], direction: str, level: int) -> int:
    return int(sum(row["carry_spikes"][direction][level] for row in rows))


def run_experiment(*, seed: int = 11, pulses: int = 68, recovery_pulses: int = 20,
                   hold_ticks: int = 84) -> dict:
    """Run seed, hold, bidirectional update, wrap/carry, and reversal tests."""

    if pulses < BASE**3:
        raise ValueError("--pulses must be at least 64 so the top-level carry is exercised")
    if recovery_pulses <= 0:
        raise ValueError("--recovery-pulses must be positive")
    # PAULA's initial dendritic adaptation vectors are sampled at build time.
    # Plasticity is off here, but fixing the seed makes the complete artifact
    # bitwise-reproducible rather than merely behaviorally reproducible.
    np.random.seed(seed)
    parameters = BuildParameters()
    compass = CylinderCompass(parameters)
    trace: list[dict] = []
    reference = 0

    _hold_phase(compass, trace, name="seed", ticks=SEED_TICKS, reference_code=reference, seed=True)
    _hold_phase(compass, trace, name="hold_initial", ticks=hold_ticks, reference_code=reference)
    initial_code = _last_code([row for row in trace if row["phase"] == "hold_initial"])

    reference = _drive_phase(compass, trace, name="ccw_primary", pulses=pulses,
                             direction=+1, reference_code=reference)
    _hold_phase(compass, trace, name="settle_after_ccw", ticks=hold_ticks, reference_code=reference)
    ccw_code = _last_code([row for row in trace if row["phase"] == "settle_after_ccw"])

    reference = _drive_phase(compass, trace, name="cw_reversal", pulses=pulses,
                             direction=-1, reference_code=reference)
    _hold_phase(compass, trace, name="settle_after_cw", ticks=hold_ticks, reference_code=reference)
    cw_code = _last_code([row for row in trace if row["phase"] == "settle_after_cw"])

    reference = _drive_phase(compass, trace, name="ccw_recovery", pulses=recovery_pulses,
                             direction=+1, reference_code=reference)
    _hold_phase(compass, trace, name="settle_final", ticks=hold_ticks, reference_code=reference)
    final_code = _last_code([row for row in trace if row["phase"] == "settle_final"])

    sections = {name: [row for row in trace if row["phase"] == name] for name in (
        "hold_initial", "ccw_primary", "settle_after_ccw", "cw_reversal",
        "settle_after_cw", "ccw_recovery", "settle_final",
    )}
    delta_ccw = None if initial_code is None or ccw_code is None else _signed_circular_delta(initial_code, ccw_code)
    delta_cw = None if ccw_code is None or cw_code is None else _signed_circular_delta(ccw_code, cw_code)
    delta_recovery = None if cw_code is None or final_code is None else _signed_circular_delta(cw_code, final_code)
    carry_expected_primary = [pulses // (BASE ** (level + 1)) for level in range(LEVELS - 1)]
    carry_expected_recovery = [recovery_pulses // (BASE ** (level + 1)) for level in range(LEVELS - 1)]
    carry_primary = [_carry_count(sections["ccw_primary"], "positive", level) for level in range(LEVELS - 1)]
    carry_reverse = [_carry_count(sections["cw_reversal"], "negative", level) for level in range(LEVELS - 1)]
    carry_recovery = [_carry_count(sections["ccw_recovery"], "positive", level) for level in range(LEVELS - 1)]
    decode_coverage = float(np.mean([row["decoded_code_offline"] is not None for row in trace]))
    hold_drift = _max_hold_deviation(sections["hold_initial"])

    # Strict exact-bin accounting belongs in this *digital isolated* test. It
    # is not proposed as an organism-level perfection criterion.  Embodied
    # evaluation should instead use task-level, drift-and-recovery budgets.
    accepted = bool(
        initial_code is not None
        and delta_ccw == pulses
        and delta_cw == -pulses
        and delta_recovery == recovery_pulses
        and hold_drift == 0
        and carry_primary == carry_expected_primary
        and carry_reverse == carry_expected_primary
        and carry_recovery == carry_expected_recovery
    )
    return {
        "experiment": "paula_cylinder_compass_isolated",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "scope": {
            "isolated": True,
            "paula_only_circuit": True,
            "embodied": False,
            "biological_fidelity_claim": False,
            "note": (
                "Raw signed current enters only level-0 PAULA update gates. "
                "The base-4 decode and reference code are offline instrumentation."
            ),
        },
        "architecture": {
            "levels": LEVELS,
            "base_per_level": BASE,
            "total_bins": TOTAL_BINS,
            "bin_degrees": BIN_DEGREES,
            "neuron_count": len(compass.nb),
            "interpretation": "four PAULA quaternary state rings with directional wrap carries",
            "parameters": parameters.__dict__,
        },
        "input_protocol": {
            "pulse_width_ticks": PULSE_WIDTH,
            "pulse_period_ticks": PULSE_PERIOD,
            "primary_pulses": pulses,
            "recovery_pulses": recovery_pulses,
            "hold_ticks": hold_ticks,
            "reason_for_two_tick_pulse": (
                "align an external vestibular dendrite with the one-extra-stage local state signal; "
                "no sample-and-hold neuron or host heading update is used"
            ),
        },
        "acceptance_metrics": {
            "initial_code": initial_code,
            "after_ccw_code": ccw_code,
            "after_cw_code": cw_code,
            "final_code": final_code,
            "hold_initial_max_deviation_bins": hold_drift,
            "ccw_delta_bins": delta_ccw,
            "ccw_expected_bins": pulses,
            "cw_delta_bins": delta_cw,
            "cw_expected_bins": -pulses,
            "recovery_delta_bins": delta_recovery,
            "recovery_expected_bins": recovery_pulses,
            "carry_positive_primary": carry_primary,
            "carry_negative_reversal": carry_reverse,
            "carry_positive_recovery": carry_recovery,
            "carry_expected_primary_or_reversal": carry_expected_primary,
            "carry_expected_recovery": carry_expected_recovery,
            "full_tick_decode_coverage": decode_coverage,
            "accepted_isolated_counter": accepted,
            "criteria": {
                "hold": "zero decoded-bin drift during the no-input hold",
                "signed_updates": "CCW, CW, and post-reversal CCW have exact requested signed bin deltas",
                "reversals": "a negative update returns from the preceding positive trajectory without re-seeding",
                "carries": "each directional wrap propagates exactly once per base-4 boundary",
            },
        },
        "tick_trace": trace,
    }


def _replay_source_rows(source_path: Path) -> tuple[dict, list[dict], str]:
    """Load a declared full-tick source trace without recreating its body."""

    source = json.loads(source_path.read_text())
    rows = source.get("tick_trace")
    if rows is None:
        # Compass replay bundles retain the body trace under this explicit
        # key.  Supporting it lets the same experiment replay either the
        # sensorimotor probe or the standard long-turn bundle.
        rows = source.get("embodied", {}).get("tick_trace")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{source_path} does not contain a non-empty full-tick trace")
    raw_key = next((key for key in ("applied_raw_yaw_rate", "input_raw_yaw_rate", "raw_yaw_rate")
                    if key in rows[0]), None)
    if raw_key is None:
        raise ValueError(f"{source_path} has no recognized raw-yaw field")
    return source, rows, raw_key


def _body_turn_degrees(rows: list[dict]) -> float | None:
    selected = [row for row in rows if row.get("turn_stimulus_active")]
    if len(selected) < 2 or "physical_yaw_radians" not in selected[0]:
        return None
    yaw = np.unwrap(np.asarray([float(row["physical_yaw_radians"]) for row in selected]))
    return float(np.degrees(yaw[-1] - yaw[0]))


def _valid_delta(rows: list[dict]) -> int | None:
    codes = [int(row["decoded_code_offline"]) for row in rows if row["decoded_code_offline"] is not None]
    return None if len(codes) < 2 else _signed_circular_delta(codes[0], codes[-1])


def run_body_yaw_replay(*, source_path: Path, seed: int = 11,
                        settle_ticks: int = 84,
                        parameters: BuildParameters | None = None,
                        replay_variant: str = "baseline") -> dict:
    """Replay recorded body yaw through the PAULA continuous transducer.

    The only replay write is the continuous signed yaw sample at the two
    opponent receptor ports.  In particular this function does *not* count
    crossings, quantize an angle, apply a Python low-pass filter, or write a
    decoded heading back into any state cell.
    """

    if replay_variant not in {"baseline", "tuned_seed11"}:
        raise ValueError("replay_variant must be baseline or tuned_seed11")
    parameters = BuildParameters() if parameters is None else parameters
    source, source_rows, raw_key = _replay_source_rows(source_path)
    raw = np.asarray([float(row[raw_key]) for row in source_rows])
    if float(np.max(np.abs(raw))) > RAW_YAW_BIAS:
        raise ValueError(
            f"source maximum |raw yaw|={float(np.max(np.abs(raw))):.3f} exceeds "
            f"declared receptor bias {RAW_YAW_BIAS}"
        )
    np.random.seed(seed)
    compass = CylinderCompass(parameters, raw_yaw_transduction=True)
    trace: list[dict] = []
    _hold_phase(compass, trace, name="seed", ticks=SEED_TICKS, reference_code=None, seed=True)
    for source_index, source_row in enumerate(source_rows):
        row = compass.tick(
            phase="body_yaw_replay",
            raw_yaw_rate=float(source_row[raw_key]),
            reference_code=None,
        )
        row["source_index"] = source_index
        row["source_neural_tick"] = int(source_row.get("neural_tick", source_row.get("tick", source_index)))
        row["source_turn_stimulus_active"] = bool(source_row.get("turn_stimulus_active", False))
        row["source_physical_yaw_radians"] = (
            None if "physical_yaw_radians" not in source_row else float(source_row["physical_yaw_radians"])
        )
        trace.append(row)
    _hold_phase(compass, trace, name="settle_after_replay", ticks=settle_ticks, reference_code=None)

    replay_rows = [row for row in trace if row["phase"] == "body_yaw_replay"]
    turn_rows = [row for row in replay_rows if row["source_turn_stimulus_active"]]
    if not turn_rows:
        raise ValueError("source trace has no turn_stimulus_active labels; refusing an unscoped body comparison")
    body_turn = _body_turn_degrees(source_rows)
    cylinder_turn = _valid_delta(turn_rows)
    expected_bins = None if body_turn is None else int(np.rint(body_turn / BIN_DEGREES))
    ccw_events = int(sum(
        row["raw_yaw_transducer"]["event_spikes"]["ccw_a"] for row in replay_rows
    ))
    cw_events = int(sum(
        row["raw_yaw_transducer"]["event_spikes"]["cw_a"] for row in replay_rows
    ))
    ccw_updates = int(sum(sum(row["update_spikes"]["positive"][0]) for row in replay_rows))
    cw_updates = int(sum(sum(row["update_spikes"]["negative"][0]) for row in replay_rows))
    direction_matches = bool(
        body_turn is not None and cylinder_turn is not None and body_turn * cylinder_turn > 0
    )
    return {
        "experiment": "paula_cylinder_compass_body_yaw_replay",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "replay_variant": replay_variant,
        "scope": {
            "isolated_replay": True,
            "body_reexecuted": False,
            "paula_only_continuous_yaw_to_event": True,
            "host_discrete_eventization": False,
            "host_heading_update": False,
            "biological_fidelity_claim": False,
            "limitation": (
                "This is a fixed recorded yaw input, not an embodied closed-loop run. "
                "It tests whether the body-generated waveform is tolerated after it reaches PAULA."
            ),
        },
        "source": {
            "path": str(source_path),
            "sha256": sha256(source_path.read_bytes()).hexdigest(),
            "format": source.get("format"),
            "seed": source.get("seed"),
            "raw_yaw_field": raw_key,
            "source_ticks": len(source_rows),
            "raw_yaw_min": float(np.min(raw)),
            "raw_yaw_max": float(np.max(raw)),
            "raw_yaw_mean": float(np.mean(raw)),
            "turn_label": "source.tick_trace[].turn_stimulus_active",
        },
        "architecture": {
            "levels": LEVELS,
            "base_per_level": BASE,
            "total_bins": TOTAL_BINS,
            "bin_degrees": BIN_DEGREES,
            "neuron_count": len(compass.nb),
            "counter": "four PAULA quaternary state rings with directional carry cells",
            "transducer": (
                "biased opponent graded receptors -> signed graded accumulators -> "
                "ordinary refractory event doublets -> level-0 PAULA update gates"
            ),
            "receptor_bias": RAW_YAW_BIAS,
            "parameters": parameters.__dict__,
        },
        "replay_metrics": {
            "body_turn_degrees_from_source": body_turn,
            "expected_quantized_turn_bins_from_body": expected_bins,
            "cylinder_turn_bins": cylinder_turn,
            "cylinder_turn_degrees": None if cylinder_turn is None else cylinder_turn * BIN_DEGREES,
            "direction_matches_body": direction_matches,
            "event_spikes_full_replay": {"ccw": ccw_events, "cw": cw_events},
            "level0_update_spikes_full_replay": {"ccw": ccw_updates, "cw": cw_updates},
            "turn_decode_coverage": float(np.mean([
                row["decoded_code_offline"] is not None for row in turn_rows
            ])),
            "status": (
                "diagnostic only: a direction match is necessary but does not establish calibration, "
                "reversal robustness, or embodied navigation"
            ),
            "calibration_status": (
                "not trace-tuned" if replay_variant == "baseline" else
                "tuned on the named sensorimotor seed-11 reference trace only; this replay remains "
                "diagnostic and requires broader body-waveform validation"
            ),
        },
        "tick_trace": trace,
    }


FLAT_BINS = 18
FLAT_STATE = [916_000 + index for index in range(FLAT_BINS)]
FLAT_BUFFER = [916_100 + index for index in range(FLAT_BINS)]
FLAT_UPDATE_POS = [916_200 + index for index in range(FLAT_BINS)]
FLAT_UPDATE_NEG = [916_300 + index for index in range(FLAT_BINS)]
FLAT_REC_POS, FLAT_REC_NEG = 916_400, 916_401
FLAT_ACC_CCW, FLAT_ACC_CW = 916_410, 916_411
FLAT_PULSE_CCW_A, FLAT_PULSE_CCW_B = 916_420, 916_421
FLAT_PULSE_CW_A, FLAT_PULSE_CW_B = 916_422, 916_423


def _build_flat_raw_yaw_network(parameters: BuildParameters) -> tuple[str, dict[int, int]]:
    """A near-equal-neuron flat state ring for one narrow comparison.

    The cylinder's raw-yaw version has 78 cells: 16 state cells, 16 local
    timing buffers, 32 update gates, six carry relays, and eight vestibular
    transducer cells.  A flat ring cannot have *exactly* that count while
    keeping its state/buffer/two-update motif integral; 18 columns gives 80
    cells, only two more.  We therefore call this a near-equal-count control,
    not a claim of exact architectural parity.
    """

    neurons: list[dict] = []
    synapses: list[dict] = []
    connections: list[dict] = []
    external_inputs: list[dict] = []
    next_synapse = {state: 0 for state in FLAT_STATE}
    seed_synapse: dict[int, int] = {}

    def state_synapse(target: int, weight: float) -> int:
        sid = next_synapse[target]
        next_synapse[target] += 1
        synapses.append(k.syn(target, sid, weight, 1))
        return sid

    for state in FLAT_STATE:
        neurons.append(k.neuron(state, r=parameters.state_threshold, b=parameters.state_threshold, c=1, lam=2))
        connections.append(k.conn(state, state, state_synapse(state, parameters.state_self_weight)))
        seed_synapse[state] = state_synapse(state, parameters.state_seed_weight)
        external_inputs.append(k.ext(state, seed_synapse[state]))
        synapses.append(k.term(state))
    for state, buffer in zip(FLAT_STATE, FLAT_BUFFER, strict=True):
        neurons.append(k.neuron(buffer, r=1e9, c=1, lam=parameters.raw_state_buffer_lambda,
                                meta={"graded_gain": 1.0, "graded_S0": 0.0}))
        synapses.append(k.syn(buffer, 0, parameters.raw_state_buffer_weight, 1))
        connections.append(k.conn(state, buffer, 0))
        synapses.append(k.term(buffer))
    for index in range(FLAT_BINS):
        for update, direction in ((FLAT_UPDATE_POS[index], +1), (FLAT_UPDATE_NEG[index], -1)):
            neurons.append(k.neuron(update, r=parameters.update_threshold,
                                    b=parameters.update_threshold, c=1, lam=1))
            synapses.append(k.syn(update, 0, parameters.update_local_weight, 1))
            synapses.append(k.syn(update, 1, parameters.update_sensor_weight, 1))
            connections.append(k.conn(FLAT_BUFFER[index], update, 0))
            synapses.append(k.term(update))
            successor = FLAT_STATE[(index + direction) % FLAT_BINS]
            connections.append(k.conn(update, successor, state_synapse(successor, parameters.update_push_weight)))
            connections.append(k.conn(update, FLAT_STATE[index],
                                      state_synapse(FLAT_STATE[index], parameters.update_pull_weight)))

    for receptor in (FLAT_REC_POS, FLAT_REC_NEG):
        neurons.append(k.neuron(receptor, r=1e9, c=1, lam=1,
                                meta={"graded_gain": 1.0, "graded_S0": 0.0}))
        synapses.append(k.syn(receptor, 0, parameters.raw_receptor_gain, 1))
        external_inputs.append(k.ext(receptor, 0))
        synapses.append(k.term(receptor))
    for accumulator, positive, negative in (
        (FLAT_ACC_CCW, FLAT_REC_POS, FLAT_REC_NEG),
        (FLAT_ACC_CW, FLAT_REC_NEG, FLAT_REC_POS),
    ):
        neurons.append(k.neuron(accumulator, r=1e9, c=1, lam=parameters.raw_accumulator_lambda,
                                meta={"graded_gain": parameters.raw_accumulator_graded_gain, "graded_S0": 0.0}))
        synapses.append(k.syn(accumulator, 0, 1.0, 1))
        synapses.append(k.syn(accumulator, 1, -1.0, 1))
        connections.append(k.conn(positive, accumulator, 0))
        connections.append(k.conn(negative, accumulator, 1))
        synapses.append(k.term(accumulator))
    for accumulator, first, second, updates in (
        (FLAT_ACC_CCW, FLAT_PULSE_CCW_A, FLAT_PULSE_CCW_B, FLAT_UPDATE_POS),
        (FLAT_ACC_CW, FLAT_PULSE_CW_A, FLAT_PULSE_CW_B, FLAT_UPDATE_NEG),
    ):
        neurons.append(k.neuron(first, r=parameters.raw_event_threshold, b=parameters.raw_event_threshold,
                                c=parameters.raw_event_refractory_ticks, lam=1))
        synapses.append(k.syn(first, 0, 1.0, 1))
        connections.append(k.conn(accumulator, first, 0))
        synapses.append(k.term(first))
        neurons.append(k.neuron(second, r=parameters.raw_event_threshold, b=parameters.raw_event_threshold, c=1, lam=1))
        synapses.append(k.syn(second, 0, 1.0, 1))
        connections.append(k.conn(first, second, 0))
        synapses.append(k.term(second))
        for update in updates:
            connections.append(k.conn(first, update, 1))
            connections.append(k.conn(second, update, 1))
    return k.build(neurons, synapses, connections, external_inputs), seed_synapse


class FlatRawYawControl:
    """Near-equal-count flat PAULA ring; decoder remains offline-only."""

    def __init__(self, parameters: BuildParameters) -> None:
        path, self.seed_synapse = _build_flat_raw_yaw_network(parameters)
        self.net, self.core = k.load(path)
        self.nb = self.net.network.neurons
        for neuron in self.nb.values():
            neuron.upper_t_ref_bound = 2.0
        self.tick_index = 0
        self.history: deque[list[int]] = deque(maxlen=DECODE_WINDOW)

    def tick(self, *, phase: str, raw_yaw_rate: float = 0.0, seed: bool = False) -> dict:
        if abs(raw_yaw_rate) > RAW_YAW_BIAS:
            raise ValueError("flat replay raw yaw exceeds declared receptor bias")
        for index, state in enumerate(FLAT_STATE):
            self.net.set_external_input(state, self.seed_synapse[state], 3.0 if seed and index == 0 else 0.0)
        self.net.set_external_input(FLAT_REC_POS, 0, RAW_YAW_BIAS + raw_yaw_rate)
        self.net.set_external_input(FLAT_REC_NEG, 0, RAW_YAW_BIAS - raw_yaw_rate)
        self.core.do_tick()
        spikes = [_spike(self.nb[state]) for state in FLAT_STATE]
        self.history.append(spikes)
        code = None
        ambiguous = True
        if len(self.history) == DECODE_WINDOW:
            counts = np.sum(np.asarray(self.history, dtype=int), axis=0)
            highest = int(np.max(counts))
            if highest > 0 and int(np.sum(counts == highest)) == 1:
                code = int(np.argmax(counts))
                ambiguous = False
        row = {
            "tick": self.tick_index,
            "phase": phase,
            "raw_yaw_rate": float(raw_yaw_rate),
            "state_spikes": spikes,
            "state_buffer_release": [float(self.nb[nid].O) for nid in FLAT_BUFFER],
            "update_spikes": {
                "positive": [_spike(self.nb[nid]) for nid in FLAT_UPDATE_POS],
                "negative": [_spike(self.nb[nid]) for nid in FLAT_UPDATE_NEG],
            },
            "transducer": {
                "accumulator": {
                    "ccw_release": float(self.nb[FLAT_ACC_CCW].O),
                    "cw_release": float(self.nb[FLAT_ACC_CW].O),
                },
                "event_spikes": {
                    "ccw_a": _spike(self.nb[FLAT_PULSE_CCW_A]),
                    "cw_a": _spike(self.nb[FLAT_PULSE_CW_A]),
                },
            },
            "decoded_code_offline": code,
            "decoded_heading_degrees_offline": None if code is None else code * (360.0 / FLAT_BINS),
            "decode_ambiguous": ambiguous,
        }
        self.tick_index += 1
        return row


def _flat_signed_delta(start: int, end: int) -> int:
    raw = (end - start) % FLAT_BINS
    return raw - FLAT_BINS if raw > FLAT_BINS // 2 else raw


def run_near_equal_flat_body_yaw_replay(*, source_path: Path, seed: int,
                                         settle_ticks: int, parameters: BuildParameters) -> dict:
    """Run the 80-neuron flat control on exactly the same raw body trace."""

    source, source_rows, raw_key = _replay_source_rows(source_path)
    np.random.seed(seed)
    control = FlatRawYawControl(parameters)
    trace: list[dict] = []
    for _ in range(SEED_TICKS):
        trace.append(control.tick(phase="seed", seed=True))
    for source_index, source_row in enumerate(source_rows):
        row = control.tick(phase="body_yaw_replay", raw_yaw_rate=float(source_row[raw_key]))
        row["source_index"] = source_index
        row["source_neural_tick"] = int(source_row.get("neural_tick", source_index))
        row["source_turn_stimulus_active"] = bool(source_row.get("turn_stimulus_active", False))
        trace.append(row)
    for _ in range(settle_ticks):
        trace.append(control.tick(phase="settle_after_replay"))
    replay_rows = [row for row in trace if row["phase"] == "body_yaw_replay"]
    turn_rows = [row for row in replay_rows if row["source_turn_stimulus_active"]]
    codes = [int(row["decoded_code_offline"]) for row in turn_rows if row["decoded_code_offline"] is not None]
    delta = None if len(codes) < 2 else _flat_signed_delta(codes[0], codes[-1])
    body_turn = _body_turn_degrees(source_rows)
    return {
        "experiment": "near_equal_count_flat_paula_raw_yaw_control",
        "scope": {
            "isolated_replay": True,
            "same_source_raw_yaw": True,
            "same_paula_transducer_motif": True,
            "not_exact_neuron_parity": "80 flat neurons versus 78 cylinder neurons (+2)",
            "not_equal_precision": f"flat {360.0 / FLAT_BINS:.1f} degree bins versus cylinder {BIN_DEGREES:.4f}",
        },
        "source": {
            "path": str(source_path),
            "sha256": sha256(source_path.read_bytes()).hexdigest(),
            "format": source.get("format"),
            "raw_yaw_field": raw_key,
        },
        "architecture": {"flat_bins": FLAT_BINS, "neuron_count": len(control.nb), "parameters": parameters.__dict__},
        "metrics": {
            "body_turn_degrees_from_source": body_turn,
            "expected_quantized_turn_bins": None if body_turn is None else int(np.rint(body_turn / (360.0 / FLAT_BINS))),
            "flat_turn_bins": delta,
            "flat_turn_degrees": None if delta is None else delta * (360.0 / FLAT_BINS),
            "direction_matches_body": bool(body_turn is not None and delta is not None and body_turn * delta > 0),
            "turn_decode_coverage": float(np.mean([row["decoded_code_offline"] is not None for row in turn_rows])),
        },
        "tick_trace": trace,
    }


def run_raw_yaw_comparison(*, source_path: Path, seed: int, settle_ticks: int,
                           parameters: BuildParameters, replay_variant: str) -> dict:
    """Retain both full traces in one explicit near-equal-count comparison."""

    cylinder = run_body_yaw_replay(
        source_path=source_path, seed=seed, settle_ticks=settle_ticks,
        parameters=parameters, replay_variant=replay_variant,
    )
    flat = run_near_equal_flat_body_yaw_replay(
        source_path=source_path, seed=seed, settle_ticks=settle_ticks, parameters=parameters,
    )
    return {
        "experiment": "paula_cylinder_vs_near_equal_flat_raw_yaw_replay",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "A narrow isolated replay comparison only. Same signed raw-yaw source and same PAULA "
            "transducer motif; flat count is +2 neurons and resolution differs by construction."
        ),
        "cylinder": cylinder,
        "near_equal_flat_control": flat,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--pulses", type=int, default=68,
                        help="CCW and CW events; >=64 exercises the highest carry level")
    parser.add_argument("--recovery-pulses", type=int, default=20,
                        help="positive events after the CW reversal")
    parser.add_argument("--hold-ticks", type=int, default=84)
    parser.add_argument(
        "--body-yaw-replay", type=Path,
        help=(
            "run the PAULA continuous-yaw replay instead of the synthetic pulse test; "
            "the input must be a full-tick sensorimotor/compass JSON trace"
        ),
    )
    parser.add_argument("--replay-settle-ticks", type=int, default=84)
    parser.add_argument(
        "--raw-replay-variant", choices=("baseline", "tuned_seed11"), default="baseline",
        help=(
            "baseline is untuned; tuned_seed11 is explicitly calibrated only on the bundled seed-11 "
            "source and must be validated on a held-out waveform"
        ),
    )
    parser.add_argument(
        "--compare-near-equal-flat", action="store_true",
        help=(
            "also run the 80-neuron flat PAULA control on the exact same raw trace; "
            "valid only with --body-yaw-replay"
        ),
    )
    parser.add_argument("--output", type=Path,
                        help="write the full-tick JSON artifact; refuses to overwrite")
    args = parser.parse_args()
    if args.body_yaw_replay:
        replay_parameters = (
            RAW_YAW_TUNED_ON_SEED11 if args.raw_replay_variant == "tuned_seed11" else BuildParameters()
        )
        record = (
            run_raw_yaw_comparison(
                source_path=args.body_yaw_replay, seed=args.seed,
                settle_ticks=args.replay_settle_ticks, parameters=replay_parameters,
                replay_variant=args.raw_replay_variant,
            )
            if args.compare_near_equal_flat else
            run_body_yaw_replay(
                source_path=args.body_yaw_replay, seed=args.seed,
                settle_ticks=args.replay_settle_ticks, parameters=replay_parameters,
                replay_variant=args.raw_replay_variant,
            )
        )
        metrics = record["cylinder"]["replay_metrics"] if args.compare_near_equal_flat else record["replay_metrics"]
        print(
            "cylinder body-yaw replay "
            f"body={metrics['body_turn_degrees_from_source']:+.2f}deg "
            f"expected={metrics['expected_quantized_turn_bins_from_body']}bins "
            f"cylinder={metrics['cylinder_turn_bins']}bins "
            f"direction_match={metrics['direction_matches_body']} "
            f"events={metrics['event_spikes_full_replay']}",
            flush=True,
        )
        if args.compare_near_equal_flat:
            flat = record["near_equal_flat_control"]["metrics"]
            print(
                "near-equal flat control "
                f"expected={flat['expected_quantized_turn_bins']}bins "
                f"flat={flat['flat_turn_bins']}bins direction_match={flat['direction_matches_body']}",
                flush=True,
            )
    elif args.compare_near_equal_flat:
        raise SystemExit("--compare-near-equal-flat requires --body-yaw-replay")
    else:
        record = run_experiment(seed=args.seed, pulses=args.pulses, recovery_pulses=args.recovery_pulses,
                                hold_ticks=args.hold_ticks)
        metrics = record["acceptance_metrics"]
        print(
            "cylinder isolated "
            f"hold_drift={metrics['hold_initial_max_deviation_bins']} "
            f"ccw={metrics['ccw_delta_bins']} cw={metrics['cw_delta_bins']} "
            f"recovery={metrics['recovery_delta_bins']} carries+={metrics['carry_positive_primary']} "
            f"carries-={metrics['carry_negative_reversal']} "
            f"accepted={metrics['accepted_isolated_counter']}",
            flush=True,
        )
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
