"""Regression checks for the shared embodied-agent configuration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulations/active_inference"))
from embodied_config import EmbodiedAgentConfig


def load_agent_module():
    spec = importlib.util.spec_from_file_location(
        "aif_agent3d_config_test", ROOT / "simulations/active_inference/aif_agent3d.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_configuration_is_explicit_and_complete():
    ag = load_agent_module()
    manifest = ag.DEFAULT_EMBODIED_CONFIG.manifest()

    assert manifest["tonic"] == {
        "amplitude": 1.0,
        "velocity_gate": 0.0,
        "synapse_weight": 0.12,
    }
    assert manifest["compass"]["yaw_filter_tau"] == 60.0
    assert manifest["compass"]["yaw_filter_compensation"] == 0.0
    assert manifest["path_integration"]["pg"] == {
        "threshold": 1.7,
        "ring_weight": 1.4,
        "speed_weight": 1.4,
    }
    assert manifest["structural_gates"]["delta7"] is True
    assert manifest["structural_gates"]["accum"] is False
    assert manifest["structural_gates"]["mb_lateral_horn"] is True
    assert manifest["structural_gates"]["forage_search_veto"] == -2.0


def test_direct_tick_and_closed_episode_use_the_same_tonic_setting():
    ag = load_agent_module()
    agent = ag.AIFAgent3D(seed=11, config=ag.DEFAULT_EMBODIED_CONFIG)
    agent.birth()
    calls = []
    drive = agent.drive_ring_tonic

    def observe(ccw, cw):
        tonic = drive(ccw, cw)
        calls.append((ccw, cw, tonic))
        return tonic

    agent.drive_ring_tonic = observe
    agent.tick(ccw=0.0, cw=0.0, speed=0.0, vision=False)
    ag.run_episode(
        agent,
        steps=1,
        sub=1,
        render_every=10**9,
        render_ticks=0,
        log_every=10**9,
    )

    assert len(calls) == 2
    assert calls[0][2] == calls[1][2] == 1.0
    assert agent.effective_config_manifest()["tonic"]["amplitude"] == 1.0


def test_legacy_direct_tick_control_remains_available_only_by_request():
    ag = load_agent_module()
    agent = ag.AIFAgent3D(seed=11, config=ag.LEGACY_DIRECT_TICK_CONFIG)
    assert agent.ring_tonic_current(0.0, 0.0) == 0.0
    assert ag.DEFAULT_EMBODIED_CONFIG.tonic_amp == 1.0


def test_package_imported_config_is_accepted_by_path_loaded_agent():
    ag = load_agent_module()
    agent = ag.AIFAgent3D(seed=11, config=EmbodiedAgentConfig(tonic_amp=0.25))
    assert agent.ring_tonic_current(0.0, 0.0) == 0.25


def test_closed_episode_uses_the_configured_angular_proprioception_filter():
    ag = load_agent_module()
    config = ag.DEFAULT_EMBODIED_CONFIG.with_overrides(wz_tau=27.0, wz_comp=3.5)
    agent = ag.AIFAgent3D(seed=11, config=config)

    ag.run_episode(
        agent,
        steps=1,
        sub=1,
        render_every=10**9,
        render_ticks=0,
        log_every=10**9,
    )

    assert agent.world.wz_tau == 27.0
    assert agent.world.wz_comp == 3.5
    assert agent.effective_config_manifest()["compass"]["yaw_filter_tau"] == 27.0


def test_opt_in_opponent_vestibular_population_is_built_and_has_signed_direct_input():
    ag = load_agent_module()
    agent = ag.AIFAgent3D(seed=11, vestibular_opponent=True, vop_tau=20.0)
    agent.birth()

    cl_spikes = 0
    for _ in range(60):
        agent.tick(ccw=0.5, cw=0.0, speed=0.0, vision=False)
        cl_spikes += sum(agent.nb[nid].O > 0 for column in ag.cc.CL for nid in column)

    assert agent._vestibular_opponent is True
    assert agent.nb[ag.VEST_CCW].O > 0.0
    assert agent.nb[ag.VEST_CW].O == 0.0
    assert cl_spikes > 0
    assert ag.DEFAULT_EMBODIED_CONFIG.vestibular_opponent is False


def test_avoid_to_steer_gain_is_forwarded_to_the_composed_network():
    ag = load_agent_module()
    default = ag.AIFAgent3D(seed=11)
    calibrated = ag.AIFAgent3D(seed=11, w_avoid_steer=9.0)

    default_weights = [point.u_i.info for point in default.nb[ag.STEER].postsynaptic_points.values()]
    calibrated_weights = [point.u_i.info for point in calibrated.nb[ag.STEER].postsynaptic_points.values()]

    assert 3.0 in default_weights
    assert 9.0 in calibrated_weights


def test_default_forage_veto_targets_search_not_the_hazard_escape_gate():
    ag = load_agent_module()
    agent = ag.AIFAgent3D(seed=11)

    search_weights = [point.u_i.info for point in agent.nb[ag.SEARCH].postsynaptic_points.values()]
    hazard_weights = [point.u_i.info for point in agent.nb[ag.STEER].postsynaptic_points.values()]

    # SEARCH receives its six HOME vetoes plus the six FORAGE vetoes.  The
    # hazard gate receives HOME vetoes only, preserving escape while hungry.
    assert search_weights.count(-2.0) == len(ag.ar.MODE[1]) + len(ag.ar.MODE[0])
    assert hazard_weights.count(-2.0) == len(ag.ar.MODE[1])


def test_paula_vestibular_comparator_rectifies_after_signed_low_pass():
    ag = load_agent_module()

    def drive(ccw: float, cw: float) -> tuple[int, int]:
        agent = ag.AIFAgent3D(seed=11, vestibular_comparator=True)
        agent.birth()
        events = [0, 0]
        for _ in range(40):
            agent.tick(ccw=ccw, cw=cw, speed=0.0, hunger_fill=0.0, eat=0.0, vision=False)
            events[0] += int(agent.nb[ag.VEST_CCW].O > 0)
            events[1] += int(agent.nb[ag.VEST_CW].O > 0)
        return tuple(events)

    ccw, cw = drive(4.0, 0.0)
    assert ccw > 0 and cw == 0
    ccw, cw = drive(0.0, 4.0)
    assert cw > 0 and ccw == 0


def test_paula_vestibular_notch_rejects_zero_and_encodes_signed_direct_current():
    """The experimental full-stroke notch is silent at rest and directional."""
    ag = load_agent_module()

    def drive(ccw: float, cw: float) -> tuple[int, int]:
        agent = ag.AIFAgent3D(seed=11, vestibular_notch=True)
        agent.birth()
        events = [0, 0]
        for _ in range(80):
            agent.tick(ccw=ccw, cw=cw, speed=0.0, hunger_fill=0.0, eat=0.0, vision=False)
            events[0] += int(agent.nb[ag.VEST_CCW].O > 0)
            events[1] += int(agent.nb[ag.VEST_CW].O > 0)
        return tuple(events)

    assert drive(0.0, 0.0) == (0, 0)
    ccw, cw = drive(4.0, 0.0)
    assert ccw > 0 and cw == 0
    ccw, cw = drive(0.0, 4.0)
    assert cw > 0 and ccw == 0


def test_paula_graded_vestibular_notch_is_opt_in_and_directional():
    ag = load_agent_module()

    agent = ag.AIFAgent3D(seed=11, vestibular_notch_graded=True)
    agent.birth()
    for _ in range(80):
        agent.tick(ccw=4.0, cw=0.0, speed=0.0, hunger_fill=0.0, eat=0.0, vision=False)

    assert agent._vestibular_notch_graded is True
    assert agent.nb[ag.VEST_NET_CCW].O > 0.0
    assert agent.nb[ag.VEST_NET_CW].O == 0.0
    assert ag.DEFAULT_EMBODIED_CONFIG.vestibular_opponent is False


def test_paula_graded_opponent_notch_subtracts_the_two_yaw_afferents():
    ag = load_agent_module()

    def drive(ccw: float, cw: float) -> tuple[float, float]:
        agent = ag.AIFAgent3D(seed=11, vestibular_notch_graded_opponent=True)
        agent.birth()
        for _ in range(100):
            agent.tick(ccw=ccw, cw=cw, speed=0.0, hunger_fill=0.0, eat=0.0, vision=False)
        return agent.nb[ag.VEST_OPP_CCW].O, agent.nb[ag.VEST_OPP_CW].O

    ccw, cw = drive(4.0, 0.0)
    assert ccw > 0.0 and cw == 0.0
    ccw, cw = drive(0.0, 4.0)
    assert cw > 0.0 and ccw == 0.0


def test_conjunctive_graded_pen_gates_on_the_paula_opponent_port():
    """Continuous P-EN release must remain bump × velocity, not global velocity."""
    ag = load_agent_module()
    agent = ag.AIFAgent3D(
        seed=11,
        vestibular_notch_graded_opponent=True,
        vop_notch_graded_opp_normalized=True,
        graded_shift=True,
        conjunctive_graded_shift=True,
    )

    pen = agent.nb[ag.cc.CL[0][0]]
    opponent = agent.nb[ag.VEST_OPP_CCW]
    assert type(pen).__name__ == "ConjunctiveGradedNeuron"
    assert type(opponent).__name__ == "ConjunctiveGradedNeuron"
    assert pen.metadata["conjunctive_ring_synapse"] == 0
    # Synapse 1 is the direct controlled-drive port.  The complete gyro route
    # appends its opponent signal at synapse 2 and the inherited class must
    # gate on that PAULA output in embodiment.
    assert pen.metadata["conjunctive_velocity_synapse"] == 2
    assert opponent.metadata["opponent_normalized_gain"] == 1.0


def test_conjunctive_graded_pen_preserves_a_signed_paula_opponent_pair():
    """The low-lag opponent route must not rectify before local subtraction."""
    ag = load_agent_module()
    agent = ag.AIFAgent3D(
        seed=11,
        vestibular_opponent=True,
        graded_shift=True,
        conjunctive_graded_shift=True,
    )

    pen = agent.nb[ag.cc.CL[0][0]]
    assert pen.metadata["conjunctive_velocity_synapse"] == 2
    assert pen.metadata["conjunctive_velocity_negative_synapse"] == 3


def test_conjunctive_graded_pen_accepts_a_population_of_signed_paula_dendrites():
    ag = load_agent_module()
    agent = ag.AIFAgent3D(
        seed=11,
        vestibular_opponent_bank=True,
        vop_bank_taus=(8.0, 24.0, 72.0),
        graded_shift=True,
        conjunctive_graded_shift=True,
    )

    pen = agent.nb[ag.cc.CL[0][0]]
    assert pen.metadata["conjunctive_velocity_synapses"] == [2, 3, 4, 5, 6, 7]
    assert all(nid in agent.nb for nid in ag.VEST_BANK_CCW[:3] + ag.VEST_BANK_CW[:3])


def test_conjunctive_graded_pen_accepts_the_pooled_relay_efference_population():
    ag = load_agent_module()
    agent = ag.AIFAgent3D(
        seed=11,
        relay_efference_population=True,
        relay_efference_only=True,
        graded_shift=True,
        conjunctive_graded_shift=True,
    )

    pen = agent.nb[ag.cc.CL[0][0]]
    assert pen.metadata["conjunctive_velocity_synapses"] == [2, 3, 4, 5, 6, 7, 8, 9]
    assert all(nid in agent.nb for nid in ag.EFF_BANK_CCW + ag.EFF_BANK_CW)


def test_sensorimotor_v2_uses_unsigned_opponent_ports_and_leaves_default_unchanged():
    """Negative PAULA ``info`` is not a sensory value; inhibition is synaptic."""
    ag = load_agent_module()
    baseline = ag.AIFAgent3D(seed=11)
    assert all(nid not in baseline.nb for nid in ag.SM2_SENS_CCW + ag.SM2_SENS_CW)

    agent = ag.AIFAgent3D(
        seed=11,
        sensorimotor_estimator_v2=True,
        graded_shift=True,
        conjunctive_graded_shift=True,
    )
    agent.birth()
    for _ in range(60):
        agent.tick(ccw=2.0, cw=0.0, speed=0.0, vision=False)

    assert agent._sensorimotor_estimator_v2 is True
    assert agent._sensorimotor_estimator is False
    assert agent.nb[ag.SM2_SENS_CCW[0]].O > 0.0
    assert agent.nb[ag.SM2_SENS_CW[0]].O == 0.0
    # P-EN coincidence is driven by the new fused PAULA update, not the old
    # direct-control synapse at index 1.
    assert agent.nb[ag.cc.CL[0][0]].metadata["conjunctive_velocity_synapse"] == 2


def test_sensorimotor_v2_rejects_combining_its_topology_with_v1():
    ag = load_agent_module()
    with pytest.raises(ValueError, match="choose at most one vestibular representation"):
        ag.AIFAgent3D(seed=11, sensorimotor_estimator=True, sensorimotor_estimator_v2=True)


def test_sensorimotor_phase_lead_is_an_opt_in_paula_population_not_a_phase_locked_cell():
    ag = load_agent_module()
    baseline = ag.AIFAgent3D(seed=11)
    assert all(nid not in baseline.nb for nid in ag.SM3_LEAD_CCW + ag.SM3_LEAD_CW)

    agent = ag.AIFAgent3D(
        seed=11,
        sensorimotor_estimator_v2=True,
        sensorimotor_v2_phase_gated_lead=True,
        sensorimotor_v2_phase_copies=4,
        sensorimotor_v2_phase_gate_tau=4.0,
        graded_shift=True,
        conjunctive_graded_shift=True,
    )
    selected = [
        nid
        for phase in range(4)
        for nid in ag.SM3_LEAD_CCW[
            phase * ag.SM3_PHASE_COPIES:phase * ag.SM3_PHASE_COPIES + 4
        ]
    ]
    pen = agent.nb[ag.cc.CL[0][0]]
    assert all(nid in agent.nb for nid in selected)
    assert type(agent.nb[selected[0]]).__name__ == "ConjunctiveGradedNeuron"
    assert not any(key.startswith("phase_locked_") for key in agent.nb[selected[0]].metadata)
    assert len(pen.metadata["conjunctive_velocity_synapses"]) == 16


def test_experimental_phase_locked_paula_gyro_holds_a_signed_stroke_estimate():
    ag = load_agent_module()
    agent = ag.AIFAgent3D(
        seed=11,
        vestibular_phase_locked=True,
        vop_phase_period=8,
        vop_phase_output_gain=0.1,
        graded_shift=True,
        conjunctive_graded_shift=True,
    )
    agent.birth()
    for _ in range(16):
        agent.tick(ccw=1.0, cw=0.0, speed=0.0, vision=False)

    assert type(agent.nb[ag.STROKE_CCW]).__name__ == "PhaseLockedGradedNeuron"
    assert agent.nb[ag.STROKE_CCW].O > 0.0
    assert agent.nb[ag.STROKE_CW].O == 0.0


def test_experimental_phase_locked_paula_gyro_can_emit_a_bounded_local_pulse():
    ag = load_agent_module()
    agent = ag.AIFAgent3D(
        seed=11,
        vestibular_phase_locked=True,
        vop_phase_period=8,
        vop_phase_output_gain=0.1,
        vop_phase_hold_ticks=2,
        graded_shift=True,
        conjunctive_graded_shift=True,
    )
    agent.birth()
    output = []
    for _ in range(16):
        agent.tick(ccw=1.0, cw=0.0, speed=0.0, vision=False)
        output.append(agent.nb[ag.STROKE_CCW].O)

    assert sum(value > 0.0 for value in output) <= 4


def test_ordinary_stroke_reset_uses_clock_synapse_not_phase_sampler_state():
    ag = load_agent_module()
    agent = ag.AIFAgent3D(
        seed=11,
        vestibular_stroke_reset=True,
        vop_reset_period=8,
        vop_reset_gain=4.0,
        vop_reset_lam=20.0,
        vop_reset_output_S0=-0.55,
        graded_shift=True,
        conjunctive_graded_shift=True,
    )
    agent.birth()
    states = []
    clocks = []
    for _ in range(24):
        agent.tick(ccw=1.0, cw=0.0, speed=0.0, vision=False)
        states.append(float(agent.nb[ag.STROKE_RESET_CCW].S))
        clocks.append(int(agent.nb[ag.STROKE_RESET_CLOCK].O > 0))

    integrator = agent.nb[ag.STROKE_RESET_CCW]
    assert type(integrator).__name__ == "ConjunctiveGradedNeuron"
    assert not any(key.startswith("phase_locked_") for key in integrator.metadata)
    assert sum(clocks) >= 2
    # The explicit negative clock synapse produces real downward membrane
    # steps during positive sensory drive; no subclass snapshot/hold exists.
    assert min(next_state - state for state, next_state in zip(states, states[1:])) < -0.05


def test_explicit_pb_eb_bridge_is_opt_in_and_builds_named_paula_relays():
    """The bridge must be a structural population, never a disguised direct edge."""
    ag = load_agent_module()
    baseline = ag.AIFAgent3D(seed=11)
    bridge = ag.AIFAgent3D(seed=11, pb_eb_bridge=True)

    pb_ids = ag.cc.PB_CL + ag.cc.PB_CR + ag.cc.PB_ML + ag.cc.PB_MR
    assert all(nid not in baseline.nb for nid in pb_ids)
    # Update relays exist whenever the bridge is selected; maintenance tracts
    # are deliberately absent until the distinct P-EG maintenance gate is on.
    assert all(nid in bridge.nb for nid in ag.cc.PB_CL + ag.cc.PB_CR)
    assert all(nid not in bridge.nb for nid in ag.cc.PB_ML + ag.cc.PB_MR)
    assert bridge.nb[ag.cc.PB_CL[0]].metadata["graded_gain"] == 1.0

    maintained = ag.AIFAgent3D(seed=11, pb_eb_bridge=True, w_peg=1.6)
    assert all(nid in maintained.nb for nid in pb_ids)


def test_trailing_penb_is_an_opt_in_offset_route_not_same_column_peg_feedback():
    ag = load_agent_module()
    baseline = ag.AIFAgent3D(seed=11)
    penb_ids = ag.cc.PENB_CL + ag.cc.PENB_CR
    assert all(nid not in baseline.nb for nid in penb_ids)

    candidate = ag.AIFAgent3D(
        seed=11,
        sensorimotor_estimator_v2=True,
        pb_eb_bridge=True,
        w_peg=1.6,
        w_peg_ring=0.0,
        p_enb_trailing=True,
    )
    assert candidate._p_enb_trailing is True
    assert all(nid in candidate.nb for nid in penb_ids)
    # Each P-ENb cell has its own local P-EG dendrite, an unused external
    # port, and a distinct V2 fused-update synapse.  The topology must not
    # write a decoded heading or reuse the old P-EG->same-column route.
    cell = candidate.nb[ag.cc.PENB_CL[0]]
    assert cell.metadata["p_enb_trailing"] is True
    incoming = [row for row in candidate.net.network.connections if row[2] == ag.cc.PENB_CL[0]]
    assert {row[0] for row in incoming} == {ag.cc.PEG[0], ag.SM2_UPDATE_CCW}
    # The CCW P-ENb column returns one spatial step *behind* its local
    # heading column; a same-column P-EG feedback edge would target ring 0.
    outgoing = [row for row in candidate.net.network.connections if row[0] == ag.cc.PENB_CL[0]]
    assert {row[2] for row in outgoing} == {ag.cc.RING[-1]}

    with pytest.raises(ValueError, match="requires the explicit PB/P-EG bridge"):
        ag.AIFAgent3D(seed=11, p_enb_trailing=True)


def test_pena_microbank_is_opt_in_heterogeneous_paula_scaling_not_a_hidden_gain():
    ag = load_agent_module()
    baseline = ag.AIFAgent3D(seed=11)
    pena_ids = [nid for columns in (ag.cc.PENA_CL, ag.cc.PENA_CR) for column in columns for nid in column]
    assert all(nid not in baseline.nb for nid in pena_ids)

    candidate = ag.AIFAgent3D(seed=11, sensorimotor_estimator_v2=True, p_ena_bank=True)
    assert candidate._p_ena_bank is True
    assert all(nid in candidate.nb for nid in pena_ids)
    first = ag.cc.PENA_CL[0][0]
    slow = ag.cc.PENA_CL[0][3]
    assert candidate.nb[first].metadata["conjunctive_velocity_tau"] == 1.0
    assert candidate.nb[slow].metadata["conjunctive_velocity_tau"] == 8.0
    incoming = [row for row in candidate.net.network.connections if row[2] == first]
    assert {row[0] for row in incoming} == {ag.cc.RING[0], ag.SM2_UPDATE_CCW}
    outgoing = [row for row in candidate.net.network.connections if row[0] == first]
    assert {row[2] for row in outgoing} == {ag.cc.RING[1]}


def test_pb_clock_reset_requires_the_explicit_bridge_population():
    ag = load_agent_module()
    with pytest.raises(ValueError, match="pb_stroke_reset requires pb_eb_bridge"):
        ag.AIFAgent3D(seed=11, vestibular_stroke_reset=True, pb_stroke_reset=True, pb_reset_gain=8.0)


def test_pb_bridge_rejects_a_delay_that_cannot_compensate_its_relay_tick():
    ag = load_agent_module()
    with pytest.raises(ValueError, match="d_pb_shift"):
        ag.AIFAgent3D(seed=11, pb_eb_bridge=True, d_push=2, d_pb_shift=1)


def test_pb_phase_update_is_explicit_paula_cpg_circuit_and_requires_its_clock():
    ag = load_agent_module()
    baseline = ag.AIFAgent3D(seed=11)
    assert all(nid not in baseline.nb for nid in ag.cc.PB_PHASE_CLOCK)

    phase = ag.AIFAgent3D(
        seed=11,
        pb_eb_bridge=True,
        vestibular_stroke_reset=True,
        pb_phase_update=True,
        d_push=4,
    )
    assert all(nid in phase.nb for nid in ag.cc.PB_PHASE_CLOCK)
    assert all(nid in phase.nb for nid in ag.cc.PB_PHASE_CL + ag.cc.PB_PHASE_CR)
    assert phase._pb_phase_update is True

    with pytest.raises(ValueError, match="pb_phase_update requires vestibular_stroke_reset"):
        ag.AIFAgent3D(seed=11, pb_eb_bridge=True, pb_phase_update=True)


def test_conjunctive_graded_release_requires_both_local_dendrites():
    ag = load_agent_module()
    path = ag.k.build(
        [ag.k.neuron(
            1,
            r=1e9,
            meta={
                "graded_gain": 1.0,
                "conjunctive_graded_gain": 2.0,
                "conjunctive_ring_synapse": 0,
                "conjunctive_velocity_synapse": 1,
            },
        )],
        [ag.k.syn(1, 0, 1.0), ag.k.syn(1, 1, 1.0), ag.k.term(1)],
        [],
        [ag.k.ext(1, 0), ag.k.ext(1, 1)],
    )
    net, core = ag.k.load(path, neuron_class=ag.k.ConjunctiveGradedNeuron)
    cell = net.network.neurons[1]

    def drive(bump: float, velocity: float) -> float:
        net.set_external_input(1, 0, bump)
        net.set_external_input(1, 1, velocity)
        core.do_tick()
        return float(cell.O)

    assert drive(2.0, 0.0) == 0.0
    assert drive(0.0, 2.0) == 0.0
    assert drive(2.0, 2.0) == 4.0


def test_conjunctive_graded_local_velocity_lead_uses_only_prior_dendritic_drive():
    """The optional lead is a local PAULA state, not an external estimator."""
    ag = load_agent_module()
    path = ag.k.build(
        [ag.k.neuron(
            1,
            r=1e9,
            meta={
                "graded_gain": 1.0,
                "conjunctive_graded_gain": 1.0,
                "conjunctive_ring_synapse": 0,
                "conjunctive_velocity_synapse": 1,
                "conjunctive_velocity_lead_gain": 2.0,
            },
        )],
        [ag.k.syn(1, 0, 1.0), ag.k.syn(1, 1, 1.0), ag.k.term(1)],
        [],
        [ag.k.ext(1, 0), ag.k.ext(1, 1)],
    )
    net, core = ag.k.load(path, neuron_class=ag.k.ConjunctiveGradedNeuron)
    cell = net.network.neurons[1]
    net.set_external_input(1, 0, 10.0)
    net.set_external_input(1, 1, 1.0)
    core.do_tick()
    # 1 + 2*(1 - 0), bounded by the bump dendrite.
    assert cell.O == 3.0
    net.set_external_input(1, 0, 10.0)
    net.set_external_input(1, 1, 1.0)
    core.do_tick()
    assert cell.O == 1.0


def test_opt_in_mb_lateral_horn_route_reverses_a_lateral_food_turn_only_with_aversion():
    ag = load_agent_module()

    def drive(mb_lh: bool):
        agent = ag.AIFAgent3D(seed=11, mb_lh=mb_lh, r_lh=0.8)
        agent.birth()
        counts = {"avoid": 0, "lh_right": 0, "turn_left": 0, "turn_right": 0}
        for _ in range(60):
            agent.net.set_external_input(ag.RFX, 0, 10.0)
            for nid in ag.FRp:
                agent.net.set_external_input(nid, 0, 3.0)
            agent.core.do_tick()
            counts["avoid"] += int(agent.nb[ag.AVOID].O > 0)
            counts["turn_left"] += int(agent.nb[ag.TL].O > 0)
            counts["turn_right"] += int(agent.nb[ag.TR].O > 0)
            if mb_lh:
                counts["lh_right"] += sum(agent.nb[nid].O > 0 for nid in ag.LH_RIGHT)
        return counts

    baseline = drive(False)
    learned = drive(True)

    assert learned["avoid"] == baseline["avoid"] > 0
    assert learned["lh_right"] > 0
    assert baseline["turn_left"] == 0
    assert learned["turn_left"] > baseline["turn_left"]
    assert learned["turn_right"] > 0
