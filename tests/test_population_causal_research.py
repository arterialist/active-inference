import numpy as np
import pytest

from simulations.active_inference.experiments.composition_probe import Neuron, encode
from simulations.active_inference.experiments.population_hierarchy import make_config, cellular
from simulations.active_inference.experiments.population_causal_probe import cut_incoming, CurrentLedger
from simulations.active_inference.experiments.multimodal_pairing_probe import (
    pairing_protocol, inputs, content_projection, fresh,
)


def test_surgical_cut_changes_only_named_connections():
    config, groups, edges = make_config(288)
    targets = set(groups["tactile_core"])
    cut, removed = cut_incoming(config, edges, targets, {"descending"})
    assert removed
    assert cut["neurons"] == config["neurons"]
    assert cut["synaptic_points"] == config["synaptic_points"]
    assert len(config["connections"])-len(cut["connections"]) == len(removed)
    assert all(tgt in targets for _, tgt, _ in removed)
    families = {(src, tgt, sid): family for src, tgt, sid, family, _ in edges}
    assert all(families[tuple(edge)] == "descending" for edge in removed)


def test_observer_is_bit_identical_and_restores_method(tmp_path):
    config, groups, edges = make_config(288)
    path = tmp_path/"config.json"
    path.write_text(encode(config))
    reference = []
    net, core, neurons, _ = fresh(path, 11)
    for t in range(40):
        for nid in groups["vision"]:
            if t % 4 == nid % 4:
                net.set_external_input(nid, 0, 2.)
        assert "error" not in core.do_tick()
        reference.append(cellular(neurons))
    net, core, neurons, _ = fresh(path, 11)
    ledger = CurrentLedger(groups["tactile_core"], edges, 40)
    original = Neuron.tick
    observed = []
    with ledger.observe():
        for t in range(40):
            for nid in groups["vision"]:
                if t % 4 == nid % 4:
                    net.set_external_input(nid, 0, 2.)
            assert "error" not in core.do_tick()
            observed.append(cellular(neurons))
    assert Neuron.tick is original
    assert np.array_equal(reference, observed)
    assert ledger.max_error < 1e-5
    assert ledger.scalar[:, :, 2].sum() > 0
    with pytest.raises(RuntimeError):
        with ledger.observe():
            raise RuntimeError("test observer cleanup")
    assert Neuron.tick is original


def test_content_swap_has_identical_per_receptor_marginals():
    _, groups, _ = make_config(1152)
    rng = np.random.default_rng(100)
    features = [{"ticks": 40, "visual": rng.random((40, 96)),
                 "auditory": rng.random((40, 96))} for _ in range(2)]
    marginal = []
    pairing = []
    for mapping in ("paired", "swapped"):
        doses = {nid: [] for nid in groups["vision"]+groups["touch"]}
        pairs = []
        for trial in pairing_protocol(40, 3, 11, mapping):
            if trial["phase"] == "experience":
                pairs.append((trial["visual_clip"], trial["audio_clip"]))
            for t in range(trial["start"], trial["stop"]):
                for nid, amplitude in inputs(features, groups, trial, t):
                    doses[nid].append(amplitude)
        marginal.append({nid: sorted(values) for nid, values in doses.items()})
        pairing.append(pairs)
    assert marginal[0] == marginal[1]
    assert [p[0] for p in pairing[0]] == [p[0] for p in pairing[1]]
    assert all(a == b for a, b in pairing[0])
    assert all(a != b for a, b in pairing[1])


def test_absent_channel_cannot_receive_stimulation():
    _, groups, _ = make_config(1152)
    features = [{"ticks": 40, "visual": np.ones((40, 96)),
                 "auditory": np.ones((40, 96))} for _ in range(2)]
    trial = {"start": 0, "stop": 40, "visual_clip": 0, "audio_clip": None}
    delivered = {nid for t in range(40) for nid, _ in inputs(features, groups, trial, t)}
    assert delivered == set(groups["vision"])


def test_content_readout_is_not_generic_excitation_or_timing_match():
    reference = np.array([[.2, 0., .1, 0.], [0., .2, 0., .1]])
    assert content_projection(reference, reference)["visual_content_contrast"] == pytest.approx(1)
    assert content_projection(reference[::-1], reference)["visual_content_contrast"] == pytest.approx(-1)
    generic = np.ones((2, 4))*.3
    assert content_projection(generic, reference)["visual_content_contrast"] == pytest.approx(0)
    assert not content_projection(generic, generic)["defined"]
    # Temporal reordering leaves population rates unchanged.
    trace = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 0.]])
    assert np.array_equal(trace.mean(axis=0), trace[::-1].mean(axis=0))


def test_regulation_factorial_changes_only_the_two_declared_parameters():
    from simulations.active_inference.experiments.population_regulation_probe import regulation_config
    config, _, _ = make_config(288)
    for condition in ("original", "sensitivity_only", "receptor_only", "combined"):
        changed = regulation_config(config, condition)
        assert changed["connections"] == config["connections"]
        assert changed["synaptic_points"] == config["synaptic_points"]
        for old, new in zip(config["neurons"], changed["neurons"]):
            role = old["metadata"]["role"]
            differences = {key for key in old["params"] if old["params"][key] != new["params"][key]}
            allowed = set()
            if role == "activity_regulator" and condition in ("sensitivity_only", "combined"):
                allowed = {"r_base", "b_base"}
            if role in ("visual_core", "tactile_core", "upper_core") and condition in ("receptor_only", "combined"):
                allowed = {"w_r", "w_b"}
            assert differences == allowed
            assert new["params"]["eta_post"] == old["params"]["eta_post"] > 0
            assert new["params"]["eta_retro"] == old["params"]["eta_retro"] > 0


def test_independent_covariance_readout_keeps_cofluctuation_not_absolute_phase():
    from simulations.active_inference.experiments.multimodal_pairing_audit import readout
    cells = np.zeros((96, 2, 8))
    cells[32:, 0, 2] = np.tile([0., 1., 0., 1.], 16)
    cells[32:, 1, 2] = cells[32:, 0, 2]
    reversed_time = cells.copy()
    reversed_time[32:] = cells[32:][::-1]
    assert np.array_equal(readout(cells, [1, 2], "state_covariance"),
                          readout(reversed_time, [1, 2], "state_covariance"))
    opposite = cells.copy()
    opposite[32:, 1, 2] = 1-cells[32:, 1, 2]
    assert not np.array_equal(readout(cells, [1, 2], "state_covariance"),
                              readout(opposite, [1, 2], "state_covariance"))
    assert not np.any(readout(cells, [1, 2], "population_centered_covariance"))


def test_centered_readout_rejects_different_uniform_activation_levels():
    from simulations.active_inference.experiments.multimodal_pairing_audit import readout, contrast
    first, second = np.zeros((96, 8, 8)), np.zeros((96, 8, 8))
    first[32::3, :, 1] = 1
    second[32::9, :, 1] = 1
    raw = np.stack([readout(x, list(range(1, 9)), "mean_rate") for x in (first, second)])
    assert contrast(raw, raw) == pytest.approx(1)
    centered = np.stack([readout(x, list(range(1, 9)), "population_centered_rate") for x in (first, second)])
    assert contrast(centered, centered) is None


def test_regional_control_preserves_degrees_and_only_scrambles_regulatory_outputs():
    from collections import Counter
    from simulations.active_inference.components.learning.regional_regulation import regional_regulation
    config, groups, edges = make_config(1152)
    local, le, lm = regional_regulation(config, groups, edges, routing="regional")
    shuffled, se, sm = regional_regulation(config, groups, edges, routing="shuffled")
    assert local["neurons"] == shuffled["neurons"]
    assert local["synaptic_points"] == shuffled["synaptic_points"] == config["synaptic_points"]
    assert local["external_inputs"] == shuffled["external_inputs"] == config["external_inputs"]
    assert len(local["connections"]) == len(shuffled["connections"]) == len(config["connections"])
    assert lm["aligned_projection_fraction"] == 1.
    assert .2 < sm["aligned_projection_fraction"] < .5
    assert sm["degree_preserving_swaps"] > 0
    for key in ("source_neuron", "target_neuron"):
        assert Counter(c[key] for c in local["connections"]) == Counter(c[key] for c in shuffled["connections"])
    assert len({(c["source_neuron"], c["target_neuron"], c["target_synapse"]) for c in shuffled["connections"]}) == len(shuffled["connections"])
    for a, b in zip(le, se):
        assert a[1:] == b[1:]
        if a[3] != "excitability_modulation":
            assert a == b
    for i, role in enumerate(("visual_core", "tactile_core", "upper_core")):
        detector_ids = set(lm["scopes"][role])
        assert detector_ids
        assert all(src in groups[role] for src, tgt, _, family, _ in le
                   if family == "observed_activity" and tgt in detector_ids)
        assert all(src in detector_ids for src, tgt, _, family, _ in le
                   if family == "excitability_modulation" and tgt in groups[role])


def test_regional_builder_does_not_mutate_input_graph():
    from simulations.active_inference.components.learning.regional_regulation import regional_regulation
    config, groups, edges = make_config(288)
    before = encode([config, groups, edges])
    regional_regulation(config, groups, edges, routing="shuffled")
    assert before == encode([config, groups, edges])


def test_weight_intervention_is_a_complete_two_factor_partition():
    from simulations.active_inference.experiments.multimodal_weight_path_probe import hybrid_weights, selected_ports
    index = [(1, 0), (1, 1), (2, 0), (2, 1)]
    edges = [[3, 1, 0, "crossmodal", True], [4, 1, 1, "descending", True],
             [3, 2, 0, "crossmodal", True], [4, 2, 1, "sensory", True]]
    selected = selected_ports(edges, {1})
    assert selected == {(1, 0), (1, 1)}
    initial, learned = np.array([1., 2., 3., 4.]), np.array([11., 12., 13., 14.])
    a = hybrid_weights(initial, learned, index, selected, "learned_selected_only")
    b = hybrid_weights(initial, learned, index, selected, "learned_remainder_only")
    assert np.array_equal(a, [11., 12., 3., 4.])
    assert np.array_equal(b, [1., 2., 13., 14.])
    assert np.array_equal(a+b-initial, learned)
    all_weights = hybrid_weights(initial, learned, index, selected, "learned_all")
    all_weights[0] = 100
    assert learned[0] == 11
    with pytest.raises(ValueError):
        hybrid_weights(initial, learned, index, {(7, 0)}, "learned_selected_only")


def test_native_soft_bound_has_asymmetric_effect_on_negative_weights():
    from simulations.active_inference.experiments.synaptic_credit_trace import legacy_update
    negative = legacy_update(-100., .005, -1., 101.)
    assert negative["soft_factor"] == 11.
    assert negative["unclipped_weight"] > 0
    assert negative["expected_weight"] == 100.
    positive = legacy_update(9., .005, 1., 8.)
    assert positive["soft_factor"] == pytest.approx(.1)
    assert 9. < positive["expected_weight"] < 10.


def test_single_port_credit_observer_preserves_actual_model_tick(tmp_path):
    from simulations.active_inference.experiments.synaptic_credit_trace import observe_port
    config, groups, _ = make_config(288)
    path = tmp_path/"config.json"
    path.write_text(encode(config))
    reference = []
    net, core, neurons, _ = fresh(path, 11)
    for t in range(32):
        for nid in groups["vision"]:
            net.set_external_input(nid, 0, 2.)
        assert "error" not in core.do_tick()
        reference.append(cellular(neurons))
    net, core, neurons, _ = fresh(path, 11)
    rows, observed = [], []
    original = Neuron.tick
    with observe_port(groups["visual_core"][0], 0, rows):
        for t in range(32):
            for nid in groups["vision"]:
                net.set_external_input(nid, 0, 2.)
            assert "error" not in core.do_tick()
            observed.append(cellular(neurons))
    assert Neuron.tick is original
    assert np.array_equal(reference, observed)
    assert any(row["active"] for row in rows)
    assert max(row["accounting_error"] for row in rows) < 1e-10


def test_per_tick_weight_observer_is_read_only(tmp_path):
    from simulations.active_inference.experiments.multimodal_pairing_probe import WeightObserver
    from simulations.active_inference.experiments.composition_probe import snapshot
    config, groups, _ = make_config(288)
    path = tmp_path/"config.json"
    path.write_text(encode(config))
    outcomes, health = [], None
    for observed in (False, True):
        net, core, neurons, synapses = fresh(path, 11)
        observer = WeightObserver(neurons, synapses) if observed else None
        for t in range(40):
            for nid in groups["vision"]:
                if t % 4 == nid % 4:
                    net.set_external_input(nid, 0, 2.)
            assert "error" not in core.do_tick()
            if observer:
                observer()
        outcomes.append(encode(snapshot(net)))
        if observer:
            health = np.array(observer.rows)
    assert outcomes[0] == outcomes[1]
    assert health.shape == (40, len(WeightObserver.fields))
    assert np.isfinite(health).all()
    assert health[:, WeightObserver.fields.index("changed_weights")].sum() > 0


def test_weight_health_audit_checks_endpoints_and_rejects_false_observer(tmp_path):
    from simulations.active_inference.experiments.multimodal_pairing_audit import weight_health
    manifest = {"trials": [{"start": 0, "stop": 2}],
                "weight_health_fields": ["max_abs_weight", "sign_changes_from_initial", "at_native_bound", "zeroed_nonzero_weights"]}
    initial, learned = np.array([1., -.8]), np.array([1.1, .2])
    np.savez(tmp_path/"parameters.npz", initial_info=initial, learned_info=learned)
    path = tmp_path/"experience-000.npz"
    health = np.array([[1., 0., 0., 0.], [1.1, 1., 0., 0.]])
    np.savez(path, incoming_info_before=initial, incoming_info_after=learned, weight_health=health)
    result = weight_health(tmp_path, manifest)
    assert result["endpoints_verified"] and result["per_tick_available"]
    assert result["episodes"][0]["sign_changes_from_initial"] == 1
    health[-1, 1] = 0  # Deliberately false claim that the synaptic sign survived.
    np.savez(path, incoming_info_before=initial, incoming_info_after=learned, weight_health=health)
    with pytest.raises(ValueError, match="disagrees"):
        weight_health(tmp_path, manifest)


def test_exploratory_weight_geometry_recovers_a_declared_toy_contrast(tmp_path):
    from simulations.active_inference.experiments.multimodal_pairing_audit import weight_assignment_geometry
    paths = [tmp_path/"paired", tmp_path/"swapped"]
    config = {"neurons": [{"id": i} for i in (1, 2, 3, 4)],
              "synaptic_points": [{"neuron_id": nid, "synapse_id": sid, "type": "postsynaptic",
                                   "u_i": {"info": .3}} for nid in (3, 4) for sid in (0, 1)]}
    edges = [[src, tgt, src-1, "crossmodal", True] for tgt in (3, 4) for src in (1, 2)]
    manifests, probes = [], [{}]
    for name, sense, offset in (("visual", "visual", 0), ("audio", "audio", 2)):
        for clip in (0, 1):
            cells = np.zeros((96, 4, 8))
            cells[32::2, offset+clip, 1] = 1
            probes[0][f"initial-{sense}-{clip}"] = cells
    for i, path in enumerate(paths):
        path.mkdir()
        (path/"config.json").write_text(encode(config))
        manifests.append({"mapping": path.name, "groups": {"visual_core": [1, 2], "tactile_core": [3, 4]}, "edges": edges})
        initial = np.full(4, .3)
        learned = initial+(1 if i == 0 else -1)*np.array([.1, -.1, -.1, .1])
        np.savez(path/"parameters.npz", initial_info=initial, learned_info=learned)
    result = weight_assignment_geometry(paths, manifests, probes)
    assert result["selected_synapses"] == 4
    assert result["conditions"]["paired"]["contrast_product_coefficient"] == pytest.approx(.4)
    assert result["conditions"]["swapped"]["contrast_product_coefficient"] == pytest.approx(-.4)
