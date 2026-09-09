import numpy as np

from simulations.active_inference.experiments.population_hierarchy import (
    make_config, protocol, input_ids, finite_cosine, read_information_weights, SIZES,
)


def test_population_sizes_and_positive_basal_adaptation():
    assert sum(SIZES.values()) == 1152
    for size in (288, 576, 1152):
        config, groups, edges = make_config(size)
        assert len(config["neurons"]) == size == sum(map(len, groups.values()))
        for n in config["neurons"]:
            assert n["params"]["eta_post"] > 0
            assert n["params"]["eta_retro"] > 0
            assert n["params"]["num_inputs"] >= 2
        assert all(edge[0] != edge[1] for edge in edges)


def test_external_world_cannot_address_upper_or_supervisor():
    config, groups, _ = make_config(288)
    sensory = set(groups["vision"] + groups["touch"])
    assert {x["target_neuron"] for x in config["external_inputs"]} == sensory
    patterns, trials = protocol(groups, 11)
    for tr in trials:
        for tick in range(tr["start"], tr["stop"]):
            assert set(input_ids(tr, tick)) <= sensory
    for role in ("vision", "touch"):
        assert len(patterns[role][0]) == len(patterns[role][1])
        assert not set(patterns[role][0]) & set(patterns[role][1])


def test_pathway_cuts_preserve_ports_weights_and_neuron_parameters():
    intact, groups, _ = make_config(288)
    for mode in ("recurrence_cut", "ascending_cut", "descending_cut"):
        cut, cut_groups, edges = make_config(288, mode=mode)
        assert cut["neurons"] == intact["neurons"]
        assert cut["synaptic_points"] == intact["synaptic_points"]
        assert cut_groups == groups
        assert len(cut["connections"]) < len(intact["connections"])
        assert sum(edge[4] for edge in edges) == len(cut["connections"])


def test_upper_only_receives_declared_neural_pathways():
    _, groups, edges = make_config(1152)
    upper = set(groups["upper_core"])
    allowed = set(groups["connector"] + groups["upper_core"] + groups["upper_inhibition"] +
                  groups["mismatch_candidate"] + groups["activity_regulator"])
    assert all(source in allowed for source, target, *_ in edges if target in upper)


def test_silence_is_not_perfect_template_matching():
    assert finite_cosine(np.zeros(4), np.zeros(4)) is None
    assert finite_cosine(np.zeros(4), np.ones(4)) is None
    assert finite_cosine(np.ones(4), np.ones(4)) == 1.


def test_information_weight_xor_is_bit_exact():
    weights = np.array([[0., -0., 1.], [1e-300, -2., 1.00000000000001], [0., -2., 1000.]], dtype=np.float64)
    bits = weights.view(np.uint64)
    coded = np.concatenate((bits[:1], np.bitwise_xor(bits[1:], bits[:-1])))
    decoded = read_information_weights({"incoming_info_xor": coded})
    assert np.array_equal(decoded.view(np.uint64), bits)


def test_media_silent_conditions_have_no_auditory_inputs():
    from simulations.active_inference.experiments.audiovisual_population import media_inputs, media_protocol
    _, groups, _ = make_config(1152)
    features = {"ticks": 120, "visual": np.ones((120, 96)), "auditory": np.ones((120, 96))}
    for tr in media_protocol(120):
        for t in range(tr["start"], tr["stop"]):
            incoming = {nid for nid, _ in media_inputs(features, groups, tr, t)}
            if not tr["audio_enabled"]:
                assert not incoming & set(groups["touch"])
            if not tr["visual_enabled"]:
                assert not incoming & set(groups["vision"])
            assert incoming <= set(groups["vision"]+groups["touch"])


def test_shifted_sound_control_preserves_delivered_audio_dose():
    from simulations.active_inference.experiments.audiovisual_population import media_inputs, media_protocol
    _, groups, _ = make_config(1152)
    rng = np.random.default_rng(14)
    features = {"ticks": 120, "visual": rng.random((120, 96)), "auditory": rng.random((120, 96))}
    doses = []
    for condition in ("aligned", "shifted"):
        dose = {nid: [] for nid in groups["touch"]}
        for tr in media_protocol(120, condition):
            if tr["phase"] != "audiovisual_experience":
                continue
            for t in range(tr["start"], tr["stop"]):
                for nid, value in media_inputs(features, groups, tr, t):
                    if nid in dose:
                        dose[nid].append(value)
        doses.append({nid: sorted(values) for nid, values in dose.items()})
    assert doses[0] == doses[1]
