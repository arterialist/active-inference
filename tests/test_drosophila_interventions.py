from pathlib import Path
import json

import numpy as np
import pytest

from simulations.drosophila.connectome import Subgraph
from simulations.drosophila.intervention_probe import TickRecorder, make_course, run_intervention
from simulations.drosophila.intervention_analysis import inspect_record, compare, verify_unobserved
from simulations.drosophila.paula import Dynamics, Neuron, GradedNeuron, build_paula

ROOTS = tuple(str(720575940000000001 + i) for i in range(3))


def graph():
    classes = ("Kenyon_Cell", "MBIN", "ALPN")
    nodes = {r: {"global_index": i, "annotation": {"root_id": r, "cell_class": classes[i],
                  "hemibrain_type": "APL" if i == 1 else ""}} for i, r in enumerate(ROOTS)}
    rows = []
    for row, (pre, post, count, sign) in enumerate(((2, 0, 10, 1), (0, 1, 3, 1), (1, 0, 4, -1))):
        rows.append([int(ROOTS[pre]), int(ROOTS[post]), pre, post, count, sign, count * sign, row, row])
    return Subgraph(ROOTS, nodes, np.asarray(rows, dtype=np.int64), {"fixture": True})


def run_small(output: Path, blocked):
    p = build_paula(graph(), Dynamics(lambda_ticks=1, cooldown_ticks=1, weight_per_count=0.5))
    r = TickRecorder(p, output, chunk_size=2)
    with r.observe(blocked):
        for t in range(8):
            if not r.arrays:
                r.begin()
            ext = np.zeros(3)
            if t == 0:
                ext[2] = 40
                p.stimulate(ROOTS[2], 40)
            p.network.run_tick()
            r.finish_tick(ext)
    r.flush()
    return p, r


def chunks(recorder, key):
    result = []
    for entry in recorder.chunks:
        with np.load(recorder.output / entry["file"], allow_pickle=False) as data:
            value = data[key]
            result.append(value[1:] if key in {"soma", "M", "post_weight", "terminal_info"} else value)
    return np.concatenate(result)


def test_block_leaves_source_spiking_and_native_return_plasticity_alive(tmp_path):
    p, r = run_small(tmp_path / "blocked-kc", {0})
    soma = chunks(r, "soma")
    emitted, blocked, returned = (chunks(r, key) for key in ("emitted", "blocked", "returned"))
    assert soma[3, 0, 1] > 0  # KC spikes normally from its PN input.
    assert emitted[3, 0] == blocked[3, 0] == 1
    assert returned[1, 0] == 1  # Return is generated on receptor arrival, before soma spike.
    assert p.network.network.neurons[2].presynaptic_points[0].u_o.info != 1
    assert np.all(soma[:, 1, 1] == 0)  # APL receives no KC release.
    assert all(c.params.eta_post > 0 and c.params.eta_retro > 0 for c in r.cells)
    assert all(not c._ablation for c in r.cells)


def test_graded_block_filters_real_release_after_super_tick(tmp_path):
    _, intact = run_small(tmp_path / "intact", set())
    _, blocked = run_small(tmp_path / "blocked-apl", {1})
    assert chunks(intact, "soma")[6, 1, 1] > 0
    assert chunks(blocked, "soma")[6, 1, 1] > 0
    assert chunks(blocked, "blocked")[6, 1] == 1
    # Actual APL input still arrives. Only its subsequent outgoing release differs.
    np.testing.assert_array_equal(chunks(intact, "soma")[:7, 1], chunks(blocked, "soma")[:7, 1])
    assert np.any(chunks(intact, "local_potential")[7] < 0)
    assert not np.any(chunks(blocked, "local_potential")[7] < 0)


def test_observer_and_chunk_boundaries_do_not_change_native_dynamics(tmp_path):
    p, r = run_small(tmp_path / "observed", set())
    plain = build_paula(graph(), Dynamics(lambda_ticks=1, cooldown_ticks=1, weight_per_count=0.5))
    expected = []
    for t in range(8):
        if t == 0:
            plain.stimulate(ROOTS[2], 40)
        plain.network.run_tick()
        expected.append([[c.S, c.O, c.F_avg, c.r, c.b, c.t_ref, c.t_last_fire]
                         for c in plain.network.network.neurons.values()])
    np.testing.assert_array_equal(chunks(r, "soma"), expected)
    assert [(c["start"], c["stop"]) for c in r.chunks] == [(0, 2), (2, 4), (4, 6), (6, 8)]
    for prev, nxt in zip(r.chunks, r.chunks[1:]):
        with np.load(r.output / prev["file"]) as a, np.load(r.output / nxt["file"]) as b:
            for key in ("soma", "M", "post_weight", "terminal_info"):
                np.testing.assert_array_equal(a[key][-1], b[key][0])
    assert p.network.current_tick == plain.network.current_tick


def test_patches_restore_after_exception(tmp_path):
    p = build_paula(graph())
    r = TickRecorder(p, tmp_path / "fault")
    base, graded = Neuron.tick, GradedNeuron.tick
    with pytest.raises(RuntimeError, match="Begin the recording"):
        with r.observe(set()):
            p.network.run_tick()
    assert Neuron.tick is base and GradedNeuron.tick is graded


def test_input_course_declares_each_level_and_does_not_drive_kcs():
    g = graph()
    p = build_paula(g)
    course, kc, pn, apl, epochs = make_course(p, g)
    assert course.shape == (224, 3)
    assert not course[:, kc].any() and not course[:, apl].any()
    for epoch in epochs:
        assert np.all(course[epoch["start"]:epoch["stop"], pn] == epoch["PN_drive"])
    assert not course[:32].any() and not course[192:].any()


@pytest.fixture
def intact_record(tmp_path):
    output = tmp_path / "course"
    run_intervention(graph(), output, "intact", 0.5)
    return output


def test_analysis_reconstructs_delayed_apl_current_and_checks_each_tick(intact_record):
    report, soma, _, _, _, groups = inspect_record(intact_record)
    assert report["recording_checks_passed"]
    assert report["positive_terminal_coefficients"]
    assert report["postsynaptic_coefficients_changed"] > 0
    assert report["terminal_coefficients_changed"] > 0
    currents = report["apl_input_by_source_each_tick"]
    assert np.any(currents["KC"])
    assert not np.any(currents["boundary"])
    assert not np.any(currents["PN"])  # This fixture has no direct PN->APL route.
    apl_s = soma[:, groups["APL"][0], 0]
    inferred = 20 * (apl_s - 0.95 * np.r_[0, apl_s[:-1]])
    np.testing.assert_allclose(sum(np.asarray(v) for v in currents.values()), inferred, atol=1e-5)
    same = compare(intact_record, intact_record)
    assert all(t is None for fields in same["first_divergence"].values() for t in fields.values())


@pytest.mark.parametrize("fault, message", [
    ("summary", "Epoch summary disagrees"),
    ("target", "Condition and blockade targets disagree"),
    ("gap", "Chunk gap"),
    ("course", "Recorded input does not match"),
])
def test_analysis_rejects_misleading_metadata(intact_record, fault, message):
    path = intact_record / "manifest.json"
    m = json.loads(path.read_text())
    if fault == "summary":
        m["epoch_summaries_not_acceptance"][0]["kc_spikes"] += 1
    elif fault == "target":
        m["protocol"]["blocked_ids"] = [0]
    elif fault == "gap":
        m["recording"]["chunks"][1]["start"] += 1
    else:
        m["protocol"]["epochs"][0]["PN_drive"] += 1
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match=message):
        inspect_record(intact_record)


def test_comparison_rejects_parameter_change_disguised_as_intervention(intact_record, tmp_path):
    other = tmp_path / "other"
    run_intervention(graph(), other, "apl_release_block", 0.4)
    with pytest.raises(ValueError, match="Unmatched assumptions"):
        compare(intact_record, other)


def test_source_files_are_sampled_before_and_after_course(intact_record):
    m = json.loads((intact_record / "manifest.json").read_text())
    sources = m["source_files"]
    assert sources["at_import"] == sources["at_start"] == sources["at_finish"]
    assert {"neuron.py", "network.py", "graded.py", "paula.py"} <= {Path(k).name for k in sources["at_start"]}


def test_full_uninstrumented_verification_checks_every_state(intact_record):
    result = verify_unobserved(graph(), intact_record)
    assert result["ticks"] == 224
    assert result["equal_values"] == {"soma": 225 * 3 * 7, "M": 225 * 3 * 2,
                                      "post_weight": 225 * 6, "terminal_info": 225 * 3}
