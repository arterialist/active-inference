"""Fast contracts for the live server/client/lab boundary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from simulations.active_inference.live.adapters import NeuronDisplayAdapter, TopologyDisplayAdapter
from simulations.active_inference.live.protocol import PROTOCOL_VERSION, hello
from simulations.active_inference.live.versions import get_version
from simulations.active_inference.live.introspection import TraceStore
from simulations.active_inference.lab.harnesses import HARNESSES, all_specs


ROOT = Path(__file__).resolve().parents[1]


def test_versions_and_protocol_are_explicit():
    assert [get_version(v).id for v in ("1", "v2", "interoceptive")] == ["v1", "v2", "v3"]
    msg = hello(get_version("v3").manifest(), [11, 12], [99])
    assert msg["protocol"] == PROTOCOL_VERSION
    assert msg["neuron_count"] == 2
    assert msg["graded"] == [99]


def test_display_adapters_are_json_safe_for_never_fired_cells():
    class Unit:
        S = 0.25
        O = 0.0
        t_last_fire = float("inf")
        t_ref = 2.0
        meta = {"kind": "graded"}

    value = NeuronDisplayAdapter().adapt(42, Unit(), incoming=[], outgoing=[])
    assert value["intracellular"]["t_last_fire"] is None
    assert json.dumps(value)


def test_live_client_is_dynamic_and_has_deep_inspection_routes():
    html = (ROOT / "simulations/active_inference/brain_live.html").read_text()
    assert '<script id="payload" type="application/json"></script>' in html
    assert "/api/topology" in html
    assert "/api/neuron/" in html
    assert "const GRADED=[" not in html
    assert "async function boot()" in html


def test_lab_defaults_to_the_tick_microscope_and_keeps_batch_tab():
    html = (ROOT / "simulations/active_inference/lab/index.html").read_text()
    assert "PAULA neural microscope" in html
    assert 'id="brain"' in html
    assert "Batch harnesses" in html
    assert "/api/introspection" in html
    assert "Evidence rail" in html
    assert 'id="tickSlider"' in html
    assert "copy record" in html
    assert 'id="runWorldField"' in html
    assert "payload.world" in html


def test_harness_registry_is_whitelisted_and_bounded(tmp_path):
    assert {"motor", "food_collection", "toxin_escape", "headon_toxin", "memory", "metabolic_rest", "compass"} <= set(HARNESSES)
    assert len(all_specs()) == len(HARNESSES)
    command = HARNESSES["metabolic_rest"].command(output=tmp_path / "run", steps=80, substeps=4, seeds=[11])
    assert str(tmp_path / "run") in command
    assert command[command.index("--version") + 1] == "v3"
    memory_command = HARNESSES["memory"].command(output=tmp_path / "memory", steps=8, substeps=4, seeds=[11], version="v2")
    assert memory_command[memory_command.index("--probe-steps") + 1] == "10"
    try:
        HARNESSES["motor"].command(output=tmp_path / "bad", steps=99999)
    except ValueError:
        pass
    else:
        raise AssertionError("harness accepted an unbounded run")
    try:
        HARNESSES["metabolic_rest"].command(output=tmp_path / "zero", steps=0, substeps=4)
    except ValueError:
        pass
    else:
        raise AssertionError("harness treated an explicit zero as the default")


def test_obstacle_harness_exposes_geometry_suite_and_forwards_world(tmp_path):
    spec = HARNESSES["obstacle_detour"]
    assert spec.default_world == "all"
    assert spec.supported_versions == ("v4",)
    assert {"head_on_wall", "corner", "chicane", "maze", "all"} <= set(spec.worlds)
    command = spec.command(output=tmp_path / "maze", steps=240, substeps=4, seeds=[11], world="maze")
    assert command[command.index("--world") + 1] == "maze"
    assert command[command.index("--version") + 1] == "v4"
    try:
        spec.command(output=tmp_path / "bad-world", world="not-a-fixture")
    except ValueError:
        pass
    else:
        raise AssertionError("obstacle harness accepted an unknown geometry")


def test_topology_adapter_requires_the_packed_wire_fields():
    try:
        TopologyDisplayAdapter().adapt({})
    except ValueError as exc:
        assert "topology payload" in str(exc)
    else:
        raise AssertionError("incomplete topology was accepted")


def test_introspection_ring_preserves_post_tick_neuron_synapse_and_body_state():
    class Point:
        def __init__(self):
            self.u_i = SimpleNamespace(info=1.25, plast=-0.2)
            self.potential = 0.7

    class Unit:
        def __init__(self, nid):
            self.id = nid
            self.metadata = {"population": "test"}
            self.params = SimpleNamespace(r_base=0.4, num_inputs=1)
            self.t_ref = 2.0; self.r = 0.4; self.b = 0.8
            self.S = 0.25; self.O = 1.0; self.F_avg = 0.1; self.t_last_fire = 3
            self.M_vector = np.array([0.2, 0.3])
            self.distances = {0: 4}; self.postsynaptic_points = {0: Point()}
            self.synapse_sources = {0: (1, 0)}

    source, target = Unit(1), Unit(2)
    network = SimpleNamespace(
        neurons={1: source, 2: target},
        connection_cache={(1, 0): [(2, 0)]},
    )
    world = SimpleNamespace(
        pose=lambda: (1.0, 2.0, 0.5), speed=lambda: 0.3, yaw_rate=lambda: -0.1,
        dist_home=lambda: 2.2, eaten=1, tox_hits=0, event="food", pending_event=None,
        foods=[[3.0, 4.0]], toxins=[], arena=8.0, barriers=[], metabolic_state=lambda: {"energy_store": 0.9},
    )
    ag = SimpleNamespace(world=world, net=SimpleNamespace(network=network), last_sensor_drives={"food_left": 2.0})
    ring = TraceStore([1, 2], [source, target], network, max_ticks=4)
    ring.capture(tick=1, step=0, ag=ag, units=[source, target])
    source.O = 0.0; source.S = 0.8; target.O = 1.0
    ring.capture(tick=2, step=1, ag=ag, units=[source, target])
    assert ring.range_public()["ticks"] == [1, 2]
    tick = ring.tick_public(2, include_synapses=True)
    assert tick["fired"] == [2]
    assert abs(tick["neurons"][0]["S"] - 0.8) < 1e-5
    assert abs(tick["synapses"][0]["potential"] - 0.7) < 1e-5
    assert abs(ring.neuron_history(1)["frames"][-1]["S"] - 0.8) < 1e-5
    assert ring.synapse_history(source=1, target=2, synapse=0)["frames"][-1]["target_firing"]
