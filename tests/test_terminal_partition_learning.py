import hashlib
import json

import numpy as np

from simulations.active_inference.experiments.composition_probe import encode
from simulations.active_inference.experiments.population_hierarchy import make_config, FIELDS
from simulations.active_inference.experiments.multimodal_pairing_probe import pairing_protocol, WeightObserver
from simulations.active_inference.components.learning.regional_regulation import regional_regulation
from simulations.active_inference.experiments.terminal_partition_learning import run


def test_runner_preserves_trials_and_distinguishes_full_state_probes(tmp_path):
    source, output = tmp_path/"source", tmp_path/"experiment"
    source.mkdir()
    config, groups, edges = make_config(288)
    config, edges, _ = regional_regulation(config, groups, edges)
    for n in config["neurons"]:
        n["metadata"]["bounded_plasticity"] = True
    (source/"config.json").write_text(encode(config))
    rng = np.random.default_rng(11)
    for i in (0, 1):
        np.savez_compressed(source/f"sensory-{i}.npz", ticks=8,
            visual=rng.random((8, len(groups["vision"]))),
            auditory=rng.random((8, len(groups["touch"]))))
    trials = pairing_protocol(8, 1, 11, "paired")
    manifest = {"weight_dynamics": "bounded", "architecture": "regional", "seed": 11,
        "mapping": "paired", "groups": groups, "edges": edges, "trials": trials,
        "clip_ticks": 8, "source_hashes": {}, "fields": FIELDS,
        "weight_health_fields": WeightObserver.fields}
    (source/"manifest.json").write_text(encode(manifest))
    result = run(source, output, "family")
    saved = json.loads((output/"manifest.json").read_text())
    assert saved["trials"] == trials and saved["groups"] == groups
    assert saved["terminal_partition"]["added_terminals"] > 0
    assert result["ticks"] == trials[-1]["stop"]+80
    assert len(result["probes"]) == 10
    for p in result["probes"]:
        assert p["parent_unchanged"]
        if p["condition"] == "continuation":
            assert p["start_state_sha256"] == result["trained_state_sha256"]
            assert p["start_tick"] == trials[-1]["stop"]
        else:
            assert p["start_tick"] == 0
        with np.load(output/f'probe-{p["condition"]}-{p["sense"]}-{p["clip"]}.npz') as raw:
            assert raw["cells"].shape == (8, 288, len(FIELDS))
    for name, expected in saved["source_hashes"].items():
        from pathlib import Path
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
