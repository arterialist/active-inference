"""Replay recorded sensory/modulatory interventions from a saved configuration.

No reconstruction of the experimental schedule or circuit builder is used.
The recorded internal state is compared after every tick, never injected into
the network. This distinguishes causal replay from animation of saved states.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .composition_probe import encode, k, observe_inputs, snapshot
from neuron.extensions.experimental.plasticity_rate import PlasticityRateNeuron


def replay(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    net, core = k.load(str(directory / "config.json"), neuron_class=PlasticityRateNeuron)
    net.record_history = False
    digest = hashlib.sha256()
    with gzip.open(directory / "ticks.jsonl.gz", "rt") as stream, observe_inputs() as inputs:
        initial = json.loads(next(stream))
        assert encode(snapshot(net)) == encode(initial["state"]), "Initial state differs"
        count = 0
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            assert row["executed_tick"] == count-1
            for nid, sid, value in row["external"]:
                net.set_external_input(nid, sid, value)
            for nid, sid, value, mod in row.get("replayed_modulator", []):
                net.set_external_input(nid, sid, value, mod=np.array(mod))
            inputs.clear()
            result = core.do_tick()
            assert "error" not in result and net.current_tick == count
            actual = snapshot(net)
            actual["plasticity_rates"] = {str(i): {"used_multiplier": n.last_tick_rate_multiplier,
                "next_multiplier": n.rate_multiplier(), "basal": n.params.eta_post}
                for i, n in net.network.neurons.items()}
            for i, n in net.network.neurons.items():
                if "basal_retro" in row["state"]["plasticity_rates"][str(i)]:
                    actual["plasticity_rates"][str(i)]["basal_retro"] = n.params.eta_retro
            assert encode(actual) == encode(row["state"]), f"First state mismatch at tick {count-1}"
            if manifest["observed"]:
                assert inputs == row["delivered_inputs"], f"Input delivery mismatch at tick {count-1}"
            digest.update(encode(actual).encode())
    assert count == summary["ticks"] and digest.hexdigest() == summary["state_digest"]
    return {"run": str(directory), "ticks": count, "exact_replay": True, "state_digest": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = replay(args.directory)
    if args.output:
        with args.output.open("x") as stream:
            stream.write(encode(result)+"\n")
    print(encode(result))


if __name__ == "__main__":
    main()
