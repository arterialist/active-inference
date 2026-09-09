"""Real audiovisual probes of aligned versus shuffled regional regulation.

Fresh network for each sensory condition. No training protocol or consciousness
claim. Raw per-tick cells, weight endpoints and final state are retained.
"""
import argparse
import gzip
import hashlib
from pathlib import Path
import time

import numpy as np

from ..components.learning.regional_regulation import regional_regulation
from .audiovisual_population import decode_media
from .composition_probe import encode, fingerprint, snapshot
from .multimodal_pairing_probe import fresh, episode
from .population_hierarchy import make_config, FIELDS, weight_values


def run(source, destination, routing, seed):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    features = decode_media(source)
    np.savez_compressed(destination/"sensory.npz", **features)
    config, groups, edges = make_config(1152, seed)
    config, edges, wiring = regional_regulation(config, groups, edges, seed=seed, routing=routing)
    path = destination/"config.json"
    path.write_text(encode(config)+"\n")
    sources = fingerprint()
    from ..components.learning import regional_regulation as component
    for p in (Path(__file__), Path(component.__file__), Path(__file__).with_name("population_hierarchy.py"),
              Path(__file__).with_name("multimodal_pairing_probe.py"), Path(__file__).with_name("audiovisual_population.py")):
        sources[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {"seed": seed, "size": 1152, "routing": routing, "wiring": wiring, "groups": groups,
        "edges": edges, "fields": FIELDS, "source_hashes": sources,
        "source_media": str(Path(source).resolve()), "media_sha256": features["source_sha256"],
        "stimulus_ticks": features["ticks"], "withdrawal_ticks": 96,
        "scope": "Isolated sensory integration and regulation, no learned association or embodied result. Each sensory condition starts fresh but plasticity remains active within it."}
    (destination/"manifest.json").write_text(encode(manifest)+"\n")
    results = {}
    started = time.perf_counter()
    for name, visual, auditory in (("vision", 0, None), ("audio", None, 0), ("both", 0, 0)):
        net, core, neurons, synapses = fresh(path, seed)
        initial = weight_values(synapses)
        trial = {"start": 0, "stop": features["ticks"], "visual_clip": visual, "audio_clip": auditory}
        cells = episode(net, core, neurons, [features], groups, trial)
        withdrawal = {"start": features["ticks"], "stop": features["ticks"]+96,
                      "visual_clip": None, "audio_clip": None}
        tail = episode(net, core, neurons, [features], groups, withdrawal)
        np.savez_compressed(destination/f"{name}.npz", cells=np.concatenate((cells, tail)),
                            incoming_info_before=initial, incoming_info_after=weight_values(synapses))
        with gzip.open(destination/f"{name}-final.json.gz", "wt") as f:
            f.write(encode(snapshot(net))+"\n")
        # Windows expose drift and onset rather than one aggregate score.
        windows = []
        spikes = cells[:, :, 1] > 0
        for start in range(0, len(cells), 48):
            part = spikes[start:start+48]
            windows.append({"start": start, "stop": min(start+48, len(cells)),
                "rates": {role: float(part[:, np.array(ids)-1].mean()) for role, ids in groups.items()}})
        results[name] = {"windows": windows, "groups": {
            role: {"spikes": int(spikes[:, np.array(ids)-1].sum()),
                   "rate_after32": float(spikes[32:, np.array(ids)-1].mean()),
                   "near_ceiling_fraction": float((spikes[32:, np.array(ids)-1].mean(axis=0) >= .30).mean()),
                   "late_withdrawal_spikes": int((tail[48:, np.array(ids)-1, 1] > 0).sum())}
            for role, ids in groups.items()}}
        print(encode({"routing": routing, "seed": seed, "probe": name,
            "seconds": round(time.perf_counter()-started, 2),
            "rates": {role: results[name]["groups"][role]["rate_after32"] for role in ("visual_core", "tactile_core", "upper_core")}}), flush=True)
    report = {"routing": routing, "seed": seed, "results": results, "wiring": wiring,
              "elapsed_seconds": time.perf_counter()-started, "ticks": 3*(features["ticks"]+96)}
    (destination/"summary.json").write_text(encode(report)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--routing", choices=("regional", "shuffled"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 44, 77])
    args = parser.parse_args()
    for seed in args.seeds:
        run(args.source, args.output/f"seed-{seed}", args.routing, seed)
