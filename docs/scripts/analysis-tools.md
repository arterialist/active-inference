# Analysis and Utility Scripts

← [Index](../INDEX.md)

## analyze_neuromod_food_seeking.py

Post-hoc analysis of logged simulation runs.

```bash
# Analyse a single run
uv run python scripts/analyze_neuromod_food_seeking.py --run-dir logs/my_run/

# Compare neuromod vs no-neuromod
uv run python scripts/analyze_neuromod_food_seeking.py --compare

# List available log dirs
uv run python scripts/analyze_neuromod_food_seeking.py --list
```

**Metrics computed:** start/end/min distance to food, fraction of steps moving toward food, chemical concentration change over the run.

---

## apply_evolved_config.py

Writes evolved parameters back to `neuron_mapping.py` class-level defaults via **regex substitution**.

```bash
uv run python scripts/apply_evolved_config.py --config evolved_food_seeking_config.json
```

Modifies `CElegansNervousSystem` class constants in-place. Use after a successful evolution run to bake the best parameters into source.

---

## download_connectome.py

One-shot connectome download and cache.

```bash
uv run python scripts/download_connectome.py
```

Downloads Cook 2019 data via `cect` and writes `data/c_elegans/connectome_cache.json`. Only needed once (or after `--no-cache`).

---

## evolve_food_seeking.py

See [Evolutionary Optimisation](../evolution/algorithm.md).

## See Also

- [Run Logs](../data-logging/run-logs.md) — format of `--run-dir` contents
- [Evolved Config Files](../evolution/output.md)
- [C. elegans Connectome Loader](../connectome/celegans-loader.md)
