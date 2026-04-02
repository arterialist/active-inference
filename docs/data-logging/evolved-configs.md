# Evolved Config Files

← [Index](../INDEX.md)

Evolved parameters live in the repo root alongside `pyproject.toml`.

| File | Description |
|------|-------------|
| `evolved_food_seeking_config.json` | Best parameters from primary evolution run |
| `evolved_food_seeking_config_2.json` | Alternative run |
| `evolved_food_seeking_checkpoint.json` | Latest checkpoint with full metadata |
| `*.bak` | Backup copies |

## Using Evolved Configs

Pass to any script via `--evol-config`:

```bash
uv run python scripts/run_c_elegans.py --evol-config evolved_food_seeking_config.json
```

Or bake into source code permanently:

```bash
uv run python scripts/apply_evolved_config.py --config evolved_food_seeking_config.json
```

## See Also

- [Evolution Output](../evolution/output.md) — full JSON schema and field descriptions
- [Configuration Reference](../config/nervous-system-constants.md) — what each key controls
