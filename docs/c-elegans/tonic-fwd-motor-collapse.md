# Research Finding: `TONIC_FWD_MOTOR` Collapses Body Curvature

<- [Index](../INDEX.md) | [C. elegans Overview](overview.md) | [Tonic Drives](../neuromodulation/tonic-drives.md)

Date: 2026-05-31

## Summary

`TONIC_FWD_MOTOR` is not a safe evolved-config parameter for the current
C. elegans locomotion stack. Non-zero values in the archived food-seeking
configs can make the worm appear active while collapsing the 13-segment body
shape into an almost straight line.

This is a motor-pattern failure, not a WebSocket or canvas-rendering issue. The
demo server sends nearly collinear segment coordinates over the wire before the
browser draws them.

## Evidence

Observed with:

```bash
cd celegans-live-demo
uv run celegans-demo-server --host 127.0.0.1 --port 8787 \
  --evol-config ../active-inference/evolved_food_seeking_config.json \
  --log-level INFO
```

Chrome raw WebSocket frames from `ws://127.0.0.1:8787` showed 13 segment points
(`sm`, protocol v3) with:

| Condition | `yRange` | `path/chord` | Interpretation |
|-----------|----------|--------------|----------------|
| Full evolved config | ~0.040 mm | ~1.008 | Nearly straight |
| No evolved config | ~0.346 mm | ~1.585 | Normal S-shaped body |

`run_c_elegans.py` can still appear to run normally because its standard plot
and GIF use COM/head trajectory plus motor-wave heatmaps. They do not render
the full 13-segment body geometry that the live demo draws.

## Ablation

At 500 ticks with `active-inference/evolved_food_seeking_config.json`:

| Config variant | `yRange` | `path/chord` | Mean D-V range |
|----------------|----------|--------------|----------------|
| Baseline, no config | 0.3465 mm | 1.5852 | 0.9541 |
| Full evolved config | 0.0399 mm | 1.0080 | 0.3884 |
| Only `TONIC_FWD_MOTOR` | 0.0451 mm | 1.0074 | 0.3872 |
| Full config minus `TONIC_FWD_MOTOR` | 0.3222 mm | 1.5742 | 0.9541 |

The single parameter `TONIC_FWD_MOTOR` is sufficient to reproduce the straight
body and removing only that key from the full evolved config restores normal
curvature.

## Mechanism

`CElegansNervousSystem._inject_tonic_forward()` adds `_TONIC_FWD_MOTOR` directly
to every forward body-wall motor neuron with prefix `DB` or `VB` on every neural
tick.

The motor decoder then maps:

- `DB` to dorsal excitation.
- `VB` to ventral excitation.

A uniform tonic lift to both populations raises mean motor activation but
reduces the travelling dorsal-versus-ventral contrast that bends the body. The
result is an active but mechanically straight worm.

## Practical Rule

Do not use archived evolved configs with non-zero `TONIC_FWD_MOTOR` as evidence
of current body locomotion quality. For the current stack:

```json
"TONIC_FWD_MOTOR": 0.0
```

or remove the key from the config so the source default is used.

If evolution is rerun, exclude `TONIC_FWD_MOTOR` from the search space or add a
body-shape/curvature penalty to the objective so forward displacement cannot be
optimized by straight-body drive.

