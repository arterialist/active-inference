"""Standalone smoke test: A-MN intrinsic oscillator + AVA gating (no MuJoCo)."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from simulations.c_elegans.connectome import load_connectome
from simulations.c_elegans.neuron_mapping import CElegansNervousSystem


def main() -> None:
    print("Loading connectome…")
    connectome = load_connectome()
    print(f"  loaded: {len(connectome.neurons)} neurons")

    print("Building nervous system…")
    ns = CElegansNervousSystem(connectome=connectome, log_level="ERROR")
    print(f"  ready: A_MN_OSC_ENABLED={ns._A_MN_OSC_ENABLED}")

    a_mn_names = [n for n in ns._name_to_id if n.startswith(("DA", "VA"))]
    print(f"  A-MNs found: {len(a_mn_names)} ({sorted(a_mn_names)[:6]}…)")
    assert "DA1" in ns._name_to_id and "DA9" in ns._name_to_id
    assert "VA1" in ns._name_to_id and "VA12" in ns._name_to_id

    # --- Phase 1: rest (AVA at default ~+0.85 from connectome+tonic) ---
    # Run a few warmup ticks, then sample DA/VA S.
    print("\n--- Phase 1: AVA at rest (no extra drive) — A-MNs should be silent ---")
    for tick in range(200):
        ns.tick({}, current_tick=tick)
    rest_S = {n: ns.get_neuron_by_name(n).S for n in a_mn_names}
    rest_ava = (ns.get_neuron_by_name("AVAL").S + ns.get_neuron_by_name("AVAR").S) / 2.0
    rest_amn_max = max(abs(s) for s in rest_S.values())
    print(f"  AVA mean S = {rest_ava:+.3f}")
    print(f"  max |A-MN S| = {rest_amn_max:.4f}")
    print(f"  DA1={rest_S['DA1']:+.3f}  DA5={rest_S['DA5']:+.3f}  DA9={rest_S['DA9']:+.3f}")
    print(f"  VA1={rest_S['VA1']:+.3f}  VA6={rest_S['VA6']:+.3f}  VA12={rest_S['VA12']:+.3f}")

    # --- Phase 2: AVA forced burst — A-MNs should oscillate ---
    print("\n--- Phase 2: force AVA into burst (S=+1.5) — A-MNs should oscillate ---")
    ava_l_id = ns._name_to_id["AVAL"]
    ava_r_id = ns._name_to_id["AVAR"]
    sample_log: dict[str, list[float]] = {n: [] for n in ["DA1", "DA5", "DA9", "VA1", "VA6", "VA12", "AVAL"]}
    for tick in range(200, 200 + 3000):
        ns.tick({}, current_tick=tick)
        # Forcibly hold AVA depolarised so the gate stays open.
        ns._network.network.neurons[ava_l_id].S = 1.5
        ns._network.network.neurons[ava_r_id].S = 1.5
        for nm in sample_log:
            sample_log[nm].append(ns.get_neuron_by_name(nm).S)
    burst_S_arrays = {nm: np.asarray(v) for nm, v in sample_log.items()}
    for nm, arr in burst_S_arrays.items():
        print(f"  {nm}: mean={arr.mean():+.3f}  std={arr.std():.3f}  "
              f"min={arr.min():+.3f}  max={arr.max():+.3f}")

    # --- Phase 3: anti-phase + spatial wave checks ---
    print("\n--- Phase 3: anti-phase + spatial wave ---")
    da1 = burst_S_arrays["DA1"]; va1 = burst_S_arrays["VA1"]
    if da1.std() > 1e-3 and va1.std() > 1e-3:
        c = np.corrcoef(da1[500:], va1[500:])[0, 1]
        print(f"  DA1 vs VA1 corr = {c:+.3f} (expect ≈ -1: anti-phase)")
    else:
        print(f"  WARNING: A-MNs not oscillating (std too small) — gate may be off")

    da9 = burst_S_arrays["DA9"]; da5 = burst_S_arrays["DA5"]
    if da9.std() > 1e-3 and da1.std() > 1e-3:
        # Cross-correlation: DA9 should lead DA1 (tail-to-head wave).
        # Compute lag at peak corr.
        x = da9[500:] - da9[500:].mean()
        y = da1[500:] - da1[500:].mean()
        n = min(len(x), len(y))
        lags = np.arange(-100, 101)
        corrs = []
        for lag in lags:
            if lag < 0:
                xx = x[:lag]; yy = y[-lag:]
            elif lag > 0:
                xx = x[lag:]; yy = y[:-lag]
            else:
                xx = x; yy = y
            if len(xx) and len(yy):
                xx = xx[:len(yy)]; yy = yy[:len(xx)]
                if xx.std() > 0 and yy.std() > 0:
                    corrs.append(np.corrcoef(xx, yy)[0, 1])
                else: corrs.append(0.0)
            else: corrs.append(0.0)
        corrs = np.asarray(corrs)
        peak = lags[corrs.argmax()]
        print(f"  DA9→DA1 cross-corr peak at lag {peak} ticks (positive = DA9 leads DA1 = tail-to-head wave)")

    # --- Phase 4: gate off again — A-MNs should decay back to rest ---
    print("\n--- Phase 4: release AVA back to rest — A-MNs should fall silent ---")
    for tick in range(3200, 3200 + 500):
        ns.tick({}, current_tick=tick)
    after_S = {n: ns.get_neuron_by_name(n).S for n in a_mn_names}
    after_ava = (ns.get_neuron_by_name("AVAL").S + ns.get_neuron_by_name("AVAR").S) / 2.0
    after_amn_max = max(abs(s) for s in after_S.values())
    print(f"  AVA mean S = {after_ava:+.3f}  (should be back at rest ≈ {rest_ava:+.3f})")
    print(f"  max |A-MN S| = {after_amn_max:.4f}  (should be small)")

    # Pass/fail summary
    print("\n=== SUMMARY ===")
    p1_pass = rest_amn_max < 0.15
    print(f"  Phase 1 (silent at rest):   {'PASS' if p1_pass else 'FAIL'}  (max|S|={rest_amn_max:.3f}, threshold 0.15)")
    osc_pass = all(burst_S_arrays[n].std() > 0.05 for n in ("DA1", "DA9", "VA1", "VA12"))
    print(f"  Phase 2 (oscillates burst): {'PASS' if osc_pass else 'FAIL'}")
    if da1.std() > 1e-3 and va1.std() > 1e-3:
        antiphase = np.corrcoef(da1[500:], va1[500:])[0, 1] < -0.5
        print(f"  Phase 3 (DA/VA anti-phase): {'PASS' if antiphase else 'FAIL'}")


if __name__ == "__main__":
    main()
