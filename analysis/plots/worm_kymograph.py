"""Render kymographs and related figures from a lab ``worm_*.npz`` capture."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def zero_crossings(yaws, deadband=0.04):
    cnt = 0; prev = None
    for y in yaws:
        if abs(y) < deadband: continue
        s = 1 if y > 0 else -1
        if prev is not None and s != prev: cnt += 1
        prev = s
    return cnt


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("WORM_NPZ", "worm_capture.npz")
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("WORM_OUTDIR", "worm_plots"))
    outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(inp, allow_pickle=True)
    ticks = d["ticks"]
    ja = d["ja"]
    jv = d["jv"]
    ma = d["ma"]
    S = d["S"]
    fired = d["fired"]
    com = d["com"]
    fe = d["fe"] if "fe" in d.files else np.zeros(len(ticks))
    nm01 = d["nm01"] if "nm01" in d.files else np.zeros((len(ticks), 2))
    joint_names = list(d["joint_names"])
    muscle_names = list(d["muscle_names"])
    neuron_names = list(d["neuron_names"])
    yaw_idx = d["yaw_idx"]
    pitch_idx = d["pitch_idx"] if "pitch_idx" in d.files else np.array([], dtype=int)
    SIM_DT = 0.002  # s/tick

    n = len(ticks)
    sec = (ticks - ticks[0]) * SIM_DT
    print(f"loaded {n} frames; sim_t {sec[0]:.1f}–{sec[-1]:.1f}s")

    n_seg = 12
    yaws = ja[:, yaw_idx]            # (n, 12)
    yaw_velocities = jv[:, yaw_idx]  # (n, 12)

    # Muscle decomposition: (n_seg, 4) per tick.
    ma3 = ma.reshape(n, n_seg, 4)    # quadrants: DL, DR, VL, VR
    dorsal = ma3[..., :2].mean(axis=-1)   # (n, 12)
    ventral = ma3[..., 2:].mean(axis=-1)  # (n, 12)
    d_minus_v = dorsal - ventral

    # ============================== 1. Joint-yaw kymograph =============================
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    im = ax.imshow(yaws.T, aspect="auto", cmap="RdBu_r",
                   extent=[sec[0], sec[-1], n_seg + 0.5, 0.5],
                   vmin=-0.35, vmax=0.35)
    ax.set_xlabel("sim time (s)")
    ax.set_ylabel("yaw joint (1=neck → 12=tail)")
    ax.set_title(f"Joint-yaw kymograph — {n} ticks ≈ {sec[-1]-sec[0]:.0f}s sim time")
    plt.colorbar(im, ax=ax, label="yaw (rad)")
    plt.tight_layout()
    plt.savefig(outdir / "01_yaw_kymograph.png", dpi=120)
    plt.close()

    # 1b. Zoomed-in 10s window
    end = int(min(n, 10.0 / SIM_DT))
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    im = ax.imshow(yaws[:end].T, aspect="auto", cmap="RdBu_r",
                   extent=[sec[0], sec[end-1], n_seg + 0.5, 0.5],
                   vmin=-0.35, vmax=0.35)
    ax.set_xlabel("sim time (s)")
    ax.set_ylabel("yaw joint (1=neck → 12=tail)")
    ax.set_title("Joint-yaw kymograph — first 10s (wave propagation visible)")
    plt.colorbar(im, ax=ax, label="yaw (rad)")
    plt.tight_layout()
    plt.savefig(outdir / "01b_yaw_kymograph_10s.png", dpi=120)
    plt.close()

    # ============================== 2. Muscle kymograph (D-V antagonism) ===============
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    im0 = axes[0].imshow(dorsal.T, aspect="auto", cmap="Reds",
                          extent=[sec[0], sec[-1], n_seg + 0.5, 0.5], vmin=0, vmax=1)
    axes[0].set_ylabel("dorsal")
    axes[0].set_title(f"Muscle activations — {n} ticks ≈ {sec[-1]-sec[0]:.0f}s")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(ventral.T, aspect="auto", cmap="Blues",
                          extent=[sec[0], sec[-1], n_seg + 0.5, 0.5], vmin=0, vmax=1)
    axes[1].set_ylabel("ventral")
    plt.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(d_minus_v.T, aspect="auto", cmap="RdBu_r",
                          extent=[sec[0], sec[-1], n_seg + 0.5, 0.5], vmin=-1, vmax=1)
    axes[2].set_ylabel("dorsal − ventral")
    axes[2].set_xlabel("sim time (s)")
    plt.colorbar(im2, ax=axes[2], label="net push")
    plt.tight_layout()
    plt.savefig(outdir / "02_muscle_kymograph.png", dpi=120)
    plt.close()

    # 2b. Zoom 10s
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    im = ax.imshow(d_minus_v[:end].T, aspect="auto", cmap="RdBu_r",
                   extent=[sec[0], sec[end-1], n_seg + 0.5, 0.5], vmin=-1, vmax=1)
    ax.set_xlabel("sim time (s)")
    ax.set_ylabel("segment")
    ax.set_title("D−V kymograph (10s) — diagonal stripes = travelling wave")
    plt.colorbar(im, ax=ax, label="net push (D−V)")
    plt.tight_layout()
    plt.savefig(outdir / "02b_dv_kymograph_10s.png", dpi=120)
    plt.close()

    # ============================== 3. Motor neuron raster ============================
    motor_db = [f"DB{i}" for i in range(1, 8)]
    motor_vb = [f"VB{i}" for i in range(1, 12)]
    motor_dd = [f"DD{i}" for i in range(1, 7)]
    motor_vd = [f"VD{i}" for i in range(1, 14)]
    motor = motor_db + motor_vb + motor_dd + motor_vd
    name_to_id = {nm: i for i, nm in enumerate(neuron_names)}
    motor_ids = [name_to_id[m] for m in motor if m in name_to_id]
    motor_used = [m for m in motor if m in name_to_id]

    color_for = lambda nm: "red" if nm.startswith("DB") else "blue" if nm.startswith("VB") else "darkorange" if nm.startswith("DD") else "purple"
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    for row, mname in enumerate(motor_used):
        nid = name_to_id[mname]
        spikes = sec[fired[:, nid].astype(bool)]
        ax.scatter(spikes, np.full_like(spikes, row), s=4, c=color_for(mname), marker="|")
    ax.set_yticks(range(len(motor_used)))
    ax.set_yticklabels(motor_used)
    ax.set_xlabel("sim time (s)")
    ax.set_title(f"Motor neuron firing raster (DB red, VB blue, DD orange, VD purple) — {n} ticks")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(outdir / "03_motor_raster.png", dpi=120)
    plt.close()

    # ============================== 4. Motor neuron S kymograph =====================
    motor_S = np.stack([S[:, name_to_id[m]] for m in motor_used], axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    im = ax.imshow(motor_S, aspect="auto", cmap="RdBu_r",
                   extent=[sec[0], sec[-1], len(motor_used) - 0.5, -0.5],
                   vmin=-1, vmax=1)
    ax.set_yticks(range(len(motor_used)))
    ax.set_yticklabels(motor_used)
    ax.set_xlabel("sim time (s)")
    ax.set_title(f"B-type motor neuron membrane potential S — {n} ticks")
    plt.colorbar(im, ax=ax, label="S")
    plt.tight_layout()
    plt.savefig(outdir / "04_motor_S_kymograph.png", dpi=120)
    plt.close()

    # ============================== 5. Wave propagation =============================
    # Extract ridge: time at which each segment's yaw crosses zero (positive going).
    # Plot zero-crossing times per segment to estimate wave speed.
    dt = SIM_DT
    crossings_per_seg = []
    for s_idx in range(n_seg):
        y = yaws[:, s_idx]
        sign = np.sign(y)
        # positive-going zero crossings
        cross = np.where((sign[:-1] <= 0) & (sign[1:] > 0))[0]
        crossings_per_seg.append(cross)

    # Per-segment, take the first ~20 zero crossings and plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    K = 30
    for s_idx in range(n_seg):
        ts = sec[crossings_per_seg[s_idx][:K]] if len(crossings_per_seg[s_idx]) > 0 else np.array([])
        ax.scatter(ts, np.full_like(ts, s_idx + 1), s=12, c="green", marker="o", alpha=0.7)
    # Draw connecting lines for each "wave" (closest crossings across segments)
    if len(crossings_per_seg[0]) > 0 and len(crossings_per_seg[-1]) > 0:
        for k in range(min(K, len(crossings_per_seg[0]))):
            t0 = sec[crossings_per_seg[0][k]]
            xs = []
            ys = []
            for s_idx in range(n_seg):
                cs = sec[crossings_per_seg[s_idx]]
                if len(cs):
                    # find closest crossing to t0 + (s_idx-0)*0.05 estimated wave delay
                    j = np.argmin(np.abs(cs - t0))
                    xs.append(cs[j])
                    ys.append(s_idx + 1)
            ax.plot(xs, ys, "g-", alpha=0.15, lw=1)
    ax.set_xlabel("sim time (s)")
    ax.set_ylabel("segment")
    ax.set_title("Yaw zero-crossings (positive-going) per segment — slope = wave speed")
    ax.set_xlim(sec[0], sec[0] + min(60.0, sec[-1]-sec[0]))
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(outdir / "05_wave_propagation.png", dpi=120)
    plt.close()

    # ============================== 6. FFT of head/mid/tail ===============
    from scipy.signal import welch
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    # WS broadcasts at ~104Hz; compute actual fs from tick spacing.
    median_tick_dt = float(np.median(np.diff(ticks))) * SIM_DT
    fs_data = 1.0 / median_tick_dt
    for ax_, name, sig in [
        (axes[0], "head (j01)", yaws[:, 0]),
        (axes[1], "mid (j67)",  yaws[:, 6]),
        (axes[2], "tail (jbc)", yaws[:, 11]),
    ]:
        f, P = welch(sig, fs=fs_data, nperseg=min(8192, len(sig)//4))
        ax_.semilogy(f, P)
        ax_.set_xlim(0, 5)
        ax_.set_ylabel(f"{name}\nPSD (rad²/Hz)")
        ax_.grid(True, alpha=0.3)
    axes[-1].set_xlabel("frequency (Hz)")
    axes[0].set_title(f"Yaw PSD — expect peak near 0.6 Hz (head CPG); fs_data={fs_data:.0f} Hz")
    plt.tight_layout()
    plt.savefig(outdir / "06_fft.png", dpi=120)
    plt.close()

    # ============================== 7. COM trajectory =============================
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(com[:, 0], com[:, 1], "b-", lw=0.5, alpha=0.6)
    ax.scatter([com[0, 0]], [com[0, 1]], c="green", s=80, label="start", zorder=5)
    ax.scatter([com[-1, 0]], [com[-1, 1]], c="red", s=80, label="end", zorder=5)
    # plate boundary
    R = 50.0  # mm
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(R*np.cos(th), R*np.sin(th), "k-", alpha=0.3, lw=1)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    ax.set_title(f"Worm COM trajectory (worm length ≈ 1mm, plate r=50mm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "07_com_trajectory.png", dpi=120)
    plt.close()

    # ============================== 8. Summary stats overlay ============
    zc = np.array([zero_crossings(y) for y in yaws])
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(sec, zc, "k-", lw=0.5)
    axes[0].set_ylabel("zero crossings\n(2 = true S)")
    axes[0].set_yticks([0, 1, 2, 3, 4])
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sec, yaws[:, 0], "C1", label="head (j01)", alpha=0.7)
    axes[1].plot(sec, yaws[:, 6], "C2", label="mid (j67)", alpha=0.7)
    axes[1].plot(sec, yaws[:, 11], "C3", label="tail (jbc)", alpha=0.7)
    axes[1].set_ylabel("yaw (rad)")
    axes[1].axhline(0.30, ls="--", c="r", alpha=0.4)
    axes[1].axhline(-0.30, ls="--", c="r", alpha=0.4)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(sec, fe, "purple", lw=0.5)
    axes[2].set_ylabel("free energy")
    axes[2].set_xlabel("sim time (s)")
    axes[2].grid(True, alpha=0.3)

    axes[0].set_title(f"Locomotion summary — {n} ticks ≈ {sec[-1]-sec[0]:.0f}s sim time")
    plt.tight_layout()
    plt.savefig(outdir / "08_summary.png", dpi=120)
    plt.close()

    # ============================== 9. body-shape gallery (8 reps) =====
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    # Sample 8 frames evenly across the long-run window
    for k, ax_ in enumerate(axes.flat):
        i = int(n * (k + 0.5) / 8)
        # Build (segment, yaw) into a curve representation. Use cumulative angles.
        cum_angle = np.concatenate([[0], np.cumsum(yaws[i])])
        x = np.cumsum(np.cos(cum_angle))
        y = np.cumsum(np.sin(cum_angle))
        ax_.plot(x, y, "b-", lw=4, alpha=0.8)
        ax_.scatter([x[0]], [y[0]], c="green", s=60, zorder=5, label="head")
        ax_.scatter([x[-1]], [y[-1]], c="red", s=60, zorder=5, label="tail")
        ax_.set_aspect("equal")
        ax_.set_title(f"t={sec[i]:.1f}s tick={ticks[i]} zc={zc[i]}")
        ax_.grid(True, alpha=0.3)
        if k == 0:
            ax_.legend(fontsize=8, loc="upper right")
    plt.suptitle(f"Body shape gallery — 8 frames sampled from {n} ticks")
    plt.tight_layout()
    plt.savefig(outdir / "09_body_shapes.png", dpi=120)
    plt.close()

    # ============================== 10. statistics print =====
    pct_eq2 = (zc == 2).mean() * 100
    pct_ge2 = (zc >= 2).mean() * 100
    pct_eq1 = (zc == 1).mean() * 100
    pct_eq0 = (zc == 0).mean() * 100
    pct_ge3 = (zc >= 3).mean() * 100
    print(f"\nzc distribution: 0={pct_eq0:.1f}%, 1={pct_eq1:.1f}%, 2={pct_eq2:.1f}%, ≥3={pct_ge3:.1f}%")
    print(f"  pct ≥2 (true S+): {pct_ge2:.1f}%")
    print(f"  per-yaw amp range: {(yaws.max(0) - yaws.min(0)).round(2)}")
    print(f"  per-yaw mean: {yaws.mean(0).round(3)}")
    com_dist = np.linalg.norm(com[-1, :2] - com[0, :2])
    com_path = np.sum(np.linalg.norm(np.diff(com[:, :2], axis=0), axis=1))
    print(f"  COM displacement: {com_dist:.2f} mm  path length: {com_path:.2f} mm")
    print(f"  saved plots → {outdir}")


if __name__ == "__main__":
    main()
