"""Run-turn-run style analysis on a long ``worm_*.npz`` capture.

Uses heading and angular velocity (Pierce-Shimomura 1999 / Berri 2009 style).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("WORM_NPZ", "worm_300k.npz")
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("WORM_OUTDIR", "worm_plots_long"))
    outdir.mkdir(parents=True, exist_ok=True)
    d = np.load(inp, allow_pickle=True)
    ticks = d["ticks"]; ja = d["ja"]; fired = d["fired"]
    com = d["com"]; sm = d["sm"]
    yaw_idx = d["yaw_idx"]
    SIM_DT = 0.002
    sec = (ticks - ticks[0]) * SIM_DT
    n = len(ticks)
    seg = sm.reshape(n, -1, 3)
    print(f"loaded {n} frames over {sec[-1]:.1f}s sim time")

    # Heading direction
    head = seg[:, 0, :2]; tail = seg[:, -1, :2]
    fwd = (head - tail) / (np.linalg.norm(head - tail, axis=1, keepdims=True) + 1e-9)
    heading_rad = np.arctan2(fwd[:, 1], fwd[:, 0])
    heading_unwrapped = np.unwrap(heading_rad)

    # Angular velocity (smooth over ~0.5s window)
    fps = 1.0 / float(np.median(np.diff(ticks))) / SIM_DT
    win = max(1, int(round(0.5 * fps)))
    heading_smooth = np.convolve(heading_unwrapped, np.ones(win)/win, mode='same')
    dh = np.diff(heading_smooth) * fps  # rad/s
    dh_deg_s = np.degrees(dh)

    # Forward speed
    com_vel = np.diff(com[:, :2], axis=0) / np.diff(sec).reshape(-1, 1)
    fwd_speed_um_s = np.einsum('ij,ij->i', com_vel, fwd[:-1]) * 1000

    print(f"\nHeading change over full run: {np.degrees(heading_unwrapped[-1] - heading_unwrapped[0]):+.0f}°")
    print(f"Mean angular velocity: {np.mean(dh_deg_s):+.2f} °/s")
    print(f"Std angular velocity:  {np.std(dh_deg_s):.2f} °/s")
    print(f"Mean forward speed:    {np.mean(fwd_speed_um_s):+.1f} µm/s")

    # Run/turn detection
    abs_dh = np.abs(dh_deg_s)
    # Threshold: turn if angular speed > 30°/s (calibrate from data)
    threshold = 30.0
    is_turn = abs_dh > threshold
    # Smooth: turn requires sustained
    sustain = max(1, int(round(0.3 * fps)))
    smoothed_turn = np.convolve(is_turn.astype(float), np.ones(sustain)/sustain, mode='same') > 0.5

    # Find run/turn segments
    transitions = np.diff(smoothed_turn.astype(int))
    run_starts = np.where(transitions == -1)[0]  # turn → run
    run_ends   = np.where(transitions == 1)[0]   # run → turn
    print(f"\nTurn detection (|dθ/dt| > {threshold}°/s, sustained):")
    print(f"  Number of run→turn transitions: {len(run_ends)}")
    print(f"  Number of turn→run transitions: {len(run_starts)}")
    print(f"  Total time turning: {smoothed_turn.sum() / fps:.1f}s of {sec[-1]:.1f}s ({smoothed_turn.mean()*100:.1f}%)")

    if len(run_starts) > 0 and len(run_ends) > 0:
        # Pair up
        if run_ends[0] < run_starts[0]:
            # First segment was a run
            pairs = list(zip(run_starts[:-1], run_ends[1:])) if len(run_ends) > 1 else []
        else:
            pairs = list(zip(run_starts, run_ends))
        run_lengths_s = np.array([(e - s) / fps for s, e in pairs])
        if len(run_lengths_s) > 0:
            print(f"  Run segments: n={len(run_lengths_s)}  mean length {run_lengths_s.mean():.2f}s  "
                  f"median {np.median(run_lengths_s):.2f}s  max {run_lengths_s.max():.2f}s")

    # Plot 1: trajectory
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    ax = axes[0, 0]
    sc = ax.scatter(com[:, 0], com[:, 1], c=sec, cmap='viridis', s=0.5)
    ax.scatter([com[0, 0]], [com[0, 1]], c='green', s=80, edgecolor='black', label='start', zorder=5)
    ax.scatter([com[-1, 0]], [com[-1, 1]], c='red', s=80, edgecolor='black', label='end', zorder=5)
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_aspect('equal'); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title(f'Full trajectory ({sec[-1]:.0f}s sim time)')
    plt.colorbar(sc, ax=ax, label='sim time (s)')

    # Plot 2: heading over time
    ax = axes[0, 1]
    ax.plot(sec, np.degrees(heading_unwrapped) % 360 - 180, 'b-', alpha=0.5, lw=0.3, label='heading (mod 360°)')
    ax.set_xlabel('sim time (s)'); ax.set_ylabel('heading (°)')
    ax.set_title('Heading angle over time')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: angular speed
    ax = axes[1, 0]
    ax.plot(sec[:-1], abs_dh, 'k-', alpha=0.4, lw=0.3, label='|dθ/dt|')
    ax.axhline(threshold, color='r', ls='--', alpha=0.5, label=f'turn threshold ({threshold}°/s)')
    ax.set_xlabel('sim time (s)'); ax.set_ylabel('|angular vel| (°/s)')
    ax.set_title('Angular speed (smoothed)')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 4: forward speed over time
    ax = axes[1, 1]
    ax.plot(sec[:-1], fwd_speed_um_s, 'g-', alpha=0.4, lw=0.3)
    ax.axhline(0, color='k', ls='-', lw=0.5)
    ax.set_xlabel('sim time (s)'); ax.set_ylabel('forward speed (µm/s)')
    ax.set_title('Forward velocity over time')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / "long_run_overview.png", dpi=120)
    plt.close()

    # Plot 5: per-segment yaw kymograph (subsampled)
    yaws = ja[:, yaw_idx]
    skip = max(1, n // 5000)
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    im = ax.imshow(yaws[::skip].T, aspect='auto', cmap='RdBu_r',
                   extent=[sec[0], sec[-1], 12.5, 0.5], vmin=-0.35, vmax=0.35)
    ax.set_xlabel('sim time (s)'); ax.set_ylabel('yaw segment')
    ax.set_title(f'Joint yaw kymograph — {sec[-1]:.0f}s')
    plt.colorbar(im, ax=ax, label='yaw (rad)')
    plt.tight_layout()
    plt.savefig(outdir / "long_yaw_kymograph.png", dpi=120)
    plt.close()

    # Plot 6: 60-second windows of trajectory (look for behavioral states)
    n_windows = 6
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, ax in enumerate(axes.flat):
        win_dur = sec[-1] / n_windows
        t_start = i * win_dur; t_end = (i+1) * win_dur
        mask = (sec >= t_start) & (sec < t_end)
        com_w = com[mask]
        if len(com_w) > 1:
            ax.plot(com_w[:, 0], com_w[:, 1], 'b-', lw=0.5, alpha=0.7)
            ax.scatter([com_w[0, 0]], [com_w[0, 1]], c='green', s=40, edgecolor='black', zorder=5)
            ax.scatter([com_w[-1, 0]], [com_w[-1, 1]], c='red', s=40, edgecolor='black', zorder=5)
            ax.set_aspect('equal')
            ax.set_title(f't={t_start:.0f}-{t_end:.0f}s')
            ax.grid(True, alpha=0.3)
    plt.suptitle('Trajectory in 6 time windows — does behavior vary?')
    plt.tight_layout()
    plt.savefig(outdir / "long_window_traj.png", dpi=120)
    plt.close()

    print(f"\nplots saved to {outdir}")


if __name__ == "__main__":
    main()
