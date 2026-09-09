"""Static plot of the robust-navigator trajectories (path spiraling into food, coloured by steering)."""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

D = json.load(open("/Users/arterialist/.claude/jobs/4630c9cc/tmp/nav_traj.json"))
runs = D["runs"]; A = D["arena"]
fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), facecolor="white")
for ax, r in zip(axes.ravel(), runs):
    tr = np.array(r["traj"]); food = r["food"]
    xs, ys, steer = tr[:,0], tr[:,1], tr[:,3]
    pts = np.array([xs, ys]).T.reshape(-1,1,2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    # colour: low steer (straight, homing) = teal ; high steer (curving, searching) = orange
    lc = LineCollection(segs, cmap="viridis_r", array=steer[:-1], linewidth=1.8, alpha=0.9)
    ax.add_collection(lc)
    ax.plot(0, 0, "o", color="#3B4CC0", ms=9, label="start (nest)")
    ax.plot(food[0], food[1], "*", color="#1F7A6D", ms=20, markeredgecolor="k", label="food")
    circ = plt.Circle((food[0], food[1]), 0.8, fill=False, ls="--", color="#1F7A6D", alpha=0.5)
    ax.add_patch(circ)
    ax.set_xlim(-A-1, A+1); ax.set_ylim(-A-1, A+1); ax.set_aspect("equal")
    ax.set_title(f"food ({food[0]:.1f}, {food[1]:.1f})   reached t={r['reached']}", fontsize=10)
    ax.grid(alpha=0.2); ax.tick_params(labelsize=7)
axes[0,0].legend(loc="upper left", fontsize=8)
cbar = fig.colorbar(lc, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
cbar.set_label("neural steering rate  (low = homing straight · high = curving/searching)", fontsize=9)
fig.suptitle("Fully-neural rower — smooth-klinokinesis + tropotaxis homes to food from every direction (6/6)",
             fontsize=13, weight="bold")
fig.savefig("/Users/arterialist/.claude/jobs/4630c9cc/tmp/nav_trajectories.png", dpi=110, bbox_inches="tight")
print("saved nav_trajectories.png")
