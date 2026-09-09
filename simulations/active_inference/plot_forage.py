"""Static plot of the multi-source foraging run: path weaving between food sources, numbered by eat order."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

D=json.load(open("/Users/arterialist/.claude/jobs/4630c9cc/tmp/forage_traj.json"))
tr=np.array(D["traj"]); foods=D["foods"]; eats=D["eats"]; A=D["arena"]
fig,ax=plt.subplots(figsize=(8.5,8.5),facecolor="white")
xs,ys,steer=tr[:,0],tr[:,1],tr[:,3]
pts=np.array([xs,ys]).T.reshape(-1,1,2); segs=np.concatenate([pts[:-1],pts[1:]],axis=1)
# colour path by TIME (progress) so the foraging order is legible
lc=LineCollection(segs,cmap="plasma",array=np.arange(len(segs)),linewidth=1.6,alpha=0.85)
ax.add_collection(lc)
ax.plot(0,0,"o",color="#3B4CC0",ms=11,label="start",zorder=5)
eat_order={fi:k for k,(ti,fi) in enumerate(eats)}
for i,(fx,fy) in enumerate(foods):
    eaten = i in eat_order
    ax.plot(fx,fy,"*",color=("#1F7A6D" if eaten else "#9A9DA5"),ms=22,markeredgecolor="k",zorder=6)
    if eaten: ax.annotate(str(eat_order[i]+1),(fx,fy),fontsize=11,weight="bold",ha="center",va="center",color="white",zorder=7)
ax.set_xlim(-A-1,A+1); ax.set_ylim(-A-1,A+1); ax.set_aspect("equal"); ax.grid(alpha=0.2)
ax.set_title(f"Neural forager visits {len(eats)}/{len(foods)} food sources  (★ numbered by eat order)",fontsize=13,weight="bold")
ax.plot([],[],"*",color="#1F7A6D",ms=16,markeredgecolor="k",label="food (eaten)")
ax.legend(loc="upper left",fontsize=9)
cbar=fig.colorbar(lc,ax=ax,shrink=0.7,pad=0.02); cbar.set_label("time (foraging progress)",fontsize=9)
fig.savefig("/Users/arterialist/.claude/jobs/4630c9cc/tmp/forage_path.png",dpi=110,bbox_inches="tight")
print("saved forage_path.png")
