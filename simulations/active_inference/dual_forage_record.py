"""Records the dual-channel agent foraging a BIGGER mixed world: many FOOD sources (attractant, consumed
on contact) and TOXINS (repellent, short-range, persistent). Saves trajectory + entity states to JSON."""
import sys, json, numpy as np, pathlib
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
import importlib.util
def _l(n,p): s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
DA=_l("da", str(pathlib.Path(__file__).parent/"dual_chemotaxis.py"))

rng=np.random.default_rng(11)
def scatter(n,rlo,rhi,avoid=(),amin=1.8):
    out=[]
    while len(out)<n:
        a=rng.uniform(0,2*np.pi); r=rng.uniform(rlo,rhi); p=(float(r*np.cos(a)),float(r*np.sin(a)))
        if all((p[0]-q[0])**2+(p[1]-q[1])**2>amin**2 for q in list(out)+list(avoid)): out.append(p)
    return out
FOODS=scatter(9,3.0,11.0)
TOXINS=scatter(6,3.0,11.0,avoid=FOODS,amin=2.2)
STRIDE=18; TMAX=30000

def run():
    ag=DA.DualAgent(foods=FOODS, toxins=TOXINS)
    frames=[]; eaten_at={}
    for t in range(TMAX):
        ag.step(); x,y,yaw=ag.pose()
        for i,f in enumerate(ag.foods):
            if not f[2] and i not in eaten_at: eaten_at[i]=t
        if t%STRIDE==0:
            frames.append([round(x,3),round(y,3),round(yaw,3),[bool(f[2]) for f in ag.foods]])
        if ag.eaten>=len(FOODS): break
    frames.append([round(ag.pose()[0],3),round(ag.pose()[1],3),round(ag.pose()[2],3),[bool(f[2]) for f in ag.foods]])
    out={"foods":[list(f) for f in FOODS],"toxins":[list(t) for t in TOXINS],"frames":frames,
         "eaten_at":eaten_at,"n_eaten":ag.eaten,"n_food":len(FOODS),"n_toxin":len(TOXINS),
         "toxin_hits":ag.hits,"stride":STRIDE,"sub":ag.sub,"sigma":ag.sigma,"sigma_tox":ag.sigma_tox}
    p=str(pathlib.Path(__file__).parent/"dual_forage_traj.json")
    json.dump(out,open(p,"w"))
    print(f"ate {ag.eaten}/{len(FOODS)} foods; toxin contacts={ag.hits}; frames={len(frames)}", flush=True)
    print(f"eaten order (idx@step): {sorted(eaten_at.items(), key=lambda kv:kv[1])}", flush=True)
    print("@@@REC DONE@@@", flush=True)

if __name__=="__main__": run()
