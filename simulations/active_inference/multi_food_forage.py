"""Multi-food foraging with the fully-neural agent (neural_agent.py). Several odour sources; the agent
senses the SUMMED field, navigates to the nearest strong source, consumes it on contact (removing it from
the field), and re-navigates to the next — all via the same neural wiring. Records the trajectory + food
states to JSON for animation."""
import sys, json, numpy as np
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
import importlib.util
def _l(n,p): s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
NA=_l("na", str(__import__("pathlib").Path(__file__).parent / "neural_agent.py"))

FOODS=[(3.5,1.5),(-3.0,2.8),(-2.0,-3.2),(3.0,-2.6),(0.2,4.2)]   # 5 sources around the start
STRIDE=15; TMAX=16000

def run():
    ag=NA.NeuralAgent(foods=FOODS)
    frames=[]; consumed_at={}
    for t in range(TMAX):
        ag.step(); x,y,yaw=ag.pose()
        active=[bool(f[2]) for f in ag.foods]
        # note consumption events
        for i,f in enumerate(ag.foods):
            if not f[2] and i not in consumed_at: consumed_at[i]=t
        if t%STRIDE==0:
            frames.append([round(x,3),round(y,3),round(yaw,3),active])
        if all(not f[2] for f in ag.foods):
            frames.append([round(x,3),round(y,3),round(yaw,3),[False]*len(FOODS)])
            break
    n_eaten=sum(1 for f in ag.foods if not f[2])
    out={"foods":[list(f) for f in FOODS],"frames":frames,"consumed_at":consumed_at,
         "n_eaten":n_eaten,"n_total":len(FOODS),"stride":STRIDE,"sub":ag.sub}
    p=str(__import__("pathlib").Path(__file__).parent / "multi_food_traj.json")
    json.dump(out,open(p,"w"))
    order=sorted(consumed_at.items(), key=lambda kv: kv[1])
    print(f"foraged {n_eaten}/{len(FOODS)} sources; consumption order (food_idx@step): "
          f"{[(i,s) for i,s in order]}", flush=True)
    print(f"frames={len(frames)} saved to {p}", flush=True)
    print("@@@FORAGE DONE@@@", flush=True)

if __name__=="__main__":
    run()
