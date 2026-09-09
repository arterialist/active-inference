"""The neural navigator foraging among MULTIPLE food sources: odour = sum of all active sources; it homes
on the nearest, consumes it (source vanishes), the gradient shifts, it moves to the next. All spiking."""
import sys, json, numpy as np
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
import importlib.util
def _l(n,p): s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
NS=_l("ns","/Users/arterialist/.claude/jobs/4630c9cc/tmp/nav_smooth.py")

class MultiForager(NS.NavSmooth):
    def __init__(self, foods, sigma=14.0, **kw):
        super().__init__(food=foods[0], sigma=sigma, **kw)
        self.foods=[[float(fx),float(fy),True] for fx,fy in foods]
    def odor(self,p):                                       # combined field over ACTIVE sources
        return float(sum(np.exp(-((p[0]-fx)**2+(p[1]-fy)**2)/self.sigma) for fx,fy,act in self.foods if act))

if __name__=="__main__":
    rng=np.random.RandomState(7)
    foods=[(rng.uniform(-4.5,4.5),rng.uniform(-4.5,4.5)) for _ in range(6)]
    f=MultiForager(foods)
    traj=[]; eats=[]; eaten=0
    for t in range(90000):
        f.step(); x,y,yaw=f.rower.pose()
        if t%25==0: traj.append([round(float(x),3),round(float(y),3),round(float(yaw),3),round(f.steer_f,2)])
        for i,(fx,fy,act) in enumerate(f.foods):
            if act and (x-fx)**2+(y-fy)**2 < 0.7**2:
                f.foods[i][2]=False; eaten+=1; eats.append([len(traj)-1, i])
                print(f"  t{t}: ate food {i} at ({fx:.1f},{fy:.1f}) — {eaten}/6 eaten", flush=True)
        if eaten>=6: print(f"  ALL EATEN at t{t}", flush=True); break
    json.dump({"foods":[[fx,fy] for fx,fy,_ in f.foods], "traj":traj, "eats":eats, "arena":5.5, "eaten":eaten},
              open("/Users/arterialist/.claude/jobs/4630c9cc/tmp/forage_traj.json","w"))
    print(f"saved forage_traj.json: {len(traj)} pts, {eaten}/6 eaten")
    print("@@@FORAGE DONE@@@")
