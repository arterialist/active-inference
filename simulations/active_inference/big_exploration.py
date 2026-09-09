"""Long open-ended exploration: dual-channel agent in a BIG DENSE mixed world (40 food + 30 toxin),
running 2,000,000 neural ticks. Eaten food RESPAWNS at a fresh location so foraging stays open-ended.
Records a strided trajectory + periodic progress to disk so it can be monitored while running."""
import sys, json, time, numpy as np, pathlib
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
import importlib.util
def _l(n,p): s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
DA=_l("da", str(pathlib.Path(__file__).parent/"dual_chemotaxis.py"))

D=pathlib.Path(__file__).parent
rng=np.random.default_rng(23)
RLO,RHI=3.0,16.0                      # big world
NFOOD,NTOX=40,30                      # dense
TARGET_TICKS=2_000_000
SUB=16
STEPS=TARGET_TICKS//SUB              # 125,000 outer steps
STRIDE=40                            # ~3125 trajectory frames
PROG_EVERY=4000                     # progress print cadence (outer steps)

def rand_pt(avoid=(), amin=1.6):
    for _ in range(200):
        a=rng.uniform(0,2*np.pi); r=rng.uniform(RLO,RHI); p=(float(r*np.cos(a)),float(r*np.sin(a)))
        if all((p[0]-q[0])**2+(p[1]-q[1])**2>amin**2 for q in avoid): return p
    return p

def run():
    foods=[];
    for _ in range(NFOOD): foods.append(rand_pt(foods))
    toxins=[]
    for _ in range(NTOX): toxins.append(rand_pt(foods+toxins, amin=2.0))
    ag=DA.DualAgent(foods=foods, toxins=toxins)
    tox_xy=[(t[0],t[1]) for t in ag.toxins]
    frames=[]; t0=time.time(); last_eaten=0; total_respawns=0
    min_tox_clear=9.0; tox_latch=[False]*len(tox_xy); discrete_cross=0
    for t in range(STEPS):
        ag.step(); x,y,yaw=ag.pose()
        # respawn any newly-eaten food at a fresh spot (keeps the field dense/open-ended)
        for f in ag.foods:
            if not f[2]:
                np_=rand_pt([(fx,fy) for fx,fy,a in ag.foods if a]+tox_xy, amin=1.6)
                f[0],f[1],f[2]=np_[0],np_[1],True; total_respawns+=1
        # closest approach + DISCRETE toxin crossings (enter<0.8 / leave>1.5 latch) -- ag.hits counts
        # every tick within radius, which over-counts lingering; discrete_cross counts distinct crossings.
        for ti,tx in enumerate(tox_xy):
            d=((x-tx[0])**2+(y-tx[1])**2)**0.5
            if d<min_tox_clear: min_tox_clear=d
            if d<0.8 and not tox_latch[ti]: tox_latch[ti]=True; discrete_cross+=1
            elif d>1.5 and tox_latch[ti]: tox_latch[ti]=False
        if t%STRIDE==0:
            fpos=[[round(f[0],1),round(f[1],1)] for f in ag.foods if f[2]]   # active food positions this frame (DualAgent food = [x,y,active])
            frames.append([round(x,2),round(y,2),round(yaw,3),ag.eaten,discrete_cross,fpos])
        if t%PROG_EVERY==0:
            el=time.time()-t0; rate=(t+1)/max(el,1e-6)
            prog={"outer_step":t,"neural_ticks":t*SUB,"eaten":ag.eaten,"respawns":total_respawns,
                  "toxin_contacts":ag.hits,"min_tox_clearance":round(min_tox_clear,2),
                  "pos":[round(x,2),round(y,2)],"elapsed_s":round(el),"steps_per_s":round(rate,1),
                  "eta_min":round((STEPS-t)/max(rate,1e-6)/60,1)}
            json.dump({"progress":prog,"foods":[list(f) for f in ag.foods],"toxins":[list(t) for t in ag.toxins]},
                      open(D/"big_exploration_progress.json","w"))
            print(f"t={t} ticks={t*SUB} eaten={ag.eaten} hits={ag.hits} clr={min_tox_clear:.2f} "
                  f"rate={rate:.0f}/s eta={prog['eta_min']}min", flush=True)
    out={"foods_final":[list(f) for f in ag.foods],"toxins":[list(t) for t in ag.toxins],
         "frames":frames,"n_eaten":ag.eaten,"n_respawns":total_respawns,
         "toxin_contacts_pertick":ag.hits,"toxin_crossings":discrete_cross,
         "min_tox_clearance":round(min_tox_clear,2),"stride":STRIDE,"sub":SUB,
         "sigma":ag.sigma,"sigma_tox":ag.sigma_tox,"total_ticks":STEPS*SUB}
    json.dump(out, open(D/"big_exploration_traj.json","w"))
    print(f"DONE ticks={STEPS*SUB} eaten={ag.eaten} respawns={total_respawns} toxin_contacts={ag.hits} "
          f"min_clearance={min_tox_clear:.2f} frames={len(frames)}", flush=True)
    print("@@@BIGEXP DONE@@@", flush=True)

if __name__=="__main__": run()
