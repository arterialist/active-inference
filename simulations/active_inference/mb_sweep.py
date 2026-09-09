"""MUSHROOM-BODY environment SWEEP — runs ONE scenario (argv[1]) of the embodied valence-learning agent and
logs its learning curve. Launch several in parallel (separate cores) to see how the new plasticity mechanism
behaves across the contact-frequency space. Key question (user): the agent should FORGET danger when it is
absent/no-longer-reinforced, but must NOT forget danger under constant exposure+reinforcement.

Scenarios:
  food_only      : no toxins            -> aversion must stay 0 (no false danger, no drift)   [control]
  sparse_toxin   : few toxins, much food-> rare contact -> aversion flickers (extinction wins)
  balanced       : equal food & toxin
  dense_toxin    : many toxins, little food -> FREQUENT contact -> aversion BUILDS & PERSISTS  [constant-exposure test]
  gauntlet       : food ring OUTSIDE a toxin ring -> forced frequent contact -> strong persistent aversion
  toxin_removed  : dense toxins for the first half, then REMOVED -> learn, then danger gone
"""
import sys, json, time, numpy as np, pathlib
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
import importlib.util
def _l(n,p): s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
MA=_l("ma", str(pathlib.Path(__file__).parent/"mushroom_agent.py"))
D=pathlib.Path(__file__).parent
GOOD,BAD=0,1; SUB=16; STRIDE=40; PROG_EVERY=2500

SCEN={
 "food_only":       dict(nf=16, nt=0,  rlo=3, rhi=9,  ticks=500000, remove_at=None,    gauntlet=False),
 "sparse_toxin":    dict(nf=14, nt=6,  rlo=3, rhi=9,  ticks=600000, remove_at=None,    gauntlet=False),
 "minefield":       dict(nf=10, nt=44, rlo=2.5,rhi=8, ticks=600000, remove_at=None,    gauntlet=False),
 "toxin_removed":   dict(nf=8,  nt=40, rlo=2.5,rhi=8, ticks=700000, remove_at=350000,  gauntlet=False),
 # sharper odorant IDENTITY (small sigma_id) so only the actually-contacted source drives the PN ->
 # should resolve the cross-contamination that made food aversive in the dense minefield.
 "minefield_sharp": dict(nf=10, nt=44, rlo=2.5,rhi=8, ticks=600000, remove_at=None,    gauntlet=False, sigma_id=1.8),
}

def build_world(cfg, rng):
    src=[]
    def rp(avoid,amin,rlo,rhi):
        for _ in range(300):
            a=rng.uniform(0,2*np.pi); r=rng.uniform(rlo,rhi); p=(float(r*np.cos(a)),float(r*np.sin(a)))
            if all((p[0]-q[0])**2+(p[1]-q[1])**2>amin**2 for q in avoid): return p
        return p
    if cfg["gauntlet"]:
        for _ in range(cfg["nf"]):                    # food in an OUTER ring
            x,y=rp([(s[0],s[1]) for s in src],1.6,7.0,9.5); src.append([x,y,GOOD,True,True])
        for _ in range(cfg["nt"]):                    # toxins in an INNER ring the agent must cross
            x,y=rp([(s[0],s[1]) for s in src],1.4,3.0,5.0); src.append([x,y,BAD,False,True])
    else:
        for _ in range(cfg["nf"]): x,y=rp([(s[0],s[1]) for s in src],1.8,cfg["rlo"],cfg["rhi"]); src.append([x,y,GOOD,True,True])
        for _ in range(cfg["nt"]): x,y=rp([(s[0],s[1]) for s in src],2.0,cfg["rlo"],cfg["rhi"]); src.append([x,y,BAD,False,True])
    return src

def run(name):
    SEED={"food_only":1,"sparse_toxin":2,"minefield":3,"minefield_sharp":4,"toxin_removed":5}
    cfg=SCEN[name]; rng=np.random.default_rng(100+SEED.get(name,0))   # fixed seed -> reproducible worlds
    src=build_world(cfg,rng); ag=MA.MBAgent(sources=src, sigma_id=cfg.get("sigma_id",4.0))
    steps=cfg["ticks"]//SUB; remove_step=(cfg["remove_at"]//SUB) if cfg["remove_at"] else None
    curve=[]; frames=[]; FRAMESTRIDE=40; t0=time.time(); last_e=last_h=0; removed=False
    tox_static=[[round(s[0],2),round(s[1],2)] for s in ag.sources if s[2]==BAD]
    print(f"[{name}] world: {cfg['nf']} food + {cfg['nt']} toxin, horizon {cfg['ticks']} ticks; "
          f"initial aversion good={ag.mbon_rate(GOOD)} toxic={ag.mbon_rate(BAD)}", flush=True)
    def respawn():
        for s in ag.sources:
            if s[2]==GOOD and not s[4]:
                if cfg["gauntlet"]: a=rng.uniform(0,2*np.pi); r=rng.uniform(7.0,9.5)
                else: a=rng.uniform(0,2*np.pi); r=rng.uniform(cfg["rlo"],cfg["rhi"])
                s[0],s[1],s[4]=float(r*np.cos(a)),float(r*np.sin(a)),True
    for t in range(steps):
        if remove_step and not removed and t>=remove_step:
            for s in ag.sources:
                if s[2]==BAD: s[4]=False   # danger GONE (odour disappears)
            removed=True
        ag.step(); respawn()
        if t%FRAMESTRIDE==0:
            x,y,yaw=ag.pose()
            fpos=[[round(s[0],1),round(s[1],1)] for s in ag.sources if s[2]==GOOD and s[4]]
            txlive=1 if any(s[2]==BAD and s[4] for s in ag.sources) else 0   # toxins present? (0 after removal)
            frames.append([round(x,2),round(y,2),round(yaw,3),fpos,ag.eaten,ag.tox_hits,txlive])
        if t%PROG_EVERY==0:
            de=ag.eaten-last_e; dh=ag.tox_hits-last_h; last_e,last_h=ag.eaten,ag.tox_hits
            vg,vb=ag.mbon_rate(GOOD),ag.mbon_rate(BAD)
            curve.append({"ticks":t*SUB,"eaten_cum":ag.eaten,"toxin_cum":ag.tox_hits,
                          "toxin_win":dh,"aversion_good":vg,"aversion_toxic":vb,"removed":removed})
            json.dump({"name":name,"cfg":cfg,"curve":curve}, open(D/f"sweep_{name}.json","w"))
            json.dump({"name":name,"cfg":cfg,"frames":frames,"toxins":tox_static,"curve":curve,
                       "framestride":FRAMESTRIDE,"sub":SUB}, open(D/f"sweep_{name}_traj.json","w"))
            if t%(PROG_EVERY*4)==0:
                rate=(t+1)/max(time.time()-t0,1e-6)
                print(f"[{name}] ticks={t*SUB} eaten={ag.eaten} toxinX={ag.tox_hits}(+{dh}) "
                      f"| aversion good={vg} toxic={vb}{' [toxin removed]' if removed else ''} eta={(steps-t)/max(rate,1e-6)/60:.0f}m", flush=True)
    fin={"good":ag.mbon_rate(GOOD),"toxic":ag.mbon_rate(BAD)}
    json.dump({"name":name,"cfg":cfg,"curve":curve,"final":fin,
               "n_eaten":ag.eaten,"toxin_crossings":ag.tox_hits}, open(D/f"sweep_{name}.json","w"))
    json.dump({"name":name,"cfg":cfg,"frames":frames,"toxins":tox_static,"curve":curve,
               "framestride":FRAMESTRIDE,"sub":SUB,"final":fin}, open(D/f"sweep_{name}_traj.json","w"))
    print(f"[{name}] DONE eaten={ag.eaten} toxinX={ag.tox_hits} final aversion good={ag.mbon_rate(GOOD)} toxic={ag.mbon_rate(BAD)}", flush=True)
    print(f"@@@{name} DONE@@@", flush=True)

if __name__=="__main__": run(sys.argv[1])
