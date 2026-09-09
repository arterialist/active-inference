"""Long-horizon EMBODIED MUSHROOM-BODY learning: the dual-odorant agent (mushroom_agent.MBAgent) forages a
BIG DENSE world of FOOD (odorant 0, nutritious, respawns) and TOXIN (odorant 1, aversive, persistent) for
2,000,000 neural ticks. The mushroom body learns odour valence from experience (cortisol on toxin contact
potentiates aversion, dopamine on food suppresses it; shock-driven teaching + slow-decay consolidation).
Over the long horizon the agent should accumulate enough toxin encounters for a LEARNING CURVE: early =
naive, many toxin crossings; later = learned aversion -> fewer crossings. Logs progress + a trajectory."""
import sys, json, time, numpy as np, pathlib
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
import importlib.util
def _l(n,p): s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
MA=_l("ma", str(pathlib.Path(__file__).parent/"mushroom_agent.py"))
D=pathlib.Path(__file__).parent

rng=np.random.default_rng(31)
RLO,RHI=3.0,11.0
NFOOD,NTOX=16,12
GOOD,BAD=0,1
TARGET_TICKS=2_000_000; SUB=16; STEPS=TARGET_TICKS//SUB
STRIDE=40; PROG_EVERY=4000

def rand_pt(avoid=(), amin=1.8):
    for _ in range(200):
        a=rng.uniform(0,2*np.pi); r=rng.uniform(RLO,RHI); p=(float(r*np.cos(a)),float(r*np.sin(a)))
        if all((p[0]-q[0])**2+(p[1]-q[1])**2>amin**2 for q in avoid): return p
    return p

def run():
    src=[]
    for _ in range(NFOOD): x,y=rand_pt([(s[0],s[1]) for s in src]); src.append([x,y,GOOD,True,True])
    for _ in range(NTOX):  x,y=rand_pt([(s[0],s[1]) for s in src],amin=2.2); src.append([x,y,BAD,False,True])
    ag=MA.MBAgent(sources=src)
    tox_xy=[(s[0],s[1]) for s in ag.sources if s[2]==BAD]
    frames=[]; t0=time.time(); curve=[]
    last_eat=0; last_hit=0; min_clear=9.0
    print(f"world: {NFOOD} food + {NTOX} toxin (radius {RLO}-{RHI}); horizon {TARGET_TICKS} ticks", flush=True)
    print(f"initial aversion: good={ag.mbon_rate(GOOD)} toxic={ag.mbon_rate(BAD)}", flush=True)
    for t in range(STEPS):
        ag.step(); x,y,yaw=ag.pose()
        for s in ag.sources:                       # respawn eaten FOOD (keeps foraging open-ended)
            if s[2]==GOOD and not s[4]:
                nx,ny=rand_pt([(q[0],q[1]) for q in ag.sources if q[4]]+tox_xy,amin=1.6); s[0],s[1],s[4]=nx,ny,True
        for tx in tox_xy:
            d=((x-tx[0])**2+(y-tx[1])**2)**0.5
            if d<min_clear: min_clear=d
        if t%STRIDE==0:
            fpos=[[round(s[0],1),round(s[1],1)] for s in ag.sources if s[2]==GOOD and s[4]]
            frames.append([round(x,2),round(y,2),round(yaw,3),ag.eaten,ag.tox_hits,fpos])
        if t%PROG_EVERY==0 and t>0:
            de=ag.eaten-last_eat; dh=ag.tox_hits-last_hit; last_eat,last_hit=ag.eaten,ag.tox_hits
            vg,vb=ag.mbon_rate(GOOD),ag.mbon_rate(BAD)
            el=time.time()-t0; rate=(t+1)/max(el,1e-6)
            curve.append({"ticks":t*SUB,"eaten_cum":ag.eaten,"toxin_cum":ag.tox_hits,
                          "eaten_win":de,"toxin_win":dh,"aversion_good":vg,"aversion_toxic":vb})
            json.dump({"curve":curve,"toxins":tox_xy}, open(D/"big_mushroom_progress.json","w"))
            print(f"t={t} ticks={t*SUB} eaten(+{de})={ag.eaten} toxinX(+{dh})={ag.tox_hits} "
                  f"| aversion good={vg} toxic={vb} | rate={rate:.0f}/s eta={(STEPS-t)/max(rate,1e-6)/60:.0f}min", flush=True)
    out={"frames":frames,"toxins":tox_xy,"curve":curve,"n_eaten":ag.eaten,"toxin_crossings":ag.tox_hits,
         "min_tox_clearance":round(min_clear,2),"stride":STRIDE,"sub":SUB,
         "sigma":ag.sigma,"sigma_tox":ag.sigma_id,"total_ticks":STEPS*SUB,
         "final_aversion":{"good":ag.mbon_rate(GOOD),"toxic":ag.mbon_rate(BAD)}}
    json.dump(out, open(D/"big_mushroom_traj.json","w"))
    print(f"DONE ticks={STEPS*SUB} eaten={ag.eaten} toxin_crossings={ag.tox_hits} "
          f"final_aversion good={out['final_aversion']['good']} toxic={out['final_aversion']['toxic']}", flush=True)
    print("@@@BIGMB DONE@@@", flush=True)

if __name__=="__main__": run()
