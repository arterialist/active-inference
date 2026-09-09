"""Insect brain — REGION 3: ACTION SELECTION (basal-ganglia-like winner-take-all over drives).
Behavioural commands compete via mutual inhibition; internal drives feed them and the winner sets the
behavioural mode, so the organism switches context-dependently (flee overrides forage when threatened)
instead of running one fixed reflex.

This module provides two selectors:
  - Selector : the original 3-way (forage/flee/explore) with the drive formula baked in (used by the
               earlier organism versions; kept for the standalone self-test below).
  - Sel4     : a general 4-way WTA (forage/flee/home/explore) whose `select(drives)` takes EXPLICIT drive
               values, so the caller owns the drive dynamics. This is what `organism.Organism` uses.

Run standalone:  python -m simulations.organism.action_selection
"""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from simulations.paula_loader import ensure_paula_available
ensure_paula_available()
from paula_agent import ckit as k

# ---- original 3-way selector ----
FORAGE,FLEE,EXPLORE = 1,2,3
def build_selector():
    ne=[k.neuron(i,r=0.6,lam=5,c=2) for i in (FORAGE,FLEE,EXPLORE)]
    sy=[]; conns=[]
    for d in (FORAGE,FLEE,EXPLORE):
        sy.append(k.syn(d,0,2.2,1))                       # syn0 = drive input
        others=[o for o in (FORAGE,FLEE,EXPLORE) if o!=d]
        for j,o in enumerate(others): sy.append(k.syn(d,1+j,-15.0,1))   # mutual inhibition
        sy.append(k.term(d,tid=900+d))
    for d in (FORAGE,FLEE,EXPLORE):
        others=[o for o in (FORAGE,FLEE,EXPLORE) if o!=d]
        for j,o in enumerate(others): conns.append(k.conn(o,d,1+j,stid=900+o))
    ex=[k.ext(d,0) for d in (FORAGE,FLEE,EXPLORE)]
    return k.build(ne,sy,conns,ex)

class Selector:
    def __init__(self):
        self.path=build_selector(); self.net,self.core=k.load(self.path); self.nb={n:u for n,u in self.net.network.neurons.items()}
    def select(self, hunger, fear, T=14):
        drives={FORAGE:1.6+2.4*hunger, FLEE:7.0*fear, EXPLORE:2.0}   # clean drives; fear weighted highest
        self.net.reset_simulation(); self.core.state.current_tick=0; self.net.current_tick=0
        cnt={FORAGE:0,FLEE:0,EXPLORE:0}
        for t in range(T):
            for d,a in drives.items(): self.net.set_external_input(d,0,a)
            self.core.do_tick()
            for d in cnt:
                if self.nb[d].O>0: cnt[d]+=1
        win=max(cnt,key=cnt.get)
        return {FORAGE:'forage',FLEE:'flee',EXPLORE:'explore'}[win], cnt

# ---- general 4-way selector: caller supplies explicit drive values ----
FOR,FLE,HOM,EXP = 1,2,3,4
def build_sel4():
    ne=[k.neuron(i,r=0.6,lam=5,c=2) for i in (FOR,FLE,HOM,EXP)]; sy=[]; conns=[]
    for d in (FOR,FLE,HOM,EXP):
        sy.append(k.syn(d,0,2.2,1))
        others=[o for o in (FOR,FLE,HOM,EXP) if o!=d]
        for j,o in enumerate(others): sy.append(k.syn(d,1+j,-15.0,1))
        sy.append(k.term(d,tid=900+d))
    for d in (FOR,FLE,HOM,EXP):
        others=[o for o in (FOR,FLE,HOM,EXP) if o!=d]
        for j,o in enumerate(others): conns.append(k.conn(o,d,1+j,stid=900+o))
    return k.build(ne,sy,conns,[k.ext(d,0) for d in (FOR,FLE,HOM,EXP)])

class Sel4:
    """4-way winner-take-all. select(drives) takes a dict {FOR/FLE/HOM/EXP: value} and returns the
    winning behaviour name."""
    def __init__(self):
        self.path=build_sel4(); self.net,self.core=k.load(self.path); self.nb={n:u for n,u in self.net.network.neurons.items()}
    def select(self, drives, T=12):
        self.net.reset_simulation(); self.core.state.current_tick=0; self.net.current_tick=0
        cnt={FOR:0,FLE:0,HOM:0,EXP:0}
        for t in range(T):
            for d,a in drives.items(): self.net.set_external_input(d,0,max(0.0,a))
            self.core.do_tick()
            for d in cnt:
                if self.nb[d].O>0: cnt[d]+=1
        w=max(cnt,key=cnt.get); return {FOR:'forage',FLE:'flee',HOM:'home',EXP:'explore'}[w]

if __name__=="__main__":
    s=Selector()
    cases=[("hungry+safe",0.9,0.0,'forage'),("threatened",0.9,1.0,'flee'),("sated+safe",0.1,0.0,'explore'),
           ("mild hunger",0.5,0.0,'forage'),("threat over hunger",1.0,0.8,'flee')]
    ok=True
    for name,h,f,exp in cases:
        mode,cnt=s.select(h,f); good=(mode==exp); ok&=good
        print(f"  {name:22s} hunger={h} fear={f} -> {mode:8s} {'OK' if good else 'EXP '+exp}  {cnt}")
    print(f"VERDICT: {'ACTION SELECTION arbitrates drives correctly (flee>forage>explore)' if ok else 'needs tuning'} | neurons=3")
    print("@@@AS DONE@@@")
