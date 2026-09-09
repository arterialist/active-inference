"""THE ORGANISM — long-horizon embodied ethology from three PAULA brain regions.

Same three regions (mushroom body, central complex, 4-way action selection) in a MuJoCo body, in a world
designed so ALL FOUR behaviours genuinely EMERGE (none are scripted):
  - FORAGE up an odour gradient when hungry and a smell is present;
  - AVOID a smell it has learned is toxic (mushroom-body valence flips the chemotaxis sign);
  - EXPLORE (wander to discover food) when fed or when no smell is nearby;
  - FLEE an ACTIVELY-PATROLLING predator that crosses the arena and gives chase;
  - and PATH-INTEGRATE home to the nest when the crop is full, then set out again.
Energy genuinely oscillates hungry<->satiated, so the behaviour is a life, not a loop.

Run standalone:  python -m simulations.organism.organism        (6 seeds, prints the ethogram)
"""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from simulations.paula_loader import ensure_paula_available
ensure_paula_available()
from simulations.organism import mushroom_body as MB
from simulations.organism import central_complex as CXM
from simulations.organism import body as BODY
from simulations.organism.action_selection import Sel4, FOR, FLE, HOM, EXP

rng0=np.random.RandomState(0); P_GOOD=MB.odor_pattern(rng0); P_BAD=MB.odor_pattern(rng0)

ARENA=5.0; SAT=18.0; CROP_CAP=2; PRED_DET=5.0; SMELL_REF=0.30

class Organism:
    def __init__(self, seed=0, nfood=7):
        self.rng=np.random.RandomState(seed)
        # each food: [x, y, sign(+/-), good(bool)];  indices 0 and 4 are toxic, the rest nutritious
        self.foods=[]
        for j in range(nfood):
            good = (j not in (0,4))
            self.foods.append([self.rng.uniform(-ARENA,ARENA),self.rng.uniform(-ARENA,ARENA), 1.0 if good else -1.0, good])
        self.world=BODY.World([(f[0],f[1],f[2]) for f in self.foods])
        self.mbp,_=MB.build_mb(seed=1); self.net,self.core,self.nb=MB.load(self.mbp)
        self.cx=CXM.CentralComplex(); self.sel=Sel4()
        self.px,self.py=4.5,4.5; self.pph=self.rng.uniform(0,6.28); self.world.set_predator(self.px,self.py)
        self.was_caught=False; self.crop=0; self.energy=16.0; self.home_streak=0
        self.good=0; self.tox=0; self.hurt=0; self.trips=0; self.abandoned=0; self.t=0
        self.avoid_events=0   # times it steered away from a learned-toxic smell
        self.modes={'forage':0,'flee':0,'home':0,'explore':0}; self.log=[]

    def _odor(self,x,y):
        best=None;bd=1e9
        for fx,fy,s,g in self.foods:
            d=(x-fx)**2+(y-fy)**2
            if d<bd: bd=d;best=g
        return (P_GOOD if best else P_BAD) if bd<40 else np.zeros(MB.N_PN)

    def step(self):
        x,y,yaw=self.world.pose(); cl,cr=self.world.antennae()
        dpred=np.hypot(self.px-x,self.py-y)
        hx,hy=self.cx.home_vector(); d_origin=float(np.hypot(x,y))
        smell=float(np.clip(max(cl,cr)/SMELL_REF,0,1))
        hunger=float(np.clip((SAT-self.energy)/14.0,0,1))
        fear=float(np.clip((PRED_DET-dpred)/PRED_DET,0,1))
        homing=1.0 if (self.crop>=CROP_CAP and d_origin>1.6) else 0.0
        # mushroom-body valence for the currently-smelled odour
        _,app,avo=MB.present(self.net,self.core,self.nb,self._odor(x,y)); valence=1 if app>avo else (-1 if avo>app else 0)
        # unified chemotaxis: approach food / investigate unknowns, but steer AWAY from a learned-toxic smell
        known_toxic = (valence<0 and smell>0.2)
        chemo=-1.0 if known_toxic else 1.0
        if known_toxic: self.avoid_events+=1
        # DRIVES (explicit): forage needs smell+hunger; explore when fed / no smell; flee on fear; home on crop-full
        drives={FOR:0.9+3.8*hunger*smell, FLE:8.0*fear, HOM:7.0*homing, EXP:1.45+0.8*(1-hunger)-0.6*smell}
        mode=self.sel.select(drives); self.modes[mode]+=1
        # motor
        if mode=='flee':
            ang=np.arctan2(y-self.py,x-self.px); turn=float(np.clip(2.6*np.sin(ang-yaw),-2.6,2.6)); speed=3.6
        elif mode=='home':
            homedir=np.arctan2(-hy,-hx); turn=float(np.clip(2.4*np.sin(homedir-yaw),-2.4,2.4)); speed=2.7
        elif mode=='forage':
            turn=float(np.clip(16.0*chemo*(cl-cr),-2.2,2.2))+self.rng.uniform(-.12,.12); speed=2.6
        else:  # explore: still investigate smells (climb toward food / away from toxic) but wander when none
            turn=float(np.clip(11.0*chemo*(cl-cr),-1.8,1.8))+self.rng.uniform(-.6,.6); speed=2.4
        self.world.step(speed,turn)
        nx,ny,_=self.world.pose(); self.cx.update((nx-x)/0.02,(ny-y)/0.02)
        # ACTIVE predator: patrol with momentum, give chase when it detects the bug (evadable: chase < flee speed)
        if dpred<PRED_DET:
            ang=np.arctan2(y-self.py,x-self.px); self.pph=ang; ps=0.048
        else:
            self.pph+=self.rng.uniform(-0.35,0.35); ps=0.030
        self.px+=ps*np.cos(self.pph); self.py+=ps*np.sin(self.pph)
        if abs(self.px)>ARENA+1: self.pph=np.pi-self.pph; self.px=float(np.clip(self.px,-ARENA-1,ARENA+1))
        if abs(self.py)>ARENA+1: self.pph=-self.pph; self.py=float(np.clip(self.py,-ARENA-1,ARENA+1))
        self.world.set_predator(self.px,self.py)
        caught=dpred<0.7
        if caught and not self.was_caught: self.energy-=3.0; self.hurt+=1
        self.was_caught=caught
        # arrived home? (generous radius); or ABANDON the trip if harassed off course too long
        self.home_streak = self.home_streak+1 if mode=='home' else 0
        if self.crop>=CROP_CAP and d_origin<1.0: self.trips+=1; self.crop=0; self.energy+=1.0; self.home_streak=0
        elif self.home_streak>170: self.crop=0; self.abandoned+=1; self.home_streak=0   # gave up returning
        # taste (only when actually moving toward it, i.e. forage/explore)
        tasted=None
        for i,(fx,fy,s,g) in enumerate(self.foods):
            if (x-fx)**2+(y-fy)**2<0.6**2 and mode in ('forage','explore'):
                tasted='good' if g else 'toxic'; self.energy+=6.0 if g else -5.0
                self.good+=g; self.tox+=(not g); self.crop+=1 if g else 0
                for _ in range(8): MB.present(self.net,self.core,self.nb, P_GOOD if g else P_BAD, teach=MB.APP if g else MB.AVO, learn=True)
                self.foods[i]=[self.rng.uniform(-ARENA,ARENA),self.rng.uniform(-ARENA,ARENA),s,g]
                self.world.set_foods([(f[0],f[1],f[2]) for f in self.foods]); self.world.set_predator(self.px,self.py); break
        self.energy=min(self.energy-0.02, SAT+4); self.t+=1
        self.log.append(dict(t=self.t,x=float(x),y=float(y),mode=mode,valence=valence,smell=round(smell,2),
                             energy=round(self.energy,1),dpred=round(float(dpred),1),fear=round(fear,2),
                             dnest=round(d_origin,2),tasted=tasted,crop=self.crop))
        return mode

if __name__=="__main__":
    for seed in range(6):
        o=Organism(seed=seed)
        for _ in range(2200): o.step()
        _,ag,vg=MB.present(o.net,o.core,o.nb,P_GOOD); _,ab,vb=MB.present(o.net,o.core,o.nb,P_BAD)
        print(f"seed{seed}: good={o.good} tox={o.tox} hurt={o.hurt} trips={o.trips} avoid={o.avoid_events} "
              f"modes={o.modes} MBgood={ag}/{vg} MBbad={ab}/{vb} E={o.energy:.0f}")
    print("@@@ORGANISM DONE@@@")
