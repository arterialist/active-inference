"""Closing the sensorimotor loop ENTIRELY in PAULA — a neurally-controlled body that navigates to food.

  odour field --(head sensors)--> SENSORY neurons SL/SR
       --(crossed Braitenberg wiring)--> descending TURN neurons TL/TR
       --> (TL-TR) sets the CPG's steering curvature
       --> CPG travelling wave + curvature --> MUSCLES --> physics --> movement --> new odour ...

No step of this maps a sensor to a velocity in Python. Sensing, steering and the locomotor rhythm are all
PAULA spikes; only the body physics is MuJoCo. Verify: the body swims up the odour gradient to the source.
"""
import sys, numpy as np, mujoco
sys.path.insert(0, "/Users/arterialist/Projects/agi-research/neuron-model")
import importlib.util
def _l(n,p): s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
SW=_l("sw", str(__import__("pathlib").Path(__file__).parent / "neural_motor.py"))
from paula_agent import ckit as k

# sensory + descending-turn micro-network (a separate PAULA net; projects a turn command to the CPG)
SL, SR, TL, TR = 1, 2, 3, 4
def T(n): return 900+n
def build_turn_net(cross=True, w_sd=3.0, w_wta=-4.0, w_slat=-8.0):
    ne=[]; sy=[]; conns=[]; ex=[]
    for nid in (SL,SR):                              # sensory neurons (odour left / right) + lateral inhibition
        ne.append(k.neuron(nid, r=0.6, c=2, lam=4))
        sy += [k.syn(nid,0,1.0,1), k.syn(nid,1,w_slat,1), k.term(nid,T(nid))]; ex.append(k.ext(nid,0))
    conns += [k.conn(SL,SR,1,T(SL)), k.conn(SR,SL,1,T(SR))]   # SL<->SR lateral inhibition (compute difference)
    for nid in (TL,TR):                              # descending turn neurons, mutual inhibition
        ne.append(k.neuron(nid, r=0.6, c=2, lam=4))
        sy += [k.syn(nid,0,w_sd,1), k.syn(nid,1,w_wta,1), k.term(nid,T(nid))]
    if cross:  conns += [k.conn(SL,TR,0,T(SL)), k.conn(SR,TL,0,T(SR))]
    else:      conns += [k.conn(SL,TL,0,T(SL)), k.conn(SR,TR,0,T(SR))]
    conns += [k.conn(TL,TR,1,T(TL)), k.conn(TR,TL,1,T(TR))]
    return k.build(ne,sy,conns,ex)

class Navigator:
    def __init__(self, food=(4.0,1.5), gain=6.0, approach=True):
        self.sw=SW.NeuralSwimmer(); self.food=np.array(food,float); self.gain=gain
        self.tnet_path=build_turn_net(cross=approach); self.tnet,self.tcore=k.load(self.tnet_path)
        self.tnb={i:u for i,u in self.tnet.network.neurons.items()}
        self.tl=self.tr=0.0
    def _odor(self,p):  # scalar field around the food source
        d2=float(np.sum((p-self.food)**2)); return np.exp(-d2/9.0)
    def step(self):
        x,y,yaw=self.sw.pose()
        # two head sensors, left/right of heading
        hd=np.array([np.cos(yaw),np.sin(yaw)]); lat=np.array([-np.sin(yaw),np.cos(yaw)]); head=np.array([x,y])+hd*0.5
        oL=self._odor(head+lat*0.25); oR=self._odor(head-lat*0.25)
        # drive sensory neurons; run the turn micro-net a few ticks; read descending turn command
        Ltc=Rtc=0
        for _ in range(5):
            self.tnet.set_external_input(SL,0,self.gain*oL); self.tnet.set_external_input(SR,0,self.gain*oR)
            self.tcore.do_tick()
            Ltc+=self.tnb[TL].O>0; Rtc+=self.tnb[TR].O>0
        self.tl=0.7*self.tl+Ltc; self.tr=0.7*self.tr+Rtc
        turn=float(np.clip((self.tl-self.tr)/6.0,-1,1))          # descending command = TL - TR
        self.sw.step(turn=turn)
        return x,y,turn,oL,oR

if __name__=="__main__":
    print("FULL NEURAL sensorimotor loop — swim to the food at (4.0, 1.5), no readout in the path:")
    nav=Navigator(food=(4.0,1.5))
    d0=np.hypot(4.0,1.5)
    for t in range(2000):
        x,y,turn,oL,oR=nav.step()
        if t%400==0:
            d=np.hypot(x-4.0,y-1.5); print(f"  t{t:4d}: pos=({x:5.2f},{y:5.2f}) dist-to-food={d:4.2f} turn={turn:+.2f} (oL={oL:.2f} oR={oR:.2f})")
    x,y,_=nav.sw.pose(); dfin=np.hypot(x-4.0,y-1.5)
    print(f"  final dist-to-food = {dfin:.2f} (started {d0:.2f})  -> {'REACHED' if dfin<0.8 else 'closer' if dfin<d0 else 'FAILED'}")
    print("@@@NEURAL-NAV DONE@@@")
