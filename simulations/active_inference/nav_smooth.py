"""Robust 360-degree navigator: TROPOTAXIS direction (range-invariant population sensor) + SMOOTH
klinokinesis steering-rate (spiral reorientation when the gradient is lost). All smooth, all spikes."""
import sys, numpy as np
sys.path.insert(0, "/Users/arterialist/Projects/agi-research/neuron-model")
import importlib.util
def _l(n,p): s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
NR=_l("nr", str(__import__("pathlib").Path(__file__).parent / "neural_rower.py"))
from paula_agent import ckit as k
N=10; POOL=500; RISE=700; STEER=701
THR=np.exp(np.linspace(np.log(0.2), np.log(4.5), N))
def Tt(n): return 900+n
def build(w_norm=-0.9, w_pool=0.5, d_del=60, w_ton=1.5, w_riseinh=-5.0):
    ne=[]; sy=[]; conns=[]; ex=[]
    for side in (0,1):
        for i in range(N):
            nid=side*100+i; ne.append(k.neuron(nid, r=float(THR[i]), c=2, lam=4))
            sy += [k.syn(nid,0,1.0,1), k.syn(nid,1,w_norm,1), k.term(nid,Tt(nid))]; ex.append(k.ext(nid,0))
            conns.append(k.conn(POOL,nid,1,Tt(POOL)))
    ne.append(k.neuron(POOL, r=0.5, c=1, lam=3)); sy += [k.syn(POOL,j,w_pool,1) for j in range(2*N)]; sy.append(k.term(POOL,Tt(POOL)))
    for side in (0,1):
        for i in range(N): conns.append(k.conn(side*100+i, POOL, side*N+i, Tt(side*100+i)))
    # RISE: total-odour trend from POOL; STEER: tonic spiral inhibited by RISE (smooth reorientation rate)
    ne.append(k.neuron(RISE, r=0.6, c=2, lam=3)); sy += [k.syn(RISE,0,3.4,1), k.syn(RISE,1,-3.4,d_del), k.term(RISE,Tt(RISE))]
    conns += [k.conn(POOL,RISE,0,Tt(POOL)), k.conn(POOL,RISE,1,Tt(POOL))]
    ne.append(k.neuron(STEER, r=0.6, c=2, lam=5)); sy += [k.syn(STEER,0,w_ton,1), k.syn(STEER,1,w_riseinh,1), k.term(STEER,Tt(STEER))]
    ex.append(k.ext(STEER,0)); conns.append(k.conn(RISE,STEER,1,Tt(RISE)))
    return k.build(ne,sy,conns,ex)
class NavSmooth:
    def __init__(self, food=(3.0,2.0), sigma=16.0, gain=5.0, kturn=0.10, k_spiral=0.05, st=20):
        self.rower=NR.NeuralRower(period_c=80); self.food=np.array(food,float); self.sigma=sigma; self.gain=gain
        self.kturn=kturn; self.k_spiral=k_spiral; self.st=st
        self.net,self.core=k.load(build()); self.nb={i:u for i,u in self.net.network.neurons.items()}
        self.turn_f=0.0; self.steer_f=0.0
    def odor(self,p): return float(np.exp(-np.sum((p-self.food)**2)/self.sigma))
    def step(self):
        x,y,yaw=self.rower.pose()
        hd=np.array([np.cos(yaw),np.sin(yaw)]); lat=np.array([-np.sin(yaw),np.cos(yaw)]); head=np.array([x,y])+hd*0.35
        oL=self.odor(head+lat*0.3); oR=self.odor(head-lat*0.3)
        cL=cR=sc=0
        for _ in range(self.st):
            for i in range(N): self.net.set_external_input(i,0,self.gain*oL); self.net.set_external_input(100+i,0,self.gain*oR)
            self.net.set_external_input(STEER,0,1.0); self.core.do_tick()
            cL+=sum(self.nb[i].O>0 for i in range(N)); cR+=sum(self.nb[100+i].O>0 for i in range(N)); sc+=self.nb[STEER].O>0
        self.turn_f=0.5*self.turn_f+(cL-cR); self.steer_f=0.9*self.steer_f+0.1*sc
        turn = self.turn_f*self.kturn + self.k_spiral*self.steer_f      # tropotaxis direction + smooth spiral
        turn=float(np.clip(turn,-1.7,1.7))
        self.rower.step(1.0-max(0.0,turn), 1.0-max(0.0,-turn))
        return oL+oR
if __name__=="__main__":
    print("Robust navigator (tropotaxis direction + smooth spiral reorientation):")
    reached=0; tot=0
    for food in [(3.0,2.0),(-2.5,3.0),(2.0,-3.0),(-3.0,-2.0),(0.0,-4.0),(-4.0,0.0)]:
        nav=NavSmooth(food=food); d0=np.hypot(*food); mind=d0; rt=None
        for t in range(45000):
            nav.step(); x,y,_=nav.rower.pose(); d=np.hypot(x-food[0],y-food[1]); mind=min(mind,d)
            if d<1.0 and rt is None: rt=t
        ok=mind<1.0; reached+=ok; tot+=1
        print(f"  food={food}: start={d0:.1f} closest={mind:.2f} reached_t={rt}  {'REACHED' if ok else ''}", flush=True)
    print(f"REACHED {reached}/{tot}")
    print("@@@NAVSMOOTH DONE@@@")
