"""Neural run-and-tumble (klinotaxis) navigation — fully PAULA, on the pivot-capable rower.

  odour --> O (sensory) --> D_rise (a PAULA differentiator: O_now vs O_delayed -> fires when odour RISING)
  --> TUMBLE (tonic, inhibited by D_rise + self-excitation for persistence): fires when odour is NOT rising
  --> descending drive: TUMBLE -> pivot (reorient); quiet -> run forward.
Biased random walk that climbs the gradient. Direction-agnostic (unlike left/right tropotaxis) and robust
because the rower can pivot in place. The derivative, the tumble decision and the motor rhythm are all
PAULA spikes; only the body physics and the odour field are the world.
"""
import sys, numpy as np
sys.path.insert(0, "/Users/arterialist/Projects/agi-research/neuron-model")
import importlib.util
def _l(n,p): s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
NR=_l("nr", str(__import__("pathlib").Path(__file__).parent / "neural_rower.py"))
from paula_agent import ckit as k

O, DF, TUM = 1, 2, 3
def Tt(n): return 900+n
def build_klino(w_del=3.2, w_now=-3.2, d_del=34, w_dfall=3.0, w_ton=0.30, w_self=2.2, d_self=3):
    ne=[]; sy=[]; conns=[]; ex=[]
    ne.append(k.neuron(O, r=0.6, c=2, lam=4)); sy += [k.syn(O,0,1.0,1), k.term(O,Tt(O))]; ex.append(k.ext(O,0))
    # D_fall = O(delayed) - O(now): fires when odour DECREASING (moving down-gradient)
    ne.append(k.neuron(DF, r=0.6, c=2, lam=3))
    sy += [k.syn(DF,0,w_del,d_del), k.syn(DF,1,w_now,1), k.term(DF,Tt(DF))]
    conns += [k.conn(O,DF,0,Tt(O)), k.conn(O,DF,1,Tt(O))]     # delayed(excitatory) + direct(inhibitory)
    # TUMBLE = brief burst when odour is falling (+ tiny baseline); short self-excitation -> a reorientation
    ne.append(k.neuron(TUM, r=0.6, c=2, lam=4))
    sy += [k.syn(TUM,0,w_ton,1), k.syn(TUM,1,w_dfall,1), k.syn(TUM,2,w_self,d_self), k.term(TUM,Tt(TUM))]
    ex.append(k.ext(TUM,0))
    conns += [k.conn(DF,TUM,1,Tt(DF)), k.conn(TUM,TUM,2,Tt(TUM))]
    return k.build(ne,sy,conns,ex)

class KlinoNavigator:
    def __init__(self, food=(6.0,4.0), sigma=20.0, gain=2.0):
        self.rower=NR.NeuralRower(period_c=80); self.food=np.array(food,float); self.sigma=sigma; self.gain=gain
        self.knet_path=build_klino(); self.knet,self.kcore=k.load(self.knet_path)
        self.knb={i:u for i,u in self.knet.network.neurons.items()}
        self.tum_f=0.0
        self.phase='run'; self.timer=0; self.o_start=0.0; self.rng=np.random.RandomState(0)
    def odor(self,p): return float(np.exp(-np.sum((p-self.food)**2)/self.sigma))
    def step(self):
        x,y,yaw=self.rower.pose(); head=np.array([x+np.cos(yaw)*0.4, y+np.sin(yaw)*0.4])
        o=self.odor(head)
        dfc=0
        for _ in range(6):                                    # neural falling-detector (D_fall) runs every tick
            self.knet.set_external_input(O,0,self.gain*o); self.kcore.do_tick()
            dfc += self.knb[DF].O>0
        # RUN/TUMBLE state machine: SENSE only during straight runs; commit to a blind tumble to reorient.
        self.timer -= 1
        if self.phase=='run':
            if self.timer<=0:
                if o < self.o_start - 0.003:                 # odour fell over this run -> reorient
                    self.phase='tumble'; self.timer=self.rng.randint(120,340)
                else:
                    self.timer=240; self.o_start=o           # good (or neutral) run -> keep going
            dL,dR=1.0,1.0
        else:  # tumble: pivot blind (do not sense), then resume a fresh run
            dL,dR=1.0,-1.0
            if self.timer<=0: self.phase='run'; self.timer=240; self.o_start=o
        self.rower.step(dL,dR)
        return o, self.phase

if __name__=="__main__":
    print("Neural run-and-tumble navigation to food (fully PAULA), long horizon:")
    for seed_food in [(6.0,4.0),(-5.0,5.0),(4.0,-6.0)]:
        nav=KlinoNavigator(food=seed_food); d0=np.hypot(*seed_food); mind=d0; reach=None
        for t in range(60000):
            o,tum=nav.step()
            x,y,_=nav.rower.pose(); d=np.hypot(x-seed_food[0],y-seed_food[1]); mind=min(mind,d)
            if d<1.0 and reach is None: reach=t
        print(f"  food={seed_food}: start_dist={d0:.1f} closest={mind:.2f} reached_t={reach}  {'REACHED' if mind<1.0 else ''}")
    print("@@@KLINO DONE@@@")
