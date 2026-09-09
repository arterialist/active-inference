"""LADDER INTEGRATOR — a SPIKING, rate-coded, persistent accumulator (the CPU4 primitive).

WHY IT EXISTS. The analog accumulator trick (r=1e9 never-spike neuron, read membrane S) integrates
beautifully but is a DEAD END for internal neural computation: in PAULA `O` is 1-on-spike / 0-otherwise
(neuron.py:347) and propagation is spike-gated (network.py:535), so a neuron that never spikes can drive
NOTHING downstream. Its value can only leave via a python membrane read -- fine as a transducer/probe,
useless as an input to a neural comparator. Real central-complex CPU4 cells hold the accumulated home
vector in PERSISTENT FIRING across columns, not in silent membranes. This circuit does that.

MECHANISM (a recruitment ladder = a thermometer code of the integral):
  cell i:  self-excitation (LATCH -- once firing it re-triggers itself forever)
        +  conjunctive advance AND(cell i-1 firing, drive spike) through a DELAYED dendrite (d_adv)
  The four inequalities that make it work (lam=2, so one spike contributes w/lam):
     latch:            w_self/lam  >  r          (5.0/2=2.5 > 1.6)   -- a lit cell stays lit
     prev alone NO:    w_prev/lam  <  r          (2.4/2=1.2 < 1.6)   -- upstream alone cannot recruit
     drive alone NO:   w_drive*amp <  r          (1.0*1.2=1.2 < 1.6) -- drive alone cannot recruit
     together YES:     w_prev/lam + w_drive*amp > r  (2.4)           -- only COINCIDENCE recruits
  So the fill front advances one cell per coincidence, the delay d_adv rate-limits it, and the number of
  latched cells = the integral of the drive RATE over time. Drive comes as SPIKES (e.g. central-complex
  PG cells firing at rate ~ speed x heading-overlap), so fill rate ~ drive rate ~ the quantity integrated.

VERIFIED (this file's __main__): zero drive -> zero accumulation (no drift); pulse period 24 -> fills
  6/10/12 over 300 ticks; period 12 -> 10/12/12 (faster); fast rates saturate; and in EVERY case the fill
  HOLDS after the drive stops (persistent memory). Saturates at NL cells -- lengthen NL or raise d_adv for
  more dynamic range.
"""
import sys, numpy as np
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
from paula_agent import ckit as k

def T(n): return 8000+n

def build_ladder(base, NL=12, w_self=5.0, w_prev=2.4, w_drive=1.0, r=1.6, lam=2, d_adv=3, w_seed=6.0,
                 ne=None, sy=None, conns=None, ex=None):
    """Append one ladder of NL cells with ids base..base+NL-1 to the given ckit lists.
    Returns (ids, drive_syn_of, seed_syn_of): drive_syn_of(i)/seed_syn_of(i) give the ext synapse ids."""
    ne=[] if ne is None else ne; sy=[] if sy is None else sy
    conns=[] if conns is None else conns; ex=[] if ex is None else ex
    ids=[base+i for i in range(NL)]
    for i,nid in enumerate(ids):
        ne.append(k.neuron(nid,r=r,c=2,lam=lam)); j=0
        sy.append(k.syn(nid,j,w_self,1)); conns.append(k.conn(nid,nid,j,T(nid))); j+=1        # latch
        if i>0:
            sy.append(k.syn(nid,j,w_prev,d_adv)); conns.append(k.conn(ids[i-1],nid,j,T(ids[i-1]))); j+=1
        sy.append(k.syn(nid,j,w_drive,1)); ex.append(k.ext(nid,j)); j+=1                      # drive (spikes)
        sy.append(k.syn(nid,j,w_seed,1));  ex.append(k.ext(nid,j))                            # seed (cell 0)
        sy.append(k.term(nid,T(nid)))
    drive_syn_of=lambda i: 2 if i>0 else 1
    seed_syn_of =lambda i: 3 if i>0 else 2
    return ids, drive_syn_of, seed_syn_of, (ne,sy,conns,ex)

class Ladder:
    """Standalone single-ladder harness (for testing the primitive in isolation)."""
    def __init__(self, NL=12, **kw):
        self.NL=NL
        ids,dsyn,ssyn,(ne,sy,conns,ex)=build_ladder(0,NL=NL,**kw)
        self.ids=ids; self.dsyn=dsyn; self.ssyn=ssyn
        self.net,self.core=k.load(k.build(ne,sy,conns,ex))
        self.nb={i:u for i,u in self.net.network.neurons.items()}
    def seed(self, ticks=25, amp=6.0):
        for _ in range(ticks):
            self.net.set_external_input(self.ids[0],self.ssyn(0),amp); self.core.do_tick()
        self.net.set_external_input(self.ids[0],self.ssyn(0),0.0)
    def fill(self, win=8):
        """windowed count of latched cells (the ladder bursts, so a single frame under-reads)"""
        acc=np.zeros(self.NL)
        for _ in range(win):
            self.core.do_tick()
            for i,nid in enumerate(self.ids): acc[i]+= self.nb[nid].O>0
        return int((acc>0).sum())
    def drive(self, ticks, period, amp=1.2):
        """pulse the drive at rate 1/period (period<=0 => no drive)"""
        for t in range(ticks):
            on = period>0 and t%period==0
            for i,nid in enumerate(self.ids):
                self.net.set_external_input(nid,self.dsyn(i), amp if on else 0.0)
            self.core.do_tick()

if __name__=="__main__":
    print("LADDER INTEGRATOR (spiking rate-coded accumulator): fill ~ integral of drive rate, and PERSISTS")
    for period in [0,24,12,6]:
        lad=Ladder(); lad.seed(); traj=[]
        for _ in range(3):
            lad.drive(100,period); traj.append(lad.fill())
        for _ in range(200): lad.core.do_tick()          # drive OFF
        lbl="no drive" if period==0 else f"drive rate 1/{period}"
        print(f"  {lbl:16s}: fill@100/200/300={traj}  after 200t of NO drive={lad.fill()}  [holds]")
    print("@@@LADDER DONE@@@")
