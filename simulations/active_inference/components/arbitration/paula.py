"""AIF ARBITER — spiking expected-free-energy action selection over BEHAVIOURAL MODES.

This is the piece that makes the mini-brain an active-inference agent rather than a bundle of reflexes.
The algorithmic T-maze agent (tmaze_agent.py) picks actions by minimising G(a) = -(pragmatic) - (epistemic);
paula_aif.py showed the spiking version of that for T-maze arms. Here the same competition runs over MODES
of an embodied forager, driven by the brain's own BELIEFS:

    FORAGE   <- hunger (interoceptive drive)              PRAGMATIC
    HOME     <- |home vector| from the central complex    PRAGMATIC (return before the PI degrades)
    EXPLORE  <- uncertainty U                             EPISTEMIC (act to reduce surprise)

MECHANISM (all wiring):
  * each mode is a POPULATION (the project's recurring principle) with within-population excitation, so a
    chosen mode PERSISTS (hysteresis) instead of chattering tick-to-tick;
  * populations CROSS-INHIBIT, so the strongest total drive wins -- a spiking argmin of expected free energy;
  * INTEROCEPTION: hunger is a ladder_integrator that fills tonically and is DRAINED by eating, so the
    pragmatic drive is itself a neural state, not a python variable.

The winner then SELECTS behaviour by VETO RELAYS (gated_agent.py's verified result: vetoing the wrong-mode
relay is far more robust than AND-gating the right one).
"""
import sys, numpy as np
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
from paula_agent import ckit as k

def T(n): return 30000+n
MODES=["FORAGE","HOME","EXPLORE"]
NM=len(MODES); NPOP=6
MODE=[[40000+m*NPOP+i for i in range(NPOP)] for m in range(NM)]
# V3 adds this population only when ``build(sleep=True)`` is requested.  The
# first three mode IDs and their default topology remain unchanged.
SLEEP_MODE=[40024+i for i in range(NPOP)]
ALL_MODES=MODE+[SLEEP_MODE]
NH=12; HUNGER=[41000+i for i in range(NH)]          # interoceptive hunger ladder

def build(w_self=1.6, w_lat=0.5, w_cross=-0.4, r_mode=1.3, lam_mode=2, w_drive=1.4,
          h_self=5.0, h_prev=2.4, h_drive=1.0, r_h=1.6, h_adv=12, w_drain=1.8,
          w_hmode=0.45, sleep=False, w_sleep_cross=-0.8, home=True):
    ne=[];sy=[];conns=[];ex=[]
    # ---- HUNGER: a ladder integrator. tonic fill (ext syn) ; EATING drains it (ext inhibitory syn).
    # latch gradient so the drain de-recruits the top cell first (bidirectional accumulator).
    for i,nid in enumerate(HUNGER):
        ne.append(k.neuron(nid,r=r_h,c=2,lam=2)); j=0
        ws=h_self-(i/max(NH-1,1))*0.8
        sy.append(k.syn(nid,j,ws,1)); conns.append(k.conn(nid,nid,j,T(nid))); j+=1              # latch
        if i>0:
            sy.append(k.syn(nid,j,h_prev,h_adv)); conns.append(k.conn(HUNGER[i-1],nid,j,T(HUNGER[i-1]))); j+=1
        sy.append(k.syn(nid,j,h_drive,1)); ex.append(k.ext(nid,j)); j+=1                        # tonic fill (port A)
        sy.append(k.syn(nid,j,-w_drain,1)); ex.append(k.ext(nid,j)); j+=1                       # EAT drain (port B)
        if i==0: sy.append(k.syn(nid,j,6.0,1)); ex.append(k.ext(nid,j))                         # birth seed
        sy.append(k.term(nid,T(nid)))
    # ---- MODE populations: within-pop excitation (persistence) + cross-pop inhibition (WTA) + drive port.
    # V3 uses the same construction for four populations; the default remains
    # the original three-way arbiter byte-for-byte in its topology.
    # A strict interoceptive profile has no path-integration component, so a
    # HOME population would be an un-driven legacy remnant.  Keep the
    # historical three/four-way construction by default, while allowing V3 to
    # compose FORAGE/EXPLORE/SLEEP only.
    base_groups = MODE if home else [MODE[0], MODE[2]]
    groups=base_groups + ([SLEEP_MODE] if sleep else [])
    n_groups=len(groups)
    for m in range(n_groups):
        for i,nid in enumerate(groups[m]):
            ne.append(k.neuron(nid,r=r_mode,c=2,lam=lam_mode)); j=0
            sy.append(k.syn(nid,j,w_self,1)); conns.append(k.conn(nid,nid,j,T(nid))); j+=1      # self latch
            for o in groups[m]:                                                                  # within-pop
                if o!=nid:
                    sy.append(k.syn(nid,j,w_lat,1)); conns.append(k.conn(o,nid,j,T(o))); j+=1
            for m2 in range(n_groups):                                                          # cross-inhibit
                if m2!=m:
                    for o in groups[m2]:
                        weight=w_sleep_cross if sleep and (m==3 or m2==3) else w_cross
                        sy.append(k.syn(nid,j,weight,1)); conns.append(k.conn(o,nid,j,T(o))); j+=1
            sy.append(k.syn(nid,j,w_drive,1)); ex.append(k.ext(nid,j)); j+=1                    # drive port
            sy.append(k.term(nid,T(nid)))
    return ne,sy,conns,ex

def compile_net(**kw):
    ne,sy,conns,ex=build(**kw)
    return k.build(ne,sy,conns,ex)

# ---- port indices (fixed by construction order above) ----
def hunger_fill_syn(i):  return 2 if i>0 else 1     # tonic fill port
def hunger_drain_syn(i): return 3 if i>0 else 2     # eat-drain port
def hunger_seed_syn():   return 3                   # cell 0 seed
def mode_drive_syn(m,i,sleep=False,home=True):
    """Drive port for a mode population built with the selected WTA width."""
    n_base=NM if home else NM-1
    n_groups=n_base+1 if sleep else n_base
    return 1+(NPOP-1)+(n_groups-1)*NPOP

class Arbiter:
    def __init__(self, **kw):
        self.net,self.core=k.load(compile_net(**kw)); self.nb={i:u for i,u in self.net.network.neurons.items()}
    def birth(self, ticks=20):
        for _ in range(ticks):
            self.net.set_external_input(HUNGER[0],hunger_seed_syn(),6.0); self.core.do_tick()
        self.net.set_external_input(HUNGER[0],hunger_seed_syn(),0.0)
    def step(self, drives, hunger_fill=0.0, eat=0.0, n=1):
        """drives: list per mode (external belief drive). hunger_fill: tonic. eat: drains hunger."""
        for _ in range(n):
            for m in range(NM):
                for i,nid in enumerate(MODE[m]):
                    self.net.set_external_input(nid,mode_drive_syn(m,i),float(drives[m]))
            for i,nid in enumerate(HUNGER):
                self.net.set_external_input(nid,hunger_fill_syn(i),hunger_fill)
                self.net.set_external_input(nid,hunger_drain_syn(i),eat)
            self.core.do_tick()
    def activity(self, win=12, drives=(0,0,0), hunger_fill=0.0, eat=0.0):
        acc=np.zeros(NM); hf=0
        for _ in range(win):
            self.step(drives,hunger_fill,eat)
            for m in range(NM): acc[m]+=sum(1 for nid in MODE[m] if self.nb[nid].O>0)
            hf+=sum(1 for nid in HUNGER if self.nb[nid].O>0)
        return acc/win, hf/win
    def winner(self,acc): return MODES[int(np.argmax(acc))] if acc.max()>0 else "none"

if __name__=="__main__":
    def fresh():
        np.random.seed(0); a=Arbiter(); a.birth(); return a
    print("AIF ARBITER — spiking WTA over behavioural modes (pragmatic + epistemic drives)")
    for dr,lbl in [((1.6,0.3,0.3),"hunger high        -> FORAGE"),
                   ((0.3,1.6,0.3),"far from home      -> HOME"),
                   ((0.3,0.3,1.6),"uncertainty high   -> EXPLORE (epistemic)")]:
        a=fresh(); acc,_=a.activity(win=18,drives=dr)
        print(f"  {lbl:36s} drives={dr} -> {np.round(acc,2)}  WINNER={a.winner(acc)}")
    a=fresh(); acc1,_=a.activity(win=15,drives=(1.6,0.3,0.3))
    acc2,_=a.activity(win=18,drives=(0.3,1.6,0.3))
    print(f"  SWITCHING: foraging ({a.winner(acc1)}) then home-vector grows -> {a.winner(acc2)}")
    print("  interoception: hunger ladder fills tonically, EATING drains it")
    a=fresh(); _,h0=a.activity(win=10,drives=(0,0,0))
    _,h1=a.activity(win=40,drives=(0,0,0),hunger_fill=1.2)
    _,h2=a.activity(win=40,drives=(0,0,0),eat=2.0)
    print(f"     hunger activity: start={h0:.1f} -> after feeding-drive={h1:.1f} -> after EATING={h2:.1f}")
    print("@@@ARBITER DONE@@@")
