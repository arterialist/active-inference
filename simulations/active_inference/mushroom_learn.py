"""MUSHROOM BODY — Phase 2: DOPAMINE-GATED ASSOCIATIVE LEARNING (KC -> MBON plasticity).

Builds on the Phase-1 Kenyon-cell sparse expansion (mushroom_body.py) and adds a mushroom-body output
neuron (MBON) = an approach/valence readout, with PLASTIC KC->MBON synapses (reward_hebb). A dopamine
signal (delivered as the MBON's neuromodulator via set_external_input mod) gates plasticity:
  reward   -> nm>1  -> active KC->MBON synapses POTENTIATE  (this odour drives approach more)
  punish   -> nm~0  -> active KC->MBON synapses DEPRESS      (this odour drives approach less = avoid)
Because the KC code is SPARSE + orthogonal, each odour trains a DISJOINT synapse set, so distinct odours
learn distinct valences without homogenising (the failure mode of non-sparse reward-hebb).

Experiment: train odour A with reward, odour B with punishment; then present each WITHOUT dopamine and
measure the MBON response. Success = MBON(A) high, MBON(B) low (learned appetitive vs aversive valence).
"""
import sys, numpy as np
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
import importlib.util, pathlib
_s=importlib.util.spec_from_file_location("mb", str(pathlib.Path(__file__).parent/"mushroom_body.py"))
mb=importlib.util.module_from_spec(_s); _s.loader.exec_module(mb)
from paula_agent import ckit as k
def Tt(n): return 900+n
NPN,NKC,KIN=mb.NPN,mb.NKC,mb.KIN; PN,KC,APL,KC_PRE=mb.PN,mb.KC,mb.APL,mb.KC_PRE
MBON=95; DOP=NKC   # MBON id; DOP = dopamine synapse index on the MBON

def build(w_pk=2.2, r_kc=4.2, w_apl_kc=-1.4, w_kc_apl=0.5, r_apl=6.0,
          w_km0=1.1, r_mbon=1.2, eta=0.06, kappa=1.6, rh_decay=1.0):
    ne=[];sy=[];conns=[];ex=[]
    # PN -> KC -> APL (Phase 1)
    for p in PN: ne.append(k.neuron(p,r=0.6,c=2,lam=4)); sy.extend([k.syn(p,0,1.0,1),k.term(p,Tt(p))]); ex.append(k.ext(p,0))
    for kc in KC:
        ne.append(k.neuron(kc,r=r_kc,c=2,lam=4)); j=0
        for p in KC_PRE[kc]: sy.append(k.syn(kc,j,w_pk,1)); conns.append(k.conn(p,kc,j,Tt(p))); j+=1
        sy.append(k.syn(kc,j,w_apl_kc,1)); conns.append(k.conn(APL,kc,j,Tt(APL)))
        sy.append(k.term(kc,Tt(kc)))
    ne.append(k.neuron(APL,r=r_apl,c=1,lam=3)); j=0
    for kc in KC: sy.append(k.syn(APL,j,w_kc_apl,1)); conns.append(k.conn(kc,APL,j,Tt(kc))); j+=1
    sy.append(k.term(APL,Tt(APL)))
    # MBON: plastic reward_hebb readout from ALL KCs + a dopamine port (ext syn index DOP)
    ne.append(k.neuron(MBON, r=r_mbon, c=2, lam=5, plasticity="reward_hebb",
                       eta_post=eta, kappa=kappa, rh_decay=rh_decay))
    for i,kc in enumerate(KC): sy.append(k.syn(MBON,i,w_km0,1)); conns.append(k.conn(kc,MBON,i,Tt(kc)))
    sy.append(k.syn(MBON,DOP,0.0,1)); ex.append(k.ext(MBON,DOP))   # dopamine port (carries mod, info stays 0)
    sy.append(k.term(MBON,Tt(MBON)))
    return k.build(ne,sy,conns,ex)

def present(net,core,nb,odor,ticks,drive=3.0,dop=None):
    """Drive PNs with `odor` for `ticks`; optionally deliver dopamine mod=[stress,reward] to the MBON.
    Returns MBON spike count over the window."""
    m=0
    for t in range(ticks):
        for i,p in enumerate(PN): net.set_external_input(p,0,drive*float(odor[i]))
        if dop is not None: net.set_external_input(MBON,DOP,0.0,mod=np.array(dop,dtype=float))
        core.do_tick()
        m+=nb[MBON].O>0
    # clear PN drive
    for p in PN: net.set_external_input(p,0,0.0)
    net.set_external_input(MBON,DOP,0.0,mod=np.array([0.0,0.0]))
    return m

def mbon_weights(nb):
    n=nb[MBON]; return np.array([n.postsynaptic_points[i].u_i.info for i in range(NKC)])

def kc_set(odor, seed_free=True):
    """Which KC indices fire for this odour (fresh network so no carry-over)."""
    net,core=k.load(build()); nb={i:u for i,u in net.network.neurons.items()}
    fired=np.zeros(NKC)
    for t in range(40):
        for i,p in enumerate(PN): net.set_external_input(p,0,3.0*float(odor[i]))
        core.do_tick()
        for i,kc in enumerate(KC): fired[i]+=nb[kc].O>0
    return np.where(fired>0)[0]

if __name__=="__main__":
    rng=np.random.default_rng(5)
    CARD=6
    def mk():
        v=np.zeros(NPN); v[rng.choice(NPN,size=CARD,replace=False)]=1.0; return v
    A,B=mk(),mk()
    while np.array_equal(A,B): B=mk()
    C=mk()  # novel (untrained) control
    kcA,kcB,kcC=kc_set(A),kc_set(B),kc_set(C)
    print(f"KC-set sizes: A={len(kcA)} B={len(kcB)} C={len(kcC)}  A∩B={len(set(kcA)&set(kcB))}")
    net,core=k.load(build()); nb={i:u for i,u in net.network.neurons.items()}
    R=[0.0,3.0]; PUN=[3.0,0.0]   # mod=[stress,reward]
    def test(): return (present(net,core,nb,A,40),present(net,core,nb,B,40),present(net,core,nb,C,40))
    def wstat():
        w=mbon_weights(nb)
        return (w[kcA].mean() if len(kcA) else 0, w[kcB].mean() if len(kcB) else 0, w[kcC].mean() if len(kcC) else 0)
    print("MBON spikes/40t (A_rew,B_pun,C_novel) | mean KC->MBON weight [wA wB wC]:")
    print(f"  pre-train  test={test()}  w={tuple(round(x,2) for x in wstat())}")
    for ep in range(30):
        present(net,core,nb,A,25,dop=R)      # appetitive
        present(net,core,nb,B,25,dop=PUN)    # aversive
        if ep%6==5:
            print(f"  ep{ep+1:2d}      test={test()}  w={tuple(round(x,2) for x in wstat())}")
    mA,mB,mC=test()
    print(f"  FINAL appetitive/aversive: A={mA} B={mB} C={mC}")
    ok1 = mA > mB + 4
    # ---- REVERSAL: now punish A, reward B -> valences should FLIP (proves genuine association) ----
    for ep in range(30):
        present(net,core,nb,A,25,dop=PUN)
        present(net,core,nb,B,25,dop=R)
    mA2,mB2,mC2=test()
    print(f"  AFTER REVERSAL:            A={mA2} B={mB2} C={mC2}  (expect A<B now)")
    ok2 = mB2 > mA2 + 4
    print(f"  VERDICT: discrimination={'YES' if ok1 else 'NO'}  reversal={'YES' if ok2 else 'NO'} "
          f"-> {'ASSOCIATIVE MEMORY WORKS' if (ok1 and ok2) else 'INCOMPLETE'}")
    print("@@@LEARN DONE@@@")
