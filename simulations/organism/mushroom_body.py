"""Insect brain — REGION 1: the MUSHROOM BODY (olfactory associative learning & memory).
Faithful to the Drosophila model: projection neurons (antennal lobe) encode an odor as a population
pattern -> a large layer of Kenyon cells forms a SPARSE, high-dimensional code (each KC samples a few
random PNs, high threshold -> only a few fire per odor) -> mushroom-body output neurons (MBONs) read
the KCs through DOPAMINE-GATED plastic synapses (reward_hebb). Reward/punishment during an odor writes
its valence onto the active KCs. Result: the brain LEARNS which odor to approach vs avoid, and
discriminates novel odors. This is a real learning+memory centre, not a reflex.

Run standalone:  python -m simulations.organism.mushroom_body
"""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root -> `simulations` importable
from simulations.paula_loader import ensure_paula_available
ensure_paula_available()
from paula_agent import ckit as k

N_PN=12; N_KC=120; KC_FANIN=4
PN0, KC0, APP, AVO, APL = 0, 100, 900, 901, 950   # id bases (APL = global inhibitory sparseness neuron)

def build_mb(seed=0, kc_r=2.65, kc_w=1.6, apl_w=0.1, apl_inh=-5.0):   # tuned: ~sparse code, learns valence
    rng=np.random.RandomState(seed)
    ne=[]; sy=[]; conns=[]; ex=[]
    # projection neurons (odor channels) — driven externally
    for i in range(N_PN):
        nid=PN0+i; ne.append(k.neuron(nid, r=0.5, lam=4, c=2))
        sy += [k.syn(nid,0,1.0,1), k.term(nid,tid=900+nid)]; ex.append(k.ext(nid,0))
    # APL: pools all KCs (exc) and inhibits all KCs (feedback) -> winner-take-most sparseness
    ne.append(k.neuron(APL, r=0.8, lam=2, c=1))
    # Kenyon cells — each samples KC_FANIN random PNs (+ APL feedback inhibition -> sparse code)
    kc_pn={}
    for j in range(N_KC):
        nid=KC0+j; kc_pn[j]=[int(x) for x in rng.choice(N_PN, KC_FANIN, replace=False)]
        ne.append(k.neuron(nid, r=kc_r, lam=4, c=3))
        for s,pn in enumerate(kc_pn[j]):
            sy.append(k.syn(nid, s, kc_w, 1)); conns.append(k.conn(PN0+pn, nid, s, stid=900+PN0+pn))
        sy.append(k.syn(nid, KC_FANIN, apl_inh, 1))                 # APL -> KC (inhibitory feedback)
        conns.append(k.conn(APL, nid, KC_FANIN, stid=900+APL))
        sy.append(k.syn(APL, j, apl_w, 1))                          # KC -> APL (excitatory pool)
        conns.append(k.conn(nid, APL, j, stid=900+nid))
        sy.append(k.term(nid, tid=900+nid))
    sy.append(k.term(APL, tid=900+APL))
    # MBONs: approach + avoid, read ALL KCs via plastic (reward_hebb) synapses + a teaching line
    for out in (APP, AVO):
        ne.append(k.neuron(out, r=1.2, lam=6, c=2, plasticity="reward_hebb", eta_post=0.0, rh_decay=0.25, kappa=2.6))
        for j in range(N_KC):
            sy.append(k.syn(out, j, 0.05, 1)); conns.append(k.conn(KC0+j, out, j, stid=900+KC0+j))
        sy.append(k.syn(out, N_KC, 1.6, 1)); ex.append(k.ext(out, N_KC))  # teaching/US line
        sy.append(k.term(out, tid=900+out))
    path=k.build(ne,sy,conns,ex)
    return path, kc_pn

def odor_pattern(rng, n_active=6):
    v=np.zeros(N_PN); v[rng.choice(N_PN, n_active, replace=False)]=1.0; return v

def load(path):
    net,core=k.load(path); nb={nid:nu for nid,nu in net.network.neurons.items()}
    return net,core,nb

def present(net,core,nb,odor,teach=None,reward=0.0,stress=0.0,learn=False,T=16):
    """Drive PNs with the odor for T ticks; optionally teach an MBON with dopamine/stress. Returns
    (kc_active_set, app_spikes, avo_spikes)."""
    for out in (APP,AVO): nb[out].params.eta_post = 0.06 if learn else 0.0
    net.reset_simulation(); core.state.current_tick=0; net.current_tick=0
    kc=set(); app=avo=0
    for t in range(T):
        if 2<=t<12:
            for i in range(N_PN):
                if odor[i]>0: net.set_external_input(PN0+i, 0, 4.0)
        if learn and teach is not None and 2<=t<12:
            nb[teach].M_vector[1] = 2.6                            # dopamine gate ON (potentiate this MBON)
            if 3<=t<10: net.set_external_input(teach, N_KC, 5.0)   # US fires the taught MBON causally
        core.do_tick()
        for j in range(N_KC):
            if nb[KC0+j].O>0: kc.add(j)
        app+=int(nb[APP].O>0); avo+=int(nb[AVO].O>0)
    return kc, app, avo

if __name__=="__main__":
    path,kc_pn=build_mb(seed=1)
    net,core,nb=load(path)
    rng=np.random.RandomState(7)
    odors=[odor_pattern(rng) for _ in range(4)]   # 4 distinct odors
    # ---- sparsity + separability of the KC code (frozen) ----
    codes=[present(net,core,nb,od)[0] for od in odors]
    sizes=[len(c) for c in codes]
    overlaps=[len(codes[0]&codes[i])/max(1,len(codes[0]|codes[i])) for i in range(1,4)]
    print(f"KC sparse code: active per odor={sizes} of {N_KC} ({100*np.mean(sizes)/N_KC:.0f}% sparse)")
    print(f"KC code overlap odor0 vs others (Jaccard): {[round(o,2) for o in overlaps]}  (low = separable)")
    # ---- associative learning: odor A -> reward(approach), odor B -> punish(avoid) ----
    A,B=odors[0],odors[1]
    for ep in range(12):
        present(net,core,nb,A,teach=APP,reward=True, learn=True)
        present(net,core,nb,B,teach=AVO,reward=False,learn=True)
    # test (frozen)
    _,aA,vA=present(net,core,nb,A); _,aB,vB=present(net,core,nb,B)
    _,aC,vC=present(net,core,nb,odors[2])  # novel odor
    print(f"after training:  odor A -> approach={aA} avoid={vA}   odor B -> approach={aB} avoid={vB}")
    print(f"novel odor C -> approach={aC} avoid={vC}")
    learned = aA>vA and vB>aB
    print(f"VERDICT: {'MUSHROOM BODY LEARNED odor valence (approach A, avoid B)' if learned else 'needs tuning'} "
          f"| neurons={N_PN+N_KC+2}")
    print("@@@MB DONE@@@")
