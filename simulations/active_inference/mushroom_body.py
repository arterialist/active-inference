"""MUSHROOM BODY — insect associative-memory centre, built in PAULA. Phase 1 (this file): the Kenyon-cell
SPARSE EXPANSION that discriminates odours.

  projection neurons PN  (odour code, NPN channels)
     -> Kenyon cells KC   (NKC >> NPN; each KC = random sparse sample of a few PNs, high threshold)
     -> APL global inhibition (feedback) -> only a few KCs fire per odour  == sparse, high-dim code
A sparse KC code makes distinct odours nearly ORTHOGONAL (low overlap), which is what later lets a single
valence readout (MBON) associate each odour with reward/punishment. Here we build it and MEASURE the
discrimination (KC overlap between odours should be far below PN overlap). No plasticity yet.
"""
import sys, numpy as np
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
from paula_agent import ckit as k
def Tt(n): return 900+n

NPN=16; NKC=160; KIN=6         # PNs, Kenyon cells, PN-inputs per KC (bigger PN space -> selective KCs)
PN=list(range(0,NPN))
KC=list(range(100,100+NKC))
APL=90                          # global feedback inhibition
rng=np.random.default_rng(7)
# fixed random PN->KC wiring: each KC samples KIN distinct PNs
KC_PRE={kc: sorted(rng.choice(NPN,size=KIN,replace=False).tolist()) for kc in KC}

def build(w_pk=2.2, r_kc=4.2, w_kc_apl=0.5, w_apl_kc=-1.4, r_apl=6.0):
    ne=[];sy=[];conns=[];ex=[]
    # projection neurons: externally driven by the odorant vector (one ext current per PN)
    for p in PN: ne.append(k.neuron(p,r=0.6,c=2,lam=4)); sy.extend([k.syn(p,0,1.0,1),k.term(p,Tt(p))]); ex.append(k.ext(p,0))
    # Kenyon cells: high threshold, sum KIN random PNs (coincidence of its specific PN combination),
    # plus APL feedback inhibition (syn index KIN)
    for kc in KC:
        ne.append(k.neuron(kc,r=r_kc,c=2,lam=4)); j=0
        for p in KC_PRE[kc]: sy.append(k.syn(kc,j,w_pk,1)); conns.append(k.conn(p,kc,j,Tt(p))); j+=1
        sy.append(k.syn(kc,j,w_apl_kc,1)); conns.append(k.conn(APL,kc,j,Tt(APL)))   # global inhibition
        sy.append(k.term(kc,Tt(kc)))
    # APL: excited by all KCs -> inhibits all KCs (feedback normalisation for sparseness)
    ne.append(k.neuron(APL,r=r_apl,c=1,lam=3)); j=0
    for kc in KC: sy.append(k.syn(APL,j,w_kc_apl,1)); conns.append(k.conn(kc,APL,j,Tt(kc))); j+=1
    sy.append(k.term(APL,Tt(APL)))
    return k.build(ne,sy,conns,ex)

def present(net,core,nb,odor_vec,ticks=40,drive=3.0):
    """Drive the PNs with an odorant vector; return KC firing counts over the window."""
    kc_count=np.zeros(NKC); pn_count=np.zeros(NPN)
    for t in range(ticks):
        for i,p in enumerate(PN): net.set_external_input(p,0,drive*float(odor_vec[i]))
        core.do_tick()
        for i,p in enumerate(PN): pn_count[i]+=nb[p].O>0
        for i,kc in enumerate(KC): kc_count[i]+=nb[kc].O>0
    return pn_count, kc_count

if __name__=="__main__":
    net,core=k.load(build()); nb={i:u for i,u in net.network.neurons.items()}
    # a set of distinct odorants = fixed-cardinality PN patterns (each = exactly CARD active PNs), so
    # total input drive is balanced and KC sparseness reflects the CODE, not the odorant intensity.
    Nod=6; CARD=6
    odors=[]
    while len(odors)<Nod:
        v=np.zeros(NPN); v[rng.choice(NPN,size=CARD,replace=False)]=1.0
        if not any(np.array_equal(v,o) for o in odors): odors.append(v)
    pn_codes=[]; kc_codes=[]
    for o in odors:
        # fresh state per odorant
        net,core=k.load(build()); nb={i:u for i,u in net.network.neurons.items()}
        pn,kc=present(net,core,nb,o)
        pn_codes.append((pn>0).astype(float)); kc_codes.append((kc>0).astype(float))
    pn_codes=np.array(pn_codes); kc_codes=np.array(kc_codes)
    sparsity=kc_codes.mean(axis=1)   # fraction of KCs active per odour
    def overlap(codes):
        n=len(codes); ov=[]
        for i in range(n):
            for j in range(i+1,n):
                a,b=codes[i],codes[j]; d=np.sqrt(a.sum()*b.sum())
                ov.append(float((a@b)/d) if d>0 else 0.0)
        return np.mean(ov)
    print(f"NPN={NPN} NKC={NKC} KIN={KIN}")
    print(f"KC sparsity (frac active) per odour: {np.round(sparsity,3).tolist()}  mean={sparsity.mean():.3f}")
    print(f"mean pairwise overlap  PN-code = {overlap(pn_codes):.3f}   KC-code = {overlap(kc_codes):.3f}")
    print(f"-> discrimination: KC overlap should be MUCH lower than PN overlap (sparse code orthogonalises)")
    print("@@@MB DONE@@@")
