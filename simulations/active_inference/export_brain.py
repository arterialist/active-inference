"""Export the selected PAULA version topology: every emitted neuron, connection, and synapse distance.
Then lay it out in 3D with a force solver whose spring REST LENGTH is the real dendritic distance, so
dendritic length literally occupies space. ``AIF_VERSION`` selects the strict version profile."""
import importlib.util,sys,json,numpy as np
sys.path.insert(0,'/Users/arterialist/Projects/agi-research/neuron-model')
spec=importlib.util.spec_from_file_location('ag','aif_agent3d.py')
AG=importlib.util.module_from_spec(spec); spec.loader.exec_module(AG)
# The topology must reflect the CONFIGURATION actually running, not a fixed default: structural gates
# (trise, vac, accum, w_peg, w_lgi) build or omit whole populations. live_brain passes the current
# override set through AIF_KWARGS so the wiring diagram matches the live network.
import os as _os
_KW=json.loads(_os.environ.get("AIF_KWARGS","{}"))
_VERSION=_os.environ.get("AIF_VERSION","v1").lower()
# The live topology exporter must build the same version as the worker.  The
# version contract is intentionally represented as ordinary builder component
# names here so this subprocess remains independent of the worker's module
# objects and never creates a second controller.
_VERSION_COMPONENTS={
    "v1": ("sensory.olfactory_valence", "motor.cpg_muscle"),
    "v2": ("sensory.olfactory_valence", "motor.cpg_muscle", "learning.mushroom_body"),
    "v3": ("sensory.olfactory_valence", "motor.cpg_muscle", "learning.mushroom_body",
           "arbitration.foraging_exploration", "body.metabolic_organs", "arbitration.metabolic_sleep"),
    "v4": ("sensory.olfactory_valence", "motor.cpg_muscle", "learning.mushroom_body",
           "arbitration.foraging_exploration", "body.metabolic_organs", "arbitration.metabolic_sleep",
           "body.obstacle_geometry", "sensory.obstacle_proximity", "motor.obstacle_reflex"),
}
if _VERSION not in _VERSION_COMPONENTS:
    raise SystemExit(f"unknown AIF_VERSION={_VERSION!r}")
_KW["components"]=_VERSION_COMPONENTS[_VERSION]
_ATTRS={k:_KW.pop(k) for k in ("tonic_amp","tonic_gate","k_ang","seed") if k in _KW}
_OUT=_os.environ.get("AIF_TOPO_OUT","brain_topology.json")
_ITERS=int(_os.environ.get("AIF_TOPO_ITERS","900"))
np.random.seed(0); a=AG.AIFAgent3D(seed=1,**_KW)
for _k,_v in _ATTRS.items():
    if _k!="seed": setattr(a,_k,_v)
net=a.net.network

REG=[]  # (test, name, colour)
def R(pred,name,col): REG.append((pred,name,col))
vc,cc,nv,ar=AG.vc,AG.cc,AG.nv,AG.ar
KCset=set(AG.KC); PNset=set(AG.PN); MBset=set(AG.MBONP)
R(lambda n: n in set(cc.RING),"compass ring","#4cc2ff")
R(lambda n: n==cc.GI,"compass GI","#2b7fb8")
R(lambda n: n in set(cc.D7),"Delta-7 inhibitors","#1d6fa5")
R(lambda n: 300<=n<300+2*cc.NR*cc.NP,"P-EN shift","#7ad3ff")
R(lambda n: n in set(cc.PB_CL+cc.PB_CR),"PB update relays","#63cdda")
R(lambda n: n in set(cc.PB_ML+cc.PB_MR),"PB maintenance tracts","#4ab6c7")
R(lambda n: n in set(cc.PB_PHASE_CL+cc.PB_PHASE_CR),"PB phase-gated update","#77dd77")
R(lambda n: n in set(cc.PB_PHASE_CLOCK),"PB phase clock","#a8e6a3")
R(lambda n: n in set(cc.PG),"PG readout","#9ae6ff")
R(lambda n: 70000<=n<71000,"retina PR","#ffd166")
R(lambda n: 71000<=n<73000,"ON/OFF","#f4a261")
R(lambda n: 73000<=n<74000,"V1","#e76f51")
R(lambda n: 74000<=n<75000,"V2","#d1495b")
R(lambda n: 75000<=n<76000,"AZ","#c04070")
R(lambda n: 76000<=n<78000,"chroma","#b5179e")
R(lambda n: 78000<=n<79000,"EMD motion","#8e44ad")
R(lambda n: 79700<=n<79800,"HS wide-field","#a06cd5")
R(lambda n: 79000<=n<79600,"salience","#7209b7")
R(lambda n: n in set(AG.VR),"visual ring","#00b4d8")
R(lambda n: n in set(AG.PE),"prediction error","#ff006e")
R(lambda n: n in set(AG.UNC),"uncertainty","#fb5607")
R(lambda n: n in PNset,"PN","#06d6a0")
R(lambda n: n in KCset,"Kenyon cells","#2a9d8f")
R(lambda n: n==AG.APL,"APL","#1b6b5f")
R(lambda n: n in MBset,"MBON","#e63946")
R(lambda n: n==AG.AVOID,"AVOID","#c1121f")
R(lambda n: n in (AG.ORN_F,AG.ORN_T,AG.ALN),"antennal lobe","#52b788")
R(lambda n: n in (AG.STG_T,AG.STG_F,AG.RFX),"sting latch","#bc4749")
R(lambda n: n in set(AG.FLp+AG.FRp),"odour sensors","#95d5b2")
R(lambda n: n in set(AG.TXL+AG.TXR),"toxin sensors","#ff8fab")
R(lambda n: n in set(AG.OBL+AG.OBR),"obstacle range","#90dbf4")
R(lambda n: n in set(AG.OBDL+AG.OBDR),"obstacle onset","#48cae4")
R(lambda n: n in (AG.OBS_LEFT,AG.OBS_RIGHT,AG.OBS_BRAKE,AG.OBS_WALL),"obstacle reflex","#00b4d8")
R(lambda n: n in (AG.POOL,AG.STEER,AG.TL,AG.TR,AG.HTL,AG.HTR),"steering","#ffb703")
R(lambda n: n in set(AG.RISEP),"RISE trend","#fdc500")
R(lambda n: n in set(AG.ENGN),"engine","#ffea00")
R(lambda n: n in set(AG.CPGP),"CPG","#8ecae6")
R(lambda n: n in (AG.PHASE_VEST_CCW,AG.PHASE_VEST_CW),"phase vestibular afferents","#a3f7bf")
R(lambda n: n in (AG.MLp,AG.MLr,AG.MRp,AG.MRr),"muscles","#f72585")
R(lambda n: any(n in v for v in AG.RLY.values()),"motor relays","#ff70a6")
R(lambda n: 9100<=n<9500,"CPU4 ladders","#80ed99")
R(lambda n: 9000<=n<9100,"CD speed cells","#57cc99")
R(lambda n: 9500<=n<9800,"CPU1 / opponent","#38a3a5")
R(lambda n: n in set(ar.SLEEP_MODE),"sleep mode","#6fe7c7")
R(lambda n: 40000<=n<41000,"arbiter modes","#c77dff")
R(lambda n: 41000<=n<42000,"hunger","#e0aaff")
R(lambda n: 88300<=n<88340,"metabolic afferents","#80ed99")
R(lambda n: n in (AG.BL,AG.BR,AG.UBEL,AG.DL,AG.DR),"belief core","#ff9e00")
# ---- populations that are OPT-IN (structural gates). Without these predicates they fell into the
# catch-all and piled onto a single blob, so the diagram silently misrepresented them.
R(lambda n: n in (cc.XACC,cc.YACC),"PI analog (XACC/YACC)","#00b4d8")
R(lambda n: n in set(cc.PEG),"P-EG maintenance","#48cae4")
R(lambda n: 9900<=n<10100,"PI accumulator","#90e0ef")
R(lambda n: n==10100,"PI accum inhibitor","#0096c7")
R(lambda n: 10200<=n<10400,"PI accum (magnitude)","#ade8f4")
R(lambda n: n==AG.TPOOL,"toxin pool","#ff4d6d")
R(lambda n: n in set(AG.TRISE),"toxin RISE trend","#ff758f")
R(lambda n: n in set(AG.VPN),"vAC colour PN","#c9184a")
R(lambda n: n in set(AG.VKC),"vAC Kenyon cells","#a4133c")
R(lambda n: n==AG.VAPL,"vAC APL","#800f2f")
def region(n):
    for pred,name,col in REG:
        try:
            if pred(n): return name,col
        except Exception: pass
    return "other","#8a96a3"

# ---- neuron groups: the functional sub-population a cell belongs to inside its region -----------
# Derived from the id arithmetic the builders use, so these are the real populations (ON vs OFF sheet,
# the four V1 orientations, the two shift directions, each CPU4 column), not cosmetic buckets.
NAZ_,NEL_,NV1AZ_,NV1EL_=vc.NAZ,vc.NEL,vc.NV1AZ,vc.NV1EL
ORI=["vertical edge","diagonal /","horizontal edge","diagonal \\"]
MUSCLE={AG.MLp:"left protractor",AG.MLr:"left retractor",
        AG.MRp:"right protractor",AG.MRr:"right retractor"}
STEERN={AG.POOL:"normalisation pool",AG.STEER:"steer command",AG.TL:"turn left",AG.TR:"turn right",
        AG.HTL:"home turn left",AG.HTR:"home turn right"}
RLYG={}
for m,lst in AG.RLY.items():
    for nid in lst: RLYG[nid]=MUSCLE.get(m,"relay")+" relay"
MODEG={}
for mi,lst in enumerate(ar.MODE):
    for nid in lst: MODEG[nid]=ar.MODES[mi].lower()+" mode"
for nid in ar.SLEEP_MODE:
    MODEG[nid]="sleep mode"
def group(nid, rname):
    if rname=="chroma":     return "red-green opponent" if nid<77000 else "blue-yellow opponent"
    if rname=="ON/OFF":     return "ON centre" if nid<72000 else "OFF centre"
    if rname=="V1":         return ORI[(nid-73000)//(NV1AZ_*NV1EL_)]
    if rname=="EMD motion": return "progressive" if nid<78100 else "regressive"
    if rname=="salience":   return "salience map" if nid<79500 else "normalisation pool"
    if rname=="P-EN shift": return "shift clockwise" if (nid-300)<cc.NR*cc.NP else "shift anticlockwise"
    if rname=="PB update relays":
        return "CCW update tract" if nid in set(cc.PB_CL) else "CW update tract"
    if rname=="PB maintenance tracts":
        return "left maintenance tract" if nid in set(cc.PB_ML) else "right maintenance tract"
    if rname=="PB phase-gated update":
        return "CCW phase gate" if nid in set(cc.PB_PHASE_CL) else "CW phase gate"
    if rname=="PB phase clock": return f"gait phase {(nid-cc.PB_PHASE_CLOCK[0]):02d}"
    if rname=="phase vestibular afferents": return "CCW fast gyro" if nid==AG.PHASE_VEST_CCW else "CW fast gyro"
    if rname=="CPU4 ladders": return f"column {(nid-9100)//nv.NL:02d}"
    if rname=="CPU1 / opponent": return ("left comparator" if nid<9600 else
                                         "right comparator" if nid<9700 else "opponent")
    if rname=="Kenyon cells": return "left calyx" if KClist.index(nid)<len(KClist)/2 else "right calyx"
    if rname=="PN":          return "left antennal tract" if PNlist.index(nid)<len(PNlist)/2 else "right antennal tract"
    if rname=="odour sensors": return "left antenna" if nid in set(AG.FLp) else "right antenna"
    if rname=="toxin sensors": return "left antenna" if nid in set(AG.TXL) else "right antenna"
    if rname=="obstacle range": return "left whisker" if nid in set(AG.OBL) else "right whisker"
    if rname=="obstacle onset": return "left onset" if nid in set(AG.OBDL) else "right onset"
    if rname=="obstacle reflex": return {AG.OBS_LEFT:"obstacle-left", AG.OBS_RIGHT:"obstacle-right",
                                           AG.OBS_BRAKE:"shared brake", AG.OBS_WALL:"wall gate"}.get(nid,"reflex")
    if rname=="antennal lobe": return {AG.ORN_F:"food receptor",AG.ORN_T:"toxin receptor"}.get(nid,"local interneuron")
    if rname=="sting latch":   return {AG.STG_T:"toxin latch",AG.STG_F:"food latch"}.get(nid,"reflex latch")
    if rname=="muscles":       return MUSCLE.get(nid,"muscle")
    if rname=="motor relays":  return RLYG.get(nid,"relay")
    if rname=="arbiter modes": return MODEG.get(nid,"mode")
    if rname=="sleep mode": return "sleep mode"
    if rname=="metabolic afferents":
        return ("gut load" if nid < 88310 else "energy store" if nid < 88320
                else "low energy" if nid < 88330 else "digestion")
    if rname=="steering":      return STEERN.get(nid,"steering")
    return rname
KClist=sorted(AG.KC); PNlist=sorted(AG.PN)

nodes=[]; idx={}
for nid in sorted(net.neurons.keys()):
    nm,col=region(nid); idx[nid]=len(nodes)
    nodes.append(dict(id=int(nid),r=nm,g=group(nid,nm),c=col))
edges=[]
for (src,term),targets in net.connection_cache.items():
    for (tgt,syn) in targets:
        if src in idx and tgt in idx:
            # dendritic distance lives in neuron.distances[synapse_id] (NOT on the PostsynapticPoint --
            # reading it there silently yields the fallback for every edge, which flattens the whole layout)
            d=float(net.neurons[tgt].distances.get(syn,1))
            w=0.0
            try: w=float(net.neurons[tgt].postsynaptic_points[syn].u_i.info)
            except Exception: pass
            edges.append((idx[src],idx[tgt],d,w))
print(f"{len(nodes)} neurons, {len(edges)} connections")
# ---- 3D layout ---------------------------------------------------------------------------------
# Two forces decide where a neuron sits:
#   (1) ANATOMY. Every region is seeded at its real place in an insect brain, in head coordinates
#       +x anterior, +y dorsal, +z right. Retinotopic sheets are seeded RETINOTOPICALLY: column a
#       sits at its own egocentric azimuth vc.az_of(a), on a shell whose radius is its processing
#       stage, so the optic lobe is a genuine layered, converging map of the visual field.
#   (2) TOPOLOGY. Springs whose REST LENGTH is the synapse's dendritic distance.
# The region CENTROID is pinned hard (areas stay where anatomy puts them) while individual neurons
# are pulled only weakly, so within an area the wiring still decides the geometry.
N=len(nodes); rng=np.random.default_rng(3)
regs=sorted({n["r"] for n in nodes})
NAZ,NEL,NV1AZ,NV1EL=vc.NAZ,vc.NEL,vc.NV1AZ,vc.NV1EL

def sheet(j,n_e,n_a,R,y0=0.0,dy=3.1):
    """retinotopic seat: azimuth column -> its true egocentric bearing on a shell of radius R"""
    a,e=j//n_e,j%n_e
    th=vc.az_of(a,n_a)                                  # + = left
    return np.array([R*np.cos(th),y0-(e-(n_e-1)/2.0)*dy,-R*np.sin(th)])
def arc(j,k,R,y0=0.0,cx=0.0):
    th=vc.az_of(j,max(k,1))
    return np.array([cx+R*np.cos(th),y0,-R*np.sin(th)])
def ring(j,k,c,R,plane="xz"):
    t=2*np.pi*j/max(k,1); c=np.asarray(c,float)
    return c+(np.array([R*np.cos(t),0,R*np.sin(t)]) if plane=="xz" else np.array([R*np.cos(t),R*np.sin(t),0]))
def blob(c,s=2.4): return np.asarray(c,float)+rng.normal(0,s,3)
def bilat(j,k,c,s=2.0):
    """left half of the population on the left side of the head, right half on the right"""
    c=np.asarray(c,float); side=-1.0 if j<k/2 else 1.0
    return np.array([c[0],c[1],side*abs(c[2])])+rng.normal(0,s,3)

def seat(rname,j,k):
    # ---- optic lobe: peripheral retina -> lamina -> medulla -> lobula -> central, all retinotopic
    if rname=="retina PR":  return sheet(j,NEL,NAZ,38.0,2.0,3.4)
    if rname=="chroma":     return sheet(j%80,NEL,NAZ,35.5-2.2*(j//80),2.0,3.4)   # RG sheet, BY sheet
    if rname=="ON/OFF":     return sheet(j%80,NEL,NAZ,31.0-2.2*(j//80),2.0,3.2)   # ON sheet, OFF sheet
    if rname=="V1":         return sheet(j%(NV1AZ*NV1EL),NV1EL,NV1AZ,25.0-1.9*(j//(NV1AZ*NV1EL)),1.5,3.0)
    if rname=="V2":         return sheet(j,NV1EL,NV1AZ,17.0,1.5,2.8)
    if rname=="EMD motion": return arc(j%20,20,15.0,-4.5+3.0*(j//20))             # progressive / regressive
    if rname=="AZ":         return arc(j,k,12.0,5.0)
    if rname=="salience":   return arc(j,19,10.0,8.5) if j<19 else blob([2,12,0],1.0)
    if rname=="visual ring":return ring(j,k,[-1,9.5,0],6.0)
    # ---- antennae and mouthparts: the most anterior, ventral structures
    if rname=="odour sensors":  return bilat(j,k,[34,-9,11],1.6)
    if rname=="toxin sensors":  return bilat(j,k,[32,-13,12],1.6)
    if rname=="obstacle range": return bilat(j,k,[31,-11,13],1.4)
    if rname=="obstacle onset": return bilat(j,k,[28,-7,10],1.2)
    if rname=="antennal lobe":  return bilat(j,k,[24,-9,6],1.4)
    if rname=="sting latch":    return blob([10,-16,0],1.6)                        # subesophageal reflex
    # ---- mushroom body: calyx dorsal-posterior, peduncle forward, lobes anterior-ventral
    if rname=="PN":            return bilat(j,k,[14,4,7],1.6)
    if rname=="Kenyon cells":  return bilat(j,k,[-6,20,13],4.2)                    # the two calyces
    if rname=="APL":           return np.array([-4.0,17.0,0.0])
    if rname=="MBON":          return bilat(j,k,[9,6,8],1.6)                       # vertical/medial lobes
    if rname=="AVOID":         return np.array([6.0,-2.0,0.0])
    # ---- central complex: strictly on the midline, dorsal
    if rname=="compass ring":  return ring(j,k,[-5,8,0],7.5)                       # ellipsoid body
    if rname=="compass GI":    return np.array([-5.0,8.0,0.0])
    if rname=="Delta-7 inhibitors": return ring(j,k,[-9,11.5,0],5.5)
    if rname=="HS wide-field": return bilat(j,k,[6,-3,9],1.4)
    if rname=="PG readout":    return ring(j,k,[-5,8,0],4.4)
    if rname=="P-EN shift":                                                        # protocerebral bridge:
        d,c=j//(36*4),(j%(36*4))//4                                                # a horizontal BAR
        return np.array([-13.0,13.5+2.2*d,(c-17.5)*1.05])+rng.normal(0,.55,3)
    if rname=="PB update relays":                                                   # explicit PB columns, bilaterally split
        side=-1.0 if j<k/2 else 1.0
        col=j%(max(k//2,1))
        return np.array([-14.0,10.8,(col-(k//2-1)/2)*1.05+side*2.1])+rng.normal(0,.38,3)
    if rname=="PB maintenance tracts":                                              # E-PG -> PB -> P-EG bridge
        side=-1.0 if j<k/2 else 1.0
        col=j%(max(k//2,1))
        return np.array([-12.0,12.4,(col-(k//2-1)/2)*1.05+side*2.1])+rng.normal(0,.38,3)
    if rname=="PB phase-gated update":
        side=-1.0 if j<k/2 else 1.0
        col=j%(max(k//2,1))
        return np.array([-14.5,8.9,(col-(k//2-1)/2)*1.05+side*2.1])+rng.normal(0,.38,3)
    if rname=="PB phase clock": return arc(j,k,4.0,2.0,cx=-18.0)
    if rname=="phase vestibular afferents": return bilat(j,k,[-16.0,4.5,3.2],.5)
    if rname=="CD speed cells":return bilat(j,k,[-11,4,5],1.4)
    if rname=="CPU4 ladders":                                                      # fan-shaped body:
        c,l=j//16,j%16                                                             # 12 columns x 16 rungs,
        return np.array([-9.0,-1.0+l*1.35,(c-5.5)*2.3])+rng.normal(0,.35,3)        # drawn as real ladders
    if rname=="CPU1 / opponent":return np.array([-6.0,-6.0,(j%12-5.5)*2.0])+rng.normal(0,.7,3)
    # ---- superior/lateral protocerebrum: the inferential layer
    if rname=="belief core":       return blob([-19,10,0],2.2)
    if rname=="prediction error":  return blob([-21,5,0],2.2)
    if rname=="uncertainty":       return blob([-23,0,0],2.2)
    if rname=="arbiter modes":     return blob([-22,-6,0],2.6)
    if rname=="hunger":            return blob([-17,-11,0],2.2)
    # ---- descending motor path: posterior, ventral, exiting toward the body
    if rname=="RISE trend":  return blob([-26,-6,0],1.8)
    if rname=="steering":    return bilat(j,k,[-29,-9,5],1.4)
    if rname=="obstacle reflex": return bilat(j,k,[-31,-8,5],1.1)
    if rname=="engine":      return blob([-32,-11,0],1.6)
    if rname=="CPG":         return blob([-35,-13,0],1.4)
    if rname=="motor relays":return bilat(j,k,[-38,-14,6],2.0)
    if rname=="muscles":     return bilat(j,k,[-43,-17,9],1.2)
    # opt-in populations, seated near the structures they belong to
    if rname=="PI analog (XACC/YACC)": return blob([-9,3,0],1.0)
    if rname=="P-EG maintenance":      return ring(j,k,[-13,11,0],6.4)
    if rname=="PI accumulator":        return np.array([-11.0,-3.0+ (j%6)*1.1,((j//6)-5.5)*2.0])+rng.normal(0,.3,3)
    if rname=="PI accum inhibitor":    return np.array([-11.0,-9.0,0.0])
    if rname=="PI accum (magnitude)":  return np.array([-14.0,-3.0+(j%6)*1.1,((j//6)-5.5)*2.0])+rng.normal(0,.3,3)
    if rname=="toxin pool":            return np.array([26.0,-15.0,0.0])
    if rname=="toxin RISE trend":      return blob([20,-17,0],1.5)
    if rname=="vAC colour PN":         return bilat(j,k,[10,10,9],1.6)
    if rname=="vAC Kenyon cells":      return bilat(j,k,[-2,22,15],3.4)
    if rname=="vAC APL":               return np.array([-1.0,19.0,0.0])
    return blob([-2,-24,0],2.0)

P=np.zeros((N,3)); SEED=np.zeros((N,3))
for rname in regs:
    ids=[i for i,n in enumerate(nodes) if n["r"]==rname]
    for j,i in enumerate(ids): SEED[i]=seat(rname,j,len(ids))
P[:]=SEED
RIDS=[np.array([i for i,n in enumerate(nodes) if n["r"]==r],dtype=int) for r in regs]
RSEEDC=[SEED[ii].mean(0) for ii in RIDS]
PAIR=[(u,v,d) for u,v,d,_ in edges if u!=v]
E=np.array([(u,v) for u,v,_ in PAIR],dtype=int)
L=np.array([max(1.0,d) for _,_,d in PAIR],dtype=float)
L=1.9*L**0.62                              # dendritic distance -> spring REST LENGTH (monotone in d)
# --- solver ------------------------------------------------------------------------------------
# ANATOMY and DENDRITIC LENGTH are not simultaneously satisfiable, and pretending otherwise is a lie:
# pinning the areas where a real brain puts them drove corr(rest length, drawn length) to -0.01, because
# an inter-area projection's drawn length is then decided by anatomy, not by its dendrite. So the solver
# runs TWICE and the page ships BOTH layouts, with a morph slider between them.
#   anatomy=True   areas rigidly at their anatomical seat; wiring shapes only the inside of each area
#   anatomy=False  no seats at all: pure springs, so drawn length IS dendritic distance
DEG=np.zeros(N,dtype=int)
for u,v,_ in PAIR: DEG[u]+=1; DEG[v]+=1
FREE=DEG==0                                 # cells no synapse touches -- wiring says nothing about them
def solve(anatomy, iters=900, seed_start=True, tag=""):
    P=(SEED.copy() if seed_start else rng.normal(0,14,(N,3)))
    # A neuron with no synapse has no spring to obey, so in the wiring layout it would keep its random
    # start and scatter as dust across the frame. Leave it on its anatomical seat instead and freeze it.
    P[FREE]=SEED[FREE]
    for it in range(iters):
        F=np.zeros_like(P)
        if anatomy: F+= (SEED-P)*(0.055*(1.0-0.72*it/iters))
        dvec=P[E[:,1]]-P[E[:,0]]; dist=np.linalg.norm(dvec,axis=1)+1e-6
        f=((dist-L)/dist)[:,None]*dvec*0.34     # strong springs so rest length actually wins
        np.add.at(F,E[:,0], f); np.add.at(F,E[:,1],-f)
        # short-range repulsion via a coarse grid (keeps neurons from piling up)
        cell=4.0; keys=np.floor(P/cell).astype(np.int64)
        order=np.lexsort((keys[:,2],keys[:,1],keys[:,0]))
        ks=keys[order]; start=0
        for i in range(1,len(order)+1):
            if i==len(order) or (ks[i]!=ks[start]).any():
                grp=order[start:i]
                if len(grp)>1:
                    q=P[grp]; c=q.mean(0); r=q-c; rn=np.linalg.norm(r,axis=1)+1e-6
                    F[grp]+= r/rn[:,None]*np.clip(2.2/rn[:,None],0,1.1)
                start=i
        F[FREE]=0.0
        P+=np.clip(F,-2.0,2.0)
        if anatomy:
            # re-imposed as a POSITIONAL correction, not a force: as a force it was destroyed by the
            # per-step clip (a 3-cell area wired hard into the mushroom body saturated the clip on
            # springs alone and drifted right out of its seat). Translating the area back is clip-proof.
            for ii,c0 in zip(RIDS,RSEEDC): P[ii]+= (c0-P[ii].mean(0))*0.9
        if it%300==299:
            seg=np.linalg.norm(P[E[:,1]]-P[E[:,0]],axis=1)
            print(f"  {tag} iter {it+1}: corr(rest,rendered)={np.corrcoef(L,seg)[0,1]:.3f}")
    P-=P.mean(0); P/= (np.abs(P).max()/48.0)
    return P
PA=solve(True , iters=_ITERS, tag="anatomy")
PW=solve(False, iters=_ITERS, tag="wiring ",seed_start=False)
def report(P,tag):
    seg=np.linalg.norm(P[E[:,1]]-P[E[:,0]],axis=1)
    dd=np.array([d for _,_,d in PAIR]); out=[]
    for k in sorted(set(dd.tolist())):
        m=dd==k
        if m.sum()>25: out.append(f"d={k:.0f}:{seg[m].mean():.1f}")
    print(f"  {tag}: corr={np.corrcoef(L,seg)[0,1]:.3f}  " + "  ".join(out))
report(PA,"anatomy"); report(PW,"wiring ")
for i,n in enumerate(nodes):
    n["p"]=[round(float(x),2) for x in PA[i]]      # anatomical layout
    n["q"]=[round(float(x),2) for x in PW[i]]      # dendritic-length layout
json.dump(dict(nodes=nodes,
               edges=[[u,v,round(dd,2)] for u,v,dd,_ in edges],
               regions=[{"name":r,"c":next(n["c"] for n in nodes if n["r"]==r),
                         "n":sum(1 for n in nodes if n["r"]==r),
                         "g":[{"name":g,"n":sum(1 for n in nodes if n["r"]==r and n["g"]==g)}
                              for g in sorted({n["g"] for n in nodes if n["r"]==r})]}
                        for r in regs]),
          open(_OUT,"w"))
print(f"wrote {_OUT} |", len(regs), "regions |", len(nodes), "nodes")
