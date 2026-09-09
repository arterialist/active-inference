"""GATED-ARBITER AGENT — the fully-neural navigator (neural_agent.py) with a MODE-GATED ARBITER cortex
in place of the fixed tropotaxis. The SAME food sensor drives OPPOSITE behaviour depending on which mode
population is active: behaviour lives in the wiring (coincidence gates), not in a Python if/else.

Arbiter cortex (all populations, one-per-function):
  food sensors FL/FR -> OPPONENT direction detectors dFL/dFR  (which side is the gradient on)
  MODE populations APP / AVO  (winner-take-all: one behaviour active)
  GATE populations = coincidence (direction AND mode):
     approach: dFL&APP -> TL,  dFR&APP -> TR      (turn TOWARD the stronger side)
     avoid   : dFL&AVO -> TR,  dFR&AVO -> TL      (turn AWAY -- crossed)
  TL/TR command -> relay-gated graded muscles -> body (unchanged motor from neural_agent).
Only non-neural steps remain the two transducers (odour->current, muscle-S->force) + one CPG birth seed.
"""
import sys, numpy as np, mujoco
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
from paula_agent import ckit as k
def Tt(n): return 900+n
P=[1,2,3,4]; MLp,MLr,MRp,MRr=10,11,12,13
NP=10; THR=np.exp(np.linspace(np.log(0.25),np.log(4.5),NP))
FL=list(range(20,20+NP)); FR=list(range(40,40+NP)); POOL=60; STEER=62; TL=70; TR=71
RISEP=list(range(63,63+6)); RISE_DEL=[35,50,65,80,95,110]
ENG=[50,51,52,53,54]; ENG_C=[2,3,5,7,11]
NR=6; THR_R=np.linspace(0.5,4.2,NR)
RLY={MLp:list(range(110,110+NR)),MLr:list(range(120,120+NR)),MRp:list(range(130,130+NR)),MRr:list(range(140,140+NR))}
# --- ARBITER cortex populations (ids >=150 to avoid the motor's ids) ---
dFL,dFR=150,151                                   # opponent food-direction detectors
NMODE=4; APP=list(range(160,160+NMODE)); AVO=list(range(164,164+NMODE))   # mode populations
NG=3                                              # gate population size
GA_L=list(range(180,180+NG)); GA_R=list(range(183,183+NG))   # approach gates (toward)
GV_L=list(range(186,186+NG)); GV_R=list(range(189,189+NG))   # avoid gates (away)
RIS1=155; POO1=156                                           # pooled "rising" / "food-present" readouts
NRLY=4                                                        # mode-vetoed klinokinesis relay POPULATIONS
TON=list(range(210,210+NRLY)); RIS_A=list(range(216,216+NRLY)); CLO_A=list(range(222,222+NRLY)); ESC_V=list(range(228,228+NRLY))
GEAR=25.0; PERIOD=40; LAM_M=6; GGAIN=8.0
XML=f"""
<mujoco><option timestep="0.004" density="1200" viscosity="0.5" integrator="RK4"><flag gravity="disable"/></option>
<worldbody><geom type="plane" size="60 60 0.1" pos="0 0 -0.4"/>
<body name="torso" pos="0 0 0"><joint name="sx" type="slide" axis="1 0 0"/><joint name="sy" type="slide" axis="0 1 0"/><joint name="rz" type="hinge" axis="0 0 1"/>
<geom type="capsule" fromto="-0.18 0 0 0.18 0 0" size="0.05"/><geom type="capsule" fromto="-0.28 0 0 -0.18 0 0" size="0.03"/>
<body name="padL" pos="-0.15 0.06 0"><joint name="pl" type="hinge" axis="0 0 1" range="-1.8 1.8" damping="0.1"/><geom type="capsule" fromto="0 0 0 -0.03 0.24 0" size="0.022"/></body>
<body name="padR" pos="-0.15 -0.06 0"><joint name="pr" type="hinge" axis="0 0 1" range="-1.8 1.8" damping="0.1"/><geom type="capsule" fromto="0 0 0 -0.03 -0.24 0" size="0.022"/></body>
</body></worldbody>
<actuator><motor joint="pl" gear="{GEAR}" name="plp"/><motor joint="pl" gear="-{GEAR}" name="plr"/>
<motor joint="pr" gear="{GEAR}" name="prp"/><motor joint="pr" gear="-{GEAR}" name="prr"/></actuator></mujoco>"""

def build(w_cpg=8.0, w_norm=-0.9, w_pool=0.5, w_sd=2.5, w_ton=2.5, w_stein=1.1, w_riseinh=-3.2, w_closeinh=-1.4,
          w_rc=14.0, w_relay_inh=-4.0, w_steer_relay=-5.0, w_brake=-0.7,
          w_gate=1.5, w_gmode=0.35, r_gate=1.7, w_g2t=3.0, w_modewta=-6.0, w_escape=3.0, swap_turn=False):
    ne=[];sy=[];conns=[];ex=[]
    # ENGINE
    for e,c in zip(ENG,ENG_C): ne.append(k.neuron(e,r=0.6,c=c,lam=5)); sy.extend([k.syn(e,0,2.0,1),k.term(e,Tt(e))]); ex.append(k.ext(e,0))
    # CPG ring
    for i in range(4): nid=P[i]; ne.append(k.neuron(nid,r=0.6,lam=3,c=PERIOD)); sy.extend([k.syn(nid,0,4.0,PERIOD),k.term(nid,Tt(nid))])
    for i in range(4): conns.append(k.conn(P[(i-1)%4],P[i],0,Tt(P[(i-1)%4])))
    ex.append(k.ext(P[0],0))
    # MUSCLES + RELAY gating (TL/TR + STEER), identical motor to neural_agent
    for m,ph in [(MLp,P[0]),(MLr,P[2]),(MRp,P[0]),(MRr,P[2])]:
        ne.append(k.neuron(m,r=1e9,lam=LAM_M,delta_decay=1.0,c=2))
        for j in range(NR): sy.append(k.syn(m,j,w_cpg/NR,1))
        sy.append(k.term(m,Tt(m)))
    for m,ph in [(MLp,P[0]),(MLr,P[2]),(MRp,P[0]),(MRr,P[2])]:
        for j,g in enumerate(RLY[m]):
            ne.append(k.neuron(g,r=float(THR_R[j]),c=2,lam=3))
            sy.append(k.syn(g,0,w_rc,1)); conns.append(k.conn(ph,g,0,Tt(ph))); si=1
            left_turn,right_turn=(TL,TR) if swap_turn else (TR,TL)
            if m in (MLp,MLr):
                sy.append(k.syn(g,si,w_relay_inh,1)); conns.append(k.conn(left_turn,g,si,Tt(left_turn))); si+=1
                sy.append(k.syn(g,si,w_steer_relay,1)); conns.append(k.conn(STEER,g,si,Tt(STEER))); si+=1
            else:
                sy.append(k.syn(g,si,w_relay_inh,1)); conns.append(k.conn(right_turn,g,si,Tt(right_turn))); si+=1
            sy.append(k.syn(g,si,w_brake,1)); conns.append(k.conn(POOL,g,si,Tt(POOL))); si+=1
            sy.append(k.term(g,Tt(g))); conns.append(k.conn(g,m,j,Tt(g)))
    # FOOD sensor populations + POOL
    for p in (FL,FR):
        for i,nid in enumerate(p): ne.append(k.neuron(nid,r=float(THR[i]),c=2,lam=4)); sy.extend([k.syn(nid,0,1.0,1),k.syn(nid,1,w_norm,1),k.term(nid,Tt(nid))]); ex.append(k.ext(nid,0)); conns.append(k.conn(POOL,nid,1,Tt(POOL)))
    ne.append(k.neuron(POOL,r=0.5,c=1,lam=3)); pj=0
    for nid in FL+FR: sy.append(k.syn(POOL,pj,w_pool,1)); conns.append(k.conn(nid,POOL,pj,Tt(nid))); pj+=1
    sy.append(k.term(POOL,Tt(POOL)))
    # OPPONENT food-direction detectors dFL/dFR (which side the gradient is on)
    for nid in (dFL,dFR): ne.append(k.neuron(nid,r=0.6,c=2,lam=4))
    dj={dFL:0,dFR:0}
    def dsyn(nid,w): j=dj[nid]; sy.append(k.syn(nid,j,w,1)); dj[nid]+=1; return j
    for s in FL: conns.append(k.conn(s,dFL,dsyn(dFL, w_sd),Tt(s)))
    for s in FR: conns.append(k.conn(s,dFL,dsyn(dFL,-w_sd),Tt(s)))
    for s in FR: conns.append(k.conn(s,dFR,dsyn(dFR, w_sd),Tt(s)))
    for s in FL: conns.append(k.conn(s,dFR,dsyn(dFR,-w_sd),Tt(s)))
    sy.append(k.term(dFL,Tt(dFL))); sy.append(k.term(dFR,Tt(dFR)))
    # MODE populations APP / AVO with mutual (population) inhibition -> one behaviour wins
    for nid in APP+AVO: ne.append(k.neuron(nid,r=0.6,c=2,lam=6))
    mj={nid:0 for nid in APP+AVO}
    def msyn(nid,w): j=mj[nid]; sy.append(k.syn(nid,j,w,1)); mj[nid]+=1; return j
    for a in APP: ex.append(k.ext(a,msyn(a,1.0)))          # syn0 = drive input
    for a in AVO: ex.append(k.ext(a,msyn(a,1.0)))
    for a in APP:                                          # APP <-> AVO cross-inhibition (population WTA)
        for b in AVO: conns.append(k.conn(b,a,msyn(a,w_modewta),Tt(b)))
    for a in AVO:
        for b in APP: conns.append(k.conn(b,a,msyn(a,w_modewta),Tt(b)))
    for nid in APP+AVO: sy.append(k.term(nid,Tt(nid)))
    # GATE populations = coincidence(direction, mode). Fire only if BOTH inputs active (threshold r_gate).
    def gate(pop, dsrc, msrc, tgt):
        # coincidence: direction (syn0, strong) AND mode (syn1..N, one synapse PER mode neuron so the mode
        # is permissive but subthreshold ALONE; only direction+mode together crosses r_gate).
        for g in pop:
            ne.append(k.neuron(g,r=r_gate,c=2,lam=4))
            sy.append(k.syn(g,0,w_gate,1)); conns.append(k.conn(dsrc,g,0,Tt(dsrc)))
            for mi,mm in enumerate(msrc): sy.append(k.syn(g,1+mi,w_gmode,1)); conns.append(k.conn(mm,g,1+mi,Tt(mm)))
            sy.append(k.term(g,Tt(g)))
    gate(GA_L,dFL,APP,TL); gate(GA_R,dFR,APP,TR)          # approach: toward
    gate(GV_L,dFL,AVO,TR); gate(GV_R,dFR,AVO,TL)          # avoid: away (crossed)
    # COMMAND neurons TL/TR: driven by the active gates (routing), plus mutual inhibition
    for nid in (TL,TR): ne.append(k.neuron(nid,r=0.6,c=2,lam=4))
    cj={TL:0,TR:0}
    def csyn(nid,w): j=cj[nid]; sy.append(k.syn(nid,j,w,1)); cj[nid]+=1; return j
    for g in GA_L: conns.append(k.conn(g,TL,csyn(TL,w_g2t),Tt(g)))   # approach food-left -> TL
    for g in GV_R: conns.append(k.conn(g,TL,csyn(TL,w_g2t),Tt(g)))   # avoid food-right  -> TL
    for g in GA_R: conns.append(k.conn(g,TR,csyn(TR,w_g2t),Tt(g)))   # approach food-right-> TR
    for g in GV_L: conns.append(k.conn(g,TR,csyn(TR,w_g2t),Tt(g)))   # avoid food-left   -> TR
    conns.append(k.conn(TR,TL,csyn(TL,-6.0),Tt(TR))); conns.append(k.conn(TL,TR,csyn(TR,-6.0),Tt(TL)))
    sy.append(k.term(TL,Tt(TL))); sy.append(k.term(TR,Tt(TR)))
    # RISE population (odour trend) + pooled readouts.
    for r,dl in zip(RISEP,RISE_DEL):
        ne.append(k.neuron(r,r=0.6,c=2,lam=3)); sy.extend([k.syn(r,0,3.4,1),k.syn(r,1,-3.4,dl),k.term(r,Tt(r))])
        conns.extend([k.conn(POOL,r,0,Tt(POOL)),k.conn(POOL,r,1,Tt(POOL))])
    ne.append(k.neuron(RIS1,r=0.5,c=2,lam=4))                 # RIS1 = smooth "odour rising"
    for i,rr in enumerate(RISEP): sy.append(k.syn(RIS1,i,1.0,1)); conns.append(k.conn(rr,RIS1,i,Tt(rr)))
    sy.append(k.term(RIS1,Tt(RIS1)))
    ne.append(k.neuron(POO1,r=0.5,c=2,lam=4))                 # POO1 = smooth "food present/close"
    for i,s in enumerate(FL+FR): sy.append(k.syn(POO1,i,0.6,1)); conns.append(k.conn(s,POO1,i,Tt(s)))
    sy.append(k.term(POO1,Tt(POO1)))
    # --- SYMMETRIC MODE-SELECTED KLINOKINESIS via VETO relays (robust: mode INHIBITS the wrong relay) ---
    # STEER (spiral) excited by:  TON (engine, approach search) + ESC (avoid, rising -> turn away)
    #             inhibited by:  RIS_A (approach, rising -> go straight) + CLO_A (approach, close -> converge)
    # Each relay is vetoed by the opposing MODE population, so the SAME machinery runs approach OR avoid.
    def veto_relay(pop, exc_src, exc_w, veto_pop, veto_w=-2.2, rr=0.6):
        for nid in pop:
            ne.append(k.neuron(nid,r=rr,c=2,lam=4)); j=0
            for s in exc_src: sy.append(k.syn(nid,j,exc_w,1)); conns.append(k.conn(s,nid,j,Tt(s))); j+=1
            for m in veto_pop: sy.append(k.syn(nid,j,veto_w,1)); conns.append(k.conn(m,nid,j,Tt(m))); j+=1
            sy.append(k.term(nid,Tt(nid)))
    veto_relay(TON,   ENG,   w_ton,  AVO)          # engine spiral, OFF in avoid
    veto_relay(RIS_A, [RIS1], 2.6,    AVO)          # rising, approach-only  -> suppress spiral (climb gradient)
    veto_relay(ESC_V, [RIS1], 2.6,    APP)          # rising, avoid-only     -> DRIVE spiral (turn away)
    # CLO_A: CLOSENESS population (approach-only, vetoed by AVO). Driven by POO1 (pooled odour presence)
    # -> suppress the spiral near food so approach converges instead of orbiting.
    veto_relay(CLO_A, [POO1], 2.0, AVO)
    # STEER: excited by TON+ESC_V populations, inhibited by RIS_A+CLO_A populations (per-neuron weights sum).
    ne.append(k.neuron(STEER,r=0.6,c=2,lam=5)); si=0
    for n in TON:   sy.append(k.syn(STEER,si,w_stein,1));   conns.append(k.conn(n,STEER,si,Tt(n))); si+=1
    for n in ESC_V: sy.append(k.syn(STEER,si,w_escape,1));  conns.append(k.conn(n,STEER,si,Tt(n))); si+=1
    for n in RIS_A: sy.append(k.syn(STEER,si,w_riseinh,1)); conns.append(k.conn(n,STEER,si,Tt(n))); si+=1
    for n in CLO_A: sy.append(k.syn(STEER,si,w_closeinh,1));conns.append(k.conn(n,STEER,si,Tt(n))); si+=1
    sy.append(k.term(STEER,Tt(STEER)))
    return k.build(ne,sy,conns,ex)

class GatedAgent:
    def __init__(self, foods, mode='approach', sigma=14.0, gain=5.0, sub=16, **bkw):
        self.model=mujoco.MjModel.from_xml_string(XML); self.data=mujoco.MjData(self.model)
        self.foods=[[float(a),float(b),True] for a,b in foods]; self.sigma=sigma; self.gain=gain; self.sub=sub
        self.mode=mode
        self.net,self.core=k.load(build(**bkw)); self.nb={i:u for i,u in self.net.network.neurons.items()}
        self.a={n:self.model.actuator(n).id for n in ("plp","plr","prp","prr")}
        self.sx=self.model.joint('sx').qposadr[0]; self.sy=self.model.joint('sy').qposadr[0]; self.rz=self.model.joint('rz').qposadr[0]
        self.kicked=False; self.t=0
    def pose(self): return float(self.data.qpos[self.sx]),float(self.data.qpos[self.sy]),float(self.data.qpos[self.rz])
    def odor(self,p): return float(sum(np.exp(-((p[0]-fx)**2+(p[1]-fy)**2)/self.sigma) for fx,fy,a in self.foods if a))
    def step(self):
        x,y,yaw=self.pose(); fwd=yaw+np.pi
        hd=np.array([np.cos(fwd),np.sin(fwd)]); lat=np.array([-np.sin(fwd),np.cos(fwd)]); head=np.array([x,y])+hd*0.45
        oL=self.odor(head+lat*0.6); oR=self.odor(head-lat*0.6)
        modepop = APP if self.mode=='approach' else AVO
        for _ in range(self.sub):
            for e in ENG: self.net.set_external_input(e,0,1.0)
            for nid in FL: self.net.set_external_input(nid,0,self.gain*oL)
            for nid in FR: self.net.set_external_input(nid,0,self.gain*oR)
            for mm in modepop: self.net.set_external_input(mm,0,2.0)      # drive the active mode (WTA does the rest)
            if not self.kicked and self.t>2: self.net.set_external_input(P[0],0,5.0); self.kicked=True
            self.core.do_tick(); self.t+=1
            self.data.ctrl[self.a["plp"]]=GGAIN*max(0,self.nb[MLp].S); self.data.ctrl[self.a["plr"]]=GGAIN*max(0,self.nb[MLr].S)
            self.data.ctrl[self.a["prp"]]=GGAIN*max(0,self.nb[MRp].S); self.data.ctrl[self.a["prr"]]=GGAIN*max(0,self.nb[MRr].S)
            mujoco.mj_step(self.model,self.data)
        for f in self.foods:
            if f[2] and (x-f[0])**2+(y-f[1])**2<0.7**2: f[2]=False
        return oL+oR

if __name__=="__main__":
    # SAME food sensor, OPPOSITE behaviour selected by which MODE population is active. The mode vetoes
    # the wrong-mode klinokinesis relay, so approach climbs the gradient and avoid flees it -- attractor
    # reversal with one shared circuit. Behaviour lives in the wiring, not a Python if/else.
    print("GATED ARBITER — SAME food, opposite behaviour by MODE (attractor reversal in shared wiring):")
    for food in [(-4.0,1.0),(-4.0,0.0),(3.0,2.0)]:
        d0=np.hypot(*food); row=f"  food=({food[0]:+.1f},{food[1]:+.1f}) start={d0:.1f}: "
        for mode in ('approach','avoid'):
            ag=GatedAgent(foods=[food], mode=mode); mind=d0; maxd=0
            for t in range(3000):
                ag.step(); x,y,_=ag.pose(); d=np.hypot(x-food[0],y-food[1]); mind=min(mind,d); maxd=max(maxd,d)
            if mode=='approach': v='REACHED' if mind<1.0 else f'near({mind:.1f})'
            else:                v='AVOIDED' if (mind>1.2 and maxd>d0) else f'FAILED({mind:.1f})'
            row+=f"{mode}->{v} [c={mind:.2f},f={maxd:.2f}]  "
        print(row, flush=True)
    print("@@@GATED DONE@@@")
