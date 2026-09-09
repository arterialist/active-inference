"""MUSHROOM BODY — Phase 3: EMBODIED VALENCE LEARNING. The agent forages a world with two ODORANT
IDENTITIES; one is nutritious (reward on eating), one is toxic (punishment on contact). It does NOT know
which. The mushroom body (PN->KC->MBON, plastic) learns each odorant's valence from experience:
  approach any source by tropotaxis; the LEARNED MBON valence gates behaviour --
     MBON high (good)  -> suppress escape, converge, eat
     MBON low  (bad)   -> AVOID neuron fires (odour present AND MBON low) -> escape spiral, turn away
  on contact the environment delivers dopamine to the MBON (reward if nutritious, punish if toxic),
     which trains the currently-active (this-odorant) KC->MBON synapses.
Naive early -> contacts toxins; after learning -> avoids the toxic odorant. Measures the learning curve.
Motor reused from the proven dual-chemotaxis / neural_agent; MB reused from mushroom_learn.
"""
import sys, numpy as np, mujoco
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
from paula_agent import ckit as k
def Tt(n): return 900+n
# ---- motor / navigation ids (as in neural_agent) ----
P=[1,2,3,4]; MLp,MLr,MRp,MRr=10,11,12,13
POOL=60; STEER=62; TL=70; TR=71
NS=10; THR=np.exp(np.linspace(np.log(0.25),np.log(4.5),NS))
SL=list(range(20,20+NS)); SR=list(range(40,40+NS))          # total-odour L/R (drives tropotaxis)
RISEP=list(range(63,63+6)); RISE_DEL=[35,50,65,80,95,110]
ENG=[50,51,52,53,54]; ENG_C=[2,3,5,7,11]
NR=6; THR_R=np.linspace(0.5,4.2,NR)
RLY={MLp:list(range(110,110+NR)),MLr:list(range(120,120+NR)),MRp:list(range(130,130+NR)),MRr:list(range(140,140+NR))}
# ---- mushroom body ids (high range, no collision) ----
NPN=16; NKC=160; KIN=6; NMBON=8
PN=list(range(400,400+NPN)); KC=list(range(500,500+NKC)); APL=490
MBONP=list(range(680,680+NMBON)); AVOID=496
DOP=NKC; TEACH=NKC+1                                    # DOP=dopamine mod port; TEACH=aversive-drive teaching port
MBON_R=np.linspace(0.7,1.6,NMBON)                      # diverse thresholds -> temporally dense aggregate
rng_mb=np.random.default_rng(7)
KC_PRE={kc: sorted(rng_mb.choice(NPN,size=KIN,replace=False).tolist()) for kc in KC}
GEAR=25.0; PERIOD=40; LAM_M=6; GGAIN=8.0
XML=f"""
<mujoco><option timestep="0.004" density="1200" viscosity="0.5" integrator="RK4"><flag gravity="disable"/></option>
<worldbody><geom type="plane" size="120 120 0.1" pos="0 0 -0.4"/>
<body name="torso" pos="0 0 0"><joint name="sx" type="slide" axis="1 0 0"/><joint name="sy" type="slide" axis="0 1 0"/><joint name="rz" type="hinge" axis="0 0 1"/>
<geom type="capsule" fromto="-0.18 0 0 0.18 0 0" size="0.05"/><geom type="capsule" fromto="-0.28 0 0 -0.18 0 0" size="0.03"/>
<body name="padL" pos="-0.15 0.06 0"><joint name="pl" type="hinge" axis="0 0 1" range="-1.8 1.8" damping="0.1"/><geom type="capsule" fromto="0 0 0 -0.03 0.24 0" size="0.022"/></body>
<body name="padR" pos="-0.15 -0.06 0"><joint name="pr" type="hinge" axis="0 0 1" range="-1.8 1.8" damping="0.1"/><geom type="capsule" fromto="0 0 0 -0.03 -0.24 0" size="0.022"/></body>
</body></worldbody>
<actuator><motor joint="pl" gear="{GEAR}" name="plp"/><motor joint="pl" gear="-{GEAR}" name="plr"/>
<motor joint="pr" gear="{GEAR}" name="prp"/><motor joint="pr" gear="-{GEAR}" name="prr"/></actuator></mujoco>"""

def build(w_cpg=8.0, w_norm=-0.9, w_pool=0.5, w_sd=2.5, w_ton=2.5, w_riseinh=-4.0, w_closeinh=-0.35,
          w_rc=14.0, w_relay_inh=-4.0, w_steer_relay=-5.0, w_brake=-0.7,
          w_pk=2.2, r_kc=4.2, w_apl_kc=-1.4, w_kc_apl=0.5, r_apl=6.0,
          w_km0=0.3, r_mbon=1.0, eta=0.08, kappa=1.8, rh_decay=0.12, w_teach=5.0,
          mbon_plast="reward_hebb", w_tref_cort=30.0,
          r_avoid=0.8, w_av_mbon=2.2, lam_avoid=4, w_avesc=6.0, swap_turn=False):
    ne=[];sy=[];conns=[];ex=[]
    # ENGINE + CPG + MUSCLES + RELAYS (proven motor)
    for e,c in zip(ENG,ENG_C): ne.append(k.neuron(e,r=0.6,c=c,lam=5)); sy.extend([k.syn(e,0,2.0,1),k.term(e,Tt(e))]); ex.append(k.ext(e,0))
    for i in range(4): nid=P[i]; ne.append(k.neuron(nid,r=0.6,lam=3,c=PERIOD)); sy.extend([k.syn(nid,0,4.0,PERIOD),k.term(nid,Tt(nid))])
    for i in range(4): conns.append(k.conn(P[(i-1)%4],P[i],0,Tt(P[(i-1)%4])))
    ex.append(k.ext(P[0],0))
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
    # TOTAL-ODOUR sensors + POOL (tropotaxis toward any source) + opponent TL/TR
    for p in (SL,SR):
        for i,nid in enumerate(p): ne.append(k.neuron(nid,r=float(THR[i]),c=2,lam=4)); sy.extend([k.syn(nid,0,1.0,1),k.syn(nid,1,w_norm,1),k.term(nid,Tt(nid))]); ex.append(k.ext(nid,0)); conns.append(k.conn(POOL,nid,1,Tt(POOL)))
    ne.append(k.neuron(POOL,r=0.5,c=1,lam=3)); pj=0
    for nid in SL+SR: sy.append(k.syn(POOL,pj,w_pool,1)); conns.append(k.conn(nid,POOL,pj,Tt(nid))); pj+=1
    sy.append(k.term(POOL,Tt(POOL)))
    for nid in (TL,TR): ne.append(k.neuron(nid,r=0.6,c=2,lam=4))
    tj={TL:0,TR:0}
    def tsyn(nid,w): j=tj[nid]; sy.append(k.syn(nid,j,w,1)); tj[nid]+=1; return j
    for s in SL: conns.append(k.conn(s,TL,tsyn(TL, w_sd),Tt(s)))
    for s in SR: conns.append(k.conn(s,TL,tsyn(TL,-w_sd),Tt(s)))
    for s in SR: conns.append(k.conn(s,TR,tsyn(TR, w_sd),Tt(s)))
    for s in SL: conns.append(k.conn(s,TR,tsyn(TR,-w_sd),Tt(s)))
    conns.append(k.conn(TR,TL,tsyn(TL,-7.0),Tt(TR))); conns.append(k.conn(TL,TR,tsyn(TR,-7.0),Tt(TL)))
    sy.append(k.term(TL,Tt(TL))); sy.append(k.term(TR,Tt(TR)))
    # MUSHROOM BODY: PN -> KC -> APL ; MBON plastic valence readout
    for p in PN: ne.append(k.neuron(p,r=0.6,c=2,lam=4)); sy.extend([k.syn(p,0,1.0,1),k.term(p,Tt(p))]); ex.append(k.ext(p,0))
    for kc in KC:
        ne.append(k.neuron(kc,r=r_kc,c=2,lam=4)); j=0
        for pi in KC_PRE[kc]:                       # KC_PRE stores PN INDICES; map to PN neuron ids
            pn=PN[pi]; sy.append(k.syn(kc,j,w_pk,1)); conns.append(k.conn(pn,kc,j,Tt(pn))); j+=1
        sy.append(k.syn(kc,j,w_apl_kc,1)); conns.append(k.conn(APL,kc,j,Tt(APL)))
        sy.append(k.term(kc,Tt(kc)))
    ne.append(k.neuron(APL,r=r_apl,c=1,lam=3)); j=0
    for kc in KC: sy.append(k.syn(APL,j,w_kc_apl,1)); conns.append(k.conn(kc,APL,j,Tt(kc))); j+=1
    sy.append(k.term(APL,Tt(APL)))
    # AVERSIVE MBON POPULATION: LOW initial weights (naive baseline low), STAGGERED thresholds so the
    # members fire at diverse times -> a temporally DENSE aggregate aversion signal (population-per-function).
    # Each is plastic (reward_hebb): potentiated by dopamine on TOXIN contact (mod reward-channel) -> learns
    # "this odorant = avoid"; depressed on FOOD contact. All read all KCs and share the dopamine port DOP.
    for mi,mbon in enumerate(MBONP):
        nd=k.neuron(mbon,r=float(MBON_R[mi]),c=2,lam=5,plasticity=mbon_plast,eta_post=eta,kappa=kappa,rh_decay=rh_decay)
        # AVERSIVE system: potentiated by CORTISOL (stress, m0), suppressed by DOPAMINE (reward, m1).
        nd["params"]["nm_reward_index"]=0   # cortisol is the potentiating signal for aversive memory
        nd["params"]["nm_stress_index"]=1   # dopamine (food) suppresses aversion
        # METAPLASTICITY: cortisol (M[0]) shifts the causal learning-window t_ref -> gates WHEN learning happens
        nd["params"]["w_tref"]=[float(w_tref_cort),0.0]
        ne.append(nd)
        for i,kc in enumerate(KC): sy.append(k.syn(mbon,i,w_km0,1)); conns.append(k.conn(kc,mbon,i,Tt(kc)))
        sy.append(k.syn(mbon,DOP,0.0,1)); ex.append(k.ext(mbon,DOP))            # neuromodulator port (cortisol/dopamine)
        sy.append(k.syn(mbon,TEACH,w_teach,1)); ex.append(k.ext(mbon,TEACH))     # aversive teaching drive (the shock)
        sy.append(k.term(mbon,Tt(mbon)))
    # AVOID: integrates the MBON POPULATION (sum). Naive/food: population low -> AVOID silent -> innate
    # approach. Learned-toxic: population dense/high -> AVOID fires -> escape spiral. syn=NMBON is the reflex.
    ne.append(k.neuron(AVOID,r=r_avoid,c=2,lam=lam_avoid))
    for mi,mbon in enumerate(MBONP): sy.append(k.syn(AVOID,mi,w_av_mbon,1)); conns.append(k.conn(mbon,AVOID,mi,Tt(mbon)))
    sy.append(k.syn(AVOID,NMBON,4.0,1)); ex.append(k.ext(AVOID,NMBON)); sy.append(k.term(AVOID,Tt(AVOID)))
    # STEER: engine tonic; FOOD-rising & close suppress (converge); AVOID drives escape (learned bad)
    ne.append(k.neuron(STEER,r=0.6,c=2,lam=5)); si=0
    for e in ENG: sy.append(k.syn(STEER,si,w_ton,1)); conns.append(k.conn(e,STEER,si,Tt(e))); si+=1
    for r,dl in zip(RISEP,RISE_DEL):
        ne.append(k.neuron(r,r=0.6,c=2,lam=3)); sy.extend([k.syn(r,0,3.4,1),k.syn(r,1,-3.4,dl),k.term(r,Tt(r))])
        conns.extend([k.conn(POOL,r,0,Tt(POOL)),k.conn(POOL,r,1,Tt(POOL))])
    for r in RISEP: sy.append(k.syn(STEER,si,w_riseinh,1)); conns.append(k.conn(r,STEER,si,Tt(r))); si+=1
    for src in SL+SR: sy.append(k.syn(STEER,si,w_closeinh,1)); conns.append(k.conn(src,STEER,si,Tt(src))); si+=1
    sy.append(k.syn(STEER,si,w_avesc,1)); conns.append(k.conn(AVOID,STEER,si,Tt(AVOID))); si+=1
    sy.append(k.term(STEER,Tt(STEER)))
    return k.build(ne,sy,conns,ex)

# two odorant identities -> distinct PN patterns
_rng=np.random.default_rng(2)
def _pat():
    v=np.zeros(NPN); v[_rng.choice(NPN,size=6,replace=False)]=1.0; return v
ODOR_PAT=[_pat(),_pat()]
while np.array_equal(ODOR_PAT[0],ODOR_PAT[1]): ODOR_PAT[1]=_pat()

class MBAgent:
    def __init__(self, sources, sigma=14.0, sigma_id=4.0, gain=5.0, pn_drive=3.0, sub=16,
                 eat_r=0.9, leave_r=2.2, tref_upper=8.0, tref_lower=1.0, **bkw):
        # sources: list of [x,y,odorant_id(0/1),nutritious(bool),active(bool)]
        self.model=mujoco.MjModel.from_xml_string(XML); self.data=mujoco.MjData(self.model)
        self.sources=[list(s) for s in sources]; self.sigma=sigma; self.sigma_id=sigma_id
        self.gain=gain; self.pn_drive=pn_drive; self.sub=sub; self.eat_r=eat_r; self.leave_r=leave_r
        self._contact=[False]*len(self.sources)   # per-source contact latch (refractory until agent leaves)
        self.net,self.core=k.load(build(**bkw)); self.nb={i:u for i,u in self.net.network.neurons.items()}
        # SMALL learning window: sparse (infrequent) inputs read as ACAUSAL -> LTD/forgetting -> no exposure
        # drift; the shock's teaching drive makes the MBON fire densely -> causal -> LTP (learning).
        for m in MBONP:
            self.nb[m].upper_t_ref_bound=float(tref_upper); self.nb[m].lower_t_ref_bound=float(tref_lower); self.nb[m].t_ref=float(tref_upper)
        self._mb_pp=[self.nb[m].postsynaptic_points for m in MBONP]   # cache for the non-negativity clamp
        self._mb_clampidx=list(range(NKC))+[TEACH]
        self.a={n:self.model.actuator(n).id for n in ("plp","plr","prp","prr")}
        self.sx=self.model.joint('sx').qposadr[0]; self.sy=self.model.joint('sy').qposadr[0]; self.rz=self.model.joint('rz').qposadr[0]
        self.kicked=False; self.t=0; self.eaten=0; self.tox_hits=0; self.reward_events=0; self.punish_events=0
        self._reflex=0                                   # nociceptive-withdrawal reflex countdown
        self._shock=0; self._shock_dop=np.array([0.0,0.0]); self._shock_teach=False  # US window (dopamine + aversive drive)
    def pose(self): return float(self.data.qpos[self.sx]),float(self.data.qpos[self.sy]),float(self.data.qpos[self.rz])
    def _conc(self,p,sig,active_only=True):
        c=np.zeros(2)
        for sx,sy,oid,nut,act in self.sources:
            if active_only and not act: continue
            c[int(oid)]+=np.exp(-((p[0]-sx)**2+(p[1]-sy)**2)/sig)
        return c
    def step(self):
        x,y,yaw=self.pose(); fwd=yaw+np.pi
        hd=np.array([np.cos(fwd),np.sin(fwd)]); lat=np.array([-np.sin(fwd),np.cos(fwd)]); head=np.array([x,y])+hd*0.45
        cL=self._conc(head+lat*0.6,self.sigma); cR=self._conc(head-lat*0.6,self.sigma)   # broad: tropotaxis
        cH=self._conc(head,self.sigma_id)                                                # SHARP: clean odorant identity
        totL,totR=cL.sum(),cR.sum()
        # PN drive = odorant-identity vector at head (normalised so identity, not intensity)
        idv=np.zeros(NPN); s=cH.sum()
        if s>1e-6:
            for oid in (0,1): idv += (cH[oid]/s)*ODOR_PAT[oid]
        gate=min(1.0, s*2.0)         # only sense identity when actually in an odour
        for _ in range(self.sub):
            for e in ENG: self.net.set_external_input(e,0,1.0)
            for nid in SL: self.net.set_external_input(nid,0,self.gain*totL)
            for nid in SR: self.net.set_external_input(nid,0,self.gain*totR)
            for i,p in enumerate(PN): self.net.set_external_input(p,0,self.pn_drive*gate*float(idv[i]))
            # US window: deliver dopamine (mod) and, for a shock, DRIVE the aversive MBONs to fire (teaching)
            # coincident with the odour's KCs -> correct-direction potentiation (one-trial aversive learning).
            for mbon in MBONP:
                self.net.set_external_input(mbon,DOP,0.0,mod=(self._shock_dop if self._shock>0 else np.array([0.0,0.0])))
                self.net.set_external_input(mbon,TEACH, 1.0 if (self._shock>0 and self._shock_teach) else 0.0)
            self.net.set_external_input(AVOID,NMBON, 1.0 if self._reflex>0 else 0.0)   # nociceptive reflex drive
            if not self.kicked and self.t>2: self.net.set_external_input(P[0],0,5.0); self.kicked=True
            self.core.do_tick(); self.t+=1
            self.data.ctrl[self.a["plp"]]=GGAIN*max(0,self.nb[MLp].S); self.data.ctrl[self.a["plr"]]=GGAIN*max(0,self.nb[MLr].S)
            self.data.ctrl[self.a["prp"]]=GGAIN*max(0,self.nb[MRp].S); self.data.ctrl[self.a["prr"]]=GGAIN*max(0,self.nb[MRr].S)
            mujoco.mj_step(self.model,self.data)
        for pp in self._mb_pp:                       # excitatory KC->MBON & TEACH synapses stay non-negative
            for idx in self._mb_clampidx:
                u=pp[idx].u_i
                if u.info<0.0: u.info=0.0
        self._reflex=max(0,self._reflex-1); self._shock=max(0,self._shock-1)
        # discrete contact EVENTS: fire the US only on ENTERING a source's radius; latch until the agent
        # LEAVES (dist>leave_r), so sitting near a toxin doesn't rack up hundreds of events.
        for i,src in enumerate(self.sources):
            sxx,syy,oid,nut,act=src; d2=(x-sxx)**2+(y-syy)**2
            if act and d2<self.eat_r**2 and not self._contact[i]:
                self._contact[i]=True
                if nut:   # FOOD: release DOPAMINE (reward, m1) -> suppress aversion for this odorant
                    self.eaten+=1; src[4]=False; self.reward_events+=1
                    self._shock=3; self._shock_dop=np.array([0.0,3.0]); self._shock_teach=False
                else:     # TOXIN: release CORTISOL (stress, m0) + SHOCK drives the aversive MBONs to fire
                    self.tox_hits+=1; self.punish_events+=1; self._reflex=15
                    self._shock=3; self._shock_dop=np.array([3.0,0.0]); self._shock_teach=True
            elif self._contact[i] and d2>self.leave_r**2:
                self._contact[i]=False
        return totL+totR
    def mbon_rate(self,odor_id,ticks=40):
        """Probe learned AVERSION for an odorant = summed spikes of the aversive-MBON population."""
        pat=ODOR_PAT[odor_id]; m=0
        for t in range(ticks):
            for i,p in enumerate(PN): self.net.set_external_input(p,0,self.pn_drive*float(pat[i]))
            self.core.do_tick(); m+=sum(self.nb[mbon].O>0 for mbon in MBONP)
        for p in PN: self.net.set_external_input(p,0,0.0)
        return m

if __name__=="__main__":
    print("EMBODIED MUSHROOM BODY — learn which odorant is food vs toxin from experience:")
    rng=np.random.default_rng(4)
    GOOD,BAD=0,1   # odorant 0 nutritious, odorant 1 toxic (agent must learn this)
    def make_sources(n_each):
        # reachable spread; toxic sources RESPAWN role implicitly (persist) so aversion learning has repeats
        S=[]
        for oid,nut in [(GOOD,True),(BAD,False)]:
            for _ in range(n_each):
                a=rng.uniform(0,2*np.pi); r=rng.uniform(3.0,6.0)
                S.append([float(r*np.cos(a)),float(r*np.sin(a)),oid,nut,True])
        return S
    ag=MBAgent(sources=make_sources(4))
    def respawn_food():
        for src in ag.sources:
            if src[2]==GOOD and not src[4]:
                a=rng.uniform(0,2*np.pi); r=rng.uniform(3.0,6.0); src[0]=float(r*np.cos(a)); src[1]=float(r*np.sin(a)); src[4]=True
    print(f"  initial aversion (MBON pop spikes/40): good-odour={ag.mbon_rate(GOOD)}  toxic-odour={ag.mbon_rate(BAD)}")
    WIN=2000; last_eat=0; last_hit=0
    for blk in range(9):
        for _ in range(WIN): ag.step(); respawn_food()   # keep food available so the agent keeps foraging
        de=ag.eaten-last_eat; dh=ag.tox_hits-last_hit; last_eat,last_hit=ag.eaten,ag.tox_hits
        vg,vb=ag.mbon_rate(GOOD),ag.mbon_rate(BAD)
        print(f"  block{blk+1:2d}: +eaten={de} +toxin_contacts={dh} | learned aversion good={vg} toxic={vb}", flush=True)
    print(f"  TOTAL eaten={ag.eaten} toxin_contacts={ag.tox_hits}")
    print("@@@MBAGENT DONE@@@")
