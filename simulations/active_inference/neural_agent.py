"""FULLY-NEURAL AGENT — one PAULA network from odour to muscle, mounted on the neuromuscular body.
Only two non-neural steps (physics transducers): odour field value at a sensor location -> neuron current;
graded muscle membrane S -> actuator force. Everything between is neurons + wiring.

  odour sensors (FL/FR log-populations) + POOL normalization
     -> TURN neurons TL/TR (tropotaxis: stronger side)   -> inhibit the OPPOSITE paddle's muscles (steer)
     -> RISE/STEER (smooth klinokinesis)                  -> inhibit muscles (spiral reorientation)
  CPG pacemaker -> antagonist muscle neurons (graded) -> [S -> actuator force] -> paddles -> physics
"""
import sys, numpy as np, mujoco
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
from paula_agent import ckit as k
def Tt(n): return 900+n
P=[1,2,3,4]; MLp,MLr,MRp,MRr=10,11,12,13
NP=10; THR=np.exp(np.linspace(np.log(0.25),np.log(4.5),NP))
FL=list(range(20,20+NP)); FR=list(range(40,40+NP)); POOL=60; STEER=62; TL=70; TR=71
RISEP=list(range(63,63+6)); RISE_DEL=[35,50,65,80,95,110]   # RISE population: staggered trend windows -> smooth "odour rising" signal
ENG=[50,51,52,53,54]; ENG_C=[2,3,5,7,11]   # engine: coprime-period pacemakers -> smooth aggregate power
# relay population between CPG and each muscle (drive-gating for graded, smooth steering)
NR=6; THR_R=np.linspace(0.5,4.2,NR)
RLY={MLp:list(range(110,110+NR)),MLr:list(range(120,120+NR)),MRp:list(range(130,130+NR)),MRr:list(range(140,140+NR))}
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
def build(w_cpg=8.0, w_norm=-0.9, w_pool=0.5, w_sd=2.5, w_wta=-7.0, w_ton=2.5, w_riseinh=-4.0, w_closeinh=-0.35,
          w_rc=14.0, w_relay_inh=-4.0, w_steer_relay=-5.0, w_brake=-0.7, swap_turn=False):
    ne=[];sy=[];conns=[];ex=[]
    # ENGINE ("arousal/drive nucleus"): coprime-period pacemakers turn one crude tonic drive into a
    # smooth, dense power train (their firings interleave like gears of different sizes). Downstream
    # modules tap it with per-module weights; the tonic drive level is a global gain on the whole brain.
    for e,c in zip(ENG,ENG_C): ne.append(k.neuron(e,r=0.6,c=c,lam=5)); sy.extend([k.syn(e,0,2.0,1),k.term(e,Tt(e))]); ex.append(k.ext(e,0))
    # CPG ring: the propagation delays (dist=PERIOD) set the phase spacing, so it must run at its own
    # natural period. Seeded by a one-time birth impulse (initial condition), then self-sequences.
    for i in range(4): nid=P[i]; ne.append(k.neuron(nid,r=0.6,lam=3,c=PERIOD)); sy.extend([k.syn(nid,0,4.0,PERIOD),k.term(nid,Tt(nid))])
    for i in range(4): conns.append(k.conn(P[(i-1)%4],P[i],0,Tt(P[(i-1)%4])))
    ex.append(k.ext(P[0],0))
    # MUSCLES (graded, analog). Driven by a RELAY population (drive-gating), NOT tonic membrane inhibition.
    for m,ph in [(MLp,P[0]),(MLr,P[2]),(MRp,P[0]),(MRr,P[2])]:
        ne.append(k.neuron(m,r=1e9,lam=LAM_M,delta_decay=1.0,c=2))
        for j in range(NR): sy.append(k.syn(m,j,w_cpg/NR,1))      # one synapse per relay, summed drive
        sy.append(k.term(m,Tt(m)))
    # RELAY populations: CPG phase excites all (distributed thresholds); steering INHIBITS -> graded fraction pass.
    # Left muscles gated by TR + STEER (klinokinesis); right muscles gated by TL. Steering weakens the DRIVE, not the membrane.
    for m,ph in [(MLp,P[0]),(MLr,P[2]),(MRp,P[0]),(MRr,P[2])]:
        for j,g in enumerate(RLY[m]):
            ne.append(k.neuron(g,r=float(THR_R[j]),c=2,lam=3))
            sy.append(k.syn(g,0,w_rc,1)); conns.append(k.conn(ph,g,0,Tt(ph)))   # CPG drive
            si=1
            left_turn, right_turn = (TL,TR) if swap_turn else (TR,TL)   # which turn neuron inhibits this side
            if m in (MLp,MLr):   # left paddle: tropotaxis turn + STEER klinokinesis spiral
                sy.append(k.syn(g,si,w_relay_inh,1)); conns.append(k.conn(left_turn,g,si,Tt(left_turn))); si+=1
                sy.append(k.syn(g,si,w_steer_relay,1)); conns.append(k.conn(STEER,g,si,Tt(STEER))); si+=1   # tight search spiral
            else:                # right paddle
                sy.append(k.syn(g,si,w_relay_inh,1)); conns.append(k.conn(right_turn,g,si,Tt(right_turn))); si+=1
            sy.append(k.syn(g,si,w_brake,1)); conns.append(k.conn(POOL,g,si,Tt(POOL))); si+=1   # brake: slow BOTH sides near food -> converge
            sy.append(k.term(g,Tt(g)))
            conns.append(k.conn(g,m,j,Tt(g)))   # relay -> muscle synapse j
    # ODOUR sensor populations + normalization POOL
    for p in (FL,FR):
        for i,nid in enumerate(p): ne.append(k.neuron(nid,r=float(THR[i]),c=2,lam=4)); sy.extend([k.syn(nid,0,1.0,1),k.syn(nid,1,w_norm,1),k.term(nid,Tt(nid))]); ex.append(k.ext(nid,0)); conns.append(k.conn(POOL,nid,1,Tt(POOL)))
    ne.append(k.neuron(POOL,r=0.5,c=1,lam=3)); pj=0
    for nid in FL+FR: sy.append(k.syn(POOL,pj,w_pool,1)); conns.append(k.conn(nid,POOL,pj,Tt(nid))); pj+=1
    sy.append(k.term(POOL,Tt(POOL)))
    # TURN neurons TL/TR : OPPONENT tropotaxis. TL = (left sensors) - (right sensors); TR = mirror.
    # Each fires only on the L/R DIFFERENCE (not absolute odour), so a small gradient still separates them.
    for nid in (TL,TR): ne.append(k.neuron(nid,r=0.6,c=2,lam=4))
    tj={TL:0,TR:0}
    def add_syn(nid,w): j=tj[nid]; sy.append(k.syn(nid,j,w,1)); tj[nid]+=1; return j
    for s in FL: conns.append(k.conn(s,TL,add_syn(TL, w_sd),Tt(s)))       # left excites TL
    for s in FR: conns.append(k.conn(s,TL,add_syn(TL,-w_sd),Tt(s)))       # right INHIBITS TL
    for s in FR: conns.append(k.conn(s,TR,add_syn(TR, w_sd),Tt(s)))       # right excites TR
    for s in FL: conns.append(k.conn(s,TR,add_syn(TR,-w_sd),Tt(s)))       # left INHIBITS TR
    conns.append(k.conn(TR,TL,add_syn(TL,w_wta),Tt(TR))); conns.append(k.conn(TL,TR,add_syn(TR,w_wta),Tt(TL)))
    sy.append(k.term(TL,Tt(TL))); sy.append(k.term(TR,Tt(TR)))
    # (TL/TR and STEER now gate the muscle RELAYS above, not the muscle membrane.)
    # STEER = search spiral. Tonic ON; SUPPRESSED by total odour (POOL). So: spiral-search when no food
    # signal, go straight/track via tropotaxis when odour is present. This gives reorientation for food
    # that is behind or lost after an overshoot.
    # RISE POPULATION: odour TREND detector. Each neuron = POOL now (+) vs POOL delayed (-) over a DIFFERENT
    # window (RISE_DEL). Individually a per-spike comparison is noisy; across staggered windows the aggregate
    # is a smooth, dense "odour rising" signal (one population per function, like the engine and the sensors).
    for r,dl in zip(RISEP,RISE_DEL):
        ne.append(k.neuron(r,r=0.6,c=2,lam=3)); sy.extend([k.syn(r,0,3.4,1),k.syn(r,1,-3.4,dl),k.term(r,Tt(r))])
        conns.extend([k.conn(POOL,r,0,Tt(POOL)),k.conn(POOL,r,1,Tt(POOL))])
    # STEER: engine-powered search spiral, SUPPRESSED BY the RISE population (klinokinesis). Spiral while
    # odour falls/flat (heading away or lost); go straight while odour rises (progress). Trend-, not presence-based.
    ne.append(k.neuron(STEER,r=0.6,c=2,lam=5)); si=0
    for e in ENG: sy.append(k.syn(STEER,si,w_ton,1)); conns.append(k.conn(e,STEER,si,Tt(e))); si+=1   # smooth tonic from engine
    for r in RISEP: sy.append(k.syn(STEER,si,w_riseinh,1)); conns.append(k.conn(r,STEER,si,Tt(r))); si+=1  # rising odour -> stop spiral (reorient)
    for src in FL+FR: sy.append(k.syn(STEER,si,w_closeinh,1)); conns.append(k.conn(src,STEER,si,Tt(src))); si+=1  # strong odour (close) -> stop spiral (converge)
    sy.append(k.term(STEER,Tt(STEER)))
    return k.build(ne,sy,conns,ex)

class NeuralAgent:
    def __init__(self, foods, sigma=14.0, gain=5.0, sub=16, **bkw):
        self.model=mujoco.MjModel.from_xml_string(XML); self.data=mujoco.MjData(self.model)
        self.foods=[[float(a),float(b),True] for a,b in foods]; self.sigma=sigma; self.gain=gain; self.sub=sub
        self.net,self.core=k.load(build(**bkw)); self.nb={i:u for i,u in self.net.network.neurons.items()}
        self.a={n:self.model.actuator(n).id for n in ("plp","plr","prp","prr")}
        self.sx=self.model.joint('sx').qposadr[0]; self.sy=self.model.joint('sy').qposadr[0]; self.rz=self.model.joint('rz').qposadr[0]
        self.kicked=False; self.t=0
    def pose(self): return float(self.data.qpos[self.sx]),float(self.data.qpos[self.sy]),float(self.data.qpos[self.rz])
    def odor(self,p): return float(sum(np.exp(-((p[0]-fx)**2+(p[1]-fy)**2)/self.sigma) for fx,fy,a in self.foods if a))
    def step(self):
        x,y,yaw=self.pose(); fwd=yaw+np.pi                      # forward is -x body axis
        hd=np.array([np.cos(fwd),np.sin(fwd)]); lat=np.array([-np.sin(fwd),np.cos(fwd)]); head=np.array([x,y])+hd*0.45
        oL=self.odor(head+lat*0.6); oR=self.odor(head-lat*0.6)  # wide antennae -> bigger L/R differential
        for _ in range(self.sub):                                       # physics steps at the NEURAL tick rate
            for e in ENG: self.net.set_external_input(e,0,1.0)           # tonic drive -> engine (global gain)
            for nid in FL: self.net.set_external_input(nid,0,self.gain*oL)   # odour transduction (every tick)
            for nid in FR: self.net.set_external_input(nid,0,self.gain*oR)
            if not self.kicked and self.t>2: self.net.set_external_input(P[0],0,5.0); self.kicked=True   # one-time CPG birth seed
            self.core.do_tick(); self.t+=1
            self.data.ctrl[self.a["plp"]]=GGAIN*max(0,self.nb[MLp].S); self.data.ctrl[self.a["plr"]]=GGAIN*max(0,self.nb[MLr].S)
            self.data.ctrl[self.a["prp"]]=GGAIN*max(0,self.nb[MRp].S); self.data.ctrl[self.a["prr"]]=GGAIN*max(0,self.nb[MRr].S)
            mujoco.mj_step(self.model,self.data)
        for f in self.foods:
            if f[2] and (x-f[0])**2+(y-f[1])**2<0.7**2: f[2]=False
        return oL+oR

if __name__=="__main__":
    # Fully-neural 360-degree navigation: only non-neural steps are the odour->current and muscle-S->force
    # transducers (+ a one-time CPG birth seed). Food placed at 8 headings around the start.
    print("FULLY-NEURAL AGENT (odour->neurons->muscles->body) — 360-degree navigation:")
    reached=0; foods=[(np.cos(a)*4.0, np.sin(a)*4.0) for a in np.linspace(0,2*np.pi,8,endpoint=False)]
    for food in foods:
        ag=NeuralAgent(foods=[food]); d0=np.hypot(*food); mind=d0
        for t in range(4000):
            ag.step(); x,y,_=ag.pose(); mind=min(mind,np.hypot(x-food[0],y-food[1]))
        ok=mind<1.0; reached+=ok
        print(f"  food=({food[0]:+.1f},{food[1]:+.1f}): closest={mind:.2f}  {'REACHED' if ok else ''}", flush=True)
    print(f"REACHED {reached}/{len(foods)}")
    print("@@@AGENT DONE@@@")
