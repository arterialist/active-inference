"""DUAL-CHANNEL CHEMOTAXIS — fully-neural agent in a mixed environment of FOOD (attractant) and TOXIN
(repellent). Two chemoreceptor channels feed the SAME motor and SUM there, like real chemotaxis:
  FOOD  channel -> tropotaxis TOWARD + klinokinesis that converges on the source (eat)
  TOXIN channel -> tropotaxis AWAY (opposite crossing) + rising-toxin DRIVES the escape spiral (flee)
No mode switch: attraction and repulsion coexist and compete at the muscles. Only non-neural steps remain
the transducers (field->current per channel; muscle-S->force) and a one-time CPG birth seed.
"""
import sys, numpy as np, mujoco
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
from paula_agent import ckit as k
def Tt(n): return 900+n
P=[1,2,3,4]; MLp,MLr,MRp,MRr=10,11,12,13
NP=10; THR=np.exp(np.linspace(np.log(0.25),np.log(4.5),NP))
# FOOD channel
FL=list(range(20,20+NP)); FR=list(range(40,40+NP)); POOL=60; STEER=62; TL=70; TR=71
RISEP=list(range(63,63+6)); RISE_DEL=[35,50,65,80,95,110]
# TOXIN channel (repellent)
XL=list(range(300,300+NP)); XR=list(range(320,320+NP)); XPOOL=340; dXL=341; dXR=342
XRISEP=list(range(343,343+6))
ENG=[50,51,52,53,54]; ENG_C=[2,3,5,7,11]
NRT=6; THR_R=np.linspace(0.5,4.2,NRT)
RLY={MLp:list(range(110,110+NRT)),MLr:list(range(120,120+NRT)),MRp:list(range(130,130+NRT)),MRr:list(range(140,140+NRT))}
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

def build(w_cpg=8.0, w_norm=-0.9, w_pool=0.5, w_sd=2.5, w_wta=-7.0, w_ton=2.5, w_riseinh=-4.0, w_closeinh=-0.35,
          w_rc=14.0, w_relay_inh=-4.0, w_steer_relay=-5.0, w_brake=-0.7,
          w_xsd=2.5, w_xesc=3.2, w_xbrake_off=0.0, swap_turn=False):
    ne=[];sy=[];conns=[];ex=[]
    # ENGINE
    for e,c in zip(ENG,ENG_C): ne.append(k.neuron(e,r=0.6,c=c,lam=5)); sy.extend([k.syn(e,0,2.0,1),k.term(e,Tt(e))]); ex.append(k.ext(e,0))
    # CPG ring (seeded once)
    for i in range(4): nid=P[i]; ne.append(k.neuron(nid,r=0.6,lam=3,c=PERIOD)); sy.extend([k.syn(nid,0,4.0,PERIOD),k.term(nid,Tt(nid))])
    for i in range(4): conns.append(k.conn(P[(i-1)%4],P[i],0,Tt(P[(i-1)%4])))
    ex.append(k.ext(P[0],0))
    # MUSCLES + RELAY gating (TL/TR + STEER + brake), identical proven motor
    for m,ph in [(MLp,P[0]),(MLr,P[2]),(MRp,P[0]),(MRr,P[2])]:
        ne.append(k.neuron(m,r=1e9,lam=LAM_M,delta_decay=1.0,c=2))
        for j in range(NRT): sy.append(k.syn(m,j,w_cpg/NRT,1))
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
            sy.append(k.syn(g,si,w_brake,1)); conns.append(k.conn(POOL,g,si,Tt(POOL))); si+=1  # brake near FOOD only
            sy.append(k.term(g,Tt(g))); conns.append(k.conn(g,m,j,Tt(g)))
    # ---- FOOD channel: sensors + POOL ----
    for p in (FL,FR):
        for i,nid in enumerate(p): ne.append(k.neuron(nid,r=float(THR[i]),c=2,lam=4)); sy.extend([k.syn(nid,0,1.0,1),k.syn(nid,1,w_norm,1),k.term(nid,Tt(nid))]); ex.append(k.ext(nid,0)); conns.append(k.conn(POOL,nid,1,Tt(POOL)))
    ne.append(k.neuron(POOL,r=0.5,c=1,lam=3)); pj=0
    for nid in FL+FR: sy.append(k.syn(POOL,pj,w_pool,1)); conns.append(k.conn(nid,POOL,pj,Tt(nid))); pj+=1
    sy.append(k.term(POOL,Tt(POOL)))
    # ---- TOXIN channel: sensors + POOL ----
    for p in (XL,XR):
        for i,nid in enumerate(p): ne.append(k.neuron(nid,r=float(THR[i]),c=2,lam=4)); sy.extend([k.syn(nid,0,1.0,1),k.syn(nid,1,w_norm,1),k.term(nid,Tt(nid))]); ex.append(k.ext(nid,0)); conns.append(k.conn(XPOOL,nid,1,Tt(XPOOL)))
    ne.append(k.neuron(XPOOL,r=0.5,c=1,lam=3)); pj=0
    for nid in XL+XR: sy.append(k.syn(XPOOL,pj,w_pool,1)); conns.append(k.conn(nid,XPOOL,pj,Tt(nid))); pj+=1
    sy.append(k.term(XPOOL,Tt(XPOOL)))
    # ---- TURN command TL/TR: FOOD toward (dFL->TL) + TOXIN away (opposite crossing: toxin-right->TL) ----
    for nid in (TL,TR): ne.append(k.neuron(nid,r=0.6,c=2,lam=4))
    tj={TL:0,TR:0}
    def tsyn(nid,w): j=tj[nid]; sy.append(k.syn(nid,j,w,1)); tj[nid]+=1; return j
    for s in FL: conns.append(k.conn(s,TL,tsyn(TL, w_sd),Tt(s)))   # food left  -> TL (toward)
    for s in FR: conns.append(k.conn(s,TL,tsyn(TL,-w_sd),Tt(s)))
    for s in FR: conns.append(k.conn(s,TR,tsyn(TR, w_sd),Tt(s)))   # food right -> TR
    for s in FL: conns.append(k.conn(s,TR,tsyn(TR,-w_sd),Tt(s)))
    for s in XR: conns.append(k.conn(s,TL,tsyn(TL, w_xsd),Tt(s)))  # toxin RIGHT -> TL (turn away = left)
    for s in XL: conns.append(k.conn(s,TL,tsyn(TL,-w_xsd),Tt(s)))
    for s in XL: conns.append(k.conn(s,TR,tsyn(TR, w_xsd),Tt(s)))  # toxin LEFT  -> TR (turn away = right)
    for s in XR: conns.append(k.conn(s,TR,tsyn(TR,-w_xsd),Tt(s)))
    conns.append(k.conn(TR,TL,tsyn(TL,w_wta),Tt(TR))); conns.append(k.conn(TL,TR,tsyn(TR,w_wta),Tt(TL)))
    sy.append(k.term(TL,Tt(TL))); sy.append(k.term(TR,Tt(TR)))
    # ---- FOOD klinokinesis: RISE(food) population + STEER (search spiral) ----
    for r,dl in zip(RISEP,RISE_DEL):
        ne.append(k.neuron(r,r=0.6,c=2,lam=3)); sy.extend([k.syn(r,0,3.4,1),k.syn(r,1,-3.4,dl),k.term(r,Tt(r))])
        conns.extend([k.conn(POOL,r,0,Tt(POOL)),k.conn(POOL,r,1,Tt(POOL))])
    # ---- TOXIN klinokinesis: RISE(toxin) population -> DRIVES the escape spiral ----
    for r,dl in zip(XRISEP,RISE_DEL):
        ne.append(k.neuron(r,r=0.6,c=2,lam=3)); sy.extend([k.syn(r,0,3.4,1),k.syn(r,1,-3.4,dl),k.term(r,Tt(r))])
        conns.extend([k.conn(XPOOL,r,0,Tt(XPOOL)),k.conn(XPOOL,r,1,Tt(XPOOL))])
    # STEER: engine tonic spiral; FOOD-rising & FOOD-close SUPPRESS (converge to eat); TOXIN-rising DRIVES (flee)
    ne.append(k.neuron(STEER,r=0.6,c=2,lam=5)); si=0
    for e in ENG: sy.append(k.syn(STEER,si,w_ton,1)); conns.append(k.conn(e,STEER,si,Tt(e))); si+=1
    for r in RISEP: sy.append(k.syn(STEER,si,w_riseinh,1)); conns.append(k.conn(r,STEER,si,Tt(r))); si+=1
    for src in FL+FR: sy.append(k.syn(STEER,si,w_closeinh,1)); conns.append(k.conn(src,STEER,si,Tt(src))); si+=1
    for r in XRISEP: sy.append(k.syn(STEER,si,w_xesc,1)); conns.append(k.conn(r,STEER,si,Tt(r))); si+=1
    sy.append(k.term(STEER,Tt(STEER)))
    return k.build(ne,sy,conns,ex)

class DualAgent:
    def __init__(self, foods, toxins, sigma=14.0, sigma_tox=6.0, gain=5.0, sub=16, eat_r=0.7, hit_r=0.8, **bkw):
        self.model=mujoco.MjModel.from_xml_string(XML); self.data=mujoco.MjData(self.model)
        self.foods=[[float(a),float(b),True] for a,b in foods]
        self.toxins=[[float(a),float(b),True] for a,b in toxins]
        # toxin acts at SHORTER range than food (repellent = local no-go; attractant = long-range seek)
        self.sigma=sigma; self.sigma_tox=sigma_tox; self.gain=gain; self.sub=sub; self.eat_r=eat_r; self.hit_r=hit_r
        self.net,self.core=k.load(build(**bkw)); self.nb={i:u for i,u in self.net.network.neurons.items()}
        self.a={n:self.model.actuator(n).id for n in ("plp","plr","prp","prr")}
        self.sx=self.model.joint('sx').qposadr[0]; self.sy=self.model.joint('sy').qposadr[0]; self.rz=self.model.joint('rz').qposadr[0]
        self.kicked=False; self.t=0; self.eaten=0; self.hits=0
    def pose(self): return float(self.data.qpos[self.sx]),float(self.data.qpos[self.sy]),float(self.data.qpos[self.rz])
    def _field(self,items,p,sig): return float(sum(np.exp(-((p[0]-fx)**2+(p[1]-fy)**2)/sig) for fx,fy,a in items if a))
    def step(self):
        x,y,yaw=self.pose(); fwd=yaw+np.pi
        hd=np.array([np.cos(fwd),np.sin(fwd)]); lat=np.array([-np.sin(fwd),np.cos(fwd)]); head=np.array([x,y])+hd*0.45
        pL=head+lat*0.6; pR=head-lat*0.6
        fL=self._field(self.foods,pL,self.sigma);   fR=self._field(self.foods,pR,self.sigma)
        xL=self._field(self.toxins,pL,self.sigma_tox); xR=self._field(self.toxins,pR,self.sigma_tox)
        for _ in range(self.sub):
            for e in ENG: self.net.set_external_input(e,0,1.0)
            for nid in FL: self.net.set_external_input(nid,0,self.gain*fL)
            for nid in FR: self.net.set_external_input(nid,0,self.gain*fR)
            for nid in XL: self.net.set_external_input(nid,0,self.gain*xL)
            for nid in XR: self.net.set_external_input(nid,0,self.gain*xR)
            if not self.kicked and self.t>2: self.net.set_external_input(P[0],0,5.0); self.kicked=True
            self.core.do_tick(); self.t+=1
            self.data.ctrl[self.a["plp"]]=GGAIN*max(0,self.nb[MLp].S); self.data.ctrl[self.a["plr"]]=GGAIN*max(0,self.nb[MLr].S)
            self.data.ctrl[self.a["prp"]]=GGAIN*max(0,self.nb[MRp].S); self.data.ctrl[self.a["prr"]]=GGAIN*max(0,self.nb[MRr].S)
            mujoco.mj_step(self.model,self.data)
        for f in self.foods:
            if f[2] and (x-f[0])**2+(y-f[1])**2<self.eat_r**2: f[2]=False; self.eaten+=1
        for tx in self.toxins:                     # touching toxin = a hit (it stays; a real toxin isn't consumed)
            if tx[2] and (x-tx[0])**2+(y-tx[1])**2<self.hit_r**2: self.hits+=1
        return fL+fR, xL+xR

if __name__=="__main__":
    rng=np.random.default_rng(3)
    def scatter(n,rlo,rhi):
        out=[]
        for _ in range(n):
            a=rng.uniform(0,2*np.pi); r=rng.uniform(rlo,rhi); out.append((float(r*np.cos(a)),float(r*np.sin(a))))
        return out
    foods=scatter(8,3.0,9.0); toxins=scatter(5,3.0,9.0)
    ag=DualAgent(foods=foods, toxins=toxins)
    for t in range(20000):
        ag.step()
        if ag.eaten>=len(foods): break
    print(f"ate {ag.eaten}/{len(foods)} foods; toxin contact-steps={ag.hits} (touches on {len(toxins)} toxins)")
    print("@@@DUAL DONE@@@")
