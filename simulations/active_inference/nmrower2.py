"""Tuned neuromuscular rower — fully neural motor. Graded muscle membrane S -> actuator force (NMJ).
Antagonist muscle pair per paddle driven by the CPG (stroke emerges); steering = an inhibitory current
to one side's muscles (weakens that paddle -> differential -> turn). No stroke(), no dL/dR, no turn scalar.
The creature swims toward -x (its 'forward'), so heading is measured accordingly."""
import sys, numpy as np, mujoco
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
from paula_agent import ckit as k
def Tt(n): return 900+n
P=[1,2,3,4]; MLp,MLr,MRp,MRr=10,11,12,13
GEAR=25.0; PERIOD=40; LAM_M=6; GGAIN=8.0
XML=f"""
<mujoco><option timestep="0.004" density="1200" viscosity="0.5" integrator="RK4"><flag gravity="disable"/></option>
<worldbody><geom type="plane" size="40 40 0.1" pos="0 0 -0.4"/>
<body name="torso" pos="0 0 0"><joint name="sx" type="slide" axis="1 0 0"/><joint name="sy" type="slide" axis="0 1 0"/><joint name="rz" type="hinge" axis="0 0 1"/>
<geom type="capsule" fromto="-0.18 0 0 0.18 0 0" size="0.05"/><geom type="capsule" fromto="-0.28 0 0 -0.18 0 0" size="0.03"/>
<body name="padL" pos="-0.15 0.06 0"><joint name="pl" type="hinge" axis="0 0 1" range="-1.8 1.8" damping="0.1"/><geom type="capsule" fromto="0 0 0 -0.03 0.24 0" size="0.022"/></body>
<body name="padR" pos="-0.15 -0.06 0"><joint name="pr" type="hinge" axis="0 0 1" range="-1.8 1.8" damping="0.1"/><geom type="capsule" fromto="0 0 0 -0.03 -0.24 0" size="0.022"/></body>
</body></worldbody>
<actuator><motor joint="pl" gear="{GEAR}" name="plp"/><motor joint="pl" gear="-{GEAR}" name="plr"/>
<motor joint="pr" gear="{GEAR}" name="prp"/><motor joint="pr" gear="-{GEAR}" name="prr"/></actuator></mujoco>"""
def build(w=8.0, w_steer=-7.0):
    ne=[];sy=[];conns=[];ex=[]
    for i in range(4): nid=P[i]; ne.append(k.neuron(nid,r=0.6,lam=3,c=PERIOD)); sy.extend([k.syn(nid,0,4.0,PERIOD),k.term(nid,Tt(nid))])
    for i in range(4): conns.append(k.conn(P[(i-1)%4],P[i],0,Tt(P[(i-1)%4])))
    ex.append(k.ext(P[0],0))
    for m,ph in [(MLp,P[0]),(MLr,P[2]),(MRp,P[0]),(MRr,P[2])]:
        ne.append(k.neuron(m,r=1e9,lam=LAM_M,delta_decay=1.0,c=2))
        sy.extend([k.syn(m,0,w,1),k.syn(m,1,w_steer,1),k.term(m,Tt(m))])   # syn1 = steering inhibition
        conns.append(k.conn(ph,m,0,Tt(ph))); ex.append(k.ext(m,1))
    return k.build(ne,sy,conns,ex)
class NMRower2:
    def __init__(self, w_cpg=8.0, w_steer=-7.0, muscle_gain=GGAIN):
        """Build the PAULA CPG-to-muscle circuit and its MuJoCo body.

        The defaults preserve the historical reference motor.  ``w_cpg=0``
        disconnects the neural CPG from the graded muscles; ``muscle_gain=0``
        leaves the neural circuit active but disconnects its allowed
        muscle-state-to-actuator transducer.  Both are useful causal controls.
        """
        self.model=mujoco.MjModel.from_xml_string(XML); self.data=mujoco.MjData(self.model)
        self.net,self.core=k.load(build(w=w_cpg,w_steer=w_steer)); self.nb={i:u for i,u in self.net.network.neurons.items()}
        self.a={n:self.model.actuator(n).id for n in ("plp","plr","prp","prr")}
        self.sx=self.model.joint('sx').qposadr[0]; self.sy=self.model.joint('sy').qposadr[0]; self.rz=self.model.joint('rz').qposadr[0]
        self.muscle_gain=float(muscle_gain); self.kicked=False; self.t=0
    def pose(self):
        x=float(self.data.qpos[self.sx]); y=float(self.data.qpos[self.sy]); yaw=float(self.data.qpos[self.rz])
        return x,y,yaw
    def heading(self):                                  # forward is -x body axis
        return float(self.data.qpos[self.rz])+np.pi
    def step(self, steerL=0.0, steerR=0.0, sub=6, tick_trace=None):
        """Advance neural and body dynamics; optional trace is observational only."""
        for _ in range(sub):
            if not self.kicked and self.t>2: self.net.set_external_input(P[0],0,5.0); self.kicked=True
            self.net.set_external_input(MLp,1,steerL); self.net.set_external_input(MLr,1,steerL)
            self.net.set_external_input(MRp,1,steerR); self.net.set_external_input(MRr,1,steerR)
            self.core.do_tick(); self.t+=1
        self.data.ctrl[self.a["plp"]]=self.muscle_gain*max(0,self.nb[MLp].S); self.data.ctrl[self.a["plr"]]=self.muscle_gain*max(0,self.nb[MLr].S)
        self.data.ctrl[self.a["prp"]]=self.muscle_gain*max(0,self.nb[MRp].S); self.data.ctrl[self.a["prr"]]=self.muscle_gain*max(0,self.nb[MRr].S)
        mujoco.mj_step(self.model,self.data)
        if tick_trace is not None:
            x,y,yaw=self.pose()
            tick_trace.append({
                "neural_tick":self.t,
                "steer_left_current":float(steerL), "steer_right_current":float(steerR),
                "cpg_spikes":{str(nid):int(self.nb[nid].O>0) for nid in P},
                "muscle_state":{str(nid):float(self.nb[nid].S) for nid in (MLp,MLr,MRp,MRr)},
                "actuator_ctrl":{name:float(self.data.ctrl[aid]) for name,aid in self.a.items()},
                "pose":{"x":x,"y":y,"yaw":yaw},
            })
if __name__=="__main__":
    print("Tuned neuromuscular rower — forward / turn / pivot (steering = muscle inhibition):")
    def run(sL,sR,tag,T=16000):
        r=NMRower2(); x0,y0,h0=r.pose(); h0=r.heading()
        for _ in range(T): r.step(sL,sR)
        x,y,_=r.pose(); print(f"  {tag:20s}: dist={np.hypot(x-x0,y-y0):5.2f} heading_change={np.degrees(r.heading()-h0):+.0f}deg")
    run(0,0,"both (forward)")
    run(1.5,0,"inhibit-L (turn)")
    run(0,1.5,"inhibit-R (turn)")
    run(1.5,-1.5,"antiphase (pivot?)")
    print("@@@NMR2 DONE@@@")
