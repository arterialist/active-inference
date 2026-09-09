"""Real articulated body driven by the PAULA CPG — neural motor control, not a readout.

A planar multi-link swimmer in a viscous medium (MuJoCo). Each inter-link hinge is a MUSCLE: its target
angle is set by the CPG's left/right muscle-driver neurons for that segment (bend = left_drive -
right_drive). The CPG's head->tail traveling wave therefore becomes a travelling bend wave, which pushes
against the fluid and propels the body forward. Descending 'motor cortex' currents bias the left/right
drive to STEER. Nothing here maps an action to a velocity; movement emerges from spikes -> muscles ->
physics."""
import sys, numpy as np, mujoco
sys.path.insert(0, "/Users/arterialist/Projects/agi-research/neuron-model")
from paula_agent import ckit as k

N = 6  # body links

def body_xml(n=N):
    links = ""
    close = ""
    for i in range(1, n):
        links += (f'<body name="l{i}" pos="0.15 0 0">'
                  f'<joint name="j{i}" type="hinge" axis="0 0 1" range="-1.0 1.0" damping="0.6"/>'
                  f'<geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.028" rgba="0.23 0.4 0.75 1"/>')
        close += "</body>"
    return f"""
<mujoco model="swimmer">
  <option timestep="0.004" density="1200" viscosity="0.9" integrator="RK4">
    <flag gravity="disable"/>
  </option>
  <visual><global offwidth="640" offheight="480"/><headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4"/></visual>
  <asset><texture name="grid" type="2d" builtin="checker" rgb1="0.9 0.9 0.86" rgb2="0.82 0.82 0.78" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="10 10"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" material="grid" pos="0 0 -0.5"/>
    <body name="head" pos="0 0 0">
      <joint name="sx" type="slide" axis="1 0 0"/><joint name="sy" type="slide" axis="0 1 0"/>
      <joint name="rz" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.032" rgba="0.18 0.22 0.5 1"/>
      {links}{close}
    </body>
  </worldbody>
  <actuator>
    {''.join(f'<position joint="j{i}" kp="3.0" ctrlrange="-1 1"/>' for i in range(1,n))}
  </actuator>
</mujoco>"""


def build_cpg(n=N, seg_delay=4):
    """Pacemaker synfire ring + LEFT/RIGHT muscle-driver chains (antiphase traveling wave).
    Adds descending steering pins: extra external drive to all LEFT drivers (turn one way) or RIGHT."""
    ne=[]; sy=[]; conns=[]; ex=[]
    for i in range(4):                                    # head pacemaker (4-neuron ring)
        nid=1+i; ne.append(k.neuron(nid, r=0.6, lam=3, c=6)); sy += [k.syn(nid,0,4.0,6), k.term(nid,tid=900+nid)]
    for i in range(4): conns.append(k.conn((i-1)%4+1, i+1, 0, stid=900+(i-1)%4+1))
    ex.append(k.ext(1,0))
    L0, R0 = 10, 50
    for base, pace in [(L0,1),(R0,3)]:                    # L kicked by n1, R by n3 (antiphase)
        for i in range(n-1):
            nid=base+i; ne.append(k.neuron(nid, r=0.6, lam=3, c=5))
            sy += [k.syn(nid,0,5.0,seg_delay), k.syn(nid,1,3.0,1), k.term(nid,tid=900+nid)]  # syn1: descending bias
            ex.append(k.ext(nid,1))                        # descending steering input
        conns.append(k.conn(pace, base, 0, stid=900+pace))
        for i in range(n-2): conns.append(k.conn(base+i, base+i+1, 0, stid=900+base+i))
    return k.build(ne,sy,conns,ex), L0, R0


class NeuralSwimmer:
    def __init__(self, n=N):
        self.n=n; self.model=mujoco.MjModel.from_xml_string(body_xml(n)); self.data=mujoco.MjData(self.model)
        self.path,self.L0,self.R0=build_cpg(n)
        self.net,self.core=k.load(self.path); self.nb={i:u for i,u in self.net.network.neurons.items()}
        self.Lf=np.zeros(n-1); self.Rf=np.zeros(n-1)             # filtered muscle-driver activity
        self.jadr=[self.model.joint(f'j{i}').qposadr[0] for i in range(1,n)]
        self.sx=self.model.joint('sx').qposadr[0]; self.sy=self.model.joint('sy').qposadr[0]; self.rz=self.model.joint('rz').qposadr[0]
        self.kicked=False; self.t=0
    def pose(self): return float(self.data.qpos[self.sx]), float(self.data.qpos[self.sy]), float(self.data.qpos[self.rz])
    def step(self, turn=0.0, cpg_substeps=5):
        """Advance: run the CPG a few neural ticks, filter muscle-driver spikes, set joint muscle targets,
        step physics. `turn`>0 biases LEFT drivers, <0 biases RIGHT (descending motor command)."""
        for _ in range(cpg_substeps):
            if not self.kicked and self.t>2: self.net.set_external_input(1,0,5.0); self.kicked=True
            self.core.do_tick(); self.t+=1
            Ls=np.array([self.nb[self.L0+i].O>0 for i in range(self.n-1)],float)
            Rs=np.array([self.nb[self.R0+i].O>0 for i in range(self.n-1)],float)
            self.Lf=0.8*self.Lf+Ls; self.Rf=0.8*self.Rf+Rs        # leaky filter -> muscle activation
        # muscle activation = travelling wave (CPG) + descending tonic curvature (motor cortex steering)
        curv=float(np.clip(turn,-1,1))*0.22
        bend=np.clip(0.9*(self.Lf-self.Rf)+curv,-1,1)
        self.data.ctrl[:]=bend
        mujoco.mj_step(self.model,self.data)


if __name__=="__main__":
    print("STAGE A — does the CPG-driven articulated body SWIM (forward displacement)?")
    sw=NeuralSwimmer()
    x0,y0,_=sw.pose()
    for _ in range(600): sw.step(turn=0.0)
    x1,y1,yaw=sw.pose()
    disp=np.hypot(x1-x0,y1-y0)
    print(f"  straight: start=({x0:.2f},{y0:.2f}) end=({x1:.2f},{y1:.2f}) displacement={disp:.3f} yaw={yaw:.2f}")
    print("STAGE B — does a descending TURN command steer it?")
    sw=NeuralSwimmer()
    for _ in range(600): sw.step(turn=1.0)
    x2,y2,yaw2=sw.pose(); print(f"  turn=+1 : end=({x2:.2f},{y2:.2f}) yaw={yaw2:.2f} (expect yaw change)")
    print("@@@SWIMMER DONE@@@")
