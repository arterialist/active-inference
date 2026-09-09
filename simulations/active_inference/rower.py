"""Pivot-capable body test — a two-paddle 'rower' in a viscous medium.
Left and right paddles are independently actuated. An asymmetric stroke (fast power stroke backward, slow
feathered recovery) gives net forward thrust via velocity-dependent fluid drag. Differential left/right
stroke amplitude STEERS; antiphase (one forward, one backward) PIVOTS in place. First we verify the plant
with SCRIPTED strokes (physics only), then drive it with PAULA."""
import sys, numpy as np, mujoco
sys.path.insert(0, "/Users/arterialist/Projects/agi-research/neuron-model")

XML = """
<mujoco model="rower">
  <option timestep="0.004" density="1200" viscosity="0.5" integrator="RK4"><flag gravity="disable"/></option>
  <visual><global offwidth="640" offheight="480"/><headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4"/></visual>
  <asset><texture name="grid" type="2d" builtin="checker" rgb1="0.9 0.9 0.86" rgb2="0.82 0.82 0.78" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="12 12"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="30 30 0.1" material="grid" pos="0 0 -0.4"/>
    <body name="torso" pos="0 0 0">
      <joint name="sx" type="slide" axis="1 0 0"/><joint name="sy" type="slide" axis="0 1 0"/>
      <joint name="rz" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="-0.18 0 0 0.18 0 0" size="0.05" rgba="0.2 0.28 0.6 1"/>
      <geom type="capsule" fromto="0.18 0 0 0.28 0 0" size="0.03" rgba="0.15 0.2 0.45 1"/>
      <body name="padL" pos="-0.15 0.06 0">
        <joint name="pl" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 -0.02 0.20 0" size="0.02" rgba="0.15 0.5 0.45 1"/>
      </body>
      <body name="padR" pos="-0.15 -0.06 0">
        <joint name="pr" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 -0.02 -0.20 0" size="0.02" rgba="0.5 0.3 0.15 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="pl" kp="8" ctrlrange="-1.6 1.6"/>
    <position joint="pr" kp="8" ctrlrange="-1.6 1.6"/>
  </actuator>
</mujoco>"""

class Rower:
    def __init__(self):
        self.model=mujoco.MjModel.from_xml_string(XML); self.data=mujoco.MjData(self.model)
        self.sx=self.model.joint('sx').qposadr[0]; self.sy=self.model.joint('sy').qposadr[0]; self.rz=self.model.joint('rz').qposadr[0]
    def pose(self): return float(self.data.qpos[self.sx]), float(self.data.qpos[self.sy]), float(self.data.qpos[self.rz])
    def set_paddles(self, aL, aR):
        self.data.ctrl[0]=np.clip(aL,-1.6,1.6); self.data.ctrl[1]=np.clip(-aR,-1.6,1.6)  # mirror R
        mujoco.mj_step(self.model,self.data)

def stroke(phase, ampl=1.0, feather=0.30):
    """Asymmetric stroke: fast power stroke (paddle sweeps back) then slow feathered recovery.
    phase in [0,1). Returns paddle angle in ~[-ampl,+ampl]; power = fast, recovery = slow."""
    p = phase % 1.0
    if p < feather:  a = -1 + 2*(p/feather)          # power stroke: front->back, FAST
    else:            a = 1 - 2*((p-feather)/(1-feather))  # recovery: back->front, SLOW
    return ampl * a

if __name__=="__main__":
    import numpy as np
    T=20000; per=140  # stroke period in physics steps
    def run(driveL, driveR, tag):
        r=Rower(); x0,y0,yaw0=r.pose()
        for t in range(T):
            ph=(t%per)/per
            r.set_paddles(stroke(ph)*driveL, stroke(ph)*driveR)
        x,y,yaw=r.pose(); print(f"  {tag:14s}: pos=({x:6.2f},{y:6.2f}) dist={np.hypot(x-x0,y-y0):5.2f} heading={np.degrees(yaw-yaw0):+.0f}deg")
    print(f"Plant test ({T} steps each):")
    run(1.0, 1.0, "both (forward)")
    run(1.0, 0.4, "L>R (turn)")
    run(1.0, -1.0, "antiphase (pivot)")
    print("@@@ROWER DONE@@@")
