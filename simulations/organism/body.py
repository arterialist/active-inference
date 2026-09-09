"""MuJoCo body for the organism — a planar 'bug' with two antennae, on a floor with food sources.
Planar 3-DOF (x, y, yaw) driven kinematically from a controller (forward speed + turn rate). Odor is a
scalar SALIENCE field (Gaussians around food); antennae sample left/right concentration. A nest disc
sits at the origin and a mocap predator body can be moved each step.

Run standalone:  python -m simulations.organism.body     (verifies physics + offscreen render)
"""
import pathlib
import numpy as np, mujoco

XML = """
<mujoco model="bug">
  <option timestep="0.02" gravity="0 0 0" integrator="Euler"><flag contact="disable"/></option>
  <visual><headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35"/><global offwidth="640" offheight="480"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.9 0.9 0.86" rgb2="0.82 0.82 0.78" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="12 12" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="12 12 0.1" material="grid"/>
    <geom name="nest" type="cylinder" size="0.7 0.02" pos="0 0 0.02" rgba="0.25 0.4 0.85 0.55"/>
    <body name="bug" pos="0 0 0.12">
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="y" type="slide" axis="0 1 0"/>
      <joint name="yaw" type="hinge" axis="0 0 1"/>
      <geom name="body" type="ellipsoid" size="0.36 0.24 0.12" rgba="0.23 0.30 0.75 1"/>
      <geom name="head" type="sphere" size="0.14" pos="0.34 0 0.02" rgba="0.18 0.22 0.5 1"/>
      <geom name="antL" type="capsule" fromto="0.42 0.12 0.03 0.7 0.62 0.06" size="0.02" rgba="0.15 0.5 0.45 1"/>
      <geom name="antR" type="capsule" fromto="0.42 -0.12 0.03 0.7 -0.62 0.06" size="0.02" rgba="0.15 0.5 0.45 1"/>
      <site name="antL_tip" pos="0.7 0.62 0.06" size="0.03"/>
      <site name="antR_tip" pos="0.7 -0.62 0.06" size="0.03"/>
    </body>
    <body name="pred" mocap="true" pos="8 8 0.25">
      <geom name="pred" type="sphere" size="0.34" rgba="0.55 0.12 0.12 1"/>
      <geom name="predeye" type="sphere" size="0.1" pos="0.22 0 0.12" rgba="0.95 0.85 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""

class World:
    def __init__(self, foods):
        # foods: list of (x,y, sign) sign=+1 nutritious / -1 toxic (odor strength)
        self.foods=foods
        markers="".join(f'<geom name="food{i}" type="cylinder" size="0.3 0.05" pos="{fx} {fy} 0.05" '
                        f'rgba="{"0.12 0.55 0.4 1" if s>0 else "0.7 0.25 0.2 1"}"/>' for i,(fx,fy,s) in enumerate(foods))
        xml=XML.replace('</worldbody>', markers+'</worldbody>')
        self.model=mujoco.MjModel.from_xml_string(xml)
        self.data=mujoco.MjData(self.model)
        self.jx=self.model.joint('x').qposadr[0]; self.jy=self.model.joint('y').qposadr[0]; self.jyaw=self.model.joint('yaw').qposadr[0]
        self.aL=self.model.site('antL_tip').id; self.aR=self.model.site('antR_tip').id
        self.pred_mid=self.model.body('pred').mocapid[0]
    def set_foods(self, foods):
        # rebuild the model with new food positions but PRESERVE the bug's pose/velocity + predator.
        qpos=self.data.qpos.copy(); qvel=self.data.qvel.copy(); mocap=self.data.mocap_pos.copy()
        self.__init__(foods)
        self.data.qpos[:]=qpos; self.data.qvel[:]=qvel; self.data.mocap_pos[:]=mocap
        mujoco.mj_forward(self.model,self.data)
    def set_predator(self,x,y):
        self.data.mocap_pos[self.pred_mid]=[x,y,0.25]
    def predator_pos(self):
        return float(self.data.mocap_pos[self.pred_mid][0]), float(self.data.mocap_pos[self.pred_mid][1])
    def pose(self):
        return self.data.qpos[self.jx], self.data.qpos[self.jy], self.data.qpos[self.jyaw]
    def odor_at(self, x, y):
        c=0.0                                     # SALIENCE field (unsigned): any food is smell-able;
        for fx,fy,s in self.foods:                # the mushroom body decides approach vs avoid separately
            d2=(x-fx)**2+(y-fy)**2; c+= abs(s)*np.exp(-d2/28.0)
        return c
    def antennae(self):
        mujoco.mj_forward(self.model,self.data)
        L=self.data.site_xpos[self.aL]; R=self.data.site_xpos[self.aR]
        return self.odor_at(L[0],L[1]), self.odor_at(R[0],R[1])
    def step(self, speed, turn):
        _,_,yaw=self.pose()
        self.data.qvel[0]=speed*np.cos(yaw); self.data.qvel[1]=speed*np.sin(yaw); self.data.qvel[2]=turn
        mujoco.mj_step(self.model,self.data)

if __name__=="__main__":
    w=World([(5,3,+1),(-4,4,-1),(3,-5,+1)])
    reached=None
    for t in range(1200):
        cl,cr=w.antennae(); turn=2.0*(cl-cr); speed=0.9
        w.step(speed,turn)
        x,y,_=w.pose()
        for fx,fy,s in w.foods:
            if s>0 and (x-fx)**2+(y-fy)**2<0.3**2: reached=(t,fx,fy); break
        if reached: break
    print("physics OK. reached nutritious food:",reached,"final pose",[round(v,2) for v in w.pose()])
    try:
        ren=mujoco.Renderer(w.model, 480, 640)
        mujoco.mj_forward(w.model,w.data); ren.update_scene(w.data)
        img=ren.render()
        print("RENDER OK: frame shape",img.shape,"mean",round(float(img.mean()),1))
        out=pathlib.Path(__file__).parent/"body_frame0.png"
        from PIL import Image
        Image.fromarray(img).save(out); print("saved",out)
    except Exception as e:
        print("RENDER FAILED:",repr(e))
    print("@@@BODY DONE@@@")
