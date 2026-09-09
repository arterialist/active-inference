import sys; sys.path.insert(0,'/Users/arterialist/Projects/agi-research/neuron-model'); import importlib.util, numpy as np, mujoco
def L(n,p): s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
NA=L("na","/Users/arterialist/Projects/agi-research/active-inference/simulations/active_inference/neural_agent.py")
from paula_agent import ckit as k
model=mujoco.MjModel.from_xml_string(NA.XML); data=mujoco.MjData(model)
net,core=k.load(NA.build(w_ton=0.0)); nb={i:u for i,u in net.network.neurons.items()}
a={n:model.actuator(n).id for n in ("plp","plr","prp","prr")}
sx=model.joint('sx').qposadr[0]; sy=model.joint('sy').qposadr[0]; rz=model.joint('rz').qposadr[0]
kicked=False; t=0; path=[]
for step in range(10000):
    if not kicked and t>2: net.set_external_input(NA.P[0],0,5.0); kicked=True
    core.do_tick(); t+=1
    data.ctrl[a["plp"]]=NA.GGAIN*max(0,nb[NA.MLp].S); data.ctrl[a["plr"]]=NA.GGAIN*max(0,nb[NA.MLr].S)
    data.ctrl[a["prp"]]=NA.GGAIN*max(0,nb[NA.MRp].S); data.ctrl[a["prr"]]=NA.GGAIN*max(0,nb[NA.MRr].S)
    mujoco.mj_step(model,data)
    if step%2000==0: path.append(f"({data.qpos[sx]:.1f},{data.qpos[sy]:.1f})yaw{np.degrees(data.qpos[rz]):.0f}")
print("forward no-steer path:", " ".join(path),flush=True)
print("@@@DONE@@@",flush=True)
