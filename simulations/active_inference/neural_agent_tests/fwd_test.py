import sys; sys.path.insert(0,'/Users/arterialist/Projects/agi-research/neuron-model'); import importlib.util, numpy as np, mujoco
def L(n,p): s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
NA=L("na","/Users/arterialist/Projects/agi-research/active-inference/simulations/active_inference/neural_agent.py")
from paula_agent import ckit as k
model=mujoco.MjModel.from_xml_string(NA.XML); data=mujoco.MjData(model)
net,core=k.load(NA.build(w_ton=0.0)); nb={i:u for i,u in net.network.neurons.items()}
a={n:model.actuator(n).id for n in ("plp","plr","prp","prr")}
sx=model.joint('sx').qposadr[0]; sy=model.joint('sy').qposadr[0]
kicked=False; t=0
for step in range(6000):       # 6000 physics steps, physics per tick
    if not kicked and t>2: net.set_external_input(NA.P[0],0,5.0); kicked=True
    core.do_tick(); t+=1
    data.ctrl[a["plp"]]=NA.GGAIN*max(0,nb[NA.MLp].S); data.ctrl[a["plr"]]=NA.GGAIN*max(0,nb[NA.MLr].S)
    data.ctrl[a["prp"]]=NA.GGAIN*max(0,nb[NA.MRp].S); data.ctrl[a["prr"]]=NA.GGAIN*max(0,nb[NA.MRr].S)
    mujoco.mj_step(model,data)
print(f"relay-motor forward (physics/tick), 6000 steps: dist={np.hypot(data.qpos[sx],data.qpos[sy]):.2f} pos=({data.qpos[sx]:.2f},{data.qpos[sy]:.2f})",flush=True)
print("@@@DONE@@@",flush=True)
