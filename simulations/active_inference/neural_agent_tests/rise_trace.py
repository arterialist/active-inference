import sys; sys.path.insert(0,'/Users/arterialist/Projects/agi-research/neuron-model'); import importlib.util, numpy as np, mujoco
def L(n,p): s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
NA=L("na","/Users/arterialist/Projects/agi-research/active-inference/simulations/active_inference/neural_agent.py")
food=(-4.0,0.0); ag=NA.NeuralAgent(foods=[food]); 
for t in range(1200):
    x,y,yaw=ag.pose(); fwd=yaw+np.pi
    hd=np.array([np.cos(fwd),np.sin(fwd)]); lat=np.array([-np.sin(fwd),np.cos(fwd)]); head=np.array([x,y])+hd*0.45
    oL=ag.odor(head+lat*0.6); oR=ag.odor(head-lat*0.6); ri=st=po=0
    for _ in range(ag.sub):
        for e in NA.ENG: ag.net.set_external_input(e,0,1.0)
        for nid in NA.FL: ag.net.set_external_input(nid,0,ag.gain*oL)
        for nid in NA.FR: ag.net.set_external_input(nid,0,ag.gain*oR)
        if not ag.kicked and ag.t>2: ag.net.set_external_input(NA.P[0],0,5.0); ag.kicked=True
        ag.core.do_tick(); ag.t+=1; ri+=ag.nb[NA.RISE].O>0; st+=ag.nb[NA.STEER].O>0; po+=ag.nb[NA.POOL].O>0
        ag.data.ctrl[ag.a["plp"]]=NA.GGAIN*max(0,ag.nb[NA.MLp].S); ag.data.ctrl[ag.a["plr"]]=NA.GGAIN*max(0,ag.nb[NA.MLr].S)
        ag.data.ctrl[ag.a["prp"]]=NA.GGAIN*max(0,ag.nb[NA.MRp].S); ag.data.ctrl[ag.a["prr"]]=NA.GGAIN*max(0,ag.nb[NA.MRr].S)
        mujoco.mj_step(ag.model,ag.data)
    if t%150==0: print(f" s{t}: pos=({x:6.2f},{y:6.2f}) d={np.hypot(x+4,y):5.2f} odor={oL+oR:.2f} POOL={po} RISE={ri} STEER={st}",flush=True)
print("@@@DONE@@@",flush=True)
