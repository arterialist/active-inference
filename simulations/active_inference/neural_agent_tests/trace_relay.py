import sys; sys.path.insert(0,'/Users/arterialist/Projects/agi-research/neuron-model'); import importlib.util, numpy as np
def L(n,p): s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
NA=L("na","/Users/arterialist/Projects/agi-research/active-inference/simulations/active_inference/neural_agent.py")
ag=NA.NeuralAgent(foods=[(3.0,2.0)])
mind=3.61
for t in range(1600):
    o=ag.step(); x,y,_=ag.pose(); d=np.hypot(x-3,y-2); mind=min(mind,d)
    if t%150==0:
        print(f"  step{t}: pos=({x:5.2f},{y:5.2f}) d={d:4.2f} odor={o:.2f} | TL={ag.nb[NA.TL].O:.0f} TR={ag.nb[NA.TR].O:.0f} ST={ag.nb[NA.STEER].O:.0f}",flush=True)
print(f"closest={mind:.2f}",flush=True); print("@@@DONE@@@",flush=True)
