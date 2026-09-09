import sys; sys.path.insert(0,'/Users/arterialist/Projects/agi-research/neuron-model'); import importlib.util, numpy as np
def L(n,p): s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
NA=L("na","/Users/arterialist/Projects/agi-research/active-inference/simulations/active_inference/neural_agent.py")
foods=[(-4.0,0.0),(-4.0,1.5),(0.0,4.0),(3.0,2.0),(4.0,-1.0)]  # front, front-L, side, BEHIND, behind-R
for food in foods:
    ag=NA.NeuralAgent(foods=[food], swap_turn=False); d0=np.hypot(*food); mind=d0
    for t in range(2200):
        ag.step(); x,y,_=ag.pose(); mind=min(mind,np.hypot(x-food[0],y-food[1]))
    print(f"  food={food} start={d0:.1f}: closest={mind:.2f} {'REACHED' if mind<1.0 else ''}",flush=True)
print("@@@DONE@@@",flush=True)
