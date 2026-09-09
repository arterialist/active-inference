import sys; sys.path.insert(0,'/Users/arterialist/Projects/agi-research/neuron-model'); import importlib.util, numpy as np
def L(n,p): s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
NA=L("na","/Users/arterialist/Projects/agi-research/active-inference/simulations/active_inference/neural_agent.py")
foods=[(1.1,-3.5),(0.5,-3.3),(-4.0,0.0),(3.0,2.0),(0.0,4.0),(4.0,-1.0)]
r=0
for food in foods:
    ag=NA.NeuralAgent(foods=[food], swap_turn=False); mind=np.hypot(*food)
    for t in range(3500):
        ag.step(); x,y,_=ag.pose(); mind=min(mind,np.hypot(x-food[0],y-food[1]))
    ok=mind<1.0; r+=ok
    print(f"  food=({food[0]:+.1f},{food[1]:+.1f}): closest={mind:.2f} {'OK' if ok else 'X'}",flush=True)
print(f"reached {r}/{len(foods)}",flush=True); print("@@@DONE@@@",flush=True)
