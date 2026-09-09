import sys; sys.path.insert(0,'/Users/arterialist/Projects/agi-research/neuron-model'); import importlib.util, numpy as np
def L(n,p): s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
NA=L("na","/Users/arterialist/Projects/agi-research/active-inference/simulations/active_inference/neural_agent.py")
foods=[(-4.0,0.0),(1.1,-3.5),(3.0,2.0)]  # front, blind-spot, behind
for wton,wri in [(1.5,-12.0),(1.0,-15.0),(2.0,-20.0)]:
    line=f"w_ton={wton} w_riseinh={wri}: "
    for food in foods:
        ag=NA.NeuralAgent(foods=[food], w_ton=wton, w_riseinh=wri); mind=np.hypot(*food)
        for t in range(3200):
            ag.step(); x,y,_=ag.pose(); mind=min(mind,np.hypot(x-food[0],y-food[1]))
        line+=f"({food[0]:+.0f},{food[1]:+.0f})={mind:.2f}{'OK' if mind<1 else 'X'} "
    print(line,flush=True)
print("@@@DONE@@@",flush=True)
