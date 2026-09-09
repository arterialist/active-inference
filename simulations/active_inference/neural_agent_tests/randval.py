import sys; sys.path.insert(0,'/Users/arterialist/Projects/agi-research/neuron-model'); import importlib.util, numpy as np
def L(n,p): s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
NA=L("na","/Users/arterialist/Projects/agi-research/active-inference/simulations/active_inference/neural_agent.py")
rng=np.random.default_rng(7); reached=0; N=12; res=[]
for i in range(N):
    ang=rng.uniform(0,2*np.pi); d=rng.uniform(3.0,4.5)
    food=(float(d*np.cos(ang)),float(d*np.sin(ang)))
    ag=NA.NeuralAgent(foods=[food], swap_turn=False); mind=d
    for t in range(4000):
        ag.step(); x,y,_=ag.pose(); mind=min(mind,np.hypot(x-food[0],y-food[1]))
    ok=mind<1.0; reached+=ok
    res.append(f"  food=({food[0]:+.1f},{food[1]:+.1f}) ang={np.degrees(ang):3.0f}deg: closest={mind:.2f} {'OK' if ok else 'MISS'}")
    print(res[-1],flush=True)
print(f"REACHED {reached}/{N} random foods (<1.0)",flush=True)
print("@@@DONE@@@",flush=True)
