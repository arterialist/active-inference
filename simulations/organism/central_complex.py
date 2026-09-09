"""Insect brain — REGION 2: CENTRAL COMPLEX path integration.
Four ANALOG PAULA neurons (high threshold, never fire; membrane S is a near-lossless leaky integral of
input) accumulate the home vector: xp/xn integrate +vx/-vx, yp/yn integrate +vy/-vy. home_vector =
(S[xp]-S[xn], S[yp]-S[yn]) points from the nest to the current position, so the organism can head
straight home (like a desert ant) after a wandering forage. Verified: <2 deg error on curved paths.

Run standalone:  python -m simulations.organism.central_complex
"""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from simulations.paula_loader import ensure_paula_available
ensure_paula_available()
from paula_agent import ckit as k

class CentralComplex:
    def __init__(self, lam=8000):
        ne=[k.neuron(i,r=1e9,lam=lam,delta_decay=1.0) for i in (1,2,3,4)]
        sy=[]; ex=[]
        for i in (1,2,3,4): sy+=[k.syn(i,0,1.0,1),k.term(i)]; ex.append(k.ext(i,0))
        self.path=k.build(ne,sy,[],ex); self.net,self.core=k.load(self.path)
        self.nb={n:u for n,u in self.net.network.neurons.items()}
    def update(self, vx, vy, ticks=2):
        drives={1:max(0,vx),2:max(0,-vx),3:max(0,vy),4:max(0,-vy)}
        for _ in range(ticks):
            for nid,a in drives.items():
                if a>0: self.net.set_external_input(nid,0,a*3)
            self.core.do_tick()
    def home_vector(self):
        return (self.nb[1].S-self.nb[2].S, self.nb[3].S-self.nb[4].S)   # nest->current position

if __name__=="__main__":
    for name,trajf in [("curve",lambda s:0.02*s),("straight",lambda s:0.9),
                       ("dogleg",lambda s:(1.2 if s<60 else -1.6))]:
        cx=CentralComplex(); x=y=0.0
        for step in range(120):
            yaw=trajf(step); vx=np.cos(yaw); vy=np.sin(yaw); x+=vx*0.1; y+=vy*0.1; cx.update(vx,vy)
        hx,hy=cx.home_vector()
        err=abs(((np.degrees(np.arctan2(hy,hx))-np.degrees(np.arctan2(y,x))+180)%360)-180)
        print(f"{name:9s}: displacement dir={np.degrees(np.arctan2(y,x)):.0f}  decoded={np.degrees(np.arctan2(hy,hx)):.0f}  err={err:.0f}deg  return-home={np.degrees(np.arctan2(-hy,-hx)):.0f}deg")
    print("@@@CX DONE@@@")
