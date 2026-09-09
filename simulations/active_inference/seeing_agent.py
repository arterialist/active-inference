"""SEEING AGENT — vision ACTING on the world: the visual cortex anchors the heading compass.

This closes the loop that the whole navigation stack was waiting on. The central complex integrates
vestibular rotation into a heading bump, but its angular gain is not perfectly constant, so heading drifts
(measured: a scripted 180deg turn left the bump ~43deg off the body). Real insects fix this the same way
this file does -- by re-anchoring the compass to a distal visual cue (the fly's visual ring neurons pulling
the E-PG bump onto the panorama).

  visual cortex SAL[a]  ->  VR[j]  ->  heading ring RING[j]

The mapping is FIXED WIRING, exactly like the PI's cos/sin weights: a feature seen at egocentric azimuth
az_of(a) while the world's reference direction is SUN_AZIMUTH implies the agent's heading is
(SUN_AZIMUTH - az_of(a)), so SAL column a is wired to the ring column for that heading. Nothing is computed
in python; seeing a landmark simply injects current into the heading columns it implies.

Run standalone:  python seeing_agent.py    (does vision pull a drifted compass back onto true heading?)
"""
import sys, importlib.util, numpy as np
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
from paula_agent import ckit as k
def _load(n,p):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
B="/Users/arterialist/Projects/agi-research/active-inference/simulations/active_inference/"
cc=_load("cc",B+"central_complex.py"); vc=_load("vc",B+"visual_cortex.py"); w3=_load("w3",B+"world3d.py")

VR=[80000+j for j in range(cc.NR)]        # visual ring neurons (the anchoring pathway)

def _next_syn(sy, nid):
    return sum(1 for s in sy if s.get("neuron_id")==nid and s.get("type")=="postsynaptic")

def build(w_vr=4.0, r_vr=0.6, lam_vr=6, w_anchor=2.0, w_veto=-2.0, veto_far=4, **cxkw):
    ne,sy,conns,ex = cc.parts(**cxkw)
    vc.parts(ne=ne,sy=sy,conns=conns,ex=ex)
    # VR[j]: driven by every salience column whose implied heading falls in ring column j (fixed map).
    # NOTE: only build VR cells that actually HAVE a source -- a neuron with zero inputs gets
    # upper_t_ref_bound = c*num_inputs = 0 and corrupts the whole network (it silently killed the ring).
    src_of={j:[] for j in range(cc.NR)}
    for a in range(vc.NV1AZ):
        implied=w3.SUN_AZIMUTH-vc.az_of(a,vc.NV1AZ)                     # absolute heading implied by a sighting
        j=int(round((implied%(2*np.pi))/(2*np.pi)*cc.NR))%cc.NR
        src_of[j].append(a)
    for j in range(cc.NR):
        if not src_of[j]: continue
        ne.append(k.neuron(VR[j],r=r_vr,c=2,lam=lam_vr)); idx=0
        for a in src_of[j]:
            sy.append(k.syn(VR[j],idx,w_vr,1)); conns.append(k.conn(vc.SAL(a),VR[j],idx,vc.T(vc.SAL(a)))); idx+=1
        sy.append(k.term(VR[j],20000+VR[j]))
    # VR -> heading ring. The fly's visual ring neurons are GABAergic: a sighting does not merely nudge the
    # implied heading, it INHIBITS every heading the scene contradicts. That is what lets a landmark move an
    # already-established bump -- pure excitation at one column cannot outcompete a settled attractor plus
    # its global inhibitor (measured: no movement even at 8x anchor gain), whereas suppressing the columns
    # the evidence rules out lets the ring's own recurrence rebuild the bump where the evidence allows.
    for j in range(cc.NR):
        if not src_of[j]: continue
        for i in range(cc.NR):
            d=abs(i-j); d=min(d,cc.NR-d)
            w=w_anchor if d<=1 else (w_veto if d>=veto_far else 0.0)
            if w==0.0: continue
            nid=cc.RING[i]; sidx=_next_syn(sy,nid)
            sy.append(k.syn(nid,sidx,w,1)); conns.append(k.conn(VR[j],nid,sidx,20000+VR[j]))
    return k.build(ne,sy,conns,ex)

class SeeingAgent:
    def __init__(self, tref_upper=2.0, **bkw):
        self.net,self.core=k.load(build(**bkw)); self.nb={i:u for i,u in self.net.network.neurons.items()}
        for nid in vc.VIS_IDS():
            if nid in self.nb:
                self.nb[nid].upper_t_ref_bound=tref_upper; self.nb[nid].lower_t_ref_bound=1.0
                self.nb[nid].t_ref=tref_upper
        self.world=w3.World3D(seed=1)
    def seed_bump(self, idx, ticks=35):
        ss={(idx+o)%cc.NR for o in range(-cc.NB,cc.NB+1)}
        for _ in range(ticks):
            for i,r in enumerate(cc.RING): self.net.set_external_input(r,cc.SEEDSYN, 4.0 if i in ss else 0.0)
            self.core.do_tick()
        for r in cc.RING: self.net.set_external_input(r,cc.SEEDSYN,0.0)
    def tick(self, ccw=0.0, cw=0.0, speed=0.0, see=True, img=None):
        if see and img is not None: vc.drive_from_image(self.net,img)
        for i in range(cc.NR):
            for p in range(cc.NP):
                self.net.set_external_input(cc.CL[i][p],1,ccw); self.net.set_external_input(cc.CR[i][p],1,cw)
            self.net.set_external_input(cc.PG[i],1,speed)
        self.core.do_tick()
    def heading(self, win=14, see=True, img=None, **dr):
        acc=np.zeros(cc.NR)
        for _ in range(win):
            self.tick(see=see,img=img,**dr)
            for i,r in enumerate(cc.RING): acc[i]+= self.nb[r].O>0
        if acc.sum()==0: return None
        return float(np.arctan2(float(np.sum(acc*np.sin(cc.PHI))),float(np.sum(acc*np.cos(cc.PHI)))))

def _dw(a,b):
    d=np.degrees(a)-np.degrees(b)
    while d>180:d-=360
    while d<-180:d+=360
    return d

if __name__=="__main__":
    np.random.seed(0)
    print("SEEING AGENT — can vision pull a DRIFTED heading bump back onto truth?")
    ag=SeeingAgent(); w=ag.world
    true_yaw=np.radians(35.0)                    # face the beacon
    w.data.qpos[w.jyaw]=true_yaw
    import mujoco; mujoco.mj_forward(w.model,w.data)
    img=w.retina()
    ag.seed_bump(cc.NR//2)                       # seed the bump DELIBERATELY WRONG (180deg off)
    h0=ag.heading(see=False)
    print(f"  seeded wrong    : bump {np.degrees(h0):+7.1f}deg | true heading {np.degrees(true_yaw):+7.1f}deg | error {_dw(h0,true_yaw):+7.1f}")
    for blk in range(6):
        h=ag.heading(win=25, see=True, img=img)
        print(f"  +vision block {blk+1} : bump {np.degrees(h):+7.1f}deg | error {_dw(h,true_yaw):+7.1f}deg")
    print("@@@SEEING DONE@@@")
