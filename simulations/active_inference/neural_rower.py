"""The pivot-capable rower driven ENTIRELY by a PAULA CPG (neural motor control).

A pacemaker ring (PAULA synfire oscillator) provides the stroke rhythm. Its four phase-neurons are read as
motor commands: each paddle's muscle activation is a weighted sum of the filtered phase-neuron spikes,
shaped so the stroke is asymmetric (fast power stroke, slow feathered recovery) -> net forward thrust in
the viscous medium. Descending drive scales the LEFT vs RIGHT stroke amplitude -> steering/pivoting.
No readout maps a command to a paddle angle; the paddle follows neural muscle activation.
"""
import sys, numpy as np, mujoco
sys.path.insert(0, "/Users/arterialist/Projects/agi-research/neuron-model")
import importlib.util
def _l(n,p): s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
RW=_l("rw", str(__import__("pathlib").Path(__file__).parent / "rower.py"))
from paula_agent import ckit as k

def build_pacemaker(period_c=6):
    """4-neuron synfire ring -> sequential phases P0..P3 (ids 1..4). c sets the period."""
    ne=[]; sy=[]; conns=[]; ex=[]
    for i in range(4):
        nid=1+i; ne.append(k.neuron(nid, r=0.6, lam=3, c=period_c)); sy += [k.syn(nid,0,4.0,period_c), k.term(nid,tid=900+nid)]
    for i in range(4): conns.append(k.conn((i-1)%4+1, i+1, 0, stid=900+(i-1)%4+1))
    ex.append(k.ext(1,0))
    return k.build(ne,sy,conns,ex)

# muscle weighting over the 4 phases: brief strong power (phase 0) then spread recovery -> asymmetric stroke
W_STROKE = np.array([ 1.0, -0.35, -0.35, -0.30])

class NeuralRower:
    def __init__(self, period_c=6):
        self.body=RW.Rower()
        self.path=build_pacemaker(period_c); self.net,self.core=k.load(self.path)
        self.nb={i:u for i,u in self.net.network.neurons.items()}
        self.kicked=False; self.t=0; self.since_p0=0; self.cyc=24.0   # neural cycle length (estimated)
    def pose(self): return self.body.pose()
    def step(self, driveL=1.0, driveR=1.0, cpg_substeps=6):
        # advance the PAULA pacemaker; recover a continuous phase from when phase-neuron P0 fires
        for _ in range(cpg_substeps):
            if not self.kicked and self.t>2: self.net.set_external_input(1,0,5.0); self.kicked=True
            self.core.do_tick(); self.t+=1; self.since_p0+=1
            if self.nb[1].O>0 and self.since_p0>4:
                self.cyc=0.7*self.cyc+0.3*self.since_p0; self.since_p0=0   # measured neural period
        phase=min(0.999, self.since_p0/max(6.0,self.cyc))
        m=RW.stroke(phase)                                          # motor pattern from the neural phase
        self.body.set_paddles(1.4*m*driveL, 1.4*m*driveR)

if __name__=="__main__":
    T=20000
    def run(dL,dR,tag):
        r=NeuralRower(); x0,y0,yaw0=r.pose()
        for _ in range(T): r.step(dL,dR)
        x,y,yaw=r.pose(); print(f"  {tag:22s}: pos=({x:6.2f},{y:6.2f}) dist={np.hypot(x-x0,y-y0):5.2f} heading={np.degrees(yaw-yaw0):+.0f}deg")
    print(f"PAULA-driven rower ({T} steps each):")
    run(1.0,1.0,"both (forward)")
    run(1.0,0.5,"L>R (turn)")
    run(0.5,1.0,"R>L (turn other way)")
    run(1.0,-1.0,"antiphase (pivot)")
    print("@@@NEURAL-ROWER DONE@@@")
