"""CX NAVIGATOR — the full central-complex navigation brain: compass + path integration + homing steering.

Stacks four layers on the verified heading compass (central_complex.parts):

  1. COMPASS (imported)   ring attractor + P-EN shift, bump = heading, integrates vestibular angular velocity.
  2. PG (imported)        speed-gated bump readout: fires at rate ~ speed x bump[i] (only while moving).
  3. CPU4 (here)          NC columnar LADDER accumulators (ladder_integrator primitive). CD[c] is a cosine-
                          tuned speed cell pooling PG with RECTIFIED cos(phi_i - phi_c) weights, so it fires
                          at rate ~ speed*[cos(heading - phi_c)]+. Its spikes fill ladder column c, whose
                          latched-cell COUNT = integral of that = the rectified projection of the outbound
                          displacement onto phi_c. Opposite columns carry the negative lobe, so the signed
                          home vector lives across the 12 columns as a sinusoid -- exactly CPU4.
  4. CPU1 (here)          steering comparator. HL[c] = AND(column c filled, heading bump at phi_c + 90deg),
                          HR[c] = AND(column c filled, bump at phi_c - 90deg). Pooling gives
                             LEFT-RIGHT  ~  2|D| sin(heading - angle(D))
                          which is POSITIVE exactly when the way home is counter-clockwise. So LEFT drives a
                          CCW turn and RIGHT a CW turn, and the agent steers home. (cos(x+90)-cos(x-90) =
                          -2 sin x is the whole trick; the +-90deg offset is built into the wiring.)

  MUSCLES: MUS_L/MUS_R are graded non-spiking cells (r=1e9); their membrane S is read as turn force -- the
  ONE sanctioned actuator transducer (same pattern as neural_agent's graded muscles). Everything upstream is
  spikes and fixed wiring; no python decides the heading, the home vector, or the steering.
"""
import sys, importlib.util, numpy as np
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
from paula_agent import ckit as k
_s=importlib.util.spec_from_file_location("cc","/Users/arterialist/Projects/agi-research/active-inference/simulations/active_inference/central_complex.py")
cc=importlib.util.module_from_spec(_s); _s.loader.exec_module(cc)

def T(n): return 20000+n
def _nsyn(sy,nid): return sum(1 for x in sy if x.get('neuron_id')==nid and x.get('type')=='postsynaptic')
NC=12                                    # CPU4 columns (NR/NC=3 ring cells per column; 90deg = 3 columns)
NL=16                                    # ladder cells per column (dynamic range of the accumulator)
STRIDE=cc.NR//NC                         # ring cells per CPU4 column
QUARTER=NC//4                            # columns per 90 degrees
CD=[9000+c for c in range(NC)]                                   # cosine-tuned speed cells (SIGNED when cd_neg>0)
# CDM: RECTIFIED twin of CD, built from the same PG pool with the same cosine weights but NEVER the
# negative lobe. MEASURED: the two L3 criteria separate on exactly this one property. Signed CD cancels
# on reversal -> the population-vector ANGLE stays clean (direction 17.7-30.0 deg) but nothing
# accumulates (r(dist,total) +0.06..+0.10). Rectified CD only ever charges -> total activity tracks
# distance (r +0.34..+0.83) but the angle blurs (79-118 deg). No readout transform bridges them
# (power sharpening p=2/4/8 and hard WTA all plateau at 55-63 deg). So run BOTH: signed feeds the
# direction readout, rectified feeds the magnitude readout. Parallel channels of different tuning
# reading a common input is what the mushroom body and visual system in this agent already do.
CDM=[9020+c for c in range(NC)]                                  # rectified twin -> magnitude channel
LAD=[[9100+c*NL+i for i in range(NL)] for c in range(NC)]        # CPU4 ladder accumulators
HL=[9500+c for c in range(NC)]; HR=[9600+c for c in range(NC)]   # CPU1 steering comparators
LGI=9690                                                         # ladder global inhibitor (normaliser)
OPP=[9700+c for c in range(NC)]                                  # opponent cells: [fill_c - fill_(c+NC/2)]+
MUS_L=9800; MUS_R=9801; MUS_F=9802                               # graded muscles (read membrane S)
PHI_C=np.array([2*np.pi*c/NC for c in range(NC)])

def parts(w_cd=1.5, r_cd=1.5, w_self=5.0, w_prev=2.4, w_drive=1.0, r_lad=1.6, d_adv=90,
          w_pool=0.30, w_bump=1.0, r_cpu1=1.5, w_mus=1.0, lam_mus=25, seed_amp=6.0,
          w_opp=1.2, r_opp=1.0, w_musf=1.0, w_self_drop=0.8, w_drain=1.6, cd_cut=0.05, cd_pow=1.0, cd_src="PG",
          cd_neg=0.0,   # wire the NEGATIVE cosine lobe as inhibition -> true signed cosine, no pedestal
          w_lgi=0.0, r_lgi=1.2, lam_lgi=4, w_lad_lgi=0.10,   # GLOBAL INHIBITION on the ladder (opt-in)
          w_cpu1=1.0,                                       # CPU4 -> CPU1 source gate (Stone PI replacement)
          wire_muscles=True, ne=None, sy=None, conns=None, ex=None, **cxkw):
    if ne is None:
        ne,sy,conns,ex = cc.parts(**cxkw)      # standalone: build the compass too
    # ---- CD: cosine-tuned speed cells. Pool PG with RECTIFIED cosine weights -> rate ~ speed*[cos(h-phi_c)]+
    for c in range(NC):
        ne.append(k.neuron(CD[c],r=r_cd,c=2,lam=3)); j=0
        # cd_src: 'PG' pools the SPEED-GATED bump readout (current design); 'RING' pools the ring
        # directly, bypassing the speed gate. PG is an AND of (bump x speed) and was inserted between
        # RING and CD by the later OPP/MUS_F speed-gate work -- if it under-fires, CD starves and the
        # ladder never fills, which is exactly what the original self-test now shows (fills all 1,
        # |home| 0.00). This switch isolates that. Default 'PG' keeps current behaviour.
        _src = cc.PG if cd_src=="PG" else cc.RING
        for i in range(cc.NR):
            w=np.cos(cc.PHI[i]-PHI_C[c])
            # cd_cut sets the tuning WIDTH. At the original 0.05 the pool spans ~170deg, so every column
            # (and its ANTIPODE) fires almost equally -- measured 611-785 spikes/col, a 1.3x spread, with
            # 0.69 of 6 antipodal pairs co-firing per tick. That co-firing is what let the drain cancel
            # every latch. cd_pow>1 narrows further while keeping the cosine shape (Stone et al. 2017).
            if w>cd_cut:
                w=w**cd_pow
                sy.append(k.syn(CD[c],j,w_cd*float(w),1)); conns.append(k.conn(_src[i],CD[c],j,cc.Tt(_src[i]))); j+=1
            elif cd_neg and w < -cd_cut:
                # NEGATIVE LOBE AS INHIBITION. Keeping only the positive lobe makes CD a RECTIFIED
                # half-cosine riding on a big DC pedestal: measured in the body, CD's tuning by heading
                # was 0.44 at peak vs 0.33 at trough -- only ~30% modulation -- so the ACC population
                # vector direction was noise (per-column spread 1.14, 1.0 = no tuning). Wiring the
                # anti-preferred half with NEGATIVE weight makes CD a TRUE SIGNED cosine, which is what
                # CPU4 integrates in Stone et al. 2017, and cancels the pedestal instead of narrowing it
                # (narrowing via cd_cut/cd_pow was measured WORSE: error 47.8 -> 82.2).
                wn=(-w)**cd_pow
                sy.append(k.syn(CD[c],j,-cd_neg*w_cd*float(wn),1)); conns.append(k.conn(_src[i],CD[c],j,cc.Tt(_src[i]))); j+=1
        sy.append(k.term(CD[c],T(CD[c])))
    # ---- CDM: identical pooling, positive lobe ONLY (no cd_neg branch) ----
    for c in range(NC):
        ne.append(k.neuron(CDM[c],r=r_cd,c=2,lam=3)); j=0
        for i in range(cc.NR):
            w=np.cos(cc.PHI[i]-PHI_C[c])
            if w>cd_cut:
                w=w**cd_pow
                sy.append(k.syn(CDM[c],j,w_cd*float(w),1)); conns.append(k.conn(_src[i],CDM[c],j,cc.Tt(_src[i]))); j+=1
        sy.append(k.term(CDM[c],T(CDM[c])))
    # ---- CPU4: one LADDER per column. latch + AND(prev, CD spike) through a DELAYED dendrite (rate limit).
    # BIDIRECTIONAL: latch strength is STAGGERED (cell 0 strongest, top weakest) and the OPPOSING column's
    # drive inhibits the whole column -- so a drain spike de-recruits the TOP (weakest-latched) cell first,
    # one at a time. That makes fill_c literally subtract when you travel the other way, so the home vector
    # SHRINKS on the way back instead of ratcheting. No extra neurons: it is just a latch-margin gradient.
    for c in range(NC):
        for i,nid in enumerate(LAD[c]):
            ne.append(k.neuron(nid,r=r_lad,c=2,lam=2)); j=0
            ws=w_self-(i/max(NL-1,1))*w_self_drop                                                   # latch gradient
            sy.append(k.syn(nid,j,ws,1)); conns.append(k.conn(nid,nid,j,T(nid))); j+=1              # latch
            if i>0:
                sy.append(k.syn(nid,j,w_prev,d_adv)); conns.append(k.conn(LAD[c][i-1],nid,j,T(LAD[c][i-1]))); j+=1
            sy.append(k.syn(nid,j,w_drive,1)); conns.append(k.conn(CD[c],nid,j,T(CD[c]))); j+=1     # drive (fill)
            sy.append(k.syn(nid,j,-w_drain,1)); conns.append(k.conn(CD[(c+NC//2)%NC],nid,j,T(CD[(c+NC//2)%NC]))); j+=1  # DRAIN
            if i==0: sy.append(k.syn(nid,j,seed_amp,1)); ex.append(k.ext(nid,j))                    # birth seed
            sy.append(k.term(nid,T(nid)))
    # ---- LADDER GLOBAL INHIBITION (opt-in, w_lgi>0) ----
    # Measured: the antipodal drain is what kills the accumulator in the BODY. Single variable in a
    # free-foraging agent: cell0 spikes 13 (w_drain=1.6) -> 4000 (w_drain=0), a 300x change, while
    # antipodal DRIVE was unchanged (903 vs 1306 spikes). But removing the drain alone SATURATES the
    # ladder (fill 16/16 on all 12 columns, |h|=0.00) -- the drain was the ONLY thing bounding it.
    # The comparative PI literature (ants, bees AND mammals) uses LOCAL EXCITATION + GLOBAL INHIBITION
    # over a circular array, never an antipodal subtraction. So: drop the drain, add a normaliser that
    # pools the whole ladder and inhibits it, bounding total activity without erasing direction.
    if w_lgi>0:
        ne.append(k.neuron(LGI,r=r_lgi,c=1,lam=lam_lgi)); gj=0
        for c in range(NC):
            for nid in LAD[c]:
                sy.append(k.syn(LGI,gj,w_lad_lgi,1)); conns.append(k.conn(nid,LGI,gj,T(nid))); gj+=1
        sy.append(k.term(LGI,T(LGI)))
        for c in range(NC):
            for nid in LAD[c]:
                j2=_nsyn(sy,nid)
                sy.append(k.syn(nid,j2,-abs(w_lgi),1)); conns.append(k.conn(LGI,nid,j2,T(LGI)))
    # ---- CPU1: AND(column filled, heading bump at +-90deg). Pool the whole ladder so input ~ fill level.
    for c in range(NC):
        for (hid,off) in ((HL[c],+QUARTER),(HR[c],-QUARTER)):
            ne.append(k.neuron(hid,r=r_cpu1,c=2,lam=3)); j=0
            for nid in LAD[c]:
                sy.append(k.syn(hid,j,w_pool*w_cpu1,1)); conns.append(k.conn(nid,hid,j,T(nid))); j+=1
            ci=(c+off)%NC                                   # the ring cells 90deg away from this column
            for d in range(STRIDE):
                ri=(ci*STRIDE+d)%cc.NR
                sy.append(k.syn(hid,j,w_bump,1)); conns.append(k.conn(cc.RING[ri],hid,j,cc.Tt(cc.RING[ri]))); j+=1
            sy.append(k.term(hid,T(hid)))
    # ---- OPPONENT magnitude: OPP[c] = [fill_c - fill_opposite]+ (excited by column c, inhibited by c+NC/2).
    # Raw total fill is NOT distance (both opposite columns fill up, so it tracks PATH LENGTH). The signed
    # vector -- and hence |D| -- only exists in the DIFFERENCE across opposite columns. Summing the rectified
    # opponent cells gives sum_c [|D|cos(theta-phi_c)]+ ~ |D|: a true "how far from home" signal.
    for c in range(NC):
        ne.append(k.neuron(OPP[c],r=r_opp,c=2,lam=4)); j=0
        for nid in LAD[c]:
            sy.append(k.syn(OPP[c],j, w_opp,1)); conns.append(k.conn(nid,OPP[c],j,T(nid))); j+=1
        for nid in LAD[(c+NC//2)%NC]:
            sy.append(k.syn(OPP[c],j,-w_opp,1)); conns.append(k.conn(nid,OPP[c],j,T(nid))); j+=1
        sy.append(k.term(OPP[c],T(OPP[c])))
    # ---- MUSCLES: graded, never spike; membrane S read as force (the sanctioned actuator transducer).
    # MUS_L/MUS_R = turn (CPU1). MUS_F = FORWARD drive gated by |D|: the agent slows as the home vector
    # shrinks and STOPS on arrival -- which is what stops the orbit (insects do exactly this).
    if not wire_muscles: return ne,sy,conns,ex     # caller supplies its own mode-GATED muscles
    for mid,src,w in ((MUS_L,HL,w_mus),(MUS_R,HR,w_mus),(MUS_F,OPP,w_musf)):
        ne.append(k.neuron(mid,r=1e9,c=2,lam=lam_mus)); j=0
        for hid in src:
            sy.append(k.syn(mid,j,w,1)); conns.append(k.conn(hid,mid,j,T(hid))); j+=1
        sy.append(k.term(mid,T(mid)))
    return ne,sy,conns,ex

def build(**kw):
    return k.build(*parts(**kw))

class Navigator:
    """Closed loop: neural steering -> muscles -> body physics -> vestibular feedback -> compass."""
    def __init__(self, kyaw=0.36, v=1.0, tref_upper=None, **bkw):
        self.net,self.core=k.load(build(**bkw)); self.nb={i:u for i,u in self.net.network.neurons.items()}
        if tref_upper is not None:
            # the agent applies this to its shift cells; the standalone never did, so LAD sat at
            # t_ref = c*num_inputs = 8 while the agent's ran at 2. Same override, opt-in.
            ids=[n for i2 in range(cc.NR) for p2 in range(cc.NP) for n in (cc.CL[i2][p2],cc.CR[i2][p2])]
            ids+=[x for c in range(NC) for x in LAD[c]]+list(CD)+list(HL)+list(HR)+list(OPP)
            for nid in ids:
                if nid in self.nb:
                    self.nb[nid].upper_t_ref_bound=tref_upper; self.nb[nid].lower_t_ref_bound=1.0
                    self.nb[nid].t_ref=tref_upper
        self.kyaw=kyaw; self.v=v
        self.x=self.y=0.0; self.yaw=0.0; self.path=[]
    # ---------- birth seeds (one-time, sanctioned) ----------
    def birth(self, idx=0, ticks=35):
        ss={(idx+o)%cc.NR for o in range(-cc.NB,cc.NB+1)}
        for _ in range(ticks):
            # per-cell seed index: with d7=True the Delta-7 coverage differs per ring cell, so a
            # SINGLE global index writes the seed into the wrong synapse on most cells.
            for i,r in enumerate(cc.RING): self.net.set_external_input(r,cc.SEEDMAP[r], 4.0 if i in ss else 0.0)
            for c in range(NC): self.net.set_external_input(LAD[c][0],self._ladseed(),6.0)
            self.core.do_tick()
        for r in cc.RING: self.net.set_external_input(r,cc.SEEDMAP[r],0.0)
        for c in range(NC): self.net.set_external_input(LAD[c][0],self._ladseed(),0.0)
    def _ladseed(self): return 3      # cell0 synapses: 0=latch, 1=drive, 2=drain, 3=seed
    # ---------- one closed-loop tick ----------
    def tick(self, speed, omega_cmd):
        """omega_cmd: commanded turn (+ = CCW). Body turns, compass sees it ONLY via vestibular drive."""
        ccw=max(0.0, omega_cmd); cw=max(0.0,-omega_cmd)
        for i in range(cc.NR):
            for p in range(cc.NP):
                self.net.set_external_input(cc.CL[i][p],1,ccw); self.net.set_external_input(cc.CR[i][p],1,cw)
            self.net.set_external_input(cc.PG[i],1,speed)
        self.core.do_tick()
        self.yaw += np.radians(self.kyaw*omega_cmd)                  # body physics
        self.x += self.v*speed*np.cos(self.yaw); self.y += self.v*speed*np.sin(self.yaw)
        self.path.append((self.x,self.y))
    def steer(self):
        """read the graded muscles -> (left, right, forward). The ONE actuator transducer."""
        return float(self.nb[MUS_L].S), float(self.nb[MUS_R].S), float(self.nb[MUS_F].S)
    # ---------- probes (measurement only) ----------
    def fills(self, win=8):
        acc=np.zeros((NC,NL))
        for _ in range(win):
            self.core.do_tick()
            for c in range(NC):
                for i,nid in enumerate(LAD[c]): acc[c,i]+= self.nb[nid].O>0
        return (acc>0).sum(axis=1)
    def home_from_fills(self, f):
        vx=float(np.sum(f*np.cos(PHI_C))); vy=float(np.sum(f*np.sin(PHI_C)))
        return np.degrees(np.arctan2(-vy,-vx)), float(np.hypot(vx,vy))   # direction BACK home
    def bump(self, win=10, speed=0.0, omega_cmd=0.0):
        """WARNING: external input is consumed every tick, so a measurement window MUST keep driving
        (pass the drive that is currently active) or the traveling wave stalls and the bump re-pins."""
        acc=np.zeros(cc.NR)
        for _ in range(win):
            self.tick(speed,omega_cmd)
            for i,r in enumerate(cc.RING): acc[i]+= self.nb[r].O>0
        if acc.sum()==0: return None
        return float(np.arctan2(float(np.sum(acc*np.sin(cc.PHI))),float(np.sum(acc*np.cos(cc.PHI)))))

if __name__=="__main__":
    # FULL NAVIGATION: an outbound L-path (commanded), then homing steered ONLY by the neural comparator.
    np.random.seed(0)
    nav=Navigator(); nav.birth()
    print("CX NAVIGATOR — outbound L-path, then PI-driven homing (steering = muscle membranes only)")
    for _ in range(300): nav.tick(speed=1.0, omega_cmd=0.0)     # leg 1
    for _ in range(280): nav.tick(speed=0.0, omega_cmd=0.9)     # turn ~90 CCW (in place)
    for _ in range(300): nav.tick(speed=1.0, omega_cmd=0.0)     # leg 2
    f=nav.fills(); hd,mag=nav.home_from_fills(f)
    true=np.degrees(np.arctan2(-nav.y,-nav.x)); d0=np.hypot(nav.x,nav.y)
    print(f"  outbound done  : pos=({nav.x:+.0f},{nav.y:+.0f}) dist={d0:.0f}")
    print(f"  CPU4 home vector: {hd:+.0f}deg   (true {true:+.0f}deg)   column fills={list(f)}")
    print("  -- homing: omega = 3.0*(MUS_L - MUS_R), nothing else --")
    best=d0
    for t in range(3000):
        L,R,_F=nav.steer()   # steer() returns (MUS_L, MUS_R, MUS_F); the old test unpacked 2
        nav.tick(speed=0.6, omega_cmd=float(np.clip(3.0*(L-R),-1.2,1.2)))
        d=np.hypot(nav.x,nav.y); best=min(best,d)
        if (t+1)%500==0: print(f"     t{t+1:4d}: dist_from_home={d:6.0f}")
    print(f"  RESULT: {d0:.0f} -> closest {best:.0f}  ({100*(1-best/d0):.0f}% of the way home)")
    print("@@@NAV DONE@@@")
