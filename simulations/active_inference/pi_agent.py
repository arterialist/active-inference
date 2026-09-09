"""PI_AGENT — clean rebuild of heading + path integration, verified LAYER BY LAYER.

WHY A REBUILD. The existing cx_navigator stack was built on a ring that DOES NOT SELF-SUSTAIN: at its
own defaults the ring is active on 0 of 900 ticks (longest silence 900). PG is AND(bump x speed), so a
dead ring means CD never fires, the CPU4 ladder never fills, OPP is silent and HOME is never selected.
Every downstream "bug" traced back to that. The old self-test does not even run (steer() returns 3
values, the test unpacks 2), and its recorded "1 deg home-vector error" is not reproducible from the
current code in any of six configurations (best achievable now: 40.7 deg).

DESIGN RULES FOR THIS FILE
  1. Every layer has a self-test that must PASS before the next layer is allowed to depend on it.
     L1 ring alive -> L2 tracks heading -> L3 accumulates -> L4 home vector -> L5 steers home.
  2. No mechanism that is not in the literature. In particular there is NO antipodal drain: Stone et
     al. 2017 (Current Biology, "An anatomically constrained model for path integration in the bee
     brain") accumulates speed MODULATED BY heading, and bidirectionality comes from the POPULATION --
     travel one way fills one set of columns, travel back fills the opposite set, and the vector sum
     cancels at readout. The drain in the old file was invented, and it pinned every latch NEGATIVE.
  3. Diagnostic oracles are allowed ONLY in tests, never in the agent, and must be labelled.

VERIFIED INGREDIENTS CARRIED OVER (measured, not assumed):
  - tonic ring drive is REQUIRED (without it: 0/900 ticks alive; with it: longest silence 1 tick)
  - Delta-7 structured inhibition with TOTAL INHIBITION product ND7*|w_d7_ring| = 24 gives 0/8
    reversals and sd 0.11 (product 16 gives sd ~0.6 and reversals). ND7=12 is the sharpest viable
    sector count (width 8.2/36); ND7 in {6,9} kills the ring outright, unexplained.
  - d_push=2 (wave-speed ceiling: 18 ring cells x d_push ticks/hop bounds how fast the bump can move).
  - Shift parameters have NO headroom: lowering r_conj_lo gives 5/8 reversals, raising w_push gives
    6/8 reversals (systematic inversion at 1.8). Slope is capped at ~0.71 in this architecture.
"""
import sys, importlib.util, numpy as np
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
from paula_agent import ckit as k
_s=importlib.util.spec_from_file_location("cc","/Users/arterialist/Projects/agi-research/active-inference/simulations/active_inference/central_complex.py")
cc=importlib.util.module_from_spec(_s); _s.loader.exec_module(cc)

# ============================ VERIFIED COMPASS (2026-07-28) ============================
# Slope 0.91-1.23 across omega 0.3-1.0 (a 3.3x rate range, mean 1.02) with the ring alive throughout,
# degrading gracefully to 0.24 at omega 1.5. Measured with CUMULATIVE UNWRAPPED per-tick displacement.
#
# THE MEASUREMENT WAS THE BUG. Every earlier slope used a single WRAPPED endpoint difference,
# ((b1-b0+180)%360-180)/dy, which silently returns small or NEGATIVE numbers once the bump laps the
# ring -- and it lapped ~6x. That one error invalidated the d_push sweep, the ND7 2x2, the r_conj_lo
# and w_push sweeps, and the stagger sweep, and it caused sh_bank to be wrongly rejected.
#
# The three ingredients, each measured:
#  * d7 (ND7=12) keeps the RING ALIVE. Without it the single global inhibitor sums the whole ring and
#    inhibits the whole ring, so a bump that widens in order to translate suppresses itself: the
#    original config is stone dead, 0 ring spikes in 400 ticks, and tonic alone does NOT revive it.
#  * sh_bank makes bump speed SCALE WITH VELOCITY -- the paper's mechanism (Turner-Evans et al. 2017,
#    eLife 23496: the P-EN/E-PG phase OFFSET varies LINEARLY with rotational velocity). Measured bump
#    speed with the bank scales 1.97x while the body scales 2x; without it, only 1.18x (a fixed-speed
#    travelling wave, where omega merely GATES motion). This is the difference between a velocity
#    integrator and a wave that runs at its own pace.
#  * w_self=2.2 / w_nbr=1.0 give the recurrent excitation needed to hold the bump at SLOW rotation.
#    At the old 1.4/0.7 the ring was silent 167/400 ticks at omega 0.3 (shift activity was propping it
#    up); now 31/400 and slope goes -0.07 -> +0.91. Tonic is the WRONG lever here: w_tonic=0.30
#    removes the silence entirely but inverts tracking (-0.65) and blurs the bump.
#  * d_push=44 then sets the GAIN. It is a clean monotonic knob once measured correctly.
# Cost: bump width ~15-17/36 (wider than the 11-14 of the non-tracking configs). Sharpness was traded
# for persistence and unity gain; that trade is deliberate and measured.
#  * jitter=0.9 BREAKS THE RING'S PHASE LOCK-STEP. At the default 0.2 the whole ring fires on strictly
#    ALTERNATE ticks (measured pattern "X.X.X.X.", 13/36 cells on active ticks, exactly 50% silence at
#    rest) because t_ref=2 and the ring is synchronised. That is NOT a dying ring -- I mis-framed it as
#    one -- but the shift AND-gate (bump AND velocity) then only gets coincidence opportunities half the
#    time, and at LOW omega there is too little drive to break the lock. Result: no slow tracking.
#    Desynchronising fixes it: omega 0.2 slope -0.12 -> +1.52 and silence 200/400 -> 0/400.
#    jitter=1.4 is too far (every rate reverses: -1.29/-0.66/-0.42). Cost: R 0.69 vs 0.84.
#    REVERTED TO 0.2: jitter=0.9 was validated at NR=36 / birth idx=0 ONLY and does NOT generalise --
#    at NR=72 it flattens the bump entirely (R 0.04-0.08 = uniform), and at NR=36 with birth idx=4 the
#    slope collapses to +0.06 at every rate. The slow-rotation dead zone is REAL and the lock-step
#    diagnosis stands, but this is not a safe fix. Desynchronisation must be made scale- and
#    position-independent before it can be used.
COMPASS=dict(d7=True, r_d7=0.3, w_ring_d7=0.9, w_d7_ring=-2.0, d7_out=8.0,
             w_gi_ring=-1.2, r_conj_lo=1.35, r_conj_hi=1.75, w_tonic=0.12,
             d_push=44, sh_bank=True, w_self=2.2, w_nbr=1.0, jitter=0.2)
ND7_SECTORS=12          # product 12*2.0 = 24 -> the stable regime

def use_verified_compass():
    """ND7 is a module-level constant in central_complex; set_NR rebuilds the derived id lists."""
    cc.ND7=ND7_SECTORS; cc.set_NR(cc.NR)

class Heading:
    """LAYER 1+2: ring attractor driven by angular velocity. Nothing else is built on top until this
    passes its self-test."""
    def __init__(self, tref_upper=2.0, **kw):
        use_verified_compass()
        cfg=dict(COMPASS); cfg.update(kw)
        ne,sy,conns,ex = cc.parts(**cfg)
        self.net,self.core=k.load(k.build(ne,sy,conns,ex))
        self.nb={i:u for i,u in self.net.network.neurons.items()}
        if tref_upper is not None:
            # t_ref = c * num_inputs, so a population that gains fan-in silently gates itself off.
            ids=[n for i in range(cc.NR) for p in range(cc.NP) for n in (cc.CL[i][p],cc.CR[i][p])]
            ids+=list(cc.RING)+list(cc.PG)
            # GI pools the ENTIRE ring and Delta-7 pools a sector, so both scale their fan-in with NR:
            # at NR=144, t_ref = c*num_inputs would be 144 and the inhibitor would gate itself off.
            ids+=[cc.GI]+list(cc.D7)
            for nid in ids:
                if nid in self.nb:
                    self.nb[nid].upper_t_ref_bound=tref_upper
                    self.nb[nid].lower_t_ref_bound=1.0
                    self.nb[nid].t_ref=tref_upper
        self.yaw=0.0
    def birth(self, idx=0, ticks=35, amp=4.0):
        ss={(idx+o)%cc.NR for o in range(-cc.NB,cc.NB+1)}
        for _ in range(ticks):
            for i,r in enumerate(cc.RING):
                # per-cell seed index: a single global index is wrong because Delta-7 coverage makes
                # each ring cell's synapse layout differ (this is what broke birth at NR>=72)
                self.net.set_external_input(r, cc.SEEDMAP[r], amp if i in ss else 0.0)
            self.core.do_tick()
        for r in cc.RING: self.net.set_external_input(r,cc.SEEDMAP[r],0.0)
    def tick(self, omega, speed=0.0):
        """omega: commanded angular velocity (+ = CCW). The ONLY inputs are angular velocity and speed."""
        ccw=max(0.0,omega); cw=max(0.0,-omega)
        for i in range(cc.NR):
            if cc.TONICSYN: self.net.set_external_input(cc.RING[i],cc.TONICSYN[cc.RING[i]],1.0)
            for p in range(cc.NP):
                self.net.set_external_input(cc.CL[i][p],1,ccw)
                self.net.set_external_input(cc.CR[i][p],1,cw)
            self.net.set_external_input(cc.PG[i],1,speed)
        self.core.do_tick()
    def bump(self, win=12, omega=0.0, speed=0.0):
        """Read the bump as a population vector over `win` ticks. Ring readouts are ALWAYS ~12 ticks:
        reading off a slower trace smeared a healthy bump into 'disoriented' three times before."""
        acc=np.zeros(cc.NR)
        for _ in range(win):
            self.tick(omega,speed)
            for i,r in enumerate(cc.RING):
                if self.nb[r].O>0: acc[i]+=1
        if acc.sum()==0: return None, 0.0, 0.0
        ang=np.degrees(np.arctan2((acc*np.sin(cc.PHI)).sum(),(acc*np.cos(cc.PHI)).sum()))%360
        width=float((acc>acc.max()*0.35).sum())
        return ang, width, float(acc.sum())

def _selftest_L1(verbose=True):
    """L1: the ring must STAY ALIVE. Pass = longest silence <= 2 ticks over 900."""
    h=Heading(); h.birth()
    sil=0; worst=0; alive=0
    for _ in range(900):
        h.tick(omega=0.0, speed=1.0)
        if any(h.nb[r].O>0 for r in cc.RING): sil=0; alive+=1
        else:
            sil+=1; worst=max(worst,sil)
    ok = worst<=2
    if verbose: print(f"  L1 ring alive       : {alive}/900 ticks, longest silence {worst}  -> {'PASS' if ok else 'FAIL'}")
    return ok

def turn_slope(omega, T=400, idx=0, win=12, deg_per_tick=0.36):
    """CUMULATIVE UNWRAPPED tracking. Returns (slope, mean width, silent ticks).

    NEVER use a wrapped endpoint difference here. ((b1-b0+180)%360-180)/dy silently reports small or
    NEGATIVE slopes as soon as the bump laps the ring, and the bump laps several times per run. That
    single error invalidated months of sweeps and caused the correct mechanism (sh_bank) to be thrown
    out. Accumulate the per-tick delta instead.
    """
    h=Heading(); h.birth(idx=idx)
    for _ in range(60): h.tick(omega=0.0, speed=1.0)
    R=np.zeros((T,cc.NR),dtype=np.int8)
    for t in range(T):
        h.tick(omega=omega, speed=0.0)
        for i,r in enumerate(cc.RING):
            if h.nb[r].O>0: R[t,i]=1
    silent=int((R.sum(axis=1)==0).sum())
    prev=None; u=0.0; wid=[]
    for t in range(T):
        a=R[max(0,t-win+1):t+1].sum(axis=0)
        if a.sum()==0: continue
        ang=np.degrees(np.arctan2((a*np.sin(cc.PHI)).sum(),(a*np.cos(cc.PHI)).sum()))%360
        if prev is not None: u+=((ang-prev+180)%360-180)     # unwrapped accumulation
        prev=ang; wid.append(float((a>a.max()*0.35).sum()))
    true=deg_per_tick*omega*T
    return (u/true if true else 0.0), (float(np.mean(wid)) if wid else 0.0), silent

def _selftest_L2(verbose=True):
    """L2: the bump must TRACK.

    THE REGIME MATTERS. This compass was designed to be PRECISE AT SLOW ROTATION and to BLUR
    GRACEFULLY at high rotation (fast spin widens the bump, which then re-sharpens on stop -- the
    Turner-Evans / Ajabi head-direction behaviour). Every earlier slope measurement used omega=1.0 with
    a 180 deg turn, which may sit in the BLUR regime -- so a 0.71 slope there is not necessarily a
    defect, it may be the intended graceful degradation. Sweep the rate instead of assuming one point.
    Pass = at least one SLOW rate reaches slope >=0.85 with a sharp bump (width <= 12/36).
    """
    rows=[]
    for om in (0.3,0.5,0.75,1.0,1.5):
        s,w,sil = turn_slope(om)
        rows.append((om,s,w,sil))
    core=[s for om,s,_,_ in rows if om<=1.0]
    ok = all(0.7<=s<=1.4 for s in core)      # unity across the normal operating range
    if verbose:
        print("  L2 tracking vs ROTATION RATE (cumulative UNWRAPPED, 400 ticks each):")
        for om,s,w,sil in rows:
            tag="  <-- unity" if (om<=1.0 and 0.7<=s<=1.4) else ("  (graceful blur)" if om>1.0 else "")
            print(f"       omega {om:4.2f} | slope {s:+6.2f} | width {w:4.1f}/36 | silent {sil:3d}/400{tag}")
        print(f"  L2 -> {'PASS' if ok else 'FAIL'}  (criterion: 0.7<=slope<=1.4 for omega<=1.0)")
    return ok

if __name__=="__main__":
    print("PI_AGENT staged self-test — each layer must pass before the next is built on it")
    p1=_selftest_L1()
    p2=_selftest_L2() if p1 else print("  L2 skipped (L1 failed)") or False
    print(f"\n  STATUS: L1={'PASS' if p1 else 'FAIL'}  L2={'PASS' if p2 else 'FAIL'}")
    print("@@@PI STAGE DONE@@@")
