"""PI_ACCUM — redesigned path-integration accumulator (CPU4 replacement).

WHY REDESIGN. Per-tick tracing showed the existing CPU4 ladder fails in the body for a specific,
measured reason: on a straight path the antipodal column is SILENT (0 spikes) so the -w_drain=-1.6
never fires and cell0 stays latched (800 spikes). Under real tropotaxis the heading oscillates
continuously, the antipode fires 268 times, each spike delivers -1.6 (alone enough to cancel
r_lad=1.6), and cell0 is de-latched (13 spikes, 60x fewer) -- even though the column receives MORE
drive (429 vs 399 CD spikes). The accumulator is erased by TURNING, not by lack of drive.

GROUNDING (comparative, not fly-specific -- ants, bees AND mammals):
compass+odometry are sustained by LEAKY NEURAL INTEGRATORS, and the home vector is computed by
LOCAL EXCITATION + GLOBAL INHIBITION over a CIRCULAR ARRAY with population-coded direction
(ScienceDirect path-integration overview; "Can the Insect PI Memory be a Bump Attractor?" bioRxiv 2022;
Stone et al. 2017 Current Biology). NO organism uses an antipodal drain -- bidirectionality comes from
the POPULATION CODE: travel one way charges one side, travel back charges the other, and the vector
sum cancels at readout.

The old design diverges on all four counts. This one implements the literature:
  * ACC[c]  : ONE leaky integrator per compass column (not a 16-cell thermometer ladder)
              - self-excitation w_acc_self BELOW runaway -> graceful leak, not a hard latch, so losing
                drive degrades the estimate instead of destroying it
              - CD[c] -> ACC[c] excitatory: charge proportional to speed x cos(heading - phi_c)
  * LOCAL EXCITATION : ACC[c] <-> ACC[c +- 1], weak. Smooths the population code and lets a column
              that momentarily loses drive be held up by its neighbours (the thing the old ladder
              could not do -- each column latched alone).
  * GLOBAL INHIBITION: AGI pools all ACC and inhibits all ACC -> normalisation, bounded total
              activity, and competition that keeps the population a lobe rather than saturating.
  * NO DRAIN. Nothing subtracts from the antipode.
Level is a RATE code across the population (spikes/window), read as a vector sum -- not a thermometer
count, so a single lost spike is not a lost level.
"""
import sys, importlib.util, numpy as np
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
from paula_agent import ckit as k
_s=importlib.util.spec_from_file_location("cc","/Users/arterialist/Projects/agi-research/active-inference/simulations/active_inference/central_complex.py")
cc=importlib.util.module_from_spec(_s); _s.loader.exec_module(cc)

def T(n): return 20000+n
NC=12
NA=6                                   # cells PER COLUMN, thresholds STAGGERED -> recruitment code
# WHY A POPULATION PER COLUMN. With ONE cell per column the magnitude of the home vector is a single
# cell's FIRING RATE, and the refractory `c` caps that rate (c=2 -> at most 0.5 spikes/tick).
# CORRECTION: this rationale originally blamed `t_ref = c*num_inputs`. That was wrong -- t_ref is the
# LTP/LTD boundary (the plasticity window), not the firing refractory, and it does not limit rate at
# all. The measured plateau below is unaffected; only the attribution was wrong. Measured in the body, |h| rose over the
# first ~4 distance units and then CLIPPED to a flat plateau (seed 23: 8.7 8.3 25.6 30.7 30.5 30.7
# 30.7 30.5; seed 77 plateaus at 20.5; seed 11 at 1.9) -- a hard rate ceiling, not a mistuned weight
# (w_acc_self 5.0 vs 6.5 moves the plateau 25.5 -> 10.2 but never removes it). Staggering thresholds
# across a population extends dynamic range by RECRUITMENT: as accumulated drive grows, progressively
# higher-threshold cells join in. This is the same device the HS cells already use here ("thresholds
# are staggered across the population so the pooled output is a GRADED rate code"), and Stone et al.
# 2017 give CPU4 16 cells per column for the same reason.
ACC_G=[[9900+c*12+n for n in range(NA)] for c in range(NC)]   # per-column populations
ACC=[nid for col in ACC_G for nid in col]                     # flat list (tref_upper, readout)
ACCM_G=[[10200+c*12+n for n in range(NA)] for c in range(NC)]   # MAGNITUDE channel (rectified CD)
ACCM=[nid for col in ACCM_G for nid in col]
AGI=10100                              # accumulator global inhibitor (moved clear of ACC_G ids)
PHI_C=np.array([2*np.pi*c/NC for c in range(NC)])

def parts(ne,sy,conns,ex, CD, CDM=None, w_cd_acc=1.2, r_acc=0.9, lam_acc=6,
          w_acc_self=0.85, w_acc_nbr=0.35, w_acc_agi=-0.9,
          r_agi=1.6, w_acc_to_agi=0.22, lam_agi=4, seed_amp=0.0, r_acc_span=3.0):
    """Attach the accumulator to an EXISTING CD population, so the identical circuit can be built
    inside the standalone net AND inside the embodied agent."""
    thr=np.linspace(r_acc, r_acc*float(r_acc_span), NA)   # STAGGERED thresholds -> recruitment
    for c in range(NC):
        for n in range(NA):
            nid=ACC_G[c][n]
            ne.append(k.neuron(nid, r=float(thr[n]), c=2, lam=lam_acc)); j=0
            # LEAKY self-excitation: below the level that would latch it permanently
            sy.append(k.syn(nid,j,w_acc_self,1)); conns.append(k.conn(nid,nid,j,T(nid))); j+=1
            # drive: speed x cos(heading - phi_c), inherited from CD. Same weight to every cell in the
            # column -- the THRESHOLD spread, not the weight, is what makes recruitment graded.
            sy.append(k.syn(nid,j,w_cd_acc,1)); conns.append(k.conn(CD[c],nid,j,20000+CD[c])); j+=1
            # LOCAL EXCITATION from the two neighbouring columns (same threshold rank)
            for off in (-1,+1):
                nb=ACC_G[(c+off)%NC][n]
                sy.append(k.syn(nid,j,w_acc_nbr,1)); conns.append(k.conn(nb,nid,j,T(nb))); j+=1
            # GLOBAL INHIBITION (normaliser)
            sy.append(k.syn(nid,j,w_acc_agi,1)); conns.append(k.conn(AGI,nid,j,T(AGI))); j+=1
            if seed_amp>0: sy.append(k.syn(nid,j,seed_amp,1)); ex.append(k.ext(nid,j)); j+=1
            sy.append(k.term(nid,T(nid)))
    # ---- MAGNITUDE channel: same integrator, driven by the RECTIFIED CD twin, NO global inhibition
    # (its whole job is to accumulate, so normalising it would erase the quantity being read).
    if CDM is not None:
        for c in range(NC):
            for n in range(NA):
                nid=ACCM_G[c][n]
                ne.append(k.neuron(nid, r=float(thr[n]), c=2, lam=lam_acc)); j=0
                sy.append(k.syn(nid,j,w_acc_self,1)); conns.append(k.conn(nid,nid,j,T(nid))); j+=1
                sy.append(k.syn(nid,j,w_cd_acc,1)); conns.append(k.conn(CDM[c],nid,j,20000+CDM[c])); j+=1
                sy.append(k.term(nid,T(nid)))
    ne.append(k.neuron(AGI,r=r_agi,c=1,lam=lam_agi)); j=0
    for nid in ACC:
        sy.append(k.syn(AGI,j,w_acc_to_agi,1)); conns.append(k.conn(nid,AGI,j,T(nid))); j+=1
    sy.append(k.term(AGI,T(AGI)))
    return ne,sy,conns,ex

def read(nb, core, win=16, tick=None):
    """RATE-CODED readout: spikes per cell over `win` ticks -> population vector = home vector.
    A rate code means one lost spike is not a lost level (the ladder's failure mode)."""
    acc=np.zeros(NC)
    for _ in range(win):
        if tick is not None: tick()
        else: core.do_tick()
        for c in range(NC):
            for nid in ACC_G[c]:
                if nb[nid].O>0: acc[c]+=1
    if acc.sum()==0: return acc, 0.0, 0.0
    vx=float((acc*np.cos(PHI_C)).sum()); vy=float((acc*np.sin(PHI_C)).sum())
    return acc, float(np.degrees(np.arctan2(-vy,-vx))%360), float(np.hypot(vx,vy))
