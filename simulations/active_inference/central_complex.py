"""SPATIAL-MEMORY CORTEX — central-complex HEADING RING-ATTRACTOR (compass) with velocity integration.

Biologically faithful (user's option 2): the only non-neural step is a VESTIBULAR transducer (rotation rate
-> CCW/CW currents; forward speed -> speed current, later). Everything else is neurons + fixed wiring.

  HEADING RING (NR neurons): local recurrent excitation (radius NB) + global inhibition -> a persistent,
     localized BUMP = current heading. Fast membrane (lam_ring=2) so one recurrent spike retriggers.

  GRADED PUSH-PULL SHIFT (the P-EN mechanism, a POPULATION per column per direction). The single binary
     velocity cell fails: below its threshold nothing moves, above it the bump jumps whole columns and
     collapses -- a positive-feedback loop with no rate limit. Fix ("more neurons"): NP cells per column
     with STAGGERED thresholds. Each is an AND-gate on (bump present AND angular velocity), and each does
     PUSH-PULL on the ring -- EXCITE the leading flank R[col+SH], INHIBIT the trailing flank R[col-SH]
     (conserving total activity so the global inhibitor stays neutral and the bump TRANSLATES rather than
     grows/dies). Because thresholds are staggered, the NUMBER of cells that fire -- hence the push force --
     scales smoothly with the velocity current: slow w -> few fire -> slow drift; fast w -> many fire ->
     fast drift. The bump position is thus the neural integral of angular velocity = heading.

  DELAYED FEEDBACK is the load-bearing trick. Without it the shift is a positive-feedback loop with no rate
     limit: below threshold the bump is pinned (no drift), above it the bump jumps whole columns and dies.
     Giving the CL/CR->ring push synapses a dendritic delay (d_push ticks) converts the runaway into a
     bounded-speed TRAVELLING WAVE that survives -- the bump drifts smoothly at ~const rate while turning.
  JITTER desynchronises the ring: lam=2 + hard reset makes the whole ring burst in lock-step, which
     periodically starves the shift AND-gate; a per-neuron threshold stagger breaks that so the bump is
     steadier under the wave.

  BIOLOGICAL DECOUPLING (the fix that made it robust). Real heading systems (fly E-PG/P-EN, mammalian
     head-direction) NEVER let the bump die: at high angular velocity it saturates, LAGS, and BLURS
     (loses directional sharpness) but keeps firing, then re-sharpens/re-anchors when motion slows or a
     landmark appears (Turner-Evans 2017; Ajabi 2023; HD disorientation studies). The trick is that
     PERSISTENCE and MOVEMENT are decoupled -- the recurrent ring maintains the bump unconditionally, and
     P-EN only EXCITES it at an offset; movement can never subtract activity. So here the shift is
     EXCITATION-ONLY (w_pull=0, biological P-EN) and the GI is a soft NORMALISER (w_gi_ring=-1.2, not the
     old bang-bang -3.0 that annihilated). Adding leading excitation makes the GI trim the trailing edge
     (bump translates = drift) instead of collapsing everything.

  STATE (verified, jitter desync + excitation-only + normalising GI): the bump NEVER DIES across hold /
     drift / fast spin / abrupt reversal (min bump >=5 everywhere). HOLD is rock-solid. Rotation integrates
     PROPORTIONALLY (|omega| 0.6/0.9/1.2 -> ~0/+114/+163deg per 25 windows) and wraps >1 full revolution
     (+435deg/70win), CW/CCW symmetric, survives reversals. GRACEFUL DEGRADATION = the biology: a FAST spin
     blurs the bump wide (up to the whole ring) but it persists and RE-SHARPENS when you stop. Spin extreme
     (|omega|~1.8) and the blur fills the ring so heading is genuinely LOST (re-sharpens at ~0) -- real
     over-spin disorientation. Operating range |omega|<=1.2 stays sharp and integrates reliably.
"""
import sys, numpy as np
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
from paula_agent import ckit as k
def Tt(n): return 5000+n
NR=36; NB=3; SH=1; NP=4                       # ring size, excitation radius, shift offset, cells/column/dir
RING=list(range(0,NR)); GI=200
def CLid(i,p): return 300+i*NP+p               # CCW shift-cell populations
def CRid(i,p): return 300+NR*NP+i*NP+p         # CW  shift-cell populations
def SRid(i,p,d): return 4000+d*NR*NP+i*NP+p    # optional graded P-EN->ring relay, d=0 CCW / 1 CW
CL=[[CLid(i,p) for p in range(NP)] for i in range(NR)]
CR=[[CRid(i,p) for p in range(NP)] for i in range(NR)]
NPEG=NR   # P-EG: offset-free maintenance loop through the bridge (one per column)
PEG=[1400+i for i in range(NR)]
# Default-off P-ENb-like cells.  Unlike the older P-EG return, these are a
# direction-specific *trailing* route: P-EG provides the local heading signal
# and a separately wired self-motion input selects which of the two spatial
# offsets can return to E-PG.  They are deliberately a distinct population,
# not an extra same-column recurrence labelled as a new cell type.
PENB_CL=[7400+i for i in range(NR)]
PENB_CR=[7500+i for i in range(NR)]
# A separate, default-off P-ENa micro-bank experiment.  It deliberately
# retains the original four P-EN cells and adds a small heterogeneous
# population per column rather than turning one synaptic weight into a hidden
# global gain.  IDs reserve four copies per column, but a candidate may choose
# fewer at construction time.
PENA_MAX_COPIES=4
def PENA_CLid(i, copy): return 7900 + i*PENA_MAX_COPIES + copy
def PENA_CRid(i, copy): return 8100 + i*PENA_MAX_COPIES + copy
PENA_CL=[[PENA_CLid(i, copy) for copy in range(PENA_MAX_COPIES)] for i in range(NR)]
PENA_CR=[[PENA_CRid(i, copy) for copy in range(PENA_MAX_COPIES)] for i in range(NR)]
# Optional explicit PB/EB bridge populations.  The baseline compact model
# deliberately leaves these absent.  When enabled, update traffic goes
# P-EN -> PB shift tract -> E-PG and maintenance goes E-PG -> bilateral PB
# tract -> P-EG -> E-PG.  They use only ordinary PAULA synapses/membranes;
# no sampling, holding, or host-side compass state is introduced.
PB_ML=[7000+i for i in range(NR)]
PB_MR=[7100+i for i in range(NR)]
PB_CL=[7200+i for i in range(NR)]
PB_CR=[7300+i for i in range(NR)]
# A separate, default-off PB update hypothesis.  It does not alter the
# accepted direct P-EN route: each cell is an ordinary PAULA three-input
# coincidence relay (local E-PG, fast vestibular afferent, and a CPG phase
# interneuron).  It is built only for the explicit phase-gating experiment.
PB_PHASE_CL=[7600+i for i in range(NR)]
PB_PHASE_CR=[7700+i for i in range(NR)]
PB_PHASE_CLOCK=[7800+i for i in range(44)]
PB_PHASE_SENSOR_SYNS={}
PB_PHASE_CLOCK_SYNS={}
ND7=8                                          # Delta-7 inhibitory population (fly: 8-9 fold periodic)
D7=[250+j for j in range(ND7)]
PG=[1000+i for i in range(NR)]   # moved off 600: at NR>=72 the CR block reaches 588+ and collided                  # PATH-INTEGRATION speed-gated bump-readout cells (one/column)
XACC=1800; YACC=1801                             # analog home-vector accumulators (never spike; read membrane S)
PHI=np.array([2*np.pi*i/NR for i in range(NR)])
def set_NR(n):
    """Change the ring's angular resolution. NR=36 gives 10 deg columns, which quantises both the
    bump's angular width and the shift's step size; more columns = finer heading. Everything below is
    derived from NR, so recompute the id lists. Call BEFORE building."""
    global NR,RING,CL,CR,PG,PEG,PENB_CL,PENB_CR,PENA_CL,PENA_CR,PB_ML,PB_MR,PB_CL,PB_CR,PB_PHASE_CL,PB_PHASE_CR,PHI,D7
    NR=int(n)
    RING=list(range(0,NR))
    CL=[[CLid(i,p) for p in range(NP)] for i in range(NR)]
    CR=[[CRid(i,p) for p in range(NP)] for i in range(NR)]
    PG=[1000+i for i in range(NR)]; PEG=[1400+i for i in range(NR)]
    PENB_CL=[7400+i for i in range(NR)]; PENB_CR=[7500+i for i in range(NR)]
    PENA_CL=[[PENA_CLid(i, copy) for copy in range(PENA_MAX_COPIES)] for i in range(NR)]
    PENA_CR=[[PENA_CRid(i, copy) for copy in range(PENA_MAX_COPIES)] for i in range(NR)]
    PB_ML=[7000+i for i in range(NR)]; PB_MR=[7100+i for i in range(NR)]
    PB_CL=[7200+i for i in range(NR)]; PB_CR=[7300+i for i in range(NR)]
    PB_PHASE_CL=[7600+i for i in range(NR)]; PB_PHASE_CR=[7700+i for i in range(NR)]
    PHI=np.array([2*np.pi*i/NR for i in range(NR)])
    D7=[250+j for j in range(ND7)]
    return NR

SEEDSYN=None
SEEDMAP={}      # ring id -> its OWN birth-seed synapse index (layouts differ per cell)
TONICSYN={}      # ring id -> tonic synapse index (differs per cell: Delta-7 coverage varies)

def _ns(sy,nid): return sum(1 for x in sy if x.get('neuron_id')==nid and x['type']=='postsynaptic')

def parts(w_self=1.4, w_nbr=0.7, w_gi_ring=-1.2, w_ring_gi=0.20, r_ring=0.9, r_gi=1.4, lam_ring=2,
          graded_ring=False, graded_ring_gain=0.0, graded_ring_S0=0.0, graded_ring_max=0.0,
          lam_conj=2, w_conj=1.4, r_conj_lo=1.7, r_conj_hi=3.5, w_r_shift=0.0,
          graded_shift=False, graded_shift_gain=1.0, graded_shift_S0=0.0, graded_shift_max=0.0,
          conjunctive_graded_shift=False, conjunctive_graded_S0=0.0,
          conjunctive_ring_tau=1.0, conjunctive_velocity_tau=1.0, conjunctive_velocity_lead_gain=0.0,
          w_push=1.1, w_push_ccw=None, w_push_cw=None, w_pull=0.0, d_push=8, d_push_hi=None, sh_bank=False, sh_max=None, bank_wta=0.0, w_lr_inhib=0.0,
          shift_relay=False, shift_relay_gain=1.0, shift_relay_S0=0.0, shift_relay_lam=4.0,
          w_seed=4.0, jitter=0.2,
          r_pg=1.7, w_pg=1.4, w_pg_s=None, k_pi=1.0, lam_acc=50000,
          w_tonic=0.0, w_peg=0.0, r_peg=0.9, lam_peg=2, d_peg=3, w_peg_ring=1.0,
          p_enb_trailing=False, lam_penb=2, w_peg_penb=0.6,
          penb_conjunctive_gain=1.0, penb_ring_tau=4.0, penb_velocity_tau=4.0,
          w_penb_ring=0.12, d_penb=3,
          p_ena_bank=False, pena_copies=PENA_MAX_COPIES, pena_gain=0.5,
          pena_ring_tau=1.0, pena_velocity_taus=(1.0, 2.0, 4.0, 8.0),
          w_pena_ring=0.35, d_pena=3,
          pb_eb_bridge=False, r_pb=0.9, lam_pb=2, w_pb_peg=0.8, d_pb=1,
          pb_shift_gain=1.0, pb_shift_S0=0.0, pb_shift_lam=1.0, d_pb_shift=1,
          w_pb_cross_inhib=0.0, w_pb_opponent=0.0,
          pb_phase_update=False, r_pb_phase=2.1, lam_pb_phase=1.0,
          w_pb_phase_ring=0.7, w_pb_phase_sensor=0.3, w_pb_phase_clock=0.7,
          pb_phase_width=3,
          d7=False, w_ring_d7=0.30, r_d7=1.4, lam_d7=3, w_d7_ring=-1.2, d7_in=2.0, d7_out=8.0, d7_cos=False, d7_uniform=0.35, d7_to_shift=False, w_d7_shift=-6.0, w_d7_d7=-2.0):
    global SEEDSYN, TONICSYN, SEEDMAP, PB_PHASE_SENSOR_SYNS, PB_PHASE_CLOCK_SYNS
    TONICSYN={}; SEEDMAP={}
    PB_PHASE_SENSOR_SYNS={}; PB_PHASE_CLOCK_SYNS={}
    ne=[];sy=[];conns=[];ex=[]
    if pb_eb_bridge and (sh_bank or d_push_hi is not None):
        raise ValueError("pb_eb_bridge presently requires one shared P-EN offset/delay")
    if pb_eb_bridge and (int(d_pb_shift) < 1 or int(d_pb_shift) + 1 >= int(d_push)):
        raise ValueError("pb_eb_bridge requires 1 <= d_pb_shift <= d_push - 2")
    if pb_eb_bridge and int(d_pb) < 1:
        raise ValueError("pb_eb_bridge requires d_pb >= 1")
    if p_enb_trailing and (not pb_eb_bridge or not w_peg):
        raise ValueError("p_enb_trailing requires the explicit PB/P-EG bridge")
    if p_enb_trailing and int(d_penb) < 1:
        raise ValueError("p_enb_trailing requires d_penb >= 1")
    if p_ena_bank and not 1 <= int(pena_copies) <= PENA_MAX_COPIES:
        raise ValueError(f"p_ena_bank pena_copies must be 1..{PENA_MAX_COPIES}")
    if p_ena_bank and int(d_pena) < 1:
        raise ValueError("p_ena_bank requires d_pena >= 1")
    if p_ena_bank:
        pena_velocity_taus=tuple(float(tau) for tau in pena_velocity_taus)
        if len(pena_velocity_taus) < int(pena_copies) or any(tau < 1.0 for tau in pena_velocity_taus):
            raise ValueError("p_ena_bank needs one positive velocity tau per selected copy")
    if pb_phase_update and not pb_eb_bridge:
        raise ValueError("pb_phase_update requires pb_eb_bridge")
    if pb_phase_update and not 1 <= int(pb_phase_width) <= len(PB_PHASE_CLOCK):
        raise ValueError("pb_phase_width must be within the explicit 44-cell CPG phase population")
    # deterministic per-neuron threshold stagger DESYNCHRONISES the ring (breaks the global burst that
    # periodically starves the shift AND-gate) -> a steadier bump that tolerates the traveling wave.
    rjit=[r_ring + jitter*np.sin(2*np.pi*3*i/NR) for i in range(NR)]
    for i,r in enumerate(RING):
        ring_meta = None
        if graded_ring:
            ring_meta = {
                "graded_gain": float(graded_ring_gain),
                "graded_S0": float(graded_ring_S0),
                "graded_max": float(graded_ring_max),
            }
        ne.append(k.neuron(r, r=rjit[i], c=2, lam=lam_ring, meta=ring_meta))
    rj={r:0 for r in RING}
    def rs(r,w,d=1): j=rj[r]; sy.append(k.syn(r,j,w,d)); rj[r]+=1; return j
    # local recurrent excitation (self + neighbours within radius NB, tapering) + global inhibition
    for i,r in enumerate(RING):
        conns.append(k.conn(r,r,rs(r,w_self),Tt(r)))
        for off in range(1,NB+1):
            w=w_nbr*(1.0-(off-1)/NB)
            conns.append(k.conn(RING[(i-off)%NR],r,rs(r,w),Tt(RING[(i-off)%NR])))
            conns.append(k.conn(RING[(i+off)%NR],r,rs(r,w),Tt(RING[(i+off)%NR])))
        conns.append(k.conn(GI,r,rs(r,w_gi_ring),Tt(GI)))
    # ---- P-EG MAINTENANCE THROUGH AN OPTIONAL EXPLICIT PB BRIDGE --------
    # The older ``w_peg`` implementation was only RING[i] -> PEG[i] ->
    # RING[i], which is extra delayed self-excitation, not a PB/EB topology.
    # It was measured to be either silent at low gain or harmful when active.
    # ``pb_eb_bridge`` is a deliberately modest but real structural test:
    # E-PG sends a coordinate-preserving signal to both bridge sides, the two
    # sides converge onto a P-EG, and P-EG returns to the same EB column.  The
    # two bilateral bridge tracts are visible populations with ordinary PAULA
    # membrane and synapse dynamics.  It is a hypothesis inspired by insect
    # PB/EB organization, not a claim that its exact cell-type mapping matches
    # a particular animal.
    if w_peg:
        if pb_eb_bridge:
            for i in range(NR):
                for nid in (PB_ML[i], PB_MR[i]):
                    ne.append(k.neuron(nid,r=r_pb,c=2,lam=lam_pb))
                    sy.append(k.syn(nid,0,w_peg,int(d_pb)))
                    conns.append(k.conn(RING[i],nid,0,Tt(RING[i])))
                    sy.append(k.term(nid,Tt(nid)))
                nid=PEG[i]; ne.append(k.neuron(nid,r=r_peg,c=2,lam=lam_peg))
                for sid,source in enumerate((PB_ML[i], PB_MR[i])):
                    sy.append(k.syn(nid,sid,w_pb_peg,1))
                    conns.append(k.conn(source,nid,sid,Tt(source)))
                sy.append(k.term(nid,Tt(nid)))
            for i,r in enumerate(RING):
                conns.append(k.conn(PEG[i],r,rs(r,w_peg_ring,d_peg),Tt(PEG[i])))
            if p_enb_trailing:
                # This is intentionally not ``PEG[i] -> RING[i]`` again.
                # Each P-EG enters a distinct, direction-specific P-ENb
                # candidate.  Its second dendrite is an external/self-motion
                # port that is connected by the owning experimental agent;
                # keeping it as a real PAULA input makes its source and
                # latency inspectable in a full trace.
                for i in range(NR):
                    for nid in (PENB_CL[i], PENB_CR[i]):
                        ne.append(k.neuron(
                            nid, r=1e9, c=2, lam=float(lam_penb),
                            meta={
                                # This is the existing PAULA inherited
                                # conjunctive release model, now applied to a
                                # separately named P-ENb hypothesis.  Its
                                # leaky dendrites explicitly tolerate sparse
                                # P-EG events versus continuous self-motion;
                                # it is not a sample/hold mechanism and must
                                # remain labelled experimental.
                                "graded_gain": float(penb_conjunctive_gain),
                                "graded_S0": 0.0,
                                "conjunctive_graded_gain": float(penb_conjunctive_gain),
                                "conjunctive_ring_synapse": 0,
                                "conjunctive_velocity_synapse": 1,
                                "conjunctive_graded_S0": 0.0,
                                "conjunctive_ring_tau": float(penb_ring_tau),
                                "conjunctive_velocity_tau": float(penb_velocity_tau),
                                "p_enb_trailing": True,
                                "p_enb_local_column": i,
                                "p_enb_claim": "experimental inherited PAULA coincidence route",
                            },
                        ))
                        sy.append(k.syn(nid, 0, float(w_peg_penb), 1))
                        conns.append(k.conn(PEG[i], nid, 0, Tt(PEG[i])))
                        # This port is deliberately zero in a normal agent.
                        # The separate V6 branch supplies it from its PAULA
                        # fused self-motion population, never from a decoded
                        # heading or host-side turn estimate.
                        sy.append(k.syn(nid, 1, 1.0, 1)); ex.append(k.ext(nid, 1))
                        sy.append(k.term(nid, Tt(nid)))
                for i, r in enumerate(RING):
                    # P-ENb has the opposite *spatial* offset from the early
                    # P-ENa-like push.  During a CCW update the delayed
                    # activity therefore reinforces the trailing flank rather
                    # than launching a second leading wave; CW is mirrored.
                    conns.append(k.conn(PENB_CL[(i + SH) % NR], r,
                                        rs(r, float(w_penb_ring), int(d_penb)),
                                        Tt(PENB_CL[(i + SH) % NR])))
                    conns.append(k.conn(PENB_CR[(i - SH) % NR], r,
                                        rs(r, float(w_penb_ring), int(d_penb)),
                                        Tt(PENB_CR[(i - SH) % NR])))
        else:
            # Retained only for backwards-compatible reproduction of the
            # former direct-relay experiment; do not call it a bridge.
            for i in range(NR):
                nid=PEG[i]; ne.append(k.neuron(nid,r=r_peg,c=2,lam=lam_peg))
                sy.append(k.syn(nid,0,w_peg,1)); conns.append(k.conn(RING[i],nid,0,Tt(RING[i])))
                sy.append(k.term(nid,Tt(nid)))
            for i,r in enumerate(RING):
                conns.append(k.conn(PEG[i],r,rs(r,w_peg_ring,d_peg),Tt(PEG[i])))

    # The update bridge always exists with ``pb_eb_bridge``, even when P-EG
    # maintenance is off.  The default bridge sums four P-EN threshold ranks
    # into a graded PB relay.  ``pb_phase_update`` is deliberately a separate
    # hypothesis: it substitutes an ordinary three-input PB coincidence cell
    # (local E-PG, fast vestibular signal, and an explicit CPG-phase neuron).
    # It is not a sample-and-hold cell or a host-side stroke summary.
    if pb_eb_bridge:
        for i in range(NR):
            if pb_phase_update:
                for nid, direction in ((PB_PHASE_CL[i], "CCW"), (PB_PHASE_CR[i], "CW")):
                    # A conventional PAULA spiking coincidence relay.  Each
                    # input is below threshold alone; an active local E-PG
                    # column, the signed fast vestibular afferent, and an
                    # explicit gait-phase interneuron are all required.
                    ne.append(k.neuron(nid, r=float(r_pb_phase), c=2, lam=float(lam_pb_phase),
                                       meta={"pb_phase_direction": direction, "pb_phase_update": True}))
                    sy.append(k.syn(nid, 0, float(w_pb_phase_ring), 1))
                    conns.append(k.conn(RING[i], nid, 0, Tt(RING[i])))
                    PB_PHASE_SENSOR_SYNS[nid] = 1
                    sy.append(k.syn(nid, 1, float(w_pb_phase_sensor), 1))
                    clock_synapses=[]
                    for phase_index in range(int(pb_phase_width)):
                        sid=2+phase_index
                        sy.append(k.syn(nid, sid, float(w_pb_phase_clock), 1))
                        clock_synapses.append(sid)
                    PB_PHASE_CLOCK_SYNS[nid]=clock_synapses
                    sy.append(k.term(nid,Tt(nid)))

            else:
                for nid, sources in ((PB_CL[i], CL[i]), (PB_CR[i], CR[i])):
                    ne.append(k.neuron(
                        nid, r=1e9, c=2, lam=float(pb_shift_lam),
                        meta={
                            "graded_gain": float(pb_shift_gain),
                            "graded_S0": float(pb_shift_S0),
                        },
                    ))
                    for sid, source in enumerate(sources):
                        sy.append(k.syn(nid,sid,1.0,1))
                        conns.append(k.conn(source,nid,sid,Tt(source)))
                    if w_pb_opponent > 0.0:
                        # The gait activates both P-EN directions on alternating
                        # half strokes.  A PB relay can therefore receive the
                        # opponent tract on ordinary negative dendrites and
                        # represent its *local directional residual* before the
                        # EB offset.  This is a circuit-level subtraction, not a
                        # host-computed yaw average.
                        opposite=CR[i] if sources is CL[i] else CL[i]
                        for p, source in enumerate(opposite):
                            sid=NP+p
                            sy.append(k.syn(nid,sid,-abs(float(w_pb_opponent)),1))
                            conns.append(k.conn(source,nid,sid,Tt(source)))
                    sy.append(k.term(nid,Tt(nid)))
        if w_pb_cross_inhib > 0.0 and not pb_phase_update:
            # Unlike the P-EN conjunction cells, these are ordinary graded
            # relay membranes.  Reciprocal inhibition here therefore affects
            # their release and can reject the common-mode CL+CR activity
            # generated by an alternating gait waveform before it reaches
            # E-PG.  This is a PB opponent hypothesis, not a hidden
            # post-processing subtraction.
            for i in range(NR):
                for source, target in ((PB_CL[i], PB_CR[i]), (PB_CR[i], PB_CL[i])):
                    sy.append(k.syn(target,NP,-abs(float(w_pb_cross_inhib)),1))
                    conns.append(k.conn(source,target,NP,Tt(source)))

    if p_ena_bank:
        # This is a scale-up test, not a new heading variable.  Every copy
        # has the normal local E-PG dendrite plus a separately attached PAULA
        # fused-update dendrite.  The copies differ only in declared local
        # velocity trace constants, providing overlapping fast-to-stroke-scale
        # evidence at the *same* spatial offset.  No CPG clock, sample/hold,
        # or Python smoothing exists here.
        for i in range(NR):
            for copy in range(int(pena_copies)):
                for cells, direction in ((PENA_CL, "CCW"), (PENA_CR, "CW")):
                    nid=cells[i][copy]
                    ne.append(k.neuron(
                        nid, r=1e9, c=2, lam=2,
                        meta={
                            "graded_gain": float(pena_gain),
                            "graded_S0": 0.0,
                            "conjunctive_graded_gain": float(pena_gain),
                            "conjunctive_ring_synapse": 0,
                            "conjunctive_velocity_synapse": 1,
                            "conjunctive_graded_S0": 0.0,
                            "conjunctive_ring_tau": float(pena_ring_tau),
                            "conjunctive_velocity_tau": pena_velocity_taus[copy],
                            "p_ena_microbank": True,
                            "p_ena_direction": direction,
                            "p_ena_copy": copy,
                            "p_ena_claim": "experimental heterogeneous PAULA population scale-up",
                        },
                    ))
                    sy.append(k.syn(nid, 0, 1.0, 1))
                    conns.append(k.conn(RING[i], nid, 0, Tt(RING[i])))
                    # The candidate owner replaces this empty direct port
                    # with a declared fused-update synapse later.
                    sy.append(k.syn(nid, 1, 1.0, 1)); ex.append(k.ext(nid, 1))
                    sy.append(k.term(nid, Tt(nid)))
        for i, r in enumerate(RING):
            for copy in range(int(pena_copies)):
                cl_source=PENA_CL[(i-SH) % NR][copy]
                cr_source=PENA_CR[(i+SH) % NR][copy]
                conns.append(k.conn(cl_source, r, rs(r, float(w_pena_ring), int(d_pena)),
                                    Tt(cl_source)))
                conns.append(k.conn(cr_source, r, rs(r, float(w_pena_ring), int(d_pena)),
                                    Tt(cr_source)))
    # ---- DELTA-7: STRUCTURED, PHASE-OFFSET INHIBITION (replaces the single global inhibitor) ----
    # The single GI sums the WHOLE ring and inhibits the WHOLE ring, so when the bump widens in order
    # to translate it suppresses ITSELF -- which is why every attempt to make the shift faster or
    # wider killed the attractor. The fly does not use one global cell: Delta-7 neurons tile the
    # protocerebral bridge and inhibit the glomeruli roughly HALF A TURN away (Hulse et al. 2021;
    # Pisokas, Heinze & Webb 2020; Kakaria & de Bivort 2017). Phase-offset inhibition means a widening
    # bump suppresses its ANTIPODE, not itself, so sharpness and shift speed stop competing.
    # Each ring cell is reached by only the few Delta-7 cells whose antipodal field covers it, which
    # also keeps fan-in (and therefore the adaptive t_ref bound) from exploding.
    if d7:
        d7phi=np.array([2*np.pi*j/ND7 for j in range(ND7)])
        def coldist(a,b): return abs(((a-b+np.pi)%(2*np.pi))-np.pi)/(2*np.pi)*NR
        for j in range(ND7):
            nid=D7[j]; ne.append(k.neuron(nid,r=r_d7,c=1,lam=lam_d7)); jj=0
            for i in range(NR):
                if coldist(PHI[i],d7phi[j])<=d7_in:
                    sy.append(k.syn(nid,jj,w_ring_d7,1))
                    conns.append(k.conn(RING[i],nid,jj,Tt(RING[i]))); jj+=1
            sy.append(k.term(nid,Tt(nid)))
        # PROFILE. My first version was a BOX at the antipode with a linear taper -- invented, not
        # taken from anywhere. The literature profile for ring-attractor inhibition is
        #     W(dphi) = -(a + b*cos(dphi))
        # a uniform term plus a cosine term (Pisokas, Heinze & Webb 2020 eLife model the locust and
        # fly Delta-7 this way; Kakaria & de Bivort 2017 give the spiking PB version). Those two terms
        # do DIFFERENT jobs: the uniform part limits how far a wave can spread locally, the cosine part
        # creates antipodal competition so only one bump survives. A box at the antipode supplies only
        # the second, which is why local spreading went unopposed and the bump delocalised the moment
        # it started moving. d7_cos=True builds the real profile over the whole ring.
        # TARGET. Pisokas, Heinze & Webb 2020 (eLife 9:e53985): Delta-7 projects to P-EN and P-EG --
        # NOT to E-PG -- and Delta-7 neurons inhibit each other (globally and uniformly in Drosophila,
        # a weakening subset in locust). I originally wired Delta-7 -> E-PG, which is the wrong cell
        # type: inhibiting the compass cells fights the bump itself, whereas inhibiting the SHIFT cells
        # stops P-EN units far from the bump from firing -- exactly the runaway that delocalised the
        # bump the moment it started moving.
        for i,r in enumerate(RING):
            for j in range(ND7):
                if d7_cos:
                    dphi=PHI[i]-d7phi[j]
                    # normalise by population size: with the full-ring profile every cell is
                    # reached by ALL ND7 inhibitors, not a subset, so the unnormalised weight
                    # delivered ~8x the intended inhibition and annihilated the bump.
                    w=(w_d7_ring/ND7)*(d7_uniform + (1.0-d7_uniform)*(1.0-np.cos(dphi))/2.0)
                    conns.append(k.conn(D7[j],r,rs(r,float(w)),Tt(D7[j])))
                else:
                    dd=coldist(PHI[i],d7phi[j]+np.pi)
                    if dd<=d7_out:
                        conns.append(k.conn(D7[j],r,rs(r,w_d7_ring*(1.0-dd/(d7_out+1.0))),Tt(D7[j])))
    # shift feedback onto the ring. EXCITATION-ONLY by default (biological P-EN: only EXCITES E-PG at an
    # offset, so movement can never subtract activity -> the bump can lag/blur/saturate but NEVER die; the
    # GI normaliser trims the trailing edge). Optional trailing inhibition (w_pull>0) = the old push-pull.
    # STRUCTURED SCALING: the shift populations already stagger THRESHOLD, which only sets WHEN each
    # engages -- every one of them then drives the ring through the same dendritic delay, so they all
    # produce the same wave speed and the compass has a single narrow linear band. Staggering the
    # DELAY as well makes wave speed itself velocity-dependent: the low-threshold population (slow
    # turns) gets the LONGEST delay and the slowest wave, the high-threshold population (fast turns)
    # the shortest delay and the fastest wave. The bump then has a bank of speeds to recruit instead
    # of one. d_push_hi=None keeps the original single-delay behaviour.
    # Staggering the DELAY was measured to destroy the wave (it needs delay coherence to propagate).
    # Staggering the SPATIAL STEP does not: every population keeps the same delay, but the one that
    # only engages at high velocity pushes FURTHER around the ring, so wave speed rises with drive
    # while each wave stays coherent. Population p (threshold rising with p) pushes SH+p columns.
    dly=(np.linspace(d_push,d_push_hi,NP) if d_push_hi is not None else np.full(NP,float(d_push)))
    for i,r in enumerate(RING):
        # The two biological P-EN tracts are separately parameterised for
        # calibration because the existing discrete ring has a measured
        # unilateral response imbalance. Defaults preserve the original
        # symmetric conductance exactly.
        push_ccw = w_push if w_push_ccw is None else w_push_ccw
        push_cw = w_push if w_push_cw is None else w_push_cw
        if pb_eb_bridge:
            # P-EN output first arrives at a linear PB relay.  A relay also
            # has one PAULA membrane/update tick before it can release.  The
            # two dendritic distances must therefore sum to ``d_push - 1``
            # to reproduce the direct P-EN->E-PG arrival phase.  The old
            # formula preserved only the distance sum and made every bridge
            # candidate one tick later than its direct control.
            dp=int(d_push) if pb_phase_update else int(d_push)-int(d_pb_shift)-1
            cl_population=PB_PHASE_CL if pb_phase_update else PB_CL
            cr_population=PB_PHASE_CR if pb_phase_update else PB_CR
            cl_source=cl_population[(i-SH)%NR]
            cr_source=cr_population[(i+SH)%NR]
            conns.append(k.conn(cl_source,r,rs(r,push_ccw,dp),Tt(cl_source)))
            conns.append(k.conn(cr_source,r,rs(r,push_cw,dp),Tt(cr_source)))
            if w_pull>0:
                conns.append(k.conn(cl_population[(i+SH)%NR],r,rs(r,-w_pull,dp),Tt(cl_population[(i+SH)%NR])))
                conns.append(k.conn(cr_population[(i-SH)%NR],r,rs(r,-w_pull,dp),Tt(cr_population[(i-SH)%NR])))
        else:
            for p in range(NP):
                # the push must land INSIDE the ring's recurrent excitation radius (NB), or it
                # seeds a second bump instead of translating the first one and the attractor dies
                _shm=(NB if sh_max is None else sh_max)
                shp=(SH+int(round(p*(_shm-SH)/max(NP-1,1)))) if sh_bank else SH
                dp=int(round(dly[p]))
                cl_source = SRid((i-shp)%NR, p, 0) if shift_relay else CL[(i-shp)%NR][p]
                cr_source = SRid((i+shp)%NR, p, 1) if shift_relay else CR[(i+shp)%NR][p]
                conns.append(k.conn(cl_source,r,rs(r, push_ccw,dp),Tt(cl_source)))  # CL lead: excite (P-EN, delayed)
                conns.append(k.conn(cr_source,r,rs(r, push_cw,dp),Tt(cr_source)))  # CR lead: excite
                if w_pull>0:
                    conns.append(k.conn(CL[(i+SH)%NR][p],r,rs(r,-w_pull,d_push),Tt(CL[(i+SH)%NR][p])))  # (optional) trailing inhibit
                    conns.append(k.conn(CR[(i-SH)%NR][p],r,rs(r,-w_pull,d_push),Tt(CR[(i-SH)%NR][p])))
        SEEDMAP[r]=rs(r,w_seed); ex.append(k.ext(r,SEEDMAP[r]))   # store PER CELL, see SEEDSYN below
        # TONIC DRIVE. The ring had none: once every cell fell silent nothing could restart it, because
        # the only excitation was recurrent (needs someone already firing) and the birth seed is applied
        # once. Measured consequence in the embodied loop: the baseline ring went silent for 2593
        # consecutive ticks out of 4800 -- the compass was simply OFF for more than half the run.
        # Real E-PG neurons are tonically driven; a small constant input makes silence unlatchable.
        TONICSYN[r]=rs(r,w_tonic); ex.append(k.ext(r,TONICSYN[r]))
        sy.append(k.term(r,Tt(r)))
    # SEEDSYN was `rj[RING[0]]-1` = the LAST synapse index of RING[0] minus one. But the TONIC synapse
    # is created AFTER the seed, so that expression pointed at the TONIC synapse: every "birth seed"
    # was injected at amplitude w_seed through a w_tonic(=0.12) weight -> 4.0*0.12 = 0.48 against
    # r_ring=0.9, i.e. SUB-THRESHOLD. The birth seed never worked. At NR=36 the ring limped along on
    # recurrence anyway; at NR=72 it was fatal (0 ring spikes from tick 0).
    # It was also a SINGLE GLOBAL index taken from RING[0], while per-cell synapse layouts differ
    # (Delta-7 sector coverage varies per cell) -- the identical bug that TONICSYN already had.
    # Keep the scalar for callers that still read cc.SEEDSYN, but make it the REAL seed index.
    SEEDSYN=SEEDMAP[RING[0]]
    # GLOBAL INHIBITOR
    ne.append(k.neuron(GI,r=r_gi,c=1,lam=3)); gj=0
    for r in RING: sy.append(k.syn(GI,gj,w_ring_gi,1)); conns.append(k.conn(r,GI,gj,Tt(r))); gj+=1
    sy.append(k.term(GI,Tt(GI)))
    # SHIFT CELL POPULATIONS: AND-gate(bump R[col], velocity ext). syn0=R[col], syn1=velocity. Staggered r.
    thr=np.linspace(r_conj_lo,r_conj_hi,NP)
    for i in range(NR):
        for p in range(NP):
            for cid,src in ((CL[i][p],RING[i]),(CR[i][p],RING[i])):
                # NEUROMODULATORY GAIN CONTROL. The substrate computes r = r_base + w_r . M, where M
                # is driven by the mod channel of incoming terminals (neuron.py:542). A negative w_r
                # therefore LOWERS this cell's firing threshold as its modulator rises. Wiring the
                # wide-field turn-rate estimate into that channel makes the shift's usable drive window
                # MOVE WITH the turn rate instead of sitting fixed -- which is the whole reason a 1.6x
                # window could not serve a 7.5x signal range. Octopamine does exactly this to gain in
                # fly optic-flow and steering neurons (Suver, Mamiya & Dickinson 2012; Longden & Krapp 2009).
                # lam_conj: the shift cell's MEMBRANE time constant. At lam=2 (8 ms) it relays each
                # tick's rectified drive, so within one 160 ms stroke BOTH CL and CR fire (measured 95-100%
                # of cycles) and their excitation-only pushes CANCEL at the ring -- the CL-CR balance
                # carries the rotation signal (r=+0.625 vs true net rotation) but the bump does not respond
                # to it (r=+0.004). Integrating over ~one stroke should make the cell's OUTPUT the net
                # rotation instead of the instantaneous sign.
                if graded_shift:
                    # Experimental continuous P-EN output.  The regular
                    # spiking route resets S at every threshold crossing,
                    # which makes a small signed gyro residual either vanish
                    # or seed a travelling-wave runaway.  A graded P-EN
                    # keeps the same ring/velocity AND inputs and fixed
                    # offset wiring, but emits in proportion to membrane
                    # depolarisation without a reset.  It is opt-in so the
                    # established spiking circuit is bit-for-bit unchanged.
                    metadata={
                        "graded_gain": float(graded_shift_gain),
                        "graded_S0": float(graded_shift_S0),
                        "graded_max": float(graded_shift_max),
                    }
                    if conjunctive_graded_shift:
                        # P-EN output must retain the local heading bump ×
                        # angular-velocity conjunction. The direct isolated
                        # route drives velocity on synapse 1; a later
                        # vestibular builder retargets that metadata to its
                        # dedicated PAULA opponent synapse.
                        metadata.update({
                            "conjunctive_graded_gain": float(graded_shift_gain),
                            "conjunctive_ring_synapse": 0,
                            "conjunctive_velocity_synapse": 1,
                            "conjunctive_graded_S0": float(conjunctive_graded_S0),
                            "conjunctive_ring_tau": float(conjunctive_ring_tau),
                            "conjunctive_velocity_tau": float(conjunctive_velocity_tau),
                            "conjunctive_velocity_lead_gain": float(conjunctive_velocity_lead_gain),
                        })
                    ne.append(k.neuron(
                        cid, r=1e9, c=2, lam=lam_conj,
                        meta=metadata,
                    ))
                else:
                    ne.append(k.neuron(cid,r=float(thr[p]),c=2,lam=lam_conj,
                                       w_r=([-abs(w_r_shift),0.0] if w_r_shift else None)))
                sy.extend([k.syn(cid,0,w_conj,1),k.syn(cid,1,w_conj,1),k.term(cid,Tt(cid))])
                conns.append(k.conn(src,cid,0,Tt(src))); ex.append(k.ext(cid,1))
                if shift_relay:
                    direction = 0 if cid == CL[i][p] else 1
                    relay = SRid(i, p, direction)
                    ne.append(k.neuron(
                        relay, r=1e9, c=2, lam=float(shift_relay_lam),
                        meta={
                            "graded_gain": float(shift_relay_gain),
                            "graded_S0": float(shift_relay_S0),
                        },
                    ))
                    sy.append(k.syn(relay, 0, 1.0, 1))
                    conns.append(k.conn(cid, relay, 0, Tt(cid)))
                    sy.append(k.term(relay, Tt(relay)))
    _nsy={}
    def _next(nid, base=2):
        _nsy[nid]=_nsy.get(nid,base-1)+1
        return _nsy[nid]
    # ---- BANK WTA: make the velocity-graded phase offset EXCLUSIVE instead of superposed ----
    # Turner-Evans et al. 2017 (eLife 23496): the P-EN/E-PG phase offset varies linearly with
    # rotational velocity -- ONE offset that moves, not several at once. sh_bank alone gives each
    # population p its own offset SH+p, but the populations have STAGGERED THRESHOLDS, so they are
    # NESTED: at high velocity every lower-threshold population is firing too, and the ring is pushed
    # at SH, SH+1, SH+2 ... simultaneously. That smears the push instead of translating the bump, and
    # it is why sh_bank measured NEGATIVE (mean slope -0.03, 5/8 conditions reversed).
    # Descending inhibition p+1 -> p makes the highest ACTIVE population suppress all slower ones, so
    # exactly one offset drives the ring at any turn rate. Default 0.0 = unchanged.
    # ---- CL <-> CR MUTUAL INHIBITION (directional winner-take-all) ----
    # MEASURED: unilateral drive rotates the bump correctly (+303 deg/1000 ticks for CL, -144 for CR),
    # so the push wiring is sound. In the closed loop BOTH sides fire within 95-100% of stroke cycles,
    # each ~12 spikes/tick, and their excitation-only pushes CANCEL -- the CL-CR balance carries the
    # rotation signal (r=+0.63 vs true net rotation) but the bump does not respond (r=+0.004).
    # Slowing the cells removes the alternation but drops firing to 0.027/tick, ~1500x too weak to move
    # the bump. So the drive must be STRONG and UNILATERAL within a stroke: let the better-driven side
    # suppress the other, so only the NET direction pushes. (bank_wta is INTRA-bank, between threshold
    # ranks of one direction -- a different mechanism.)
    if w_lr_inhib>0:
        for i in range(NR):
            for p in range(NP):
                for src,dst in ((CR[i][p],CL[i][p]),(CL[i][p],CR[i][p])):
                    sid=_next(dst)
                    sy.append(k.syn(dst,sid,-abs(w_lr_inhib),1))
                    conns.append(k.conn(src,dst,sid,Tt(src)))
    if bank_wta>0:
        for i in range(NR):
            for p in range(NP-1):
                for lo,hi in ((CL[i][p],CL[i][p+1]),(CR[i][p],CR[i][p+1])):
                    sid=_next(lo)
                    sy.append(k.syn(lo,sid,-abs(bank_wta),1))
                    conns.append(k.conn(hi,lo,sid,Tt(hi)))
    # ---- DELTA-7 -> P-EN (the paper's target), added AFTER the shift cells exist ----
    # Pisokas, Heinze & Webb 2020 (eLife 9:e53985): Delta-7 projects to P-EN and P-EG, NOT to E-PG,
    # and Delta-7 neurons inhibit each other (global and uniform in Drosophila, a weakening subset in
    # locust). Inhibiting the SHIFT cells is what stops P-EN units far from the bump from firing --
    # the runaway that delocalised the bump the moment it moved. Inhibiting E-PG (what I built first)
    # fights the bump itself instead.
    # ORDER MATTERS: the shift cells' ring and velocity inputs are hard-coded as synapse ids 0 and 1,
    # so any synapse added before they are built collides with them and destroys the AND gate.
    if d7 and d7_to_shift:
        _phi7=np.array([2*np.pi*j/ND7 for j in range(ND7)])
        for i in range(NR):
            for p in range(NP):
                for cid in (CL[i][p],CR[i][p]):
                    for j in range(ND7):
                        w=(w_d7_shift/ND7)*(d7_uniform+(1.0-d7_uniform)*(1.0-np.cos(PHI[i]-_phi7[j]))/2.0)
                        sid=_next(cid)
                        sy.append(k.syn(cid,sid,float(w),1))
                        conns.append(k.conn(D7[j],cid,sid,Tt(D7[j])))
        for j in range(ND7):                       # Delta-7 <-> Delta-7 mutual inhibition
            sid=64
            for j2 in range(ND7):
                if j2==j: continue
                sid+=1
                sy.append(k.syn(D7[j],sid,w_d7_d7/ND7,1))
                conns.append(k.conn(D7[j2],D7[j],sid,Tt(D7[j2])))

    # ---- PATH INTEGRATION (the NEURAL readout of the compass) ----
    # PG[i] = speed-gated bump readout: AND-gate(bump R[i], forward-speed ext). syn0=ring, syn1=speed.
    # Its spike rate ~ speed x bump[i], so the bump is read out ONLY while moving forward.
    # The speed synapse must be WEAKER than the ring one: with both at w_pg the speed term alone reached
    # w_pg*spd_max = 2.1 > r_pg and PG fired on all 36 columns at once, which flattened CD's cosine tuning
    # and left the CPU4 ladders empty. The standalone test never saw it because it drives speed=0.6.
    w_pg_s = w_pg if w_pg_s is None else w_pg_s
    for i in range(NR):
        ne.append(k.neuron(PG[i],r=r_pg,c=2,lam=2))
        sy.extend([k.syn(PG[i],0,w_pg,1),k.syn(PG[i],1,w_pg_s,1),k.term(PG[i],Tt(PG[i]))])
        conns.append(k.conn(RING[i],PG[i],0,Tt(RING[i]))); ex.append(k.ext(PG[i],1))
    # XACC/YACC = analog home-vector accumulators. Never spike (r huge); near-lossless (lam_acc huge) so the
    # membrane S integrates its input. FIXED cos/sin synaptic weights from PG => S_X=integral(speed*cos(head)),
    # S_Y=integral(speed*sin(head)) = the running displacement vector. All in wiring, no python arithmetic.
    ne.append(k.neuron(XACC,r=1e9,c=2,lam=lam_acc)); ne.append(k.neuron(YACC,r=1e9,c=2,lam=lam_acc))
    for i in range(NR):
        sy.append(k.syn(XACC,i,k_pi*float(np.cos(PHI[i])),1)); conns.append(k.conn(PG[i],XACC,i,Tt(PG[i])))
        sy.append(k.syn(YACC,i,k_pi*float(np.sin(PHI[i])),1)); conns.append(k.conn(PG[i],YACC,i,Tt(PG[i])))
    sy.append(k.term(XACC,Tt(XACC))); sy.append(k.term(YACC,Tt(YACC)))
    return ne,sy,conns,ex

def build(**kw):
    """Compile the central complex to a network file. (parts() returns the raw ckit lists so downstream
    layers -- CPU4 accumulators, CPU1 steering -- can be appended before compiling; see cx_navigator.py)"""
    return k.build(*parts(**kw))

class Compass:
    def __init__(self, tref_upper=2.0, **bkw):
        self.net,self.core=k.load(build(**bkw)); self.nb={i:u for i,u in self.net.network.neurons.items()}
        # t_ref = c * num_inputs, so adding synapses silently gates a population off. AIFAgent3D
        # overrides this; the standalone Compass did not, which invalidated a whole transfer sweep
        # (the full-ring Delta-7 profile pushed ring fan-in to 25 -> t_ref 50 -> no bump, and I read
        # that as "the profile does not work"). Keep the two harnesses equivalent.
        for nid in list(RING)+[n for i in range(NR) for p in range(NP) for n in (CL[i][p],CR[i][p])]+list(D7):
            if nid in self.nb: self.nb[nid].upper_t_ref_bound=tref_upper
        self.seedsyn=SEEDSYN
    def seed(self, idx, ticks=35, amp=4.0):
        ss={(idx+o)%NR for o in range(-NB,NB+1)}
        for _ in range(ticks):
            for i,r in enumerate(RING): self.net.set_external_input(r,self.seedsyn, amp if i in ss else 0.0)
            self.core.do_tick()
        for r in RING: self.net.set_external_input(r,self.seedsyn,0.0)
    def _drive(self, ccw, cw, speed=0.0):
        for i in range(NR):
            for p in range(NP):
                self.net.set_external_input(CL[i][p],1,ccw); self.net.set_external_input(CR[i][p],1,cw)
            self.net.set_external_input(PG[i],1,speed)
    def step(self, ccw=0.0, cw=0.0, speed=0.0, sub=1):
        for _ in range(sub):
            self._drive(ccw,cw,speed); self.core.do_tick()
    def home_vector(self):
        # read the analog accumulators (membrane S). Position vector = (x,y); vector back home = its negation.
        x=float(self.nb[XACC].S); y=float(self.nb[YACC].S)
        return x, y, float(np.degrees(np.arctan2(-y,-x))), float(np.hypot(x,y))
    def heading(self, win=12, ccw=0.0, cw=0.0, speed=0.0):
        # NOTE: external input is CONSUMED every tick (network.run_tick clears it), so the drive MUST be
        # re-applied on every measurement tick too -- measuring without drive stalls the traveling wave and
        # the bump re-pins to its column (looks like the compass is "stuck" one column from where it began).
        acc=np.zeros(NR)
        for _ in range(win):
            self._drive(ccw,cw,speed); self.core.do_tick()
            for i,r in enumerate(RING): acc[i]+=self.nb[r].O>0
        if acc.sum()==0: return None,0
        vx=float(np.sum(acc*np.cos(PHI))); vy=float(np.sum(acc*np.sin(PHI)))
        return float(np.arctan2(vy,vx)), int((acc>0).sum())

def _unwrap(prev,h):
    d=np.degrees(h)-np.degrees(prev)
    while d>180:d-=360
    while d<-180:d+=360
    return d

if __name__=="__main__":
    # BIO-FAITHFUL vestibular heading compass. Excitation-only P-EN shift + normalising GI => the bump
    # NEVER dies: it holds, integrates angular velocity, and under fast spin it BLURS (disorientation) but
    # persists and RE-SHARPENS when you stop -- exactly how real head-direction systems behave.
    np.random.seed(0)
    print(f"CENTRAL-COMPLEX COMPASS (NR={NR} ring, excitation-only P-EN shift + normalising GI):")
    cp=Compass(); cp.seed(0); prev=cp.heading()[0]; cum=0.0
    def block(ccw,cw,n,lbl,note):
        global prev,cum,mn,mx
        mn,mx=99,0
        for _ in range(n):
            h,a=cp.heading(win=14,ccw=ccw,cw=cw); mn,mx=min(mn,a),max(mx,a)
            if h is not None and prev is not None: cum+=_unwrap(prev,h)
            if h is not None: prev=h
        print(f"  {lbl:16s}: heading={cum:+5.0f}deg  bump {mn}-{mx} neurons   {note}")
    block(0,0,6,      "hold",           "[holds heading, tight bump]")
    block(0.9,0,14,   "slow CCW turn",  "[integrates, stays sharp]")
    block(1.8,0,10,   "FAST spin",      "[BLURS wide but PERSISTS = disorientation]")
    block(0,0,10,     "stop / observe", "[RE-SHARPENS]")
    block(0,0.9,14,   "slow CW turn",   "[reverses heading back down, survives]")
    print("@@@COMPASS DONE@@@")
