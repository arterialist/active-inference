"""VISUAL CORTEX — a GENERIC retinotopic hierarchy for the 3D world (retina -> ON/OFF -> V1 -> V2 -> AZ).

Design rule: NO task-specific cells. There is no "beacon cell", no "sky cell", no "food cell". Every layer
computes a generic image property over a RETINOTOPIC map (azimuth x elevation), exactly as an early visual
system does, and anything task-relevant is only a READOUT of that map further downstream:

  PR[a,e]      photoreceptors on a retinotopic grid. The one transducer: luminance -> current.
  ON/OFF[a,e]  centre-surround opponency built by WIRING (centre excitatory, 4 neighbours inhibitory).
               ON = brighter-than-surround, OFF = darker-than-surround. Uniform fields vanish; only local
               CONTRAST survives -- illumination-invariant, rather than tuned to absolute brightness.
  V1[o,a,e]    oriented simple cells: NORI orientations, each a fixed elongated +/- kernel stamped over
               ON/OFF in a local patch (data-free, Gabor-like -- no learning, no labels).
  V2[a,e]      complex cells: pool V1 across ALL orientations and a local neighbourhood -> orientation- and
               phase-tolerant "there is structure here" (the standard simple->complex step).
  AZ[a]        a V2 column pooled over elevation: "how much structure at this bearing". Still generic --
               it is just the retinotopic map read along one axis.
  CH_RG/CH_BY  the two standard chromatic opponent channels on the same grid (generic colour opponency,
               not "amber food" detectors).

WHY THIS GENERALISES: a pillar, a food blob and the beacon are all just structure at some (azimuth,
elevation) with some chromatic signature. Downstream circuits (compass anchoring, object approach)
subscribe to the SAME map instead of each owning a private detector, so a new object in the world needs no
new visual machinery.

Run standalone:  python visual_cortex.py   (does the generic map localise real objects in rendered pixels?)
"""
import sys, importlib.util, numpy as np
sys.path.insert(0,"/Users/arterialist/Projects/agi-research/neuron-model")
from paula_agent import ckit as k
_w=importlib.util.spec_from_file_location("w3","/Users/arterialist/Projects/agi-research/active-inference/simulations/active_inference/world3d.py")
w3=importlib.util.module_from_spec(_w); _w.loader.exec_module(w3)

def T(n): return 60000+n
NAZ, NEL = 20, 4                      # retinotopic grid (azimuth x elevation)
NV1AZ, NV1EL = NAZ-1, NEL-1           # V1/V2 grid. STRIDE 1 = densely OVERLAPPING receptive fields, as
                                      # in real V1 (tiling with stride 2 halves angular resolution, and the
                                      # readout error was pinned at exactly one column because of it).
NORI = 4                              # orientations: vertical, /, horizontal, \
HFOV = 2*np.arctan((72/20)*np.tan(np.radians(52.0)/2))    # from the eye's fovy+aspect (=120.6deg)

def _id(base,a,e,n_e): return base+a*n_e+e
def PR(a,e):    return _id(70000,a,e,NEL)
def ON(a,e):    return _id(71000,a,e,NEL)
def OFF(a,e):   return _id(72000,a,e,NEL)
def V1(o,a,e):  return 73000+o*NV1AZ*NV1EL+a*NV1EL+e
def V2(a,e):    return _id(74000,a,e,NV1EL)
def AZ(a):      return 75000+a
def CH_RG(a,e): return _id(76000,a,e,NEL)
def CH_BY(a,e): return _id(77000,a,e,NEL)
def EMD_P(a):   return 78000+a          # Reichardt correlator, progressive (front-to-back) motion
def EMD_R(a):   return 78100+a          # Reichardt correlator, regressive
def SAL(a):     return 79000+a          # salience map with lateral inhibition (attention / WTA)
NHS = 4                                 # cells per direction in the wide-field motion integrators
def HS_CCW(i): return 79700+i           # lobula-plate tangential cells (fly HS): pool the EMD array
def HS_CW(i):  return 79710+i           # over the WHOLE field -> an estimate of self-rotation (yaw)
NORM_R = 79500                          # normalisation pool for the retina/ON-OFF stage
NORM_V = 79501                          # normalisation pool for V1/V2

def az_of(a, n=NAZ):
    """Egocentric azimuth of retinotopic column a (+ = left). A pinhole camera is TANGENT-distorted, so a
    linear column->angle map compresses eccentric bearings (a true +45deg reads as +26deg). This inverts it."""
    u=2.0*(0.5-(a+0.5)/n)
    return float(np.arctan(u*np.tan(HFOV/2)))

def _ori_kernel(o):
    """fixed data-free oriented kernels over a 2x2 patch: (da,de)->sign"""
    return {0:{(0,0):+1,(0,1):+1,(1,0):-1,(1,1):-1},     # vertical edge (contrast across azimuth)
            1:{(0,0):+1,(1,1):-1},                        # diagonal /
            2:{(0,0):+1,(1,0):+1,(0,1):-1,(1,1):-1},      # horizontal edge (contrast across elevation)
            3:{(0,1):+1,(1,0):-1}}[o]                     # diagonal \

def parts(w_pr=1.5, r_pr=1.2,
          w_cen=5.0, w_sur=-1.1, r_onoff=0.8, lam_onoff=3,
          w_v1=4.0, r_v1=0.55, lam_v1=10,
          w_v2=3.5, r_v2=0.65, lam_v2=10,
          w_az=3.5, r_az=0.65, lam_az=10,
          w_chr=1.6, r_chr=1.2,
          w_norm_in=0.16, w_norm_out=-0.25, r_norm=1.4,
          # d_emd sets the Reichardt VELOCITY TUNING (peak response near delta_phi/d_emd). d_emd=4 was
          # tuned to fast flow and left a large DC OFFSET in the pooled HS signed output (zero-omega
          # bias -0.080), which is what made HS look sign-inverted at low turn rates -- HS_CW exceeded
          # HS_CCW during CCW rotation purely from the offset. Measured per-tick against a time-varying
          # omega, d_emd=16 removes the offset (bias -0.010) AND raises low-omega discriminability
          # (d' at |omega| 0.05-0.15: 0.24 -> 0.58) with NO detection floor in any omega bin, per-tick
          # r(omega,HS) 0.433 -> 0.575. d_emd 24/32 score higher r but open holes in the mid range.
          w_emd=1.0, r_emd=0.7, d_emd=16, lam_emd=2, w_emd_opp=-1.2,
          w_hs=1.6, r_hs_lo=0.35, r_hs_hi=2.2, lam_hs=6, w_hs_opp=-0.7, hs_mod=0.0, w_hs_anti=0.0, hs_prof=0.0,
          w_sal=5.0, r_sal=0.5, lam_sal=10, w_sal_lat=-0.65,
          ne=None, sy=None, conns=None, ex=None):
    ne=[] if ne is None else ne; sy=[] if sy is None else sy
    conns=[] if conns is None else conns; ex=[] if ex is None else ex
    # ---- RETINA: photoreceptors + chromatic opponent channels (all transducer port 0)
    for a in range(NAZ):
        for e in range(NEL):
            for nid,w,r in ((PR(a,e),w_pr,r_pr),(CH_RG(a,e),w_chr,r_chr),(CH_BY(a,e),w_chr,r_chr)):
                ne.append(k.neuron(nid,r=r,c=2,lam=3))
                sy.append(k.syn(nid,0,w,1)); ex.append(k.ext(nid,0)); sy.append(k.term(nid,T(nid)))
    # ---- ON/OFF: centre-surround by WIRING -> local contrast (uniform illumination cancels)
    for a in range(NAZ):
        for e in range(NEL):
            nb4=[(a+d,e) for d in (-1,1) if 0<=a+d<NAZ]+[(a,e+d) for d in (-1,1) if 0<=e+d<NEL]
            for nid,cs in ((ON(a,e),+1),(OFF(a,e),-1)):
                ne.append(k.neuron(nid,r=r_onoff,c=2,lam=lam_onoff)); j=0
                sy.append(k.syn(nid,j,cs*w_cen,1)); conns.append(k.conn(PR(a,e),nid,j,T(PR(a,e)))); j+=1
                # RF normalisation at the borders: an edge cell has only 2-3 neighbours, so an unscaled
                # surround leaves it under-inhibited and it fires spuriously -- which made the salience map
                # attend to the image border no matter where the agent looked. Scale to a constant total.
                wsur=cs*w_sur*4.0/len(nb4)
                for (na,nee) in nb4:
                    sy.append(k.syn(nid,j,wsur,1)); conns.append(k.conn(PR(na,nee),nid,j,T(PR(na,nee)))); j+=1
                sy.append(k.term(nid,T(nid)))
    # ---- V1: oriented simple cells (fixed kernels on ON/OFF over a 2x2 retinal patch)
    for o in range(NORI):
        K=_ori_kernel(o)
        for a in range(NV1AZ):
            for e in range(NV1EL):
                nid=V1(o,a,e); ne.append(k.neuron(nid,r=r_v1,c=2,lam=lam_v1)); j=0
                for (da,de),wk in K.items():
                    ra,re=a+da,e+de
                    if ra>=NAZ or re>=NEL: continue
                    src=ON(ra,re) if wk>0 else OFF(ra,re)
                    sy.append(k.syn(nid,j,w_v1,1)); conns.append(k.conn(src,nid,j,T(src))); j+=1
                sy.append(k.term(nid,T(nid)))
    # ---- V2: complex cells -- pool V1 over ALL orientations + neighbouring positions
    for a in range(NV1AZ):
        for e in range(NV1EL):
            nid=V2(a,e); ne.append(k.neuron(nid,r=r_v2,c=2,lam=lam_v2)); j=0
            for o in range(NORI):
                for da in (-1,0,1):
                    if 0<=a+da<NV1AZ:
                        s=V1(o,a+da,e)
                        sy.append(k.syn(nid,j,w_v2,1)); conns.append(k.conn(s,nid,j,T(s))); j+=1
            sy.append(k.term(nid,T(nid)))
    # ---- DIVISIVE NORMALISATION (Carandini & Heeger's canonical cortical computation): a pool neuron
    # sums the whole layer and inhibits every unit in it, so each response is scaled by TOTAL activity.
    # This is what makes the map contrast-INVARIANT and stops the saturation failure mode (threshold below
    # baseline -> every unit fires -> lateral inhibition cancels everything). It is the same trick as the
    # heading ring's global inhibitor, applied to vision.
    ne.append(k.neuron(NORM_R,r=r_norm,c=1,lam=4)); j=0
    for a in range(NAZ):
        for e in range(NEL):
            for src in (ON(a,e),OFF(a,e)):
                sy.append(k.syn(NORM_R,j,w_norm_in,1)); conns.append(k.conn(src,NORM_R,j,T(src))); j+=1
    sy.append(k.term(NORM_R,T(NORM_R)))
    ne.append(k.neuron(NORM_V,r=r_norm,c=1,lam=4)); j=0
    for a in range(NV1AZ):
        for e in range(NV1EL):
            sy.append(k.syn(NORM_V,j,w_norm_in*2,1)); conns.append(k.conn(V2(a,e),NORM_V,j,T(V2(a,e)))); j+=1
    sy.append(k.term(NORM_V,T(NORM_V)))
    # ---- EMD: Hassenstein-Reichardt correlators. AND(this receptor NOW, neighbour DELAYED) -> direction-
    # selective motion. Its magnitude is MOTION PARALLAX: during translation a near object sweeps fast and
    # a distal one barely moves, so this is the principled near/far test -- the cue that says which visual
    # feature is stable enough to anchor a compass to.
    # A single correlator ARM is NOT direction selective -- measured: spinning the body CCW vs CW gave
    # EMD_P-EMD_R = +1 vs +6, i.e. noise. Two things were wrong and both are textbook:
    #   (1) lam_emd=12 with d_emd=4: the membrane held input three times longer than the delay it was
    #       meant to discriminate, so both arms integrated the same thing. The membrane must be
    #       TRANSIENT relative to the delay. lam=2, and then the AND-gate inequality has to hold:
    #       one input alone w/lam < r (1.0/2 = 0.5 < 0.7) and coincidence 2w/lam > r (1.0 > 0.7).
    #   (2) no OPPONENCY. Hassenstein & Reichardt (1956) is a SUBTRACTION of two mirror-image arms;
    #       direction selectivity lives in that subtraction, not in either arm. In the fly it is T4/T5
    #       converging with opposite sign on the lobula plate (Borst, Haag & Reiff 2010). Here the two
    #       arms already exist as EMD_P/EMD_R, so opponency is mutual inhibition between them.
    for a in range(NAZ):
        for nid,other in ((EMD_P(a),(a+1)%NAZ),(EMD_R(a),(a-1)%NAZ)):
            ne.append(k.neuron(nid,r=r_emd,c=2,lam=lam_emd)); j=0
            sy.append(k.syn(nid,j,w_emd,1)); conns.append(k.conn(ON(a,1),nid,j,T(ON(a,1)))); j+=1
            sy.append(k.syn(nid,j,w_emd,d_emd)); conns.append(k.conn(ON(other,1),nid,j,T(ON(other,1)))); j+=1
            sy.append(k.term(nid,T(nid)))
    for a in range(NAZ):                                   # the opponent subtraction
        sy.append(k.syn(EMD_P(a),2,w_emd_opp,1)); conns.append(k.conn(EMD_R(a),EMD_P(a),2,T(EMD_R(a))))
        sy.append(k.syn(EMD_R(a),2,w_emd_opp,1)); conns.append(k.conn(EMD_P(a),EMD_R(a),2,T(EMD_P(a))))
    # ---- HS: WIDE-FIELD MOTION INTEGRATORS (lobula plate tangential cells).
    # A single EMD is a local motion detector; pooling the whole array turns local motion into an
    # estimate of SELF-ROTATION, because a yaw sweeps the entire visual field coherently in one
    # direction while translation produces a non-uniform flow field (near objects sweep fast, distant
    # ones barely move). This is the fly's HS/horizontal-system pathway (Hausen 1982; Krapp &
    # Hengstenberg 1996; Borst & Haag 2002), and it is how a fly's compass gets a VISUAL self-motion
    # signal alongside the proprioceptive one (Green, Vijayan & Maimon 2017; Turner-Evans 2017).
    # Measured direction mapping on this body: yaw CCW -> EMD_R dominates, yaw CW -> EMD_P dominates.
    # Thresholds are staggered across the population so the pooled output is a GRADED rate code of
    # turn speed rather than a binary "turning" flag -- the shift cells need a graded drive.
    hthr=np.linspace(r_hs_lo,r_hs_hi,NHS); _HSJ={}
    for i in range(NHS):
        # WITHIN-CELL OPPONENCY (fly HS): a real HS cell is EXCITED by its preferred direction and
        # INHIBITED by the anti-preferred one in the SAME field (Borst/Haag). The subtraction happens
        # BEFORE the cell's threshold, which is what gives common-mode rejection. HS_CCW/HS_CW here
        # pooled only their preferred EMD arm, so when body oscillation drives BOTH arms both pools
        # fired and the opponent difference collapsed to <4% of total (measured: CCW 1273 / CW 1187,
        # difference +86). HS-to-HS inhibition (w_hs_opp) acts AFTER threshold -- too late.
        for nid,src,anti in ((HS_CCW(i),EMD_R,EMD_P),(HS_CW(i),EMD_P,EMD_R)):
            ne.append(k.neuron(nid,r=float(hthr[i]),c=2,lam=lam_hs)); j=0
            for a in range(NAZ):
                # ROTATION MATCHED FILTER. Forward translation gives RADIAL flow from the focus of
                # expansion: near-zero HORIZONTAL flow straight ahead and straight behind, maximal at
                # the sides. Yaw rotation gives UNIFORM horizontal flow at EVERY azimuth. So weighting
                # by |cos(azimuth)| (heavy fore/aft, light lateral) passes rotation and rejects
                # translation. Uniform pooling (hs_prof=0) is a matched filter for nothing, and cannot
                # recover the signal downstream because the per-azimuth EMD is RECTIFIED: where
                # translation flow opposes rotation flow the NET flow reverses and the rectified EMD
                # reports the reversal. Measured: turn +0.3 with speed 1.0 -> EMD sel -5.8% (INVERTED).
                _ang=2*np.pi*a/NAZ
                _wa=w_hs*((1.0-hs_prof)+hs_prof*abs(np.cos(_ang)))
                sy.append(k.syn(nid,j,_wa,1)); conns.append(k.conn(src(a),nid,j,T(src(a)))); j+=1
                if w_hs_anti:
                    _wi=w_hs_anti*((1.0-hs_prof)+hs_prof*abs(np.cos(_ang)))
                    sy.append(k.syn(nid,j,-abs(_wi),1)); conns.append(k.conn(anti(a),nid,j,T(anti(a)))); j+=1
            _HSJ[nid]=j                                       # remember the next FREE synapse index
            sy.append(k.term(nid,T(nid),mod=[hs_mod,0.0]))   # gain-control channel
    for i in range(NHS):                       # opponency again at the wide-field stage
        for i2 in range(NHS):
            # was hardcoded NAZ+i2 -- with w_hs_anti the excitatory/inhibitory pairs occupy 0..2*NAZ-1,
            # so NAZ+i2 landed ON TOP of anti synapses and silently overwrote them (population went
            # SILENT at every w_hs_anti, even 0.05). Use the running free index instead.
            ja=_HSJ[HS_CCW(i)]; _HSJ[HS_CCW(i)]=ja+1
            sy.append(k.syn(HS_CCW(i),ja,w_hs_opp,1))
            conns.append(k.conn(HS_CW(i2),HS_CCW(i),ja,T(HS_CW(i2))))
            jb=_HSJ[HS_CW(i)]; _HSJ[HS_CW(i)]=jb+1
            sy.append(k.syn(HS_CW(i),jb,w_hs_opp,1))
            conns.append(k.conn(HS_CCW(i2),HS_CW(i),jb,T(HS_CCW(i2))))
    # ---- SALIENCE MAP + WTA: V2 drives it, normalisation divides it, and lateral inhibition across
    # azimuth selects ONE winner = the attended bearing. Averaging a map that reports ALL structure blurs
    # objects together; selection is what turns a feature map into something you can act on.
    for a in range(NV1AZ):
        nid=SAL(a); ne.append(k.neuron(nid,r=r_sal,c=2,lam=lam_sal)); j=0
        for e in range(NV1EL):
            sy.append(k.syn(nid,j,w_sal,1)); conns.append(k.conn(V2(a,e),nid,j,T(V2(a,e)))); j+=1
        sy.append(k.syn(nid,j,w_norm_out,1)); conns.append(k.conn(NORM_V,nid,j,T(NORM_V))); j+=1
        for b in range(NV1AZ):
            if abs(b-a)>1:
                for e in range(NV1EL):
                    sy.append(k.syn(nid,j,w_sal_lat,1)); conns.append(k.conn(V2(b,e),nid,j,T(V2(b,e)))); j+=1
        sy.append(k.term(nid,T(nid)))
    # ---- AZ: V2 pooled over elevation = "structure at this bearing"
    for a in range(NV1AZ):
        nid=AZ(a); ne.append(k.neuron(nid,r=r_az,c=2,lam=lam_az)); j=0
        for e in range(NV1EL):
            s=V2(a,e); sy.append(k.syn(nid,j,w_az,1)); conns.append(k.conn(s,nid,j,T(s))); j+=1
        sy.append(k.term(nid,T(nid)))
    return ne,sy,conns,ex

def drive_from_image(net, img_rgb, gain=2.0, chroma_gain=3.0):
    """TRANSDUCER: rendered pixels -> photoreceptor currents (the only non-neural step in vision)."""
    h,w,_=img_rgb.shape
    lum=img_rgb.mean(axis=2)
    cols=np.array_split(np.arange(w),NAZ); rows=np.array_split(np.arange(h),NEL)
    for a,ci in enumerate(cols):
        for e,ri in enumerate(rows):
            patch=img_rgb[np.ix_(ri,ci)]
            net.set_external_input(PR(a,e),0, float(lum[np.ix_(ri,ci)].max())*gain)
            R,G,B=(float(patch[:,:,c].mean()) for c in range(3))
            net.set_external_input(CH_RG(a,e),0, max(0.0,R-G)*chroma_gain)
            net.set_external_input(CH_BY(a,e),0, max(0.0,B-(R+G)/2)*chroma_gain)

def population_azimuth(counts, n=None):
    n=len(counts) if n is None else n
    if counts.sum()<=0: return None
    A=np.array([az_of(a,n) for a in range(n)])
    return float(np.arctan2(float(np.sum(counts*np.sin(A))),float(np.sum(counts*np.cos(A)))))

def upper_azimuth(v2_counts):
    """Azimuth of structure in the UPPER elevation band. Still just reading the retinotopic map -- but the
    upper band is where DISTAL cues sit (a near object's bearing shifts as you translate, a far one's does
    not), which is what makes it usable as a compass anchor."""
    return population_azimuth(v2_counts[:,0])   # band 0 = TOP of the image (row 0 is the top)

if __name__=="__main__":
    np.random.seed(0)
    ne,sy,conns,ex=parts()
    net,core=k.load(k.build(ne,sy,conns,ex)); nb={i:u for i,u in net.network.neurons.items()}
    print(f"GENERIC VISUAL CORTEX: retina {NAZ}x{NEL} -> ON/OFF -> V1 {NORI}ori x {NV1AZ}x{NV1EL} -> V2 -> AZ")
    print(f"  {len(ne)} neurons, FOV {np.degrees(HFOV):.0f}deg, no task-specific cells")
    import mujoco
    world=w3.World3D(seed=1); errs=[]
    for yaw_deg in [35,20,5,-10,55,70]:
        world.data.qpos[world.jyaw]=np.radians(yaw_deg); mujoco.mj_forward(world.model,world.data)
        img=world.retina()
        v2=np.zeros((NV1AZ,NV1EL)); rg=0; on=0
        for _ in range(16):
            drive_from_image(net,img); core.do_tick()
            for a in range(NV1AZ):
                for e in range(NV1EL): v2[a,e]+= nb[V2(a,e)].O>0
            on+=sum(1 for a in range(NAZ) for e in range(NEL) if nb[ON(a,e)].O>0)
            rg+=sum(1 for a in range(NAZ) for e in range(NEL) if nb[CH_RG(a,e)].O>0)
        up=upper_azimuth(v2); true_b=world.sun_bearing()
        if up is None:
            print(f"  yaw={yaw_deg:+4d}: no upper-field structure (beacon true {np.degrees(true_b):+.0f}deg)  ON={on}")
        else:
            e2=np.degrees(np.arctan2(np.sin(up-true_b),np.cos(up-true_b))); errs.append(abs(e2))
            print(f"  yaw={yaw_deg:+4d}: upper structure {np.degrees(up):+6.1f}deg | beacon {np.degrees(true_b):+6.1f}deg | ERR={e2:+6.1f}  ON={on} V2={int((v2>0).sum())} RG={rg}")
    if errs: print(f"  mean |error| = {np.mean(errs):.1f}deg  (generic map, zero beacon-specific wiring)")
    print("@@@VISION DONE@@@")


# ---------------------------------------------------------------------------------------------------
# THE DYNAMIC-RANGE FIX. In PAULA a neuron's refractory bound is upper_t_ref_bound = c * num_inputs, so a
# POOLING cell is crippled by its own fan-in: V2 pools 12 inputs -> t_ref up to 24 ticks -> a ceiling of
# ~0.04 spikes/tick. That is why every deep layer here was starved and needed hand-tuned thresholds. The
# mushroom body hit the same wall and solved it the same way: override the bounds after load so each cell
# has a usable firing-RATE range and codes intensity gradedly instead of teetering on threshold.
VIS_IDS=lambda: ([PR(a,e) for a in range(NAZ) for e in range(NEL)]
               + [ON(a,e) for a in range(NAZ) for e in range(NEL)]
               + [OFF(a,e) for a in range(NAZ) for e in range(NEL)]
               + [CH_RG(a,e) for a in range(NAZ) for e in range(NEL)]
               + [CH_BY(a,e) for a in range(NAZ) for e in range(NEL)]
               + [V1(o,a,e) for o in range(NORI) for a in range(NV1AZ) for e in range(NV1EL)]
               + [V2(a,e) for a in range(NV1AZ) for e in range(NV1EL)]
               + [AZ(a) for a in range(NV1AZ)] + [SAL(a) for a in range(NV1AZ)]
               + [EMD_P(a) for a in range(NAZ)] + [EMD_R(a) for a in range(NAZ)]
               + [HS_CCW(i) for i in range(NHS)] + [HS_CW(i) for i in range(NHS)]
               + [NORM_R, NORM_V])

class Vision:
    """The visual cortex as a usable component: build, load, fix the refractory bounds, then see()."""
    def __init__(self, tref_upper=2.0, tref_lower=1.0, **kw):
        ne,sy,conns,ex=parts(**kw)
        self.net,self.core=k.load(k.build(ne,sy,conns,ex))
        self.nb={i:u for i,u in self.net.network.neurons.items()}
        for nid in VIS_IDS():
            if nid in self.nb:
                self.nb[nid].upper_t_ref_bound=tref_upper
                self.nb[nid].lower_t_ref_bound=tref_lower
                self.nb[nid].t_ref=tref_upper
    def see(self, img, ticks=16):
        """run the cortex on one frame; returns the population activity of each stage"""
        out=dict(on=np.zeros((NAZ,NEL)), v2=np.zeros((NV1AZ,NV1EL)), sal=np.zeros(NV1AZ),
                 emd=np.zeros(NAZ), rg=np.zeros((NAZ,NEL)))
        for _ in range(ticks):
            drive_from_image(self.net,img); self.core.do_tick()
            for a in range(NAZ):
                for e in range(NEL):
                    out["on"][a,e]+= self.nb[ON(a,e)].O>0
                    out["rg"][a,e]+= self.nb[CH_RG(a,e)].O>0
                out["emd"][a]+= (self.nb[EMD_P(a)].O>0)+(self.nb[EMD_R(a)].O>0)
            for a in range(NV1AZ):
                out["sal"][a]+= self.nb[SAL(a)].O>0
                for e in range(NV1EL): out["v2"][a,e]+= self.nb[V2(a,e)].O>0
        return out
    def attended_azimuth(self, act):
        """the selected bearing = peak of the salience map (WTA), read as a population vector"""
        return population_azimuth(act["sal"], NV1AZ)
