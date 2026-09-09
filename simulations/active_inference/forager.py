"""Active inference in a COMPLEX, OPEN-ENDED world — running ENTIRELY on PAULA spiking neurons.

The T-maze (`tmaze_agent.py`) proved the principle; this puts the same active-inference computation into
the continuous MuJoCo foraging world, on PAULA neurons only. There is no numpy decision logic in the loop:

  BELIEF / world model  -> the MUSHROOM BODY (learned odour->value). For the currently-smelled odour it
                           reports approach-evidence (app) and avoid-evidence (avo). A NOVEL odour gives
                           app=avo=0 -> the agent does not yet know its value.
  EPISTEMIC value       -> an uncertainty neuron U (tonic, inhibited by app+avo). It fires only when the
                           odour is UNRESOLVED, and drives INVESTIGATE -> the agent approaches novel food
                           to taste it and LEARN. Curiosity as information-gain, not a hand-set drive.
  PRAGMATIC value       -> APPROACH is driven by app (believe-nutritious), AVOID by avo (believe-toxic).
  survival              -> FLEE (predator) and HOME (path-integrated by the central complex) compete too.
  ACTION                -> a spiking winner-take-all over {investigate, approach, avoid, flee, home}.

Learning closes the loop: investigating a novel food and tasting it teaches the mushroom body, so its
belief resolves and the agent switches from epistemic (investigate) to pragmatic (approach / avoid) --
open-ended: it keeps meeting new food, resolving uncertainty, and foraging on what it has learned.

Run standalone:  python -m simulations.active_inference.forager
"""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from simulations.paula_loader import ensure_paula_available
ensure_paula_available()
from paula_agent import ckit as k
from simulations.organism import mushroom_body as MB
from simulations.organism import body as BODY
from simulations.organism import central_complex as CXM

# WTA neuron ids
U = 1
INVEST, APPROACH, AVOID, FLEE, HOME, WANDER = 2, 3, 4, 5, 6, 7
ACT = [INVEST, APPROACH, AVOID, FLEE, HOME, WANDER]
ANAME = {INVEST: 'investigate', APPROACH: 'approach', AVOID: 'avoid', FLEE: 'flee', HOME: 'home', WANDER: 'wander'}
def T(n): return 900 + n


def build_wta(w_tonic=1.4, w_uinh=-2.6, w_epi=3.4, w_wta=-5.0, w_base=0.5):
    """PAULA winner-take-all: U (uncertainty) + 6 action neurons. U fires only for a novel/unresolved
    odour (tonic minus the mushroom body's app+avo evidence) and drives INVESTIGATE (epistemic).
    APPROACH/AVOID/FLEE/HOME get their drives as external currents; WANDER is the tonic default (the
    behaviour when NOTHING is driven — so without the epistemic drive the agent never investigates)."""
    ne = []; sy = []; conns = []; exts = []
    ne.append(k.neuron(U, r=0.6, c=2, lam=5))
    sy += [k.syn(U, 0, w_tonic, 1), k.syn(U, 1, w_uinh, 1), k.syn(U, 2, w_uinh, 1), k.term(U, T(U))]
    exts += [k.ext(U, 0), k.ext(U, 1), k.ext(U, 2)]   # syn0 tonic, syn1 -app, syn2 -avo (projections)
    w0 = {INVEST: w_epi, WANDER: w_base}
    for a in ACT:
        ne.append(k.neuron(a, r=0.6, c=2, lam=5))
        sy.append(k.syn(a, 0, w0.get(a, 1.0), 1))                      # syn0: EFE drive
        if a == INVEST: conns.append(k.conn(U, a, 0, T(U)))            # investigate <- uncertainty
        else:           exts.append(k.ext(a, 0))                        # others: external current
        for j, o in enumerate([x for x in ACT if x != a]):
            sy.append(k.syn(a, 1 + j, w_wta, 1))                        # mutual inhibition
        sy.append(k.term(a, T(a)))
    for a in ACT:
        for j, o in enumerate([x for x in ACT if x != a]):
            conns.append(k.conn(o, a, 1 + j, T(o)))
    return k.build(ne, sy, conns, exts)


class ForagerBrain:
    """PAULA action-selection brain. select(app, avo, fear, home) returns the winning action name."""
    def __init__(self, w_epi=3.4, **kw):
        self.path = build_wta(w_epi=w_epi, **kw); self.net, self.core = k.load(self.path)
        self.nb = {n: u for n, u in self.net.network.neurons.items()}
    def select(self, app, avo, fear, home, W=16, ev_scale=3.0):
        self.net.reset_simulation(); self.core.state.current_tick = 0; self.net.current_tick = 0
        cnt = {a: 0 for a in ACT}; ucnt = 0
        a_cur, v_cur = ev_scale * app, ev_scale * avo
        for _ in range(W):
            self.net.set_external_input(U, 0, 1.0)             # tonic epistemic baseline
            self.net.set_external_input(U, 1, a_cur)           # inhibited by approach-evidence (w_uinh<0)
            self.net.set_external_input(U, 2, v_cur)           # inhibited by avoid-evidence
            self.net.set_external_input(APPROACH, 0, a_cur)    # pragmatic: believe-good
            self.net.set_external_input(AVOID, 0, v_cur)       # pragmatic: believe-toxic
            self.net.set_external_input(FLEE, 0, 7.0 * fear)   # survival
            self.net.set_external_input(HOME, 0, 6.0 * home)
            self.net.set_external_input(WANDER, 0, 1.0)         # tonic default (weak baseline)
            self.core.do_tick()
            for a in ACT:
                if self.nb[a].O > 0: cnt[a] += 1
            if self.nb[U].O > 0: ucnt += 1
        win = max(cnt, key=cnt.get)
        return ANAME[win], cnt, ucnt


# ---------------- embodied forager in the continuous MuJoCo world ----------------
rng0 = np.random.RandomState(0); P_GOOD = MB.odor_pattern(rng0); P_BAD = MB.odor_pattern(rng0)
ARENA = 5.0

class Forager:
    """The PAULA active-inference organism in the continuous foraging world."""
    def __init__(self, seed=0, nfood=7, epistemic=True):
        self.rng = np.random.RandomState(seed)
        self.foods = [[self.rng.uniform(-ARENA, ARENA), self.rng.uniform(-ARENA, ARENA),
                       1.0 if (j not in (0, 4)) else -1.0, (j not in (0, 4))] for j in range(nfood)]
        self.world = BODY.World([(f[0], f[1], f[2]) for f in self.foods])
        self.mbp, _ = MB.build_mb(seed=1); self.net, self.core, self.nb = MB.load(self.mbp)
        self.cx = CXM.CentralComplex()
        self.brain = ForagerBrain(w_epi=3.4 if epistemic else 0.0)   # ablation: kill the epistemic drive
        self.px, self.py = 4.5, 4.5; self.pph = self.rng.uniform(0, 6.28); self.world.set_predator(self.px, self.py)
        self.energy = 16.0; self.crop = 0; self.was_caught = False
        self.good = 0; self.tox = 0; self.hurt = 0; self.trips = 0
        self.investigate_novel = 0; self.approach_learned = 0; self.avoid_learned = 0
        self.modes = {m: 0 for m in ANAME.values()}; self.t = 0; self.log = []
    def _odor(self, x, y):
        best = None; bd = 1e9
        for fx, fy, s, g in self.foods:
            d = (x - fx) ** 2 + (y - fy) ** 2
            if d < bd: bd = d; best = g
        return (P_GOOD if best else P_BAD) if bd < 40 else np.zeros(MB.N_PN)
    def step(self):
        x, y, yaw = self.world.pose(); cl, cr = self.world.antennae()
        dpred = np.hypot(self.px - x, self.py - y)
        hx, hy = self.cx.home_vector(); d_origin = float(np.hypot(x, y))
        fear = float(np.clip((5.0 - dpred) / 5.0, 0, 1))
        hunger = float(np.clip((18 - self.energy) / 14.0, 0, 1))
        home = 1.0 if (self.crop >= 2 and d_origin > 1.6) else 0.0
        # BELIEF: mushroom body reports app/avo for the currently-smelled odour (0/0 = novel = uncertain)
        _, app, avo = MB.present(self.net, self.core, self.nb, self._odor(x, y))
        mode, cnt, ucnt = self.brain.select(app, avo, fear, home)     # PAULA action selection
        self.modes[mode] += 1
        # motor
        if mode == 'flee':
            ang = np.arctan2(y - self.py, x - self.px); turn = float(np.clip(2.6 * np.sin(ang - yaw), -2.6, 2.6)); speed = 3.6
        elif mode == 'home':
            hd = np.arctan2(-hy, -hx); turn = float(np.clip(2.4 * np.sin(hd - yaw), -2.4, 2.4)); speed = 2.7
        elif mode == 'avoid':
            turn = float(np.clip(-16.0 * (cl - cr), -2.2, 2.2)) + self.rng.uniform(-.15, .15); speed = 2.4  # away
        elif mode == 'approach':
            turn = float(np.clip(16.0 * (cl - cr), -2.2, 2.2)) + self.rng.uniform(-.12, .12); speed = 2.6   # toward
        else:  # investigate: approach the novel smell to taste it, with more wander
            turn = float(np.clip(13.0 * (cl - cr), -1.9, 1.9)) + self.rng.uniform(-.5, .5); speed = 2.5
        self.world.step(speed, turn)
        nx, ny, _ = self.world.pose(); self.cx.update((nx - x) / 0.02, (ny - y) / 0.02)
        # active predator
        if dpred < 5.0:
            ang = np.arctan2(y - self.py, x - self.px); self.pph = ang; ps = 0.048
        else:
            self.pph += self.rng.uniform(-0.35, 0.35); ps = 0.030
        self.px += ps * np.cos(self.pph); self.py += ps * np.sin(self.pph)
        if abs(self.px) > ARENA + 1: self.pph = np.pi - self.pph; self.px = float(np.clip(self.px, -ARENA - 1, ARENA + 1))
        if abs(self.py) > ARENA + 1: self.pph = -self.pph; self.py = float(np.clip(self.py, -ARENA - 1, ARENA + 1))
        self.world.set_predator(self.px, self.py)
        caught = dpred < 0.7
        if caught and not self.was_caught: self.energy -= 3.0; self.hurt += 1
        self.was_caught = caught
        if self.crop >= 2 and d_origin < 1.0: self.trips += 1; self.crop = 0; self.energy += 1.0
        # taste + LEARN (only when moving toward food: investigate or approach)
        tasted = None; novel = (app == 0 and avo == 0)
        for i, (fx, fy, s, g) in enumerate(self.foods):
            if (x - fx) ** 2 + (y - fy) ** 2 < 0.6 ** 2 and mode in ('investigate', 'approach'):
                tasted = 'good' if g else 'toxic'; self.energy += 6.0 if g else -5.0
                self.good += g; self.tox += (not g); self.crop += 1 if g else 0
                if mode == 'investigate' and novel: self.investigate_novel += 1
                elif mode == 'approach': self.approach_learned += 1
                for _ in range(8): MB.present(self.net, self.core, self.nb, P_GOOD if g else P_BAD,
                                              teach=MB.APP if g else MB.AVO, learn=True)
                self.foods[i] = [self.rng.uniform(-ARENA, ARENA), self.rng.uniform(-ARENA, ARENA), s, g]
                self.world.set_foods([(f[0], f[1], f[2]) for f in self.foods]); self.world.set_predator(self.px, self.py); break
        if mode == 'avoid': self.avoid_learned += 1
        self.energy = min(self.energy - 0.02, 22.0); self.t += 1
        self.log.append(dict(t=self.t, x=float(x), y=float(y), mode=mode, app=app, avo=avo, U=ucnt,
                             energy=round(self.energy, 1), dpred=round(float(dpred), 1), dnest=round(d_origin, 2),
                             tasted=tasted, crop=self.crop, px=round(float(self.px), 2), py=round(float(self.py), 2),
                             foods=[[round(f[0], 1), round(f[1], 1), int(bool(f[3]))] for f in self.foods],
                             hx=round(float(hx), 2), hy=round(float(hy), 2)))
        return mode


if __name__ == "__main__":
    b = ForagerBrain()
    print("PAULA active-inference action selection — belief/uncertainty -> behaviour:")
    cases = [("novel odour, safe, fed",      0, 0, 0.0, 0.0, 'investigate'),
             ("believe GOOD (learned)",      1, 0, 0.0, 0.0, 'approach'),
             ("believe TOXIC (learned)",     0, 1, 0.0, 0.0, 'avoid'),
             ("novel odour + predator",      0, 0, 0.9, 0.0, 'flee'),
             ("believe good but hungry+home",1, 0, 0.0, 0.9, 'home'),
             ("believe toxic + predator",    0, 1, 0.8, 0.0, 'flee')]
    ok = True
    for name, app, avo, fear, home, exp in cases:
        w, cnt, u = b.select(app, avo, fear, home)
        good = (w == exp); ok &= good
        print(f"  {name:32s} app={app} avo={avo} fear={fear} home={home} -> {w:11s} "
              f"(U={u}) {'OK' if good else 'EXP '+exp}")
    print(f"VERDICT: {'PAULA AIF action selection correct' if ok else 'needs tuning'}")
    print("-" * 74)
    print("EMBODIED in the continuous foraging world (1200 steps) — is the epistemic drive load-bearing?")
    for tag, epi in [("epistemic ON  (active inference)", True), ("epistemic OFF (ablation)", False)]:
        f = Forager(seed=2, epistemic=epi)
        for _ in range(1200): f.step()
        m = f.modes
        print(f"  {tag}:")
        print(f"      investigate-novel={f.investigate_novel}  approach-learned={f.approach_learned}  "
              f"avoid={f.avoid_learned} | good={f.good} toxic={f.tox} trips={f.trips}")
        print(f"      modes: investigate={m['investigate']} approach={m['approach']} avoid={m['avoid']} "
              f"flee={m['flee']} home={m['home']}")
    print("@@@FORAGER DONE@@@")
