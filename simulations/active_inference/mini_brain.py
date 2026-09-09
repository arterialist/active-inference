"""THE MINI-BRAIN — a fully-PAULA foraging organism in an open world, on the pivot-capable rower body.

Puts every verified region together into one closed sensorimotor loop:

  SENSORY        odour at the head; predator distance; interoception (energy).
  MUSHROOM BODY  learns odour -> value; reports approach-evidence (app) / avoid-evidence (avo). A NOVEL
                 odour gives app=avo=0.
  UNCERTAINTY U  (epistemic) high when the odour is novel -> drives INVESTIGATE.
  ACTION WTA     spiking winner-take-all over {investigate, approach, avoid, flee, home, wander},
                 driven by U (epistemic), app/avo (pragmatic), fear, and a home drive.
  CENTRAL CPLX   path-integrates the home vector for returning to the nest.
  NAVIGATION     each behaviour climbs its own signal by neural run-and-tumble:
                   investigate/approach -> climb odour;  avoid -> descend odour;
                   flee -> climb predator-distance;      home -> climb nest-closeness.
  MOTOR CORTEX   the run/tumble policy issues descending drive (dL,dR) to ...
  CPG + BODY     the PAULA pacemaker CPG drives the two paddles -> the rower moves through fluid physics.
  LEARNING       tasting a food teaches the mushroom body, so INVESTIGATE (novel) turns into APPROACH
                 (good) or AVOID (toxic): curiosity -> learning -> foraging. Open-ended.

Run:  python -m simulations.active_inference.mini_brain
"""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from simulations.paula_loader import ensure_paula_available
ensure_paula_available()
import importlib.util
def _l(n, p): s=importlib.util.spec_from_file_location(n, p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
HERE = pathlib.Path(__file__).parent
NR = _l("nr", str(HERE / "neural_rower.py"))
FG = _l("fg", str(HERE / "forager.py"))
from simulations.organism import mushroom_body as MB
from simulations.organism import central_complex as CXM

rng0 = np.random.RandomState(0); P_GOOD = MB.odor_pattern(rng0); P_BAD = MB.odor_pattern(rng0)
ARENA = 4.0

class MiniBrain:
    def __init__(self, seed=0, nfood=4):
        self.rng = np.random.RandomState(seed)
        self.foods = [[self.rng.uniform(-ARENA, ARENA), self.rng.uniform(-ARENA, ARENA),
                       1.0 if (j != 0) else -1.0, (j != 0)] for j in range(nfood)]   # index 0 toxic, rest good
        self.rower = NR.NeuralRower(period_c=80)
        self.net, self.core, self.nb = MB.load(MB.build_mb(seed=1)[0])       # mushroom body
        self.cx = CXM.CentralComplex()                                        # central complex
        self.wta = FG.ForagerBrain()                                          # PAULA action selection
        self.px, self.py = 6.0, 6.0; self.pph = self.rng.uniform(0, 6.28); self.was_caught = False
        self.energy = 20.0; self.crop = 0; self.hurt = 0
        self.good = 0; self.tox = 0; self.trips = 0
        self.investigate_novel = 0; self.approach_learned = 0; self.avoid_learned = 0
        self.modes = {m: 0 for m in FG.ANAME.values()}
        self.phase = 'run'; self.timer = 0; self.s_start = 0.0; self.prev_mode = None; self.home_streak = 0
        self.t = 0; self.log = []
    # ---- world ----
    def odor_at(self, x, y):
        c = 0.0
        for fx, fy, s, g in self.foods: c += abs(s) * np.exp(-((x-fx)**2+(y-fy)**2)/12.0)
        return float(c)
    def _pattern(self, x, y):
        best=None; bd=1e9
        for fx, fy, s, g in self.foods:
            d=(x-fx)**2+(y-fy)**2
            if d<bd: bd=d; best=g
        return (P_GOOD if best else P_BAD) if bd < 30 else np.zeros(MB.N_PN)
    # ---- run/tumble navigation on an arbitrary scalar signal S (climb it) ----
    def _run_tumble(self, S, mode):
        if mode != self.prev_mode:                                           # behaviour switched -> fresh run
            self.phase='run'; self.timer=240; self.s_start=S; self.prev_mode=mode
        if S is None:                                                        # wander: run with occasional tumble
            self.timer -= 1
            if self.timer <= 0:
                self.phase = 'tumble' if self.phase == 'run' else 'run'
                self.timer = self.rng.randint(250, 500) if self.phase == 'run' else self.rng.randint(100, 260)
            return (1.0, 1.0) if self.phase == 'run' else (1.0, -1.0)
        self.timer -= 1
        if self.phase == 'run':
            if self.timer <= 0:
                if S < self.s_start - 0.02*max(0.2, abs(self.s_start)):       # signal fell over the run -> reorient
                    self.phase='tumble'; self.timer=self.rng.randint(120,340)
                else:
                    self.timer=240; self.s_start=S
            return 1.0, 1.0
        else:
            if self.timer <= 0: self.phase='run'; self.timer=240; self.s_start=S
            return 1.0, -1.0
    def step(self):
        x, y, yaw = self.rower.pose(); head = (x+np.cos(yaw)*0.35, y+np.sin(yaw)*0.35)
        dpred = float(np.hypot(self.px-x, self.py-y))
        hx, hy = self.cx.home_vector(); dnest = float(np.hypot(x, y))
        fear = float(np.clip((2.6-dpred)/2.6, 0, 1))
        hunger = float(np.clip((18-self.energy)/14.0, 0, 1))
        home_drive = 1.0 if (self.crop >= 2 and dnest > 1.6) else 0.0
        # BELIEF (mushroom body) + ACTION SELECTION (PAULA WTA)
        _, app, avo = MB.present(self.net, self.core, self.nb, self._pattern(x, y))
        mode, _, ucnt = self.wta.select(app, avo, fear, home_drive)
        self.modes[mode] += 1
        # NAVIGATION SIGNAL for this behaviour (run-and-tumble climbs it)
        odor = self.odor_at(*head)
        if   mode in ('investigate', 'approach'): S = odor
        elif mode == 'avoid':                     S = -odor
        elif mode == 'flee':                      S = dpred / 10.0
        elif mode == 'home':                      S = -dnest / 10.0
        else:                                     S = None                    # wander
        dL, dR = self._run_tumble(S, mode)
        self.rower.step(dL, dR)
        # central complex integrates actual motion
        nx, ny, _ = self.rower.pose(); self.cx.update((nx-x)/(0.004*6), (ny-y)/(0.004*6))
        # active predator — scaled to the rower's speed (~0.0013 u/step) so fleeing can escape it
        if dpred < 2.6:
            ang=np.arctan2(y-self.py, x-self.px); self.pph=ang; ps=0.0011
        else:
            self.pph += self.rng.uniform(-0.4, 0.4); ps=0.0008
        self.px += ps*np.cos(self.pph); self.py += ps*np.sin(self.pph)
        if abs(self.px) > ARENA: self.pph=np.pi-self.pph
        if abs(self.py) > ARENA: self.pph=-self.pph
        self.px=float(np.clip(self.px,-ARENA,ARENA)); self.py=float(np.clip(self.py,-ARENA,ARENA))
        caught = dpred < 0.7
        if caught and not self.was_caught: self.energy -= 3.0; self.hurt += 1     # count per encounter
        self.was_caught = caught
        # arrived home? (generous radius) — or ABANDON the trip if it can't get there
        self.home_streak = self.home_streak + 1 if mode == 'home' else 0
        if self.crop >= 2 and dnest < 1.5: self.trips += 1; self.crop = 0; self.energy += 1.0; self.home_streak = 0
        elif self.home_streak > 1400: self.crop = 0; self.home_streak = 0        # gave up returning
        # taste + LEARN (investigate / approach reaching a food)
        novel = (app == 0 and avo == 0)
        for i, (fx, fy, s, g) in enumerate(self.foods):
            if (x-fx)**2 + (y-fy)**2 < 0.7**2 and mode in ('investigate', 'approach'):
                self.energy += 6.0 if g else -5.0; self.good += g; self.tox += (not g); self.crop += 1 if g else 0
                if mode == 'investigate' and novel: self.investigate_novel += 1
                elif mode == 'approach': self.approach_learned += 1
                for _ in range(8): MB.present(self.net, self.core, self.nb, P_GOOD if g else P_BAD,
                                              teach=MB.APP if g else MB.AVO, learn=True)
                self.foods[i] = [self.rng.uniform(-ARENA, ARENA), self.rng.uniform(-ARENA, ARENA), s, g]
                break
        if mode == 'avoid': self.avoid_learned += 1
        self.energy = min(self.energy - 0.006, 24.0); self.t += 1
        if self.t % 20 == 0:
            self.log.append(dict(t=self.t, x=round(x,2), y=round(y,2), mode=mode, app=app, avo=avo, U=ucnt,
                                 energy=round(self.energy,1), dpred=round(dpred,1), dnest=round(dnest,2),
                                 crop=self.crop, px=round(self.px,2), py=round(self.py,2),
                                 foods=[[round(f[0],1), round(f[1],1), int(bool(f[3]))] for f in self.foods]))
        return mode


if __name__ == "__main__":
    print("MINI-BRAIN foraging (fully PAULA) — 80000 ticks:")
    o = MiniBrain(seed=1)
    for _ in range(80000): o.step()
    m = o.modes
    print(f"  investigate-novel={o.investigate_novel}  approach-learned={o.approach_learned}  avoid={o.avoid_learned}")
    print(f"  good={o.good} toxic={o.tox} trips_home={o.trips} caught={o.hurt} energy={o.energy:.0f}")
    print(f"  modes: " + " ".join(f"{k}={v}" for k, v in m.items()))
    _, ag, vg = MB.present(o.net, o.core, o.nb, P_GOOD); _, ab, vb = MB.present(o.net, o.core, o.nb, P_BAD)
    print(f"  learned belief: P_GOOD->app{ag}/avo{vg}  P_BAD->app{ab}/avo{vb}")
    print("@@@MINIBRAIN DONE@@@")
