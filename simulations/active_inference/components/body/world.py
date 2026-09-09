"""WORLD3D — a complex MuJoCo world for the fully-PAULA active-inference agent.

Contains everything the brain's beliefs are ABOUT, so no behaviour has to be scripted:
  * NEST at the origin (the home the central complex path-integrates back to)
  * FOOD sources (respawn when eaten) and TOXIN sources -- two odour IDENTITIES so the mushroom body has
    something to learn a valence for
  * a distant SUN BEACON -- the compass anchor. Its egocentric bearing gives an absolute heading fix, which
    is what real insects (and the fly's visual ring neurons) use to stop path-integration drift. This is the
    landmark that makes the epistemic drive meaningful: "go where I can see it and my uncertainty drops".
  * PILLAR landmarks for visual structure/occlusion and for a richer scene.
  * deterministic V4 geometry fixtures (wall, corner, chicane, maze) for
    constraint-driven embodied tests.

TRANSDUCERS (the only non-neural steps, both sanctioned):
  IN : physical field -> neuron current. Odour concentration at each antenna (per identity), and the sun's
       egocentric bearing rendered as a Gaussian of current across heading columns.
  OUT: graded-muscle membrane S -> actuator force (turn + forward), applied to the body.

Body is planar-kinematic (x, y, yaw slide/slide/hinge, gravity off) -- the same proven pattern as
simulations/organism/body.py.  Baseline worlds keep contacts disabled; the V4 barrier layout enables only
hull↔barrier contacts so the new environmental rule is physical without introducing paddle self-collision.

Run standalone:  python world3d.py     (physics + sensor sanity + offscreen render)
"""
import numpy as np, mujoco

GOOD, BAD = 0, 1
GEAR = 26.0                          # muscle->actuator gain (NMJ), from the tuned rower
SUN_AZIMUTH = np.radians(35.0)      # absolute world direction of the beacon (the compass reference)

def make_xml(foods, toxins, pillars, arena=14.0, barriers=None):
    barrier_mode = bool(barriers)
    contact_flag = "enable" if barrier_mode else "disable"
    source_contact = ' contype="0" conaffinity="0"' if barrier_mode else ""
    pillar_contact = ' contype="0" conaffinity="0"' if barrier_mode else ""
    floor_contact = ' contype="0" conaffinity="0"' if barrier_mode else ""
    hull_contact = ' contype="1" conaffinity="1"' if barrier_mode else ""
    appendage_contact = ' contype="0" conaffinity="0"' if barrier_mode else ""
    def src(i,x,y,kind):
        col = "0.95 0.72 0.15 1" if kind==GOOD else "0.85 0.15 0.25 1"
        return (f'<body name="{"food" if kind==GOOD else "tox"}{i}" mocap="true" pos="{x} {y} 0">'
                f'<geom type="sphere" size="0.26" rgba="{col}"{source_contact}/></body>')
    def pil(i,x,y,h,col):
        return (f'<body name="pillar{i}" pos="{x} {y} {h/2-0.4}">'
                f'<geom type="cylinder" size="0.35 {h/2}" rgba="{col}"{pillar_contact}/></body>')
    def barrier(i, item):
        x, y, hx, hy, hz = (float(item[k]) for k in ("x", "y", "hx", "hy", "hz"))
        col = item.get("rgba", "0.17 0.20 0.24 1")
        name = item.get("name", f"barrier{i}")
        return (f'<body name="{name}" pos="{x} {y} 0">'
                f'<geom type="box" size="{hx} {hy} {hz}" rgba="{col}" '
                f'contype="1" conaffinity="1"/></body>')
    bodies  = "".join(src(i,x,y,GOOD) for i,(x,y) in enumerate(foods))
    bodies += "".join(src(i,x,y,BAD)  for i,(x,y) in enumerate(toxins))
    bodies += "".join(pil(i,x,y,h,c) for i,(x,y,h,c) in enumerate(pillars))
    bodies += "".join(barrier(i,b) for i,b in enumerate(barriers or ()))
    return f"""
<mujoco model="aif_world">
  <option timestep="0.004" gravity="0 0 0" density="1200" viscosity="0.5" integrator="RK4"><flag contact="{contact_flag}"/></option>
  <visual><headlight diffuse="0.75 0.75 0.75" ambient="0.35 0.35 0.35"/>
          <global offwidth="900" offheight="620"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.35 0.5 0.72" rgb2="0.05 0.07 0.12" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.88 0.87 0.83" rgb2="0.78 0.77 0.73" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="16 16" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0 0 9" dir="0 0 -1" diffuse="0.65 0.65 0.65"/>
    <light pos="{40*np.cos(SUN_AZIMUTH)} {40*np.sin(SUN_AZIMUTH)} 18" dir="{-np.cos(SUN_AZIMUTH)} {-np.sin(SUN_AZIMUTH)} -0.45"
           diffuse="0.5 0.47 0.38"/>
    <geom name="floor" type="plane" size="{arena} {arena} 0.1" pos="0 0 -0.4" material="grid"{floor_contact}/>
    <geom name="nest" type="cylinder" size="0.9 0.02" pos="0 0 0.02" rgba="0.25 0.45 0.9 0.55"{floor_contact}/>
    <!-- the SUN BEACON: far away, so its bearing is an absolute heading reference -->
    <body name="sun" pos="{70*np.cos(SUN_AZIMUTH)} {70*np.sin(SUN_AZIMUTH)} 14">
      <geom type="sphere" size="6.0" rgba="1 0.95 0.6 1"/>
    </body>
    <!-- BODY: the EXACT geometry of the proven 12/12 navigator (neural_agent.py / nmrower2.py). A slim
         capsule torso with two paddles. My earlier hand-drawn body (fat ellipsoid + head sphere + long
         antenna capsules) had so much drag in this density-1200 fluid that the same muscles produced
         0.7 units of travel where this hull produces >20. Antennae are SITES, not geoms: odour sampling is
         computed from the pose, so they must not add drag. -->
    <body name="bug" pos="0 0 0">
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="y" type="slide" axis="0 1 0"/>
      <joint name="yaw" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="-0.18 0 0 0.18 0 0" size="0.05" rgba="0.22 0.28 0.72 1"{hull_contact}/>
      <geom type="capsule" fromto="-0.28 0 0 -0.18 0 0" size="0.03" rgba="0.16 0.20 0.48 1"{appendage_contact}/>
      <site name="antL_tip" pos="0.66 0.58 0" size="0.02"/>
      <site name="antR_tip" pos="0.66 -0.58 0" size="0.02"/>
      <camera name="eye" pos="-0.24 0 0.10" xyaxes="0 1 0 0 0 1" fovy="52"/>
      <!-- NAMING WARNING: forward is the -x body axis, so body +y is the agent's RIGHT and body -y is
           its LEFT. "padL"/"pl"/MLp/MLr therefore drive the paddle on the agent's RIGHT-HAND side, and
           "padR"/"pr"/MRp/MRr the one on its LEFT. The control loop is consistent because the navigator
           gates each paddle from the OPPOSITE-side turn neuron (turn_src = TR if left else TL), so the
           two inversions cancel and steering is correct -- only the names are misleading. -->
      <body name="padL" pos="-0.15 0.06 0">
        <joint name="pl" type="hinge" axis="0 0 1" range="-1.8 1.8" damping="0.1"/>
        <geom type="capsule" fromto="0 0 0 -0.03 0.24 0" size="0.022" rgba="0.15 0.5 0.45 1"{appendage_contact}/></body>
      <body name="padR" pos="-0.15 -0.06 0">
        <joint name="pr" type="hinge" axis="0 0 1" range="-1.8 1.8" damping="0.1"/>
        <geom type="capsule" fromto="0 0 0 -0.03 -0.24 0" size="0.022" rgba="0.15 0.5 0.45 1"{appendage_contact}/></body>
    </body>
    {bodies}
  </worldbody>
  <actuator>
    <motor joint="pl" gear="{GEAR}" name="plp"/><motor joint="pl" gear="-{GEAR}" name="plr"/>
    <motor joint="pr" gear="{GEAR}" name="prp"/><motor joint="pr" gear="-{GEAR}" name="prr"/>
  </actuator>
</mujoco>"""

class World3D:
    """MuJoCo world plus physical and physiological transducers.

    The metabolic variables are body state, not policy.  They are updated from
    contact, digestion, and realized movement and are exposed to the brain
    only through explicit afferent currents.
    """
    def __init__(self, n_food=7, n_tox=6, arena=11.0, seed=0, sigma_odor=14.0, sigma_id=1.1, sigma_tox=1.7,
                 barrier=None):
        self.rng=np.random.default_rng(seed); self.arena=arena
        self.sigma=sigma_odor; self.sigma_id=sigma_id; self.sigma_tox=sigma_tox
        self.barrier_name = None
        self.barriers = []
        self._obstacle_prev = [0.0, 0.0]
        challenge = self._barrier_config(barrier) if barrier is not None else None
        if challenge is not None:
            self.barrier_name = challenge.get("name", str(barrier))
            self.barriers = [dict(item) for item in challenge.get("segments", ())]
            self.arena = float(challenge.get("arena", arena))
            # Challenge fixtures may deliberately contain no food, or may
            # place one fixed target that must not respawn into an easier
            # route after contact.  The ordinary meadow keeps its historical
            # respawning behaviour.
            self.respawn_food = bool(challenge.get("respawn_food", True))
            food = challenge.get("food")
            tox = challenge.get("toxins")
            self.foods = [list(map(float, p)) for p in (food if food is not None else
                                                        [self._rp(2.5, self.arena-1.5) for _ in range(n_food)])]
            self.toxins = [list(map(float, p)) for p in (tox if tox is not None else
                                                         [self._rp(2.5, self.arena-1.5) for _ in range(n_tox)])]
            self._start_pose = tuple(float(v) for v in challenge.get("start", (0.0, 0.0, 0.0)))
        else:
            self.respawn_food = True
            self.foods=[self._rp(2.5,arena-1.5) for _ in range(n_food)]
            self.toxins=[self._rp(2.5,arena-1.5) for _ in range(n_tox)]
            self._start_pose = None
        self.pillars=[(x,y,h,c) for (x,y,h,c) in [
            ( 8.5, 3.0, 2.2, "0.55 0.5 0.45 1"), (-7.0, 6.5, 1.7, "0.5 0.45 0.42 1"),
            (-6.0,-7.5, 2.6, "0.45 0.42 0.4 1"), ( 4.0,-8.5, 1.5, "0.52 0.48 0.44 1")]]
        self.model=mujoco.MjModel.from_xml_string(make_xml(self.foods,self.toxins,self.pillars,self.arena+3,self.barriers))
        self.data=mujoco.MjData(self.model)
        self.eaten=0; self.tox_hits=0; self._tox_in=False
        # V3 physiology: food becomes a gut load first, then usable energy.
        # Values are normalized so the neural afferent scale is stable across
        # worlds and seeds.  One MuJoCo/neural tick is the unit of this model.
        self.gut_load=0.0
        self.energy_store=0.42
        # Deliberately slower than a meal/contact transient: digestion is a
        # state that can justify rest over an immediately repeated search.
        self.digestion_rate=0.002
        self.meal_load=0.82
        self.energy_yield=0.78
        self.basal_cost=0.00035
        self.activity_cost=0.0018
        self.metabolic_ticks=0
        self.digested_total=0.0
        self.event=None                      # 'food'/'toxin' on the tick a source is ENTERED (the US)
        self.pending_event=None              # the SAME event, latched until the brain actually reads it.
                                             # `event` alone is useless as a US: _consume() clears it on
                                             # EVERY world step, and there are `sub`(=16) world steps per
                                             # agent step, while run_episode samples it once per agent
                                             # step. Measured: 7 toxin contacts -> 0 STG_T spikes, i.e.
                                             # the unconditioned stimulus never reached the neurons at all.
        self._in=[False]*(n_food+n_tox)
        self.jx,self.jy,self.jyaw=(mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in ("x","y","yaw"))
        self.food_bid=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,f"food{i}") for i in range(len(self.foods))]
        self.tox_bid=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,f"tox{i}") for i in range(len(self.toxins))]
        self.act_id={n:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in ("plp","plr","prp","prr")}
        self._sync_mocap()
        if self._start_pose is not None:
            self.data.qpos[self.jx], self.data.qpos[self.jy], self.data.qpos[self.jyaw] = self._start_pose
            mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _barrier_config(value):
        """Normalize the deterministic V4 challenge or a caller-provided layout."""
        if isinstance(value, dict):
            return value
        # The acceptance matrix keeps a monotone catalogue of ten physical
        # obstacle worlds.  These small layouts are intentionally distinct
        # fixtures (not random perturbations of a trajectory), so a reflex
        # cannot pass by exploiting one memorized wall pose.
        simple_walls = {
            "wall_short": ("wall_short", -1.90, 0.0, 0.72),
            "wall_medium": ("wall_medium", -2.15, 0.0, 1.25),
            "wall_offset_left": ("wall_offset_left", -2.30, 0.72, 1.15),
            "wall_offset_right": ("wall_offset_right", -2.30, -0.72, 1.15),
        }
        if value in simple_walls:
            name, x, y, hy = simple_walls[value]
            return {
                "name": name, "arena": 7.0, "start": (0.0, 0.0, 0.0),
                "food": [], "toxins": [], "respawn_food": False,
                "segments": [{"name": f"{name}_segment", "x": x, "y": y,
                               "hx": 0.16, "hy": hy, "hz": 0.42,
                               "rgba": "0.12 0.18 0.28 1"}],
            }
        if value == "corner_small":
            return {
                "name": "corner_small", "arena": 7.0, "start": (0.0, 0.0, 0.0),
                "food": [], "toxins": [], "respawn_food": False,
                "segments": [
                    {"name": "corner_small_main", "x": -2.25, "y": -0.55,
                     "hx": 0.16, "hy": 1.10, "hz": 0.42,
                     "rgba": "0.16 0.20 0.30 1"},
                    {"name": "corner_small_wing", "x": -3.05, "y": 0.60,
                     "hx": 0.80, "hy": 0.16, "hz": 0.42,
                     "rgba": "0.16 0.20 0.30 1"},
                ],
            }
        if value == "chicane_short":
            return {
                "name": "chicane_short", "arena": 7.0, "start": (0.0, 0.0, 0.0),
                "food": [], "toxins": [], "respawn_food": False,
                "segments": [
                    {"name": "chicane_short_lower", "x": -2.35, "y": -0.35,
                     "hx": 0.16, "hy": 0.95, "hz": 0.42,
                     "rgba": "0.20 0.24 0.34 1"},
                    {"name": "chicane_short_upper", "x": -4.20, "y": 1.35,
                     "hx": 0.16, "hy": 0.95, "hz": 0.42,
                     "rgba": "0.20 0.24 0.34 1"},
                ],
            }
        if value in ("head_on_wall", "v4_head_on_wall", "obstacle_detour"):
            # A full-width wall turns the V4 question into the smallest
            # justified embodied constraint: can a bilateral whisker/reflex
            # route prevent a collision and deflect the body?  There is no
            # food in this fixture, so success cannot come from stumbling onto
            # a randomly respawned source elsewhere in the arena.
            return {
                "name": "head_on_wall",
                "arena": 7.0,
                "start": (0.0, 0.0, 0.0),
                "food": [],
                "toxins": [],
                "respawn_food": False,
                "segments": [
                    {"name": "wall_full_width", "x": -2.40, "y": 0.0,
                     "hx": 0.16, "hy": 7.20, "hz": 0.42,
                     "rgba": "0.12 0.18 0.28 1"},
                ],
            }
        if value == "corner":
            # A compound L-corner: the first wall reaches just into the
            # ordinary V3 trajectory and the return wing closes the high-side
            # escape.  It is still a local tactile constraint, not a target
            # planner; the fixture contains no food or toxin source.
            return {
                "name": "corner",
                "arena": 7.0,
                "start": (0.0, 0.0, 0.0),
                "food": [],
                "toxins": [],
                "respawn_food": False,
                "segments": [
                    {"name": "corner_main", "x": -2.40, "y": -0.65,
                     "hx": 0.16, "hy": 1.65, "hz": 0.42,
                     "rgba": "0.16 0.20 0.30 1"},
                    {"name": "corner_wing", "x": -3.35, "y": 1.05,
                     "hx": 1.10, "hy": 0.16, "hz": 0.42,
                     "rgba": "0.16 0.20 0.30 1"},
                ],
            }
        if value == "chicane":
            # Alternating lower/upper bars create repeated bilateral cues.
            # The first bar is deliberately on the proven V3 trajectory so
            # the unchanged prior must fail before the later stress geometry
            # can be interpreted.
            return {
                "name": "chicane",
                "arena": 7.0,
                "start": (0.0, 0.0, 0.0),
                "food": [],
                "toxins": [],
                "respawn_food": False,
                "segments": [
                    {"name": "chicane_1_lower", "x": -2.40, "y": -0.40,
                     "hx": 0.16, "hy": 1.10, "hz": 0.42,
                     "rgba": "0.20 0.24 0.34 1"},
                    {"name": "chicane_2_upper", "x": -4.40, "y": 1.60,
                     "hx": 0.16, "hy": 1.10, "hz": 0.42,
                     "rgba": "0.20 0.24 0.34 1"},
                    {"name": "chicane_3_lower", "x": -6.40, "y": -0.40,
                     "hx": 0.16, "hy": 1.10, "hz": 0.42,
                     "rgba": "0.20 0.24 0.34 1"},
                ],
            }
        if value == "maze":
            # A compact alternating-bar maze with a cross-cap.  V4 can be
            # evaluated for safe tactile response here, while route depth is
            # reported separately as a deliberate stress/diagnostic measure
            # for the next version rather than silently treated as success.
            return {
                "name": "maze",
                "arena": 7.0,
                "start": (0.0, 0.0, 0.0),
                "food": [],
                "toxins": [],
                "respawn_food": False,
                "segments": [
                    {"name": "maze_1_lower", "x": -2.40, "y": -0.40,
                     "hx": 0.16, "hy": 1.10, "hz": 0.42,
                     "rgba": "0.24 0.26 0.36 1"},
                    {"name": "maze_2_upper", "x": -3.50, "y": 1.60,
                     "hx": 0.16, "hy": 1.10, "hz": 0.42,
                     "rgba": "0.24 0.26 0.36 1"},
                    {"name": "maze_3_lower", "x": -4.60, "y": -0.40,
                     "hx": 0.16, "hy": 1.10, "hz": 0.42,
                     "rgba": "0.24 0.26 0.36 1"},
                    {"name": "maze_4_upper", "x": -5.70, "y": 1.60,
                     "hx": 0.16, "hy": 1.10, "hz": 0.42,
                     "rgba": "0.24 0.26 0.36 1"},
                    {"name": "maze_cross_cap", "x": -3.50, "y": 0.0,
                     "hx": 1.25, "hy": 0.16, "hz": 0.42,
                     "rgba": "0.24 0.26 0.36 1"},
                ],
            }
        if value in ("l_gap", "v4_detour"):
            # The forward axis is -x.  The first wall ends just above the
            # right antenna at y=+0.55, making the upper gap physically
            # detectable by the bilateral whiskers; the return wall closes
            # the lower route.  Food is beyond the gap, not teleported there.
            return {
                "name": "l_gap",
                "arena": 7.0,
                "start": (0.0, 0.0, 0.0),
                "food": [(-3.0, 1.4)],
                "toxins": [],
                "segments": [
                    {"name": "barrier_main", "x": -2.35, "y": -1.75, "hx": 0.12, "hy": 1.55, "hz": 0.42,
                     "rgba": "0.12 0.18 0.28 1"},
                    {"name": "barrier_return", "x": -3.95, "y": -3.40, "hx": 1.72, "hy": 0.12, "hz": 0.42,
                     "rgba": "0.12 0.18 0.28 1"},
                ],
            }
        raise ValueError(f"unknown barrier layout {value!r}")

    def update_metabolism(self, event=None, speed=None):
        """Advance the body energy state by one neural/physics tick.

        ``event`` and ``speed`` are physical observations.  No behavioural
        branch is taken here: the same update runs whether the brain is
        awake, exploring, or silent.  Sleep emerges when the neural motor
        output is suppressed, causing realized speed/activity cost to fall.
        """
        if event == "food":
            self.gut_load=min(1.0, self.gut_load+self.meal_load)
        speed=self.speed() if speed is None else max(0.0, float(speed))
        digested=min(self.gut_load, self.digestion_rate)
        self.gut_load=max(0.0, self.gut_load-digested)
        self.digested_total+=digested
        self.energy_store=min(1.0, max(0.0,
            self.energy_store + self.energy_yield*digested
            - self.basal_cost - self.activity_cost*min(speed, 2.0)))
        self.metabolic_ticks+=1

    def metabolic_state(self):
        """Return normalized body afferents: gut, energy, deficit, digestion."""
        return {
            "gut_load":float(self.gut_load),
            "energy_store":float(self.energy_store),
            "low_energy":float(max(0.0,1.0-self.energy_store)),
            "digestion_rate":float(self.digestion_rate if self.gut_load>0 else 0.0),
        }
    def _rp(self,rlo,rhi):
        a=self.rng.uniform(0,2*np.pi); r=self.rng.uniform(rlo,rhi)
        return [float(r*np.cos(a)),float(r*np.sin(a))]
    def _sync_mocap(self):
        for i,b in enumerate(self.food_bid):
            self.data.mocap_pos[self.model.body_mocapid[b]]=[self.foods[i][0],self.foods[i][1],0.0]
        for i,b in enumerate(self.tox_bid):
            self.data.mocap_pos[self.model.body_mocapid[b]]=[self.toxins[i][0],self.toxins[i][1],0.0]
    # ---------- state ----------
    def pose(self):
        return (float(self.data.qpos[self.jx]),float(self.data.qpos[self.jy]),float(self.data.qpos[self.jyaw]))
    def heading(self):
        """FORWARD IS THE -x BODY AXIS on this hull (the proven navigator's convention)."""
        return self.pose()[2]+np.pi
    def antennae(self):
        x,y,yaw=self.pose(); fwd=yaw+np.pi
        hd=np.array([np.cos(fwd),np.sin(fwd)]); lat=np.array([-np.sin(fwd),np.cos(fwd)])
        head=np.array([x,y])+hd*0.45
        return [tuple(head+lat*0.6), tuple(head-lat*0.6)]          # [left, right], wide -> bigger differential

    @staticmethod
    def _rect_distance(point, obstacle):
        px, py = point
        x, y = float(obstacle["x"]), float(obstacle["y"])
        hx, hy = float(obstacle["hx"]), float(obstacle["hy"])
        dx = max(abs(float(px) - x) - hx, 0.0)
        dy = max(abs(float(py) - y) - hy, 0.0)
        return float(np.hypot(dx, dy))

    def obstacle_proximity(self):
        """Physical bilateral whisker/range transducer for V4.

        The signal is computed from the agent pose and the actual barrier
        rectangles.  It contains no preferred turn or target direction: only
        distance, onset, and contact are exposed to the PAULA fragment.
        """
        if not self.barriers:
            return {"left": 0.0, "right": 0.0, "left_onset": 0.0,
                    "right_onset": 0.0, "distance_left": float("inf"),
                    "distance_right": float("inf"), "contact": 0.0}
        values=[]; distances=[]
        for antenna in self.antennae():
            d=min(self._rect_distance(antenna, item) for item in self.barriers)
            distances.append(d)
            # A finite whisker range is a sensory field, not a controller.
            values.append(float(np.clip((2.35-d)/2.35, 0.0, 1.0)))
        onset=[max(0.0, values[i]-self._obstacle_prev[i]) for i in range(2)]
        self._obstacle_prev=list(values)
        return {"left": values[0], "right": values[1],
                "left_onset": onset[0], "right_onset": onset[1],
                "distance_left": distances[0], "distance_right": distances[1],
                "contact": float(min(distances) < 0.10)}
    # ---------- TRANSDUCER IN: odour field -> currents ----------
    def odour(self):
        """returns {GOOD:(L,R), BAD:(L,R)} concentrations at the antennae"""
        res={}
        # toxin is SHORT-range on purpose (dual_chemotaxis result): a repellent is a LOCAL no-go while an
        # attractant is a long-range seek. With equal broad sigma the escape response fires everywhere and
        # suppresses all foraging.
        for kind,srcs in ((GOOD,self.foods),(BAD,self.toxins)):
            sig=self.sigma if kind==GOOD else self.sigma_tox
            vals=[]
            for (ax,ay) in self.antennae():
                s=0.0
                for (sx,sy) in srcs:
                    d2=(ax-sx)**2+(ay-sy)**2
                    s+=np.exp(-d2/sig) if kind==GOOD else np.exp(-d2/(2*sig**2))
                vals.append(float(s))
            res[kind]=tuple(vals)
        return res
    def odour_identity(self):
        """SHARP, head-centred per-odorant concentrations = the KC IDENTITY channel, deliberately separate
        from the broad tropotaxis fields. Building the identity code from the broad fields blends food and
        toxin at every point, so the mushroom body cannot separate them and learns to fear everything
        (measured: learned aversion food=80 toxin=80). sigma_id is small so only the source you are ACTUALLY
        at contributes."""
        x,y,yaw=self.pose(); fwd=yaw+np.pi
        hx=x+0.45*np.cos(fwd); hy=y+0.45*np.sin(fwd)
        out=[]
        for srcs in (self.foods,self.toxins):
            v=sum(np.exp(-((hx-sx)**2+(hy-sy)**2)/(2*self.sigma_id**2)) for sx,sy in srcs)
            out.append(float(v))
        return out

    def sun_bearing(self):
        """egocentric bearing of the beacon (radians, + = to the left). The compass anchor."""
        _,_,yaw=self.pose()
        b=SUN_AZIMUTH-(yaw+np.pi)
        return float(np.arctan2(np.sin(b),np.cos(b)))
    def implied_heading(self):
        """absolute heading implied by the beacon sighting = SUN_AZIMUTH - bearing (fixed wiring maps it)"""
        return float(np.arctan2(np.sin(SUN_AZIMUTH-self.sun_bearing()),np.cos(SUN_AZIMUTH-self.sun_bearing())))
    # ---------- TRANSDUCER OUT: muscle S -> body ----------
    def act_muscles(self, sLp, sLr, sRp, sRr, ggain=8.0):
        """THE sanctioned actuator transducer: graded muscle membrane S -> actuator force (NMJ).
        The stroke, the thrust and the turn all EMERGE from the muscle activity and the physics -- nothing
        here sets a velocity or a yaw. Steering happens upstream, by inhibiting one side's muscles."""
        d=self.data
        d.ctrl[self.act_id["plp"]]=ggain*max(0.0,sLp); d.ctrl[self.act_id["plr"]]=ggain*max(0.0,sLr)
        d.ctrl[self.act_id["prp"]]=ggain*max(0.0,sRp); d.ctrl[self.act_id["prr"]]=ggain*max(0.0,sRr)
        mujoco.mj_step(self.model,self.data)
        x,y,_=self.pose()
        r=np.hypot(x,y)
        if r>self.arena:
            self.data.qpos[self.jx]=x*self.arena/r; self.data.qpos[self.jy]=y*self.arena/r
            mujoco.mj_forward(self.model,self.data); x,y,_=self.pose()
        self._consume(x,y)
        self.update_metabolism(event=self.event)

    def act(self, turn, speed, kyaw=0.36, kv=0.06):
        x,y,yaw=self.pose()
        yaw+=np.radians(kyaw*turn)
        x+=kv*speed*np.cos(yaw); y+=kv*speed*np.sin(yaw)
        r=np.hypot(x,y)
        if r>self.arena: x*=self.arena/r; y*=self.arena/r          # keep inside the arena
        self.data.qpos[self.jx]=x; self.data.qpos[self.jy]=y; self.data.qpos[self.jyaw]=yaw
        mujoco.mj_forward(self.model,self.data)
        self._consume(x,y)
    def take_event(self):
        """TRANSDUCER OUT (US): hand the pending contact event to the brain exactly once, then clear it.
        Read-and-clear rather than a bare flag, so an event raised on any of the sub-steps between two
        agent steps survives to be delivered instead of being overwritten by the next _consume()."""
        e=self.pending_event; self.pending_event=None; return e
    def _consume(self,x,y):
        # discrete ENTER events (latched until the agent leaves) so sitting on a source does not rack up
        # hundreds of unconditioned stimuli -- the mushroom body needs events, not a continuous drip.
        self.event=None
        for i,(fx,fy) in enumerate(self.foods):
            if (x-fx)**2+(y-fy)**2 < 0.7**2:      # 0.7 = the proven navigator's eat radius; at 0.45 the
                if self.respawn_food:
                    self.foods[i]=self._rp(2.5,self.arena-1.5)
                    self._sync_mocap()
                else:
                    # Keep the source outside the arena while preserving a
                    # stable body count and an inspectable consumed target.
                    self.foods[i]=[float(self.arena + 10.0), float(self.arena + 10.0)]
                    self._sync_mocap()
                self.eaten+=1; self.event="food"; self.pending_event="food"   # agent closes to ~0.7 then overshoots
        near=False
        for j,(tx,ty) in enumerate(self.toxins):
            d2=(x-tx)**2+(y-ty)**2
            idx=len(self.foods)+j
            if d2<0.55**2:
                near=True
                if not self._in[idx]:
                    self.tox_hits+=1; self.event="toxin"; self.pending_event="toxin"   # toxin overrides a
                    self._in[idx]=True                                                 # pending food: aversive wins
            elif d2>0.9**2: self._in[idx]=False
        self._tox_in=near
    def speed(self):
        """PROPRIOCEPTION: actual forward speed from the physics, a sensor like any other. Feeding the path
        integrator a CONSTANT makes it integrate TIME instead of DISTANCE, which is not path integration."""
        vx=float(self.data.qvel[self.jx]); vy=float(self.data.qvel[self.jy])
        return float(np.hypot(vx,vy))

    def yaw_rate(self):
        """PROPRIOCEPTION (angular), raw: the body's turn rate straight from the physics."""
        return float(self.data.qvel[self.jyaw])

    def yaw_rate_net(self, tau=None):
        """The turn rate the compass can actually use, low-passed over one stroke cycle.

        The raw signal is dominated by the STROKE, not by turning: |yaw_rate| averages 7.6 rad/s and
        peaks at 35 while the body's net rotation is a small fraction of that. The shift cells take a
        RECTIFIED pair (ccw, cw), so an oscillating input drives both directions alternately and, since
        the P-EN shift is excitation-only, the bump just blurs instead of rotating -- measured gain
        0.09 against the body. One stroke is CPG_PERIOD*4 = 40 ticks, so averaging over tau=60 ticks
        leaves net rotation and removes the rowing. Real angular proprioceptors are band-limited the
        same way. Call once per neural tick: it advances its own filter state."""
        # BOXCAR over exactly one stroke cycle, NOT an exponential lag. The stroke is periodic at
        # CPG_PERIOD*4 = 40 ticks and ~30x larger than the net turn. A first-order filter with tau=60
        # attenuates a 40-tick component only ~9x, so the stroke still outweighs the real signal ~3:1
        # and any gain high enough to move the bump also slams both shift directions alternately.
        # A boxcar of exactly one period is a NOTCH at the stroke frequency and its harmonics: the
        # stroke integrates to zero identically while net rotation passes. Insects face the same
        # problem rejecting their own wingbeat. Call once per neural tick; it advances its own state.
        n=int(getattr(self,"wz_box",0) or 0)
        if n>1:
            b=getattr(self,"_wzbuf",None)
            if b is None or b.maxlen!=n:
                from collections import deque
                b=deque([0.0]*n,maxlen=n); self._wzbuf=b
            b.append(self.yaw_rate())
            return float(sum(b)/n)
        tau = getattr(self,"wz_tau",60.0) if tau is None else tau
        if tau<=1.0: return self.yaw_rate()
        _prev = getattr(self,"_wzf",0.0)
        self._wzf = _prev + (self.yaw_rate()-_prev)/tau
        # GROUP-DELAY COMPENSATION. A first-order lag rejects the stroke but delays the estimate: measured
        # against NET rotation (true dyaw averaged over one stroke), tau=60 gives r=+0.239 at zero lag and
        # +0.754 at a 9-tick lag -- the signal is strong but LATE, and the shift cells consume it now. The
        # compass therefore tracked the body with r~0.00 and every shift-parameter sweep was run on a stale
        # input. Adding k*d/dt cancels the phase lag causally: tau=120,k=20 gives r=+0.848 at ZERO lag with
        # LOWER stroke leakage than the uncompensated tau=60 (15.6% vs 28.2%).
        k = float(getattr(self,"wz_comp",0.0) or 0.0)
        if k:
            out = self._wzf + k*(self._wzf-_prev)
            return float(out)
        return float(self._wzf)

    def dist_home(self): x,y,_=self.pose(); return float(np.hypot(x,y))
    def retina(self, w=72, h=20):
        """TRANSDUCER IN (vision): render the world from the agent's eye -> luminance per photoreceptor.
        Low-res and wide, like an insect eye: what matters is WHERE bright/edgy things are in azimuth."""
        cam=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_CAMERA,"eye")
        with mujoco.Renderer(self.model,height=h,width=w) as r:
            r.update_scene(self.data,camera=cam)
            img=r.render().astype(np.float32)/255.0
        return img                                   # (h,w,3) RGB -- the cortex does luminance/chroma

    def render(self, w=900, h=620, cam_dist=20.0, elev=-38.0, azim=115.0):
        cam=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
        cam.lookat[:]=[0,0,0.4]; cam.distance=cam_dist; cam.elevation=elev; cam.azimuth=azim
        with mujoco.Renderer(self.model,height=h,width=w) as r:
            r.update_scene(self.data,camera=cam); return r.render()

if __name__=="__main__":
    w=World3D()
    print(f"WORLD3D: {len(w.foods)} food, {len(w.toxins)} toxins, {len(w.pillars)} pillars, sun az={np.degrees(SUN_AZIMUTH):.0f}deg")
    print(f"  bodies={w.model.nbody} geoms={w.model.ngeom}  pose={tuple(round(v,2) for v in w.pose())}")
    od=w.odour(); print(f"  odour GOOD L/R={od[GOOD][0]:.3f}/{od[GOOD][1]:.3f}  BAD L/R={od[BAD][0]:.3f}/{od[BAD][1]:.3f}")
    print(f"  sun bearing={np.degrees(w.sun_bearing()):+.0f}deg  implied heading={np.degrees(w.implied_heading()):+.0f}deg (true yaw={np.degrees(w.pose()[2]):+.0f})")
    for _ in range(60): w.act(turn=0.6, speed=1.0)
    print(f"  after 60 steps: pose={tuple(round(v,2) for v in w.pose())} dist_home={w.dist_home():.2f}")
    print(f"  sun bearing={np.degrees(w.sun_bearing()):+.0f}deg implied={np.degrees(w.implied_heading()):+.0f}deg true yaw={np.degrees(w.pose()[2]):+.0f}deg  <-- must MATCH")
    img=w.render(); print(f"  render OK: {img.shape}, mean pixel {img.mean():.1f}")
    print("@@@WORLD3D DONE@@@")
