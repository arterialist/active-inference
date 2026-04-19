# Virtual lab, demo server, and config parity

← [Index](../INDEX.md) | [C. elegans Overview](overview.md)

This note clarifies how the **same** `active-inference` C. elegans stack is used by multiple backends, and when their behaviour can **diverge** even though they share one codebase.

## Single sources of truth

| What | Where | Loaded when |
|------|-------|----------------|
| MuJoCo kinematics / collision / default `mjOption` | `simulations/c_elegans/body_model.xml` | Each process constructs `CElegansBody` → `mj_loadXML` |
| Python constants (joint limit rad, muscle filter α, plate radius, …) | `simulations/c_elegans/config.py` | Import time; **some** lab parameters mutate this module at runtime |
| Neural / tonic / CPG defaults | `simulations/c_elegans/neuron_mapping.py` | Instance attributes on `CElegansNervousSystem` |
| Sensory mapping | `simulations/c_elegans/sensors.py` | `SensorEncoder` class body |

There is **no second copy** of the MJCF or `config.py` inside `celegans-live-demo` for physics: both the **lab** (`celegans-lab-server`, `lab/`) and the **WebSocket demo** (`celegans-demo-server`, `celegans_live_demo/`) call `build_c_elegans_simulation(...)` from `active-inference`.

## Two backends, one engine

- **`celegans-lab-server`** adds HTTP (`/api/schema`, `/api/patch`, `/api/body/patch`, …) so the **virtual lab** can change `mjModel.opt`, actuator limits, contact friction, and registry-backed parameters between ticks.
- **`celegans-demo-server`** only exposes the compact WebSocket protocol; it does **not** load the lab parameter registry. Tuning is via code, `evol_config`, or editing files and restarting.

If both servers are installed from the **same** checkout with **`active-inference` editable** (`celegans-live-demo/pyproject.toml` → `../active-inference`), they always import the **same** Python modules and the **same** `body_model.xml` path resolved from the `active-inference` package.

## When things *can* go out of sync

1. **Different checkouts or installs**  
   One environment points at `active-inference` commit A, another at B → MJCF, `config.py`, or `sensors.py` can differ. A non-editable wheel of `active-inference` pinned to an old version will **not** pick up repo edits until you reinstall.

2. **Separate OS processes**  
   Lab and demo are **two processes** with **two** `mjModel` instances and **two** copies of imported `config` / neuron state. Editing physics in the lab UI **does not** change the demo process until you apply equivalent patches there or restart both from the same on-disk files.

3. **Runtime-only mutations (lab)**  
   - **`/api/patch`** (registry): live setters may update `config` module globals (e.g. neuromod gains) or `mjModel` fields depending on the spec. These exist **only in memory** for that lab process.  
   - **`/api/body/patch`**: edits the live `mjModel` (joint damping, `opt.*`, pair friction, …). Again, **not** written back to `body_model.xml` by default.  
   The demo server never receives these calls.

4. **Queued “rebuild” parameters (lab only)**  
   Some schema paths are tagged `rebuild`: they stash pending values and take full effect after **Apply pending** + reset, which may reload logic that re-reads `config` or rebuilds the body. The demo server has **no** equivalent queue; it always reflects whatever the factory read at `reset()`.

5. **`CElegansBody._apply_joint_limits()`**  
   At init/reset, hinge ranges in `mjModel` are overwritten from `JOINT_ANGLE_MAX_RAD` even if the XML still contains a wider placeholder range. Both backends run the same `CElegansBody` code, so they stay consistent **for a given** `JOINT_ANGLE_MAX_RAD` value in that process.

## Practical rule

- **On-disk parity:** Keep one `active-inference` tree; reinstall / restart after changing MJCF or Python defaults.  
- **Runtime parity:** Treat lab hot-patches as **session-local** unless you export values into versioned config or MJCF. The demo server will only match after you mirror those values in code/XML or drive the same patches.

## See also

- [MuJoCo body](mujoco-body.md) — MJCF structure and runtime joint limits  
- [Biological constants](biological-constants.md) — `config.py` table aligned with the lab  
- `celegans-live-demo/lab/parameters/mujoco_engine_params.py` — every exposed `mjOption` path for the lab UI  
