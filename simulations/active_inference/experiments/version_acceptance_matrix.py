"""Run the complete fail-closed V1--V4 embodied acceptance matrix.

The matrix is the handoff command for a trustworthy result.  Every strict
child is a ten-world, five-seed embodied suite; its own catalog derives the
horizon from the hardest declared world.  Independent children run in
parallel processes, while each writes raw tick traces and ``acceptance.json``.
This coordinator never converts a failed or inconclusive child into a pass and
records its stdout/stderr.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from simulations.active_inference.experiments.matrix_protocol import PROTOCOL_SEEDS, WORLD_COUNT


HERE = Path(__file__).resolve().parent
COMMON = (
    ("food_collection_causal", "simulations.active_inference.experiments.embodied_food_collection_causal"),
    ("toxin_escape_causal", "simulations.active_inference.experiments.embodied_toxin_escape_causal"),
    ("headon_toxin_causal", "simulations.active_inference.experiments.embodied_headon_toxin_causal"),
)


def _experiments(version: str):
    items = list(COMMON)
    if version in {"v2", "v3"}:
        items.append(("mb_valence_causal", "simulations.active_inference.experiments.embodied_mb_valence_causal"))
    if version == "v3":
        items.extend((
            ("arbiter_explore_causal", "simulations.active_inference.experiments.embodied_arbiter_explore_causal"),
            ("metabolic_rest_causal", "simulations.active_inference.experiments.embodied_metabolic_rest_causal"),
        ))
    if version == "v4":
        items.append(("obstacle_detour_causal", "simulations.active_inference.experiments.embodied_obstacle_detour_causal"))
    return items


def _acceptance(path: Path) -> dict:
    value = path / "acceptance.json"
    if not value.exists():
        return {"passed": False, "error": f"missing acceptance.json at {value}"}
    try:
        return json.loads(value.read_text())
    except json.JSONDecodeError as exc:
        return {"passed": False, "error": f"invalid acceptance.json: {exc}"}


def _file_hashes(root: Path) -> dict[str, str]:
    """Attest every child evidence file after it has been validated."""
    hashes: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[str(path.relative_to(root))] = digest
    return hashes


def _run_child(*, args, version: str, label: str, module: str, target: Path) -> tuple[str, dict]:
    """Run one complete causal suite in a separate process.

    The matrix parallelizes at suite boundaries.  Each child still serializes
    its own world/seed records, which keeps MuJoCo and PAULA state isolated and
    makes a partial child resumable without sharing mutable simulator state.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [args.python, "-m", module, "--version", version, "--protocol",
               "--seeds", *map(str, args.seeds), "--output", str(target)]
    try:
        run = subprocess.run(command, cwd=HERE.parents[2], text=True, capture_output=True)
    except OSError as exc:
        run = None
        stdout, stderr, returncode = "", str(exc), 127
    else:
        stdout, stderr, returncode = run.stdout, run.stderr, run.returncode
    acceptance = _acceptance(target)
    validator_path = target / "validator.json"
    validator_cmd = [args.python, "-m", "simulations.active_inference.experiments.version_evidence_validator",
                     str(target), "--output", str(validator_path)]
    validator_run = subprocess.run(validator_cmd, cwd=HERE.parents[2], text=True, capture_output=True)
    try:
        validator = json.loads(validator_path.read_text())
    except (OSError, json.JSONDecodeError):
        validator = {"valid": False, "failures": ["validator did not write a JSON report"]}
    record = {
        "command": command,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "acceptance": acceptance,
        "validator": {
            "command": validator_cmd,
            "returncode": validator_run.returncode,
            "stdout": validator_run.stdout,
            "stderr": validator_run.stderr,
            "report": validator,
        },
        "attested_files": _file_hashes(target) if target.exists() else {},
    }
    return f"{version}/{label}", record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", nargs="+", choices=("v1", "v2", "v3", "v4"), default=("v1", "v2", "v3", "v4"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PROTOCOL_SEEDS))
    parser.add_argument("--workers", type=int, default=10,
                        help="parallel causal-suite processes (bounded to the task count)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable, help="interpreter used for child experiments")
    args = parser.parse_args(argv)
    if tuple(args.seeds) != PROTOCOL_SEEDS:
        parser.error(f"acceptance matrix requires exactly five protocol seeds: {list(PROTOCOL_SEEDS)}")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = [(version, label, module, args.output / version / label)
             for version in args.versions for label, module in _experiments(version)]
    worker_count = min(args.workers, len(tasks))
    manifest = {
        "experiment": "version_acceptance_matrix",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": args.versions,
        "seeds": args.seeds,
        "protocol": {
            "name": "embodied_matrix",
            "version": "1.0",
            "world_count_per_strict_suite": WORLD_COUNT,
            "seed_count": len(PROTOCOL_SEEDS),
            "seeds": list(PROTOCOL_SEEDS),
            "horizon_policy": "each child derives max(world_catalog[*].completion_steps)",
            "parallel": True,
            "workers": worker_count,
        },
        "children": {},
    }
    audit_dir = args.output / "integrity"
    audit_dir.mkdir()
    audit_cmd = [args.python, "-m", "simulations.active_inference.experiments.version_integrity_audit",
                 "--output", str(audit_dir / "audit.json")]
    audit_run = subprocess.run(audit_cmd, cwd=HERE.parents[2], text=True, capture_output=True)
    integrity_record = json.loads((audit_dir / "audit.json").read_text()) if (audit_dir / "audit.json").exists() else {"passed": False}
    manifest["integrity_audit"] = {"command": audit_cmd, "returncode": audit_run.returncode,
                                   "stdout": audit_run.stdout, "stderr": audit_run.stderr,
                                   "acceptance": integrity_record}
    failures = []
    if audit_run.returncode != 0 or not integrity_record.get("passed", False):
        failures.append("version_integrity_audit failed")

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="embodied-matrix") as pool:
        futures = [pool.submit(_run_child, args=args, version=version, label=label,
                               module=module, target=target)
                   for version, label, module, target in tasks]
        for future in as_completed(futures):
            key, record = future.result()
            manifest["children"][key] = record
            if record["returncode"] != 0 or not record["acceptance"].get("passed", False):
                failures.append(f"{key} failed")
            # A behaviourally red child deliberately makes the validator's
            # process exit non-zero too.  Only structural invalidity is an
            # additional integrity failure here; neither condition is hidden.
            if not record["validator"]["report"].get("valid", False):
                failures.append(f"{key} evidence integrity failed")

    manifest["passed"] = not failures
    manifest["failures"] = failures
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": not failures, "failures": failures, "output": str(args.output)}, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
