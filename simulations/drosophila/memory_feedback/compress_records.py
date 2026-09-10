"""Reclaim space with transparent APFS compression of completed raw arrays.

Logical file bytes, paths and hashes remain unchanged; numpy memory maps keep
working normally. Only completed memory-feedback courses are eligible. Each
copy is hashed before atomic replacement, and every replacement is audited.
Active/incomplete records, source data and checkpoints are excluded.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from ..connectome import sha256


def run(base, *, free_gib=6.):
    if sys.platform != "darwin":
        raise RuntimeError("This utility requires macOS transparent filesystem compression")
    base = Path(base)
    candidates = []
    for p in base.glob("memory-*/summary.json"):
        r = json.loads(p.read_text())
        complete = (isinstance(r, list) and r and r[-1].get("name") == "final_recovery") or (
            isinstance(r, dict) and r.get("phases") and r["phases"][-1].get("name") == "retention" and "artifacts" in r)
        if complete:
            candidates.extend(f for f in p.parent.glob("*.npy") if not (f.stat().st_flags & 32))
    candidates.sort(key=lambda f: (-f.stat().st_blocks, str(f)))
    audit = base/"transparent-compression.jsonl"
    saved = 0; files = 0
    for path in candidates:
        if shutil.disk_usage(base).free >= free_gib*1024**3:
            break
        before = path.stat(); digest = sha256(path)
        with tempfile.TemporaryDirectory(prefix=".compression-", dir=path.parent) as d:
            temporary = Path(d)/path.name
            subprocess.run(["/usr/bin/ditto", "--hfsCompression", "--noclone", str(path), str(temporary)], check=True)
            if sha256(temporary) != digest:
                raise RuntimeError("Compression changed logical bytes: "+str(path))
            after = temporary.stat()
            if after.st_blocks >= before.st_blocks:
                continue
            # Refuse a concurrently modified source. Completed records should
            # have no writers; this also protects against unexpected changes.
            current = path.stat()
            if (current.st_size, current.st_mtime_ns, current.st_ino) != (before.st_size, before.st_mtime_ns, before.st_ino):
                raise RuntimeError("Source changed during compression: "+str(path))
            os.replace(temporary, path)
            assert sha256(path) == digest
            record = dict(path=str(path.relative_to(base)), sha256=digest, logical_bytes=before.st_size,
                allocated_before=before.st_blocks*512, allocated_after=after.st_blocks*512,
                compressed_flag=bool(path.stat().st_flags & 32), utility_sha256=sha256(Path(__file__)))
            with audit.open("a") as out:
                out.write(json.dumps(record, sort_keys=True)+"\n"); out.flush(); os.fsync(out.fileno())
            files += 1; saved += record["allocated_before"]-record["allocated_after"]
    result = dict(files=files, allocated_bytes_saved=saved, free_bytes=shutil.disk_usage(base).free,
        audit=str(audit), logical_contents="unchanged, SHA-256 verified")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path); p.add_argument("--free-gib", type=float, default=6.)
    a=p.parse_args(); run(a.records, free_gib=a.free_gib)
