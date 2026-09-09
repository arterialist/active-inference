"""Manifested cleanup of explicitly selected, completed raw-data replications.

Only the 18 older evidence-population replications listed below are in scope.
Seed 11 runs, all recent real-media work, code, configuration, checkpoints,
summaries, schedules, audit outputs and source media are never deletion targets.
Per pruned run retain first/last training traces and synchronous clean probes.
Default is read-only; --apply requires the exact previously generated manifest.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT = Path('/Users/arterialist/Projects/agi-research/active-inference/.live/research')
TARGETS = tuple(f'20260909_{family}_{mapping}_seed{seed}'
    for family in ('association_evidence_diverse','association_evidence_homogeneous','evidence_embedded')
    for mapping in ('paired','swapped') for seed in (23,44,77))
RAW = re.compile(r'(?:train-\d{3}|(?:32|128)-.+)\.npz$')
GOLDEN = re.compile(r'(?:32|128)-(?:trained-)?clean-cue[01]-sample0--synchronous\.npz$')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def candidate_files(root=ROOT):
    result=[]; retained=[]
    for name in TARGETS:
        run=root/name
        if run.is_symlink() or not run.is_dir(): raise ValueError(f'Unsafe or missing run: {run}')
        for required in ('manifest.json','config.json','summary.json'):
            if not (run/required).is_file(): raise ValueError(f'Incomplete run: {run}')
        trains=sorted(p.name for p in run.iterdir() if re.fullmatch(r'train-\d{3}\.npz',p.name))
        if not trains: raise ValueError(f'No training record: {run}')
        for p in sorted(run.iterdir()):
            if p.is_symlink(): raise ValueError(f'Symlink requires manual inspection: {p}')
            if not p.is_file(): continue
            if RAW.fullmatch(p.name) and p.name not in (trains[0],trains[-1]) and not GOLDEN.fullmatch(p.name):
                result.append(p)
            else: retained.append(p)
    return result,retained


def plan(destination):
    destination=Path(destination).resolve()
    if destination.exists(): raise FileExistsError(destination)
    files,retained=candidate_files()
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),root=str(ROOT),
        authority='User requested old raw-data cleanup because only 1 GiB remained',
        recovery='Permanent removal. Regenerate missing ticks from retained code/configuration and stimuli; exact reproducibility must be rechecked against stored hashes. No backup is asserted.',
        evidence_status='Historical summaries remain. Full raw re-audit of pruned replications requires regeneration. Seed 11 and per-run golden traces remain fully readable.',
        free_bytes_before=shutil.disk_usage(ROOT).free,
        files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files],
        retained=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in retained])
    result['bytes_selected']=sum(p['bytes'] for p in result['files'])
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(dict(manifest=str(destination),files=len(files),bytes=result['bytes_selected'],retained_files=len(retained))))


def apply(manifest):
    manifest=Path(manifest).resolve(); result=json.loads(manifest.read_text())
    if result['root']!=str(ROOT): raise ValueError('Unexpected root')
    selected,_=candidate_files(); expected={str(p) for p in selected}
    if {r['path'] for r in result['files']}!=expected: raise ValueError('Candidates changed')
    # Verify all files and retained evidence before the first deletion.
    for row in result['files']+result['retained']:
        p=Path(row['path'])
        if p.parent.name not in TARGETS or p.parent.parent!=ROOT or p.is_symlink():
            raise ValueError('Target escaped the explicitly resolved run list')
        if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']: raise ValueError(f'File changed: {p}')
    journal=manifest.with_suffix('.deleted.jsonl')
    with journal.open('x') as f:
        for row in result['files']:
            p=Path(row['path']);p.unlink()
            f.write(json.dumps(row)+'\n');f.flush()
    for row in result['retained']:
        if sha(Path(row['path']))!=row['sha256']: raise ValueError('Retained evidence changed')
    removed=sum(row['bytes'] for row in result['files'])
    completion=dict(removed_files=len(result['files']),removed_bytes=removed,
        free_bytes_after=shutil.disk_usage(ROOT).free,manifest_sha256=sha(manifest),journal_sha256=sha(journal))
    with manifest.with_suffix('.complete.json').open('x') as f: json.dump(completion,f,indent=2)
    print(json.dumps(completion))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);p.add_argument('--apply',action='store_true')
    a=p.parse_args();(apply if a.apply else plan)(a.manifest)
