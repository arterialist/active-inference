// One-off, scoped cleanup. Run without arguments to inventory; --apply to delete
// only unchanged files named by that inventory. No recursive removal.+ 
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = '/Users/arterialist/Projects/agi-research/active-inference/.live/research';
const stem = path.join(__dirname, '2026-09-09-intermediate-raw-cleanup');
const families = /^20260909_(association_evidence_|eligibility_media_|inhibitory_capacity_|graded_media_|contact_media_order_|mean_pooled_media_order_|media_order_|evidence_embedded_)/;
const trial = /^(train|experience)-\d+\.npz$/;
const hash = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const save = (p, data) => fs.writeFileSync(p, JSON.stringify(data, null, 2) + '\n', {flag: 'wx'});
if (!process.argv.includes('--apply')) {
  const files = [];
  for (const name of fs.readdirSync(root).sort()) {
    const dir = path.join(root, name);
    if (!families.test(name) || !fs.lstatSync(dir).isDirectory() || !fs.existsSync(path.join(dir, 'summary.json'))) continue;
    for (const prefix of ['train-', 'experience-']) {
      const names = fs.readdirSync(dir).filter(n => trial.test(n) && n.startsWith(prefix)).sort();
      // Keep the first and last training recording, every recall/intervention,
      // all checkpoints, stimuli, source/config files, summaries and audits.
      for (const file of names.slice(1, -1)) {
        const absolute = path.join(dir, file), stat = fs.lstatSync(absolute);
        if (!stat.isFile() || stat.isSymbolicLink()) throw Error('Unexpected file type: ' + absolute);
        files.push({path: absolute, bytes: stat.size, sha256: hash(absolute)});
      }
    }
  }
  const manifest = {
    created_utc: new Date().toISOString(), root,
    authority: 'User requested cleanup of old raw data due to disk pressure.',
    scope: 'Intermediate training traces in explicit older experiment families only. Current sensory timing, transplant, contrast prediction, predictive bridge and media course dependency families excluded.',
    retained: 'All non-selected files, including first/last training traces, recall/intervention traces, checkpoints, input media, configurations, summaries, audits and code.',
    evidence_limit: 'Permanent deletion, no backup asserted. Historical summaries are retained but a full training-trajectory re-audit now requires regeneration and reproducibility checks. This supersedes any earlier assertion that all seed-11 raw traces remain available for affected runs.',
    files, total_bytes: files.reduce((s, f) => s + f.bytes, 0)
  };
  save(stem + '.json', manifest);
  console.log(JSON.stringify({files: files.length, bytes: manifest.total_bytes}));
} else {
  const manifest = JSON.parse(fs.readFileSync(stem + '.json', 'utf8'));
  if (manifest.root !== root) throw Error('Wrong root');
  for (const f of manifest.files) {
    const relative = path.relative(root, f.path), parts = relative.split(path.sep);
    if (parts.length !== 2 || !families.test(parts[0]) || !trial.test(parts[1])) throw Error('Out of scope: ' + f.path);
    const stat = fs.lstatSync(f.path);
    if (!stat.isFile() || stat.isSymbolicLink() || fs.realpathSync(f.path) !== f.path || stat.size !== f.bytes || hash(f.path) !== f.sha256) throw Error('Changed file: ' + f.path);
  }
  const fd = fs.openSync(stem + '.deleted.jsonl', 'wx');
  let bytes = 0, count = 0;
  try {
    for (const f of manifest.files) {
      fs.unlinkSync(f.path);
      fs.writeSync(fd, JSON.stringify(f) + '\n');
      bytes += f.bytes; count++;
    }
    fs.fsyncSync(fd);
  } finally { fs.closeSync(fd); }
  for (const dir of new Set(manifest.files.map(f => path.dirname(f.path)))) {
    save(path.join(dir, 'INTERMEDIATE_RAW_PRUNED.json'), {
      manifest: stem + '.json', removed_files: manifest.files.filter(f => path.dirname(f.path) === dir).map(f => path.basename(f.path)),
      evidence_limit: manifest.evidence_limit
    });
  }
  save(stem + '.complete.json', {removed_files: count, removed_bytes: bytes, manifest_sha256: hash(stem + '.json'), journal_sha256: hash(stem + '.deleted.jsonl')});
  console.log(JSON.stringify({removed_files: count, removed_bytes: bytes}));
}
