# Left APL spatial reconstruction, 10 September 2026

The spatial mapping is now resolved for the left APL used by the PAULA
preparation. Every linked neuron-pair count agrees with the existing graph.
This adds anatomical constraints for a local APL model. It is not a new neural
simulation or a reproduction of calcium imaging.

## Source and identity

The anonymous public [FlyWire CATMAID server](https://fafb-flywire.catmaid.org/)
identifies project 1 as **FlyWire m783**. An exact neuron-name query maps
FlyWire root `720575940624547622` to CATMAID skeleton `48201714`, neuron
`48201715`, with APL and left-side annotations. These identifiers are different
namespaces. No nearest-neighbor guess or cross-specimen ID conversion was used.

The acquisition stores the raw project, identity, relation-type, compact-tree,
connector/partner and partner-name responses. It uses the documented read-only
[CATMAID endpoints](https://catmaid.readthedocs.io/en/stable/_static/api/index.html).
The public session requires the normal CSRF exchange for read-only POST queries.
No account or authentication token is needed. Request metadata, timestamps and
compressed/uncompressed hashes accompany the snapshot. This is a captured public
server state, not a claim that the live API is immutable.

## What the actual data establish

| Check | Result |
| --- | ---: |
| Tree nodes | 360,309 |
| Connected components / tree roots | 1 / 1 |
| Summed cable length | 81.722 mm |
| Local connector links | 171,007 |
| Incoming links / with linked partners | 70,914 / 68,332 |
| Outgoing links / with linked partners | 100,093 / 60,103 |
| Linked directed neuron pairs | 6,315 |
| Linked contacts | 128,435 |
| Pair-count disagreements with the imported Shiu table | 0 |
| Unlinked incoming / outgoing connectors | 2,582 / 39,990 |

The summed cable length includes all branches; it is not a soma-to-tip distance.
All 6,315 pairs agree individually, not just in their total. A separate direct
count of the raw partner records reproduces that equality. Expanding the stored
original source-row bindings also recovers every aggregate count exactly.
Weak pairs, incident cut-boundary pairs and both directions are included.

The 42,572 open connectors are unlinked in this export. Their other neurons
are not inferred. They are additional unresolved anatomy, not simulated silent
cells and not the same thing as the existing identified graph boundary. The
snapshot and parsed arrays preserve their locations. Root and treenode zero
mean an absent partner; local index and pair-row minus one mean absent bindings.
These sentinels must never be passed to a neural builder as biological IDs.

Connector coordinates and attached skeleton nodes are separate measurements in
the export. Their median separation is 0.327 µm, maximum 4.435 µm. The tree has
no zero-length edges, median edge length 0.184 µm, maximum 6.299 µm. This does
not establish that segmentation, skeleton healing, radii or attachment choices
are free of reconstruction error. Neither a connector nor a nearest tree node
specifies a measured electrical compartment.

## Why this changes the next experiment

The three glomerulus groups used to investigate calyx locality have different
input distributions on this APL:

| Selected ALPN group | Contacts onto APL | Distinct attachment nodes |
| --- | ---: | ---: |
| DM3 | 3 | 3 |
| VA1d | 26 | 26 |
| DC3 | 102 | 95 |

These groupings are anatomical queries, not odor-response vectors. VA1d includes
two cholinergic adPNs providing 10 and 15 contacts and a GABAergic vPN providing
one. The per-root identities and known/predicted transmitter labels are retained.
Equal positive stimulation of all members would be an experimental assumption,
not a measured odor response.

For each VA1d attachment, the median distance to its nearest DM3 attachment is
135.94 µm **along the tree**, whereas the median distance from DC3 attachments
to their nearest VA1d attachment is 28.71 µm. These are directed nearest-set
statistics with different sample sizes, not pairwise symmetry violations or
electrical transmission distances. One VA1d site is 754.03 µm from its nearest
DM3 site along the tree. Full distance arrays, site identities and geometry are
retained, including that extreme; no trimming or arbitrary calyx boundary was
applied. Local branch structure therefore offers a real constraint that the
pair aggregate alone could not provide.

[Amin et al., 2020](https://doi.org/10.7554/eLife.56954) measured spatially
restricted APL activity and inhibition. Their phenomenological exponential
attenuation model used neurite distances and radii; a normalized 50 µm space
constant fit their measurements better than the tested longer values. It did
not model dynamic feedback or branch-loading effects. That fit belongs to their
preparation and hemibrain reconstruction, not this FlyWire specimen.
[Prisco et al., 2021](https://doi.org/10.7554/eLife.74172) measured calyx APL
locality and PN-bouton/KC-claw calcium responses. KC soma spikes cannot substitute
for a claw-calcium assay. These studies constrain the next model, but do not
license treating a distance kernel as a complete cellular mechanism.

The next neural change can now distribute input and read local release using
the identified contacts while retaining one APL identity, actual partner
connections and plastic return paths. Intracellular propagation, release and
the calcium observation model still require explicit assumptions and tests.
The existing global graded preparation is unchanged and remains the comparison.

## Reproduce and inspect

From `active-inference/`, acquisition is explicit and bounded to 96 MiB per
response. Output directories must be new. Analysis is offline and checks the
saved raw bytes before parsing them.

```sh
uv run python -m simulations.drosophila.spatial acquire \
  .live/research/flywire783/apl-left-spatial-new
uv run python -m simulations.drosophila.spatial analyze \
  .live/research/flywire783/apl-left-spatial-new \
  .live/research/flywire783/kc-apl-left-v1 \
  .live/research/flywire783/apl-left-spatial-analysis-new
uv run python -m pytest tests/test_drosophila_connectome.py \
  tests/test_drosophila_interventions.py tests/test_drosophila_spatial.py -q
```

The analyzed acquisition is locally under
`.live/research/flywire783/apl-left-spatial-20260910/`; the complete bound analysis
is `apl-left-spatial-bound-20260910/`. The preceding `apl-left-spatial-audit-20260910/`
retains the initial audit, before full distance arrays and pair-row bindings
were added. No previous recordings were overwritten.

The final `anatomy.npz` contains exact node IDs, parent indices, coordinates,
radii, per-contact endpoint/attachment identities, connector coordinates,
original pair-table source rows and one full tree-distance array per input group.
`analysis.json` declares columns, units, identities and hashes. A mismatched
pair table can be reported but cannot be automatically joined to the contacts.

The final analysis took 5.82 seconds, with peak resident memory about 747 MiB.
Acquisition occupies 16 MiB compressed; final analysis occupies 19 MiB. The
29 new fixture tests cover topology, exact IDs, polyads, autapses, incomplete
partners, inconsistent endpoint records, release identity, raw-byte integrity
and count-preserving joins. All 68 fly-package tests pass. These are software
and anatomical checks, not biological acceptance or evidence of learning.

Snapshot manifest SHA256:
`fae817c12afd9949aa66cf831b84f1574e7bb332a04df153a6e325067a88ee90`.
Final anatomy SHA256:
`5b67141392dae7cf96d6830011ce2a90fbe6d3d472f23634d8ba1db03438e076`.
Pair-graph manifest SHA256:
`243c8594373273d0dc18c70e0c9b29162199fb024e95f4eadf8b1f0679c806d2`.
