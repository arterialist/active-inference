# Data Policy

Do not commit bulk datasets, PDFs under restrictive licenses, CAVE credentials,
Google-account-only data exports, or multi-GB imaging/EM archives.

Safe to commit:

- Source links, DOIs, dataset IDs, citations, access notes, checksums published
  by repositories, and small hand-written summaries.
- Scripts that download public data into ignored local cache directories.
- Derived tiny fixtures created for tests, if their license permits reuse.

This directory also includes `article_sources.json` and
`source_open_articles.py` for fetching public article pages and PDFs. Running
the script writes fetched files under `research/articles/` plus an `index.json`
with success/failure status. Re-check publisher terms before redistributing
those fetched files outside the local workspace.

Use local ignored paths for bulk data:

- `data/zebrafish/raw/`
- `data/zebrafish/cache/`
- `data/zebrafish/derived/`

Recommended future downloader behavior:

1. Require an explicit source ID from `source_manifest.json`.
2. Print expected download size before fetching anything over 100 MB.
3. Store raw files under `data/zebrafish/raw/<source-id>/`.
4. Store reduced simulation artifacts under `data/zebrafish/derived/`.
5. Preserve source URL, DOI, checksum, and transform code in metadata.
