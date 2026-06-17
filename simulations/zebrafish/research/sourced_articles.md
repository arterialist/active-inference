# Sourced Articles

Fetched on 2026-05-13 with `source_open_articles.py`.

Local fetched files live under:

- `articles/pdfs/`: verified PDF files only.
- `articles/html/`: article landing pages and non-PDF responses from attempted
  PDF links.
- `articles/index.json`: machine-readable fetch status for every attempted
  article source.

Verified PDFs fetched:

- `svara-2022-nat-methods.pdf`
- `dunn-2016-exploratory-locomotion.pdf`
- `fishnet-2007.pdf`
- `fishchip-2023.pdf`
- `pyzebrascope-2022.pdf`
- `campari2-2026.pdf`

HTML article pages fetched:

- Ahrens 2012 motor adaptation.
- Ahrens 2013 light-sheet whole-brain imaging.
- Dunn 2016 exploratory locomotion.
- FishChip 2023 chemosensory behavior.
- FishNet 2007 anatomy database.
- Hildebrand 2017 whole-brain ssEM.
- PyZebrascope 2022.
- Randlett/Z-Brain 2015.
- Svara 2022 synapse-level reconstruction.
- CaMPARI2 2026.

Blocked or redirected sources:

- bioRxiv Fish1 and Boulanger-Weill preprints returned Cloudflare 403 from this
  command-line environment, though the DOI/source URLs remain in the manifest.
- Cell Press article/PDF URLs returned redirect/challenge responses here.
- Journal of Experimental Biology PDF/article URLs returned Cloudflare 403 here.
- GigaScience/OUP PDF/article URL returned Cloudflare 403 here.
- Some Nature PDF URLs returned HTML rather than PDF; those responses were
  stored as `*-pdf-response.html`, not as PDFs.

