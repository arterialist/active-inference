#!/usr/bin/env python3
"""Download open/public zebrafish article HTML and PDFs into this directory."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
SOURCE_FILE = BASE / "article_sources.json"
ARTICLES = BASE / "articles"
PDFS = ARTICLES / "pdfs"
HTML = ARTICLES / "html"
INDEX = ARTICLES / "index.json"
USER_AGENT = "active-inference-zebrafish-source-corpus/0.1"


def slug(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    return text.strip("-")[:120]


def fetch(
    url: str,
    path: Path,
    *,
    expect_pdf: bool = False,
    fallback_path: Path | None = None,
) -> dict[str, Any]:
    if path.exists() and path.stat().st_size > 0:
        if expect_pdf and not path.read_bytes()[:4] == b"%PDF":
            path.unlink()
        else:
            return {
                "status": "exists",
                "bytes": path.stat().st_size,
                "path": str(path),
            }
    if fallback_path is not None and fallback_path.exists() and fallback_path.stat().st_size > 0:
        return {
            "status": "exists",
            "bytes": fallback_path.stat().st_size,
            "path": str(fallback_path),
        }
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            status = getattr(resp, "status", 200)
            content_type = resp.headers.get("content-type", "")
            data = resp.read()
    except urllib.error.HTTPError as exc:
        return {"status": "http_error", "code": exc.code, "url": url}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": repr(exc), "url": url}
    if status >= 400:
        return {"status": "http_error", "code": status, "url": url}
    if expect_pdf and not data[:4] == b"%PDF":
        if fallback_path is not None:
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            fallback_path.write_bytes(data)
        return {
            "status": "not_pdf",
            "bytes": len(data),
            "content_type": content_type,
            "path": str(fallback_path) if fallback_path is not None else None,
            "url": url,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "status": "downloaded",
        "bytes": len(data),
        "content_type": content_type,
        "path": str(path),
        "url": url,
    }


def main() -> None:
    payload = json.loads(SOURCE_FILE.read_text(encoding="utf-8"))
    PDFS.mkdir(parents=True, exist_ok=True)
    HTML.mkdir(parents=True, exist_ok=True)
    records = []
    for item in payload["sources"]:
        sid = item["id"]
        rec: dict[str, Any] = {
            "id": sid,
            "title": item.get("title"),
            "doi": item.get("doi"),
            "license_note": item.get("license_note"),
            "html_url": item.get("html_url"),
            "pdf_url": item.get("pdf_url"),
        }
        if item.get("html_url"):
            html_path = HTML / f"{slug(sid)}.html"
            rec["html"] = fetch(item["html_url"], html_path)
            time.sleep(0.25)
        if item.get("pdf_url"):
            pdf_path = PDFS / f"{slug(sid)}.pdf"
            fallback_path = HTML / f"{slug(sid)}-pdf-response.html"
            rec["pdf"] = fetch(
                item["pdf_url"],
                pdf_path,
                expect_pdf=True,
                fallback_path=fallback_path,
            )
            time.sleep(0.25)
        records.append(rec)
        print(
            f"{sid}: html={rec.get('html', {}).get('status')} "
            f"pdf={rec.get('pdf', {}).get('status')}"
        )

    INDEX.write_text(json.dumps({"sources": records}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {INDEX}")


if __name__ == "__main__":
    main()
