"""Download and trim corpus texts listed in corpus/sources*.csv (sources.csv, sources_main.csv, ...).

sources.csv columns:
  text_id, title, kind (pg | ia | own), ref (Gutenberg number or archive.org identifier; several joined by ';'),
  translator, year, start (first words of the work proper), end (first words after it ends; empty = to the end),
  drop (regexes of lines to remove, joined by ' || '), note,
  filter (optional: "english" drops paragraphs that are not English, for sources printed with the
  original text on facing pages)

The untouched download goes to corpus/source/, the trimmed work to corpus/raw/<text_id>.txt.
Trimming keeps only the work itself: no title pages, translator prefaces, introductions, indexes, or ads.
"""
from __future__ import annotations

import csv
import json
import re
import time
import urllib.request
from pathlib import Path

UA = {"User-Agent": "lgdb-corpus/0.1 (research text download)"}


def _get(url: str, tries: int = 3) -> bytes:
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def download(kind: str, ref: str) -> str:
    if kind == "pg":
        txt = _get(f"https://www.gutenberg.org/cache/epub/{ref}/pg{ref}.txt").decode("utf-8", "replace")
        m = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", txt)
        e = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", txt)
        return txt[m.end() if m else 0: e.start() if e else len(txt)]
    if kind == "ia":
        meta = json.loads(_get(f"https://archive.org/metadata/{ref}"))
        names = [f["name"] for f in meta.get("files", []) if f["name"].endswith("_djvu.txt")]
        if not names:
            names = [f["name"] for f in meta.get("files", []) if f["name"].endswith(".txt")]
        if not names:
            raise FileNotFoundError(f"no text layer for {ref}")
        return _get(f"https://archive.org/download/{ref}/{urllib.request.quote(names[0])}").decode("utf-8", "replace")
    if kind == "own":
        raise FileNotFoundError("own copy: put the plain text at corpus/source/<text_id>.txt")
    raise ValueError(kind)


def _find(text: str, marker: str, start: int = 0) -> int:
    """Locate a marker, tolerant to whitespace and OCR line breaks."""
    pat = r"\s+".join(re.escape(w) for w in marker.split())
    m = re.search(pat, text[start:])
    if not m:
        raise ValueError(f"marker not found: {marker!r}")
    return start + m.start()


EN = set("the and of to in that is it you he for not with his be as on are this but they all have from by".split())


def english_only(text: str, min_share: float = 0.12) -> str:
    """Keep paragraphs whose share of common English words is at least min_share."""
    keep = []
    for para in re.split(r"\n\s*\n", text):
        words = re.findall(r"[A-Za-z']+", para.lower())
        if len(words) < 4 or sum(w in EN for w in words) / len(words) >= min_share:
            keep.append(para)
    return "\n\n".join(keep)


def trim(text: str, start: str, end: str, drop: str) -> str:
    text = text.replace("\f", "\n")  # page breaks from PDF extraction would hide line starts from ^
    a = _find(text, start) if start else 0
    b = _find(text, end, a + 1) if end else len(text)
    body = text[a:b]
    for pat in [p for p in (drop or "").split(" || ") if p]:
        body = re.sub(pat, "", body, flags=re.M)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip() + "\n"


def run(corpus: Path, only: set[str] | None = None, refresh: bool = False) -> list[dict]:
    src_dir, raw_dir = corpus / "source", corpus / "raw"
    src_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    report = []
    rows = [row for f in sorted(corpus.glob("sources*.csv")) for row in csv.DictReader(open(f, encoding="utf-8"))]
    for row in rows:
        tid = row["text_id"]
        if only and tid not in only:
            continue
        cache = src_dir / f"{tid}.txt"
        try:
            if row["kind"] == "own" and not cache.exists():
                raise FileNotFoundError(f"own copy missing: put the plain text at {cache}")
            if row["kind"] != "own" and (refresh or not cache.exists()):
                parts = [download(row["kind"], r.strip()) for r in row["ref"].split(";") if r.strip()]
                cache.write_text("\n\n".join(parts), encoding="utf-8")
            body = trim(cache.read_text(encoding="utf-8"), row["start"], row["end"], row.get("drop", ""))
            if (row.get("filter") or "").strip() == "english":
                body = english_only(body)
            if len(body.split()) < 100:
                raise ValueError(f"only {len(body.split())} words after trimming")
            (raw_dir / f"{tid}.txt").write_text(body, encoding="utf-8")
            report.append({"text_id": tid, "words": len(body.split()), "ok": True})
        except Exception as e:  # keep going, report the failure
            report.append({"text_id": tid, "ok": False, "error": str(e)[:200]})
    return report
