"""Mechanical corpus selection from P1 candidates (preregistered rule, no judgment calls).

1. A candidate counts for its cell only if the format the search agent reported equals the cell format.
2. A title found in several cells stays only in the cell where it ranked highest (smallest rank);
   ties go to the collection cell, then to the earlier cell in the grid.
3. A text is available if a full English text is legally usable: a translation (or an English
   original) published before 1931 (US public domain as of 2026), or the title is listed in
   human/owned_texts.csv (copies the researcher owns, used for analysis only).
   corpus/availability.csv overrides this by title after a bibliographic check: yes for a verified
   other full translation or a public-domain status found (for example copyright not renewed), no when
   the only old translation covers less than 2/3 of the work or is a translator's selection. A selection
   made by a modern translator would carry the translator's era into the sample.
4. Per cell, the first 6 available texts by rank are taken. round(n / 3), at most 2, go to the
   pilot at random (seeded); the rest go to the main corpus.
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import numpy as np

PD_BEFORE = 1931
PER_CELL = 6
PILOT_MAX = 2


def _year(text: str) -> int | None:
    years = [int(y) for y in re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", text or "")]
    return min(years) if years else None


def rule_year(row: dict) -> int | None:
    """Year of the English text: the translation year, or the original date for English originals."""
    tr = row["translation"]
    if "original english" in tr.lower():
        return _year(row["date"])
    return _year(tr)


def _key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def select(candidates: Path, owned: Path | None, seed: int, grid: list[tuple[str, str, str]],
           overrides: Path | None = None) -> list[dict]:
    rows = list(csv.DictReader(open(candidates, encoding="utf-8")))
    owned_titles = set()
    if owned and owned.exists():
        owned_titles = {_key(r["title"]) for r in csv.DictReader(open(owned, encoding="utf-8"))}
    override = {}
    if overrides and overrides.exists():
        override = {_key(r["title"]): r for r in csv.DictReader(open(overrides, encoding="utf-8"))}
    order = {cell: i for i, cell in enumerate(grid)}

    def cell(r):
        return (r["period"], r["region"], r["cell_format"])

    for r in rows:
        r["rank"] = int(r["rank"])
        r["status"] = ""
        if r["format_reported"].strip().lower() != r["cell_format"]:
            r["status"] = "format mismatch"

    best = {}
    for r in rows:
        if r["status"]:
            continue
        k = _key(r["title"])
        score = (r["rank"], 0 if r["cell_format"] == "collection" else 1, order.get(cell(r), 99))
        if k not in best or score < best[k][0]:
            best[k] = (score, r)
    for r in rows:
        if not r["status"] and best[_key(r["title"])][1] is not r:
            b = best[_key(r["title"])][1]
            r["status"] = f"kept in {b['period']} / {b['region']} / {b['cell_format']}"

    for r in rows:
        y = rule_year(r)
        r["english_year"] = y if y is not None else ""
        r["available"] = (y is not None and y < PD_BEFORE) or _key(r["title"]) in owned_titles
        o = override.get(_key(r["title"]))
        if o is not None and _key(r["title"]) not in owned_titles:
            r["available"] = o["available"].strip().lower() == "yes"
            r["translation"] = o.get("translation") or r["translation"]
        if not r["status"] and not r["available"]:
            r["status"] = "no usable full text yet"

    out = []
    for c in grid:
        ok = sorted([r for r in rows if cell(r) == c and not r["status"]], key=lambda r: r["rank"])[:PER_CELL]
        n_pilot = min(PILOT_MAX, int(round(len(ok) / 3)))
        h = int(hashlib.sha256(f"{seed}|{'|'.join(c)}".encode()).hexdigest()[:8], 16)
        pilot = set(np.random.default_rng(h).choice(len(ok), size=n_pilot, replace=False).tolist()) if ok else set()
        for i, r in enumerate(ok):
            r["status"] = "selected"
            r["split"] = "pilot" if i in pilot else "main"
    for r in rows:
        if r["status"] == "" and r["available"]:
            r["status"] = "available, beyond first 6"
        r.setdefault("split", "")
        out.append(r)
    return out


def write(rows: list[dict], path: Path) -> dict:
    cols = ["period", "region", "cell_format", "rank", "title", "author", "date", "format_reported",
            "translation", "english_year", "available", "status", "split"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return {"pilot": sum(r["split"] == "pilot" for r in rows), "main": sum(r["split"] == "main" for r in rows)}


META_COLS = ["text_id", "title", "author", "year", "period", "region", "format",
             "translator", "translation_year", "split", "mask_terms"]


def build_meta(corpus: Path) -> dict:
    """corpus/meta.csv from selected candidates that have a trimmed text in corpus/raw/."""
    sel = {_key(r["title"]): r for r in csv.DictReader(open(corpus / "selection.csv", encoding="utf-8"))
           if r["status"] == "selected"}
    masks = {}
    if (corpus / "masks.csv").exists():
        masks = {r["text_id"]: r["mask_terms"] for r in csv.DictReader(open(corpus / "masks.csv", encoding="utf-8"))}
    rows, missing = [], []
    for src in [r for f in sorted(corpus.glob("sources*.csv")) for r in csv.DictReader(open(f, encoding="utf-8"))]:
        s = sel.get(_key(src["title"]))
        if s is None or not (corpus / "raw" / f"{src['text_id']}.txt").exists():
            missing.append(src["text_id"])
            continue
        rows.append({"text_id": src["text_id"], "title": re.sub(r"\s*\(.*?\)", "", s["title"]).strip(),
                     "author": s["author"].replace("attributed to ", ""), "year": s["date"],
                     "period": s["period"], "region": s["region"], "format": s["cell_format"],
                     "translator": src["translator"], "translation_year": src["year"], "split": s["split"],
                     "mask_terms": masks.get(src["text_id"], "")})
    with open(corpus / "meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=META_COLS)
        w.writeheader()
        w.writerows(rows)
    return {"texts": len(rows), "not in selection or no text": missing,
            "pilot": sum(r["split"] == "pilot" for r in rows), "main": sum(r["split"] == "main" for r in rows)}
