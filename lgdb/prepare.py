"""Corpus preparation: fragments, anonymization, blinded ids.

Input: corpus/meta.csv (1 row per text) and corpus/raw/<text_id>.txt.
Output: work/fragments.jsonl (what blind stages see: id and text only) and
keys/fragments_key.csv (the mapping back to texts, never shown to agents).
"""
from __future__ import annotations

import csv
import json
import random
import re
import secrets
from pathlib import Path

META_COLUMNS = ["text_id", "title", "author", "year", "period", "region", "format",
                "translator", "translation_year", "split", "mask_terms"]


def read_meta(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in META_COLUMNS if c not in rows[0]] if rows else META_COLUMNS
    if missing:
        raise ValueError(f"meta.csv is missing columns: {missing}")
    return rows


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def word_count(s: str) -> int:
    return len(s.split())


def chunk(paras: list[str], target: int) -> list[str]:
    """Consecutive paragraph-aligned chunks of about `target` words."""
    out, cur, n = [], [], 0
    for p in paras:
        cur.append(p)
        n += word_count(p)
        if n >= target:
            out.append("\n\n".join(cur))
            cur, n = [], 0
    if cur:
        if out and n < target // 3:
            out[-1] += "\n\n" + "\n\n".join(cur)
        else:
            out.append("\n\n".join(cur))
    return out


def fragments_for_text(text: str, whole_max_words: int = 12000, n_fragments: int = 8,
                       fragment_words: int = 1500) -> list[tuple[int, str]]:
    """Whole text in ~1500-word chunks if short; otherwise 8 evenly spaced chunks.
    Returns (position_index, fragment) pairs."""
    chunks = chunk(paragraphs(text), fragment_words)
    if word_count(text) <= whole_max_words or len(chunks) <= n_fragments:
        return list(enumerate(chunks))
    step = len(chunks) / n_fragments
    picks = sorted({int(step * i + step / 2) for i in range(n_fragments)})
    return [(i, chunks[i]) for i in picks]


def parse_mask_terms(spec: str) -> list[tuple[str, str]]:
    """'Krishna:DEITY; Arjuna:PERSON; Atomic Habits:METHOD' -> [(term, tag), ...]"""
    out = []
    for part in (spec or "").split(";"):
        if ":" in part:
            term, tag = part.rsplit(":", 1)
            if term.strip():
                out.append((term.strip(), tag.strip().upper() or "NAME"))
    return sorted(out, key=lambda t: -len(t[0]))  # longest first


def anonymize(text: str, row: dict, use_ner: bool = False) -> str:
    terms = [(row.get("title", ""), "TITLE"), (row.get("author", ""), "PERSON")]
    terms += parse_mask_terms(row.get("mask_terms", ""))
    for term, tag in terms:
        if term and len(term) > 1:
            text = re.sub(r"(?<!\w)" + re.escape(term) + r"(?!\w)", f"[{tag}]", text, flags=re.I)
    if use_ner:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            for ent in reversed(doc.ents):
                if ent.label_ == "PERSON":
                    text = text[:ent.start_char] + "[PERSON]" + text[ent.end_char:]
        except Exception as e:  # spaCy optional; manual mask_terms still apply
            print(f"NER skipped: {e}")
    return text


def prepare(corpus_dir: Path, work_dir: Path, keys_dir: Path, split: str | None = None,
            synthetic: Path | None = None, use_ner: bool = False, seed: int = 0, **frag_kw):
    meta = read_meta(corpus_dir / "meta.csv")
    if split:
        meta = [r for r in meta if r["split"] == split]
    work_dir.mkdir(parents=True, exist_ok=True)
    keys_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for row in meta:
        raw = (corpus_dir / "raw" / f"{row['text_id']}.txt").read_text(encoding="utf-8")
        for pos, frag in fragments_for_text(raw, **frag_kw):
            items.append((row["text_id"], pos, anonymize(frag, row, use_ner)))
    if synthetic and synthetic.exists():
        for line in synthetic.read_text(encoding="utf-8").splitlines():
            if line.strip():
                p = json.loads(line)
                items.append((f"SYN::{p['passage_id']}", 0, p["text"]))
    random.Random(seed).shuffle(items)
    with open(work_dir / "fragments.jsonl", "w", encoding="utf-8") as fo, \
         open(keys_dir / "fragments_key.csv", "w", newline="", encoding="utf-8") as fk:
        w = csv.writer(fk)
        w.writerow(["fragment_id", "text_id", "position"])
        for text_id, pos, frag in items:
            fid = secrets.token_hex(4)
            fo.write(json.dumps({"fragment_id": fid, "text": frag}, ensure_ascii=False) + "\n")
            w.writerow([fid, text_id, pos])
    return len(items)
