"""Trial: how well do models recall the recommendations of a book from memory (P9)?

Inputs: work_memory/books_key.json (ids, titles, which books are invented controls) and
work_memory/<model>/<id>.json (P9 answers). Checks:
  1. invented controls: does each model admit it does not know them?
  2. agreement between models on the same book, against agreement across different books;
  3. grounding: for books with a cleaned full text, how many recalled recommendations have a passage
     in the text that is closer than chance (chance = recommendations recalled for other books).
"""
from __future__ import annotations

import csv
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np


def _key(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def load(root: Path):
    books = json.loads((root / "work_memory" / "books_key.json").read_text())
    models = sorted(p.name for p in (root / "work_memory").iterdir() if p.is_dir())
    ans = {}
    for m in models:
        for b in books:
            f = root / "work_memory" / m / f"{b['id']}.json"
            if f.exists():
                try:
                    ans[(m, b["id"])] = json.loads(f.read_text())
                except json.JSONDecodeError:
                    ans[(m, b["id"])] = {"knowledge_basis": "parse error", "recommendations": []}
    return books, models, ans


def embedder():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return lambda xs: np.asarray(model.encode(xs, batch_size=64, normalize_embeddings=True))
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        state = {}

        def enc(xs):
            if "v" not in state:
                raise RuntimeError("tfidf fallback needs fit; install sentence-transformers")
            return state["v"].transform(xs).toarray()
        return enc


def _sym(a: np.ndarray, b: np.ndarray) -> float:
    s = a @ b.T
    return float(0.5 * (s.max(1).mean() + s.max(0).mean()))


def windows(text: str, size: int = 120, step: int = 60) -> list[str]:
    w = text.split()
    return [" ".join(w[i:i + size]) for i in range(0, max(1, len(w) - size + 1), step)]


def run(root: Path) -> dict:
    books, models, ans = load(root)
    enc = embedder()
    real = [b for b in books if b["real"]]
    out = {"models": models, "controls": {}, "knowledge_basis": {}, "agreement": {}, "grounding": {}}

    for b in books:
        if not b["real"]:
            out["controls"][b["title"]] = {m: {"basis": ans.get((m, b["id"]), {}).get("knowledge_basis"),
                                               "n_recs": len(ans.get((m, b["id"]), {}).get("recommendations") or [])}
                                           for m in models}
    for m in models:
        kb = [str(ans.get((m, b["id"]), {}).get("knowledge_basis", "")).lower() for b in real]
        recs = [r for b in real for r in (ans.get((m, b["id"]), {}).get("recommendations") or [])]
        cond = [str(r.get("conditions", "none")).strip().lower() not in ("none", "", "n/a") for r in recs]
        word = [str(r.get("wording", "")).lower() for r in recs]
        out["knowledge_basis"][m] = {"text": kb.count("text"), "secondary": kb.count("secondary"),
                                     "unknown": kb.count("unknown"), "n_recs": len(recs),
                                     "share_with_conditions": round(float(np.mean(cond)), 2) if cond else None,
                                     "wording": {w: word.count(w) for w in ("absolute", "direct", "hedged", "tentative")}}

    emb = {}
    for (m, bid), a in ans.items():
        texts = [r.get("recommendation", "") for r in (a.get("recommendations") or []) if r.get("recommendation")]
        if texts:
            emb[(m, bid)] = enc(texts)

    for m1, m2 in combinations(models, 2):
        same, diff, per_book = [], [], {}
        for b in real:
            if (m1, b["id"]) in emb and (m2, b["id"]) in emb:
                s = _sym(emb[(m1, b["id"])], emb[(m2, b["id"])])
                same.append(s)
                per_book[b["title"]] = round(s, 2)
                diff += [_sym(emb[(m1, b["id"])], emb[(m2, c["id"])]) for c in real
                         if c["id"] != b["id"] and (m2, c["id"]) in emb]
        out["agreement"][f"{m1} vs {m2}"] = {"same_book": round(float(np.mean(same)), 3) if same else None,
                                              "different_books": round(float(np.mean(diff)), 3) if diff else None,
                                              "per_book": per_book}

    src = {_key(r["title"]): r["text_id"] for f in sorted((root / "corpus").glob("sources*.csv"))
           for r in csv.DictReader(open(f, encoding="utf-8"))}
    for b in real:
        tid = src.get(_key(b["title"]))
        raw = root / "corpus" / "raw" / f"{tid}.txt" if tid else None
        if not raw or not raw.exists():
            continue
        win = enc(windows(raw.read_text(encoding="utf-8")))
        others = np.vstack([emb[(m, c["id"])] for m in models for c in real
                            if c["id"] != b["id"] and (m, c["id"]) in emb])
        thr = float(np.quantile((others @ win.T).max(1), 0.9))  # 90th percentile of chance matches
        res = {"threshold": round(thr, 3)}
        for m in models:
            if (m, b["id"]) in emb:
                best = (emb[(m, b["id"])] @ win.T).max(1)
                res[m] = round(float(np.mean(best > thr)), 2)
        out["grounding"][b["title"]] = res
    (root / "work_memory" / "report.json").write_text(json.dumps(out, indent=1))
    return out
