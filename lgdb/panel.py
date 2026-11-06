"""Gold labels for the quality gates from a panel of models plus a small human review.

Why: the gates compare the annotator with an independent reference. A panel of models is cheaper than
2 human coders, but models share blind spots, so 2 safeguards stay human:
  1. every item the panel disagrees on goes to the researcher;
  2. a random share of the items the panel agrees on also goes to the researcher, mixed in so the
     reviewer cannot tell them apart (a blind audit). If the reviewer often overrules a unanimous
     panel, the panel is not a valid reference and the gate fails.

3 checks use this scheme:
  pairs:   do 2 recommendations go together (P5p, sign classes as in the graph)
  claims:  is an extracted recommendation supported by its passage (P10). Wording force is validated
           on the planted synthetic texts instead: asking the panel for it leaked the hypothesis.
  missing: recommendations the extractor missed in a passage (P11 proposes, P12 verifies)

Files:
  work/panel_<check>.jsonl     panel votes and the hidden audit flag
  human/review_<check>.csv     what the researcher fills in (no panel labels shown)
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

from .graph import CONDITIONAL, NEGATIVE, POSITIVE
from .llm import render
from .pipeline import load_jsonl, prompt, save_jsonl

PAIR_OPTIONS = "compatible / incompatible / conditional / unrelated"


def sign(label: str) -> str:
    label = (label or "").strip().lower()
    if label in POSITIVE:
        return "compatible"
    if label in NEGATIVE:
        return "incompatible"
    if label in CONDITIONAL:
        return "conditional"
    return "unrelated"


def _consensus(votes: list[str]) -> str | None:
    votes = [v for v in votes if v]
    return votes[0] if votes and len(set(votes)) == 1 and len(votes) >= 2 else None


def _fragments(work: Path) -> dict[str, str]:
    return {f["fragment_id"]: f["text"] for f in load_jsonl(work / "fragments.jsonl")}


# ---------- sampling ----------

def sample(work: Path, n_pairs: int = 200, n_claims: int = 100, n_passages: int = 10, seed: int = 0) -> dict:
    rng = random.Random(seed)
    pairs = [p for p in load_jsonl(work / "pairs.jsonl") if p["kind"] == "within"]
    claims = load_jsonl(work / "claims.jsonl")
    frag_ids = sorted({c["fragment_id"] for c in claims})
    out = {"pairs": rng.sample(pairs, min(n_pairs, len(pairs))),
           "claims": rng.sample(claims, min(n_claims, len(claims))),
           "passages": rng.sample(frag_ids, min(n_passages, len(frag_ids)))}
    save_jsonl(work / "panel_sample.jsonl", [{"check": k, "item": v} for k, vs in out.items() for v in vs])
    return {k: len(v) for k, v in out.items()}


def _sampled(work: Path, check: str) -> list:
    return [r["item"] for r in load_jsonl(work / "panel_sample.jsonl") if r["check"] == check]


# ---------- panel runs ----------

def run_pairs(llms: list, work: Path) -> int:
    pairs = _sampled(work, "pairs")
    tpl = prompt("P5p")
    votes = [[sign((o or {}).get("label")) if o else "" for o in
              llm.map([render(tpl, {"PAIR_ID": p["pair_id"], "REC_1": p["rec_1"], "REC_2": p["rec_2"]},
                              shuffle_seed=f"{p['pair_id']}-{i}") for p in pairs])]
             for i, llm in enumerate(llms)]
    save_jsonl(work / "panel_pairs.jsonl",
               [{"item_id": p["pair_id"], "rec_1": p["rec_1"], "rec_2": p["rec_2"],
                 "votes": [v[j] for v in votes]} for j, p in enumerate(pairs)])
    return len(pairs)


def run_claims(llms: list, work: Path) -> int:
    claims, frags, tpl = _sampled(work, "claims"), _fragments(work), prompt("P10")
    rows = [{"item_id": c["claim_id"], "recommendation": c["recommendation"],
             "quote": c.get("evidence_quote", ""), "passage": frags[c["fragment_id"]]} for c in claims]
    outs = [llm.map([render(tpl, {"ITEM_ID": r["item_id"], "PASSAGE": r["passage"], "RECOMMENDATION": r["recommendation"],
                                  "QUOTE": r["quote"]}) for r in rows]) for llm in llms]
    for j, r in enumerate(rows):
        r["supported_votes"] = [str((o[j] or {}).get("supported", "")).lower() for o in outs]
    save_jsonl(work / "panel_claims.jsonl", rows)
    return len(rows)


def run_missing(proposer, verifiers: list, work: Path) -> int:
    frags, claims = _fragments(work), load_jsonl(work / "claims.jsonl")
    by_frag = {}
    for c in claims:
        by_frag.setdefault(c["fragment_id"], []).append(c["recommendation"])
    rows = []
    passages = _sampled(work, "passages")
    lists = {f: "\n".join(f"- {r}" for r in by_frag.get(f, [])) or "(empty)" for f in passages}
    props = proposer.map([render(prompt("P11"), {"ITEM_ID": f, "PASSAGE": frags[f], "LIST": lists[f]})
                          for f in passages])
    for f, out in zip(passages, props):
        for k, m in enumerate((out or {}).get("missing", []) or [], 1):
            if isinstance(m, dict) and m.get("recommendation"):
                rows.append({"item_id": f"{f}-m{k}", "fragment_id": f, "candidate": m["recommendation"],
                             "quote": m.get("quote", ""), "passage": frags[f], "list": lists[f]})
    tpl = prompt("P12")
    outs = [v.map([render(tpl, {"ITEM_ID": r["item_id"], "PASSAGE": r["passage"], "LIST": r["list"],
                                "CANDIDATE": r["candidate"]}) for r in rows]) for v in verifiers]
    for j, r in enumerate(rows):
        r["votes"] = [str((o[j] or {}).get("verdict", "")).lower() for o in outs]
    save_jsonl(work / "panel_missing.jsonl", rows)
    save_jsonl(work / "panel_passages.jsonl",
               [{"fragment_id": f, "n_extracted": len(by_frag.get(f, []))} for f in passages])
    return len(rows)


# ---------- human review sheets ----------

def review_sheets(work: Path, human: Path, audit_share: float = 0.25, seed: int = 0) -> dict:
    """Write review CSVs: every panel disagreement plus a hidden random audit of agreements."""
    rng = random.Random(seed)
    human.mkdir(exist_ok=True)
    counts = {}
    specs = {
        "pairs": (lambda r: _consensus(r["votes"]),
                  lambda r: {"item_id": r["item_id"], "advice_A": r["rec_1"], "advice_B": r["rec_2"],
                             f"answer ({PAIR_OPTIONS})": ""}),
        "claims": (lambda r: _consensus(r["supported_votes"]),
                   lambda r: {"item_id": r["item_id"], "passage_excerpt": _excerpt(r["passage"], r["quote"]),
                              "extracted_recommendation": r["recommendation"],
                              "supported (yes / no)": ""}),
        "missing": (lambda r: _consensus(r["votes"]),
                    lambda r: {"item_id": r["item_id"], "passage_excerpt": _excerpt(r["passage"], r["quote"]),
                               "already_extracted": r["list"], "candidate": r["candidate"],
                               "missed_recommendation (yes / no)": ""}),
    }
    for check, (agreed, sheet_row) in specs.items():
        path = work / f"panel_{check}.jsonl"
        if not path.exists():
            continue
        rows = load_jsonl(path)
        out = []
        for r in rows:
            if "to_human" not in r:  # decided once; rerunning does not redraw the audit
                r["audit"] = bool(agreed(r)) and rng.random() < audit_share
                r["to_human"] = not agreed(r) or r["audit"]
            if r["to_human"]:
                out.append(sheet_row(r))
        save_jsonl(path, rows)
        rng.shuffle(out)
        if out:
            with open(human / f"review_{check}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
                w.writeheader()
                w.writerows(out)
        counts[check] = {"items": len(rows), "to_human": len(out)}
    return counts


def _excerpt(passage: str, quote: str, width: int = 600) -> str:
    i = passage.find(quote[:60]) if quote else -1
    if i < 0:
        return passage[:2 * width]
    return passage[max(0, i - width): i + len(quote) + width]


def _human(human: Path, check: str) -> dict[str, dict]:
    path = human / f"review_{check}.csv"
    if not path.exists():
        return {}
    rows = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        vals = {k.split(" (")[0]: (v or "").strip().lower() for k, v in r.items()}
        if any(vals[k] for k in vals if k not in ("item_id", "advice_A", "advice_B", "passage_excerpt",
                                                    "extracted_recommendation", "already_extracted", "candidate")):
            rows[r["item_id"]] = vals
    return rows


# ---------- gold labels ----------

def gold(work: Path, human: Path, check: str) -> dict:
    """Gold label per item, and the audit result (share of audited agreements the reviewer confirmed)."""
    rows = load_jsonl(work / f"panel_{check}.jsonl") if (work / f"panel_{check}.jsonl").exists() else []
    hum = _human(human, check)
    labels, audit_hits, audit_n, pending = {}, 0, 0, 0
    for r in rows:
        h = hum.get(r["item_id"])
        if check == "pairs":
            panel, mine = _consensus(r["votes"]), h and sign(h.get("answer"))
        elif check == "claims":
            panel = _consensus(r["supported_votes"])
            mine = h and h.get("supported")
        else:
            panel, mine = _consensus(r["votes"]), h and h.get("missed_recommendation")
        if r.get("to_human") and not h:
            pending += 1
            continue
        if r.get("audit"):
            audit_n += 1
            audit_hits += (mine == panel)
        label = mine if r.get("to_human") else panel
        labels[r["item_id"]] = {"label": label}
    return {"labels": labels, "pending": pending,
            "audit_agreement": audit_hits / audit_n if audit_n else None, "audit_n": audit_n}
