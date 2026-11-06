"""Planted controls: synthetic passages with a known structure.

Built on the unblinded side (P8). The spec stays in keys/; only the passage
texts enter the corpus, mixed in and anonymized like real fragments.
"""
from __future__ import annotations

import json
from pathlib import Path

from .llm import render

STYLES = ["an ancient collection of sayings", "an 18th-century moral essay",
          "a 19th-century self-improvement book", "a contemporary bestselling self-help book",
          "a contemporary book of short maxims"]
FORCES = ["absolute", "direct", "hedged", "tentative"]

SUBJECTS = {
    "S1": {"subject": "lending money to friends", "structure": "two_camps", "camps": {
        "L": ["lend money to a friend who is in need", "help a friend with a loan when asked",
              "treat lending to friends as a duty of friendship"],
        "N": ["refuse to lend money to friends", "keep money and friendship apart",
              "decline when a friend asks to borrow"]}},
    "S2": {"subject": "keeping a diary", "structure": "unanimous", "items": [
        "keep a diary", "write down the events of each day", "record your thoughts in a notebook"]},
    "S3": {"subject": "dealing with neighbors", "structure": "complementary", "items": [
        "greet your neighbors", "help neighbors in an emergency", "keep noise down at night",
        "share surplus food with neighbors", "return borrowed tools promptly", "learn your neighbors' names"]},
    "S4": {"subject": "the hour of rising", "structure": "degree", "items": [
        "rise a little earlier than you must", "rise at sunrise", "rise well before dawn",
        "rise at 4 in the morning"]},
}


def build_spec(per_subject: int = 6, recs_per_passage: int = 3) -> list[dict]:
    rows, k = [], 0
    for sid, s in SUBJECTS.items():
        for p in range(per_subject):
            if s["structure"] == "two_camps":
                camp = "L" if p < per_subject // 2 else "N"
                pool = s["camps"][camp]
            else:
                camp, pool = "", s["items"]
            recs = []
            for r in range(recs_per_passage):
                recs.append({"content": pool[(p + r) % len(pool)], "wording": FORCES[k % len(FORCES)]})
                k += 1
            rows.append({"passage_id": f"SYN-{sid}-{p + 1}", "subject_id": sid, "subject": s["subject"],
                         "structure": s["structure"], "camp": camp, "style": STYLES[p % len(STYLES)],
                         "recommendations": recs})
    return rows


def spec_lines(rows: list[dict]) -> str:
    out = []
    for r in rows:
        recs = "; ".join(f"{x['content']} [{x['wording']}]" for x in r["recommendations"])
        out.append(f"{r['passage_id']} | {r['style']} | {r['subject']} | {recs} | none")
    return "\n".join(out)


def generate(llm, prompts_dir: Path, corpus_dir: Path, keys_dir: Path) -> int:
    spec = build_spec()
    tpl = (prompts_dir / "P8_synthetic.txt").read_text(encoding="utf-8")
    texts = {}
    for sid in SUBJECTS:
        rows = [r for r in spec if r["subject_id"] == sid]
        out = llm.call(render(tpl, {"N": len(rows), "SPEC": spec_lines(rows)})) or {}
        for p in out.get("passages", []):
            texts[p["passage_id"]] = p["text"]
    keys_dir.mkdir(parents=True, exist_ok=True)
    (keys_dir / "synthetic_spec.json").write_text(json.dumps(spec, indent=1))
    with open(corpus_dir / "synthetic.jsonl", "w", encoding="utf-8") as f:
        for pid, text in texts.items():
            f.write(json.dumps({"passage_id": pid, "text": text}, ensure_ascii=False) + "\n")
    return len(texts)
