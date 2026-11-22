"""Full stance-matrix run (plan v2, work_full/plan.md): about 120 books from memory dossiers.

Reuses the pilot's prompts and helpers (lgdb/stance.py). Differences from the pilot:
memory dossiers for every book (text dossiers kept only for the pilot classics, as a check),
a cluster needs 3 books, no reversed statements (acquiescence was 0% in the pilot).
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import stance as st
from . import topics as tp
from .llm import render
from .pipeline import load_jsonl, prompt, save_jsonl

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261122
PER_CELL = 7
ELIGIBLE = {"selected", "no usable full text yet", "available, beyond first 6"}
THRESHOLDS = {"tight": 0.40, "loose": 0.50}
MIN_BOOKS = 3
MAX_PER_THRESHOLD = 100
PILOT_TEXT = {"Analects (Lunyu)": "analects", "Bhagavad Gita": "gita", "Vivekachudamani": "vivekachudamani",
              "Meditations": "meditations", "Republic": "republic",
              "Classic of Filial Piety (Xiaojing)": "xiaojing", "The Art of Worldly Wisdom": "worldly_wisdom",
              "Ramcharitmanas": "ramcharitmanas", "The Buddha and His Dhamma": "buddha_dhamma"}


def books() -> list[dict]:
    """Mechanical: per (period, region, format) cell, the first 7 eligible candidates by rank."""
    rows = list(csv.DictReader(open(ROOT / "corpus" / "selection.csv", encoding="utf-8")))
    cells = defaultdict(list)
    for r in rows:
        if r["status"] in ELIGIBLE:
            cells[(r["period"], r["region"], r["cell_format"])].append(r)
    # rule 2 again on normalized titles (parenthetical subtitles differ between cells)
    best = {}
    for r in rows:
        if r["status"] in ELIGIBLE:
            k = r["title"].split(" (")[0].strip().lower()
            score = (int(r["rank"]), 0 if r["cell_format"] == "collection" else 1)
            if k not in best or score < best[k][0]:
                best[k] = (score, id(r))
    for c in cells:
        cells[c] = [r for r in cells[c] if best[r["title"].split(" (")[0].strip().lower()][1] == id(r)]
    out = []
    for c in sorted(cells):
        for r in sorted(cells[c], key=lambda r: int(r["rank"]))[:PER_CELL]:
            out.append({"title": r["title"], "author": r["author"], "year": r["date"], "period": r["period"],
                        "region": r["region"], "format": r["cell_format"], "group": "real"})
    for n, b in enumerate(out, 1):
        b["id"] = f"f{n:03d}"
    out += [dict(b, period="invented", region="invented", format="invented", group="invented") for b in st.INVENTED]
    return out


def recall(q, work: Path) -> dict:
    tpl = prompt("S1")
    res, missing = {}, 0
    for b in books():
        r = q.call(render(tpl, {"TITLE": b["title"], "AUTHOR": b["author"], "YEAR": b["year"]}))
        if r is None:
            missing += 1
        else:
            res[b["id"]] = r
    (work / "dossiers").mkdir(parents=True, exist_ok=True)
    if not missing:
        (work / "dossiers" / "memory.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    basis = Counter((r or {}).get("knowledge_basis", "?") for r in res.values())
    return {"done": len(res), "missing": missing, "knowledge_basis": dict(basis)}


def known(mem: dict, b: dict) -> bool:
    d = mem.get(b["id"]) or {}
    return d.get("knowledge_basis") in ("text", "secondary") and bool(st.entries(d))


def build_pool(work: Path, provider="sentence-transformers") -> dict:
    mem = json.loads((work / "dossiers" / "memory.json").read_text())
    items = []
    for b in books():
        if b["group"] != "real" or not known(mem, b):
            continue
        for e in st.entries(mem[b["id"]])[:30]:
            items.append({"book": b["id"], "entry": e["id"], "kind": "advised", "text": e["advice"]})
            if st._real(e.get("argued_against", "")):
                items.append({"book": b["id"], "entry": e["id"], "kind": "criticized", "text": e["argued_against"]})
    z = tp.embed([i["text"] for i in items], provider=provider)
    groups, seen = [], {}
    for kind in ("advised", "criticized"):
        idx = [k for k, i in enumerate(items) if i["kind"] == kind]
        for lab, thr in THRESHOLDS.items():
            cl = defaultdict(list)
            for k, l in zip(idx, tp.cluster(z[idx], thr)):
                cl[int(l)].append(k)
            for m in cl.values():
                bk = {items[k]["book"] for k in m}
                if len(bk) < MIN_BOOKS:
                    continue
                sig = tuple(sorted(m))
                if sig in seen:
                    seen[sig]["thresholds"].append(lab)
                    continue
                g = {"kind": kind, "thresholds": [lab], "members": [items[k] for k in m], "books": sorted(bk)}
                seen[sig] = g
                groups.append(g)
    keep = set()
    for lab in THRESHOLDS:
        ranked = sorted([i for i, g in enumerate(groups) if lab in g["thresholds"]],
                        key=lambda i: (-len(groups[i]["books"]), -len(groups[i]["members"])))
        keep.update(ranked[:MAX_PER_THRESHOLD])
    groups = [g for i, g in enumerate(groups) if i in keep]
    random.Random(SEED).shuffle(groups)
    for n, g in enumerate(groups, 1):
        g["group"] = f"G{n:03d}"
        # members shown to the statement writer: at most 12, spread over books
        rng = random.Random(f"{SEED}-{g['group']}")
        rng.shuffle(g["members"])
    (work / "pool").mkdir(exist_ok=True)
    save_jsonl(work / "pool" / "groups.jsonl", groups)
    return {"items": len(items), "groups": len(groups), "kinds": dict(Counter(g["kind"] for g in groups)),
            "per_threshold": {lab: sum(lab in g["thresholds"] for g in groups) for lab in THRESHOLDS}}


def final_list(work: Path) -> list[dict]:
    path = work / "pool" / "final.jsonl"
    if path.exists():
        return load_jsonl(path)
    rows = [{"src": r["group"], "statement": r["statement"], "kind": r["kind"], "thresholds": r["thresholds"],
             "books": r["books"], "reversal_of": None} for r in load_jsonl(work / "pool" / "statements.jsonl")]
    random.Random(SEED + 1).shuffle(rows)
    for n, r in enumerate(rows, 1):
        r["sid"] = f"S{n:03d}"
    save_jsonl(path, rows)
    return rows


def rows(work: Path) -> list[dict]:
    mem = json.loads((work / "dossiers" / "memory.json").read_text())
    text = json.loads((ROOT / "work_stance" / "dossiers" / "text.json").read_text())
    out = []
    for b in books():
        if b["group"] == "real" and not known(mem, b):
            continue
        out.append({"book": b["id"], "source": "memory", "group": b["group"], "period": b["period"],
                    "region": b["region"], "entries": st.entries(mem.get(b["id"]))})
        if b["title"] in PILOT_TEXT:
            out.append({"book": b["id"], "source": "text", "group": "real", "period": b["period"],
                        "region": b["region"], "entries": st.entries(text[PILOT_TEXT[b["title"]]])})
    for k in (1, 2):
        out.append({"book": f"null{k}", "source": "null", "group": "null", "period": "null", "region": "null",
                    "entries": st._null_dossier(SEED + k)})
    random.Random(SEED + 2).shuffle(out)
    for n, r in enumerate(out, 1):
        r["row"] = f"D{n:03d}"
    (work / "rows_key.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "entries"} for r in out],
                                                   indent=1))
    return out


def code(q_by_rater: dict, work: Path) -> dict:
    stmts = final_list(work)
    sids = {s["sid"] for s in stmts}
    cells, missing = [], Counter()
    for rater, q in q_by_rater.items():
        for r in rows(work):
            ans = q.call(st.coding_prompt(rater, r, stmts))
            if ans is None:
                missing[rater] += 1
                continue
            parsed = st.parse_codes(ans, {e["id"].lower() for e in r["entries"]}, sids)
            for sid in sorted(sids):
                c = parsed.get(sid, {"code": "N", "cond": False, "ev": "", "flag": "missing_line"})
                cells.append({"rater": rater, "row": r["row"], "sid": sid, **c})
    if not missing:
        save_jsonl(work / "codes.jsonl", cells)
    return {"cells": len(cells), "missing": dict(missing)}


def tiebreak(q, work: Path) -> dict:
    stmts = {s["sid"]: s for s in final_list(work)}
    rs = {r["row"]: r for r in rows(work)}
    by = defaultdict(dict)
    for c in load_jsonl(work / "codes.jsonl"):
        by[(c["row"], c["sid"])][c["rater"]] = c["code"]
    disputes = defaultdict(list)
    for (row, sid), v in by.items():
        if {v.get("r1"), v.get("r2")} == {"A", "B"}:
            disputes[row].append(sid)
    out, missing = [], 0
    for row, sl in sorted(disputes.items()):
        r = rs[row]
        ans = q.call(st.coding_prompt("r1", r, [stmts[s] for s in sl]))
        if ans is None:
            missing += 1
            continue
        parsed = st.parse_codes(ans, {e["id"].lower() for e in r["entries"]}, set(sl))
        out += [{"rater": "tie", "row": row, "sid": s,
                 **parsed.get(s, {"code": "N", "cond": False, "ev": "", "flag": "missing_line"})} for s in sl]
    if not missing:
        save_jsonl(work / "codes_tie.jsonl", out)
    return {"disputes": sum(map(len, disputes.values())), "missing": missing}
