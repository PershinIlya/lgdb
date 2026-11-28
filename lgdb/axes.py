"""Axes as explicit dilemmas (follow-up to the full stance run, work_axes/plan.md).

1. tensions: cluster the dossiers' "advice || instead of" pairs (books name these tensions themselves).
2. axes: one writer turns the clusters into 8 to 12 questions with two answers each (A1).
3. place: two blind raters put every known book on every question (A2), on a 5-point scale
   P2 P1 M Q1 Q2 plus N/U. Rater 2 sees the answers swapped (P and Q exchanged), so position bias
   cancels and the spread between raters measures how certain a placement is.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import stance as st
from . import stance_full as sf
from . import topics as tp
from .llm import render
from .pipeline import load_jsonl, prompt, save_jsonl

SEED = 20261128
THRESHOLD = 0.45
MIN_BOOKS, MIN_CELLS, MAX_MEMBERS = 4, 2, 40
MAX_GROUPS, SHOW = 70, 8
VAL = {"P2": 2, "P1": 1, "M": 0, "Q1": -1, "Q2": -2}
CODE_RE = re.compile(r"^\s*(K\d{2})\s+(P2|P1|M|Q1|Q2|N|U)\b\s*([tm]\d+)?", re.I)


def tensions(work: Path, provider="sentence-transformers") -> dict:
    mem = json.loads((sf.ROOT / "work_full" / "dossiers" / "memory.json").read_text())
    B = {b["id"]: b for b in sf.books()}
    items = []
    for b in sf.books():
        if b["group"] != "real" or not sf.known(mem, b):
            continue
        for e in st.entries(mem[b["id"]])[:30]:
            if st._real(e.get("argued_against", "")):
                items.append({"book": b["id"], "entry": e["id"], "advice": e["advice"].strip(),
                              "against": e["argued_against"].strip()})
    z = tp.embed([f"{i['advice']} Instead of: {i['against']}" for i in items], provider=provider)
    cl = defaultdict(list)
    for i, l in zip(items, tp.cluster(z, THRESHOLD)):
        cl[int(l)].append(i)
    groups = []
    for m in cl.values():
        bk = {i["book"] for i in m}
        cells = {(B[x]["period"], B[x]["region"]) for x in bk}
        if len(bk) >= MIN_BOOKS and len(cells) >= MIN_CELLS and len(m) <= MAX_MEMBERS:
            groups.append({"books": sorted(bk), "members": m})
    groups.sort(key=lambda g: (-len(g["books"]), -len(g["members"])))
    groups = groups[:MAX_GROUPS]
    random.Random(SEED).shuffle(groups)
    for n, g in enumerate(groups, 1):
        g["gid"] = f"T{n:03d}"
    work.mkdir(parents=True, exist_ok=True)
    save_jsonl(work / "tensions.jsonl", groups)
    return {"pairs": len(items), "groups": len(groups), "books_covered": len({b for g in groups for b in g["books"]})}


def _groups_text(groups) -> str:
    out = []
    for g in groups:
        rng = random.Random(f"{SEED}-{g['gid']}")
        seen, shown = set(), []
        for i in rng.sample(g["members"], len(g["members"])):  # one record per book first
            if i["book"] not in seen:
                seen.add(i["book"])
                shown.append(i)
        lines = [f"- {i['advice']} || INSTEAD OF: {i['against']}" for i in shown[:SHOW]]
        out.append(f"{g['gid']} ({len(g['books'])} books)\n" + "\n".join(lines))
    return "\n\n".join(out)


def axes(q, work: Path) -> dict:
    groups = load_jsonl(work / "tensions.jsonl")
    ans = q.call(render(prompt("A1"), {"GROUPS": _groups_text(groups)}))
    if ans is None:
        return {"missing": 1}
    gids = {g["gid"] for g in groups}
    gb = {g["gid"]: set(g["books"]) for g in groups}
    out = []
    for a in ans.get("axes", []):
        xg = [g for g in a.get("x_groups", []) if g in gids]
        yg = [g for g in a.get("y_groups", []) if g in gids]
        if not xg or not yg:
            continue
        a["x_books"] = len(set().union(*[gb[g] for g in xg]))
        a["y_books"] = len(set().union(*[gb[g] for g in yg]))
        out.append(a)
    for n, a in enumerate(out, 1):
        a["kid"] = f"K{n:02d}"
    save_jsonl(work / "axes.jsonl", out)
    return {"axes": len(out), "dropped": len(ans.get("axes", [])) - len(out),
            "list": [(a["kid"], a["question"], a["x"], a["y"], a["x_books"], a["y_books"]) for a in out]}


def rows(work: Path) -> list[dict]:
    return sf.rows(sf.ROOT / "work_full")  # same rows, ids and blinding as the full run


def place_prompt(rater: str, row: dict, axs: list[dict]) -> str:
    order = list(axs)
    random.Random(f"{SEED}-{rater}-{row['row']}").shuffle(order)
    swap = rater == "r2"
    qs = []
    for a in order:
        p, qq = (a["y_advice"], a["x_advice"]) if swap else (a["x_advice"], a["y_advice"])
        qs.append(f"{a['kid']}. {a['question']}\n   P: {p}\n   Q: {qq}")
    return render(prompt("A2"), {"DOSSIER": st._dossier_text(row["entries"]), "QUESTIONS": "\n".join(qs)})


def parse(ans, valid_ids, kids, swap) -> dict:
    out = {}
    for line in (ans or {}).get("answers", []):
        m = CODE_RE.match(str(line))
        if not m:
            continue
        k, code, ev = m.group(1).upper(), m.group(2).upper(), (m.group(3) or "").lower()
        if k not in kids:
            continue
        if code in VAL and ev not in valid_ids:
            code = "N"
        v = VAL.get(code)
        out[k] = {"code": code, "v": (-v if swap and v is not None else v), "ev": ev if code in VAL else ""}
    return out


RATERS = {"r1": "sonnet", "r2": "opus"}


def place(q_by_rater: dict, work: Path) -> dict:
    axs = load_jsonl(work / "axes.jsonl")
    kids = {a["kid"] for a in axs}
    cells, missing = [], Counter()
    for rater, q in q_by_rater.items():
        for r in rows(work):
            ans = q.call(place_prompt(rater, r, axs))
            if ans is None:
                missing[rater] += 1
                continue
            p = parse(ans, {e["id"].lower() for e in r["entries"]}, kids, rater == "r2")
            for k in sorted(kids):
                cells.append({"rater": rater, "row": r["row"], "kid": k, **p.get(k, {"code": "N", "v": None, "ev": ""})})
    if not missing:
        save_jsonl(work / "placements.jsonl", cells)
    return {"cells": len(cells), "missing": dict(missing)}


MIN_PLACED, MIN_SIDE, MIN_R, MAX_OPPOSITE = 20, 5, 0.6, 0.10


def consensus(work: Path) -> dict:
    by = defaultdict(dict)
    for c in load_jsonl(work / "placements.jsonl"):
        by[(c["row"], c["kid"])][c["rater"]] = c
    out = {}
    for k, v in by.items():
        a, b = (v.get("r1") or {}).get("v"), (v.get("r2") or {}).get("v")
        if a is not None and b is not None:
            out[k] = {"v": (a + b) / 2, "spread": abs(a - b), "weak": False, "r1": a, "r2": b}
        elif a is not None or b is not None:
            out[k] = {"v": a if a is not None else b, "spread": None, "weak": True, "r1": a, "r2": b}
    return out


def _eta2(x, lab, n=1000, seed=SEED):
    x, lab = np.asarray(x, float), np.asarray(lab)

    def e(l):
        tot = ((x - x.mean()) ** 2).sum()
        return sum((l == g).sum() * (x[l == g].mean() - x.mean()) ** 2 for g in np.unique(l)) / tot if tot else 0.0
    obs = e(lab)
    rng = np.random.default_rng(seed)
    null = [e(rng.permutation(lab)) for _ in range(n)]
    return round(float(obs), 3), round(float((np.sum(np.array(null) >= obs) + 1) / (n + 1)), 4)


def analyze(work: Path) -> dict:
    key = {r["row"]: r for r in json.loads((sf.ROOT / "work_full" / "rows_key.json").read_text())}
    axs = load_jsonl(work / "axes.jsonl")
    con = consensus(work)
    res = {"axes": []}
    ctrl = [k for k in con if key[k[0]]["group"] in ("invented", "null")]
    res["controls_placed"] = len(ctrl)
    for a in axs:
        k = a["kid"]
        mem = {r: con[(r, k)] for r in key if key[r]["group"] == "real" and key[r]["source"] == "memory" and (r, k) in con}
        both = [c for c in mem.values() if not c["weak"]]
        r1 = np.array([c["r1"] for c in both], float)
        r2 = np.array([c["r2"] for c in both], float)
        rr = float(np.corrcoef(r1, r2)[0, 1]) if len(both) > 2 and r1.std() and r2.std() else None
        opp = float(np.mean((r1 * r2) < 0)) if len(both) else None
        vals = np.array([c["v"] for c in mem.values()])
        raw = [x for c in mem.values() for x in (c["r1"], c["r2"]) if x is not None]
        rows = list(mem)
        txt = {key[r]["book"]: r for r in key if key[r]["source"] == "text"}
        mb = {key[r]["book"]: r for r in rows}
        pairs = [(con[(mb[b], k)]["v"], con[(t, k)]["v"]) for b, t in txt.items() if b in mb and (t, k) in con]
        d = {**{x: a[x] for x in ("kid", "question", "x", "y", "x_advice", "y_advice")},
             "placed": len(mem), "both_raters": len(both), "side_x": int((vals >= 1).sum()), "side_y": int((vals <= -1).sum()),
             "middle": int((abs(vals) < 1).sum()), "rater_r": None if rr is None else round(rr, 3),
             "opposite_share": None if opp is None else round(opp, 3),
             "mean_spread": round(float(np.mean([c["spread"] for c in both])), 3) if both else None,
             "raw_extreme_share": round(float(np.mean(np.abs(raw) == 2)), 3) if raw else None,
             "raw_middle_share": round(float(np.mean(np.array(raw) == 0)), 3) if raw else None,
             "eta2_period": _eta2(vals, [key[r]["period"] for r in rows]) if len(rows) > 5 else None,
             "eta2_region": _eta2(vals, [key[r]["region"] for r in rows]) if len(rows) > 5 else None,
             "text_vs_memory_sign": (round(float(np.mean([np.sign(x) == np.sign(y) for x, y in pairs])), 3), len(pairs)) if pairs else None}
        d["keep"] = bool(d["placed"] >= MIN_PLACED and d["side_x"] >= MIN_SIDE and d["side_y"] >= MIN_SIDE
                         and rr is not None and rr >= MIN_R and opp <= MAX_OPPOSITE and not res["controls_placed"])
        res["axes"].append(d)
    res["books"] = {}
    for (r, k), c in con.items():
        if key[r]["group"] == "real":
            res["books"].setdefault(r, {"book": key[r]["book"], "source": key[r]["source"], "pos": {}})["pos"][k] = c
    (work / "analysis.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return {"controls_placed": res["controls_placed"],
            "axes": [{x: d[x] for x in d if x not in ("x_advice", "y_advice", "question")} for d in res["axes"]]}
