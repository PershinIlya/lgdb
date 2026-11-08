"""Stance-matrix pilot: dossiers, statement pool, book x statement codes, go criteria.

Stages read and write files in work_stance/. Model calls go through QueueLLM, so a
stage run queues missing prompts; agents answer them in batches (tools/stance.py).
Titles and authors never reach the stance coders: rows carry opaque ids, and the
key from row id to book lives only in work_stance/rows_key.json.
"""
from __future__ import annotations

import csv
import json
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import topics as tp
from .llm import render
from .pipeline import load_jsonl, prompt, save_jsonl

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261108

CLASSICS = ["analects", "gita", "vivekachudamani", "meditations", "republic", "xiaojing",
            "worldly_wisdom", "ramcharitmanas", "vemana", "buddha_dhamma"]
INVENTED = [
    {"id": "i1", "title": "The Quiet Ledger: Letters to a Young Householder", "author": "Aurelio Venn", "year": "1888"},
    {"id": "i2", "title": "Twelve Lamps of the Mountain School", "author": "Kosen Haruda", "year": "1702"},
    {"id": "i3", "title": "Sayings at the River Ford", "author": "Bhadranath Sastri", "year": "c. 1350"},
]
THRESHOLDS = {"tight": 0.40, "loose": 0.50}  # cosine distance; set before any coding
MAX_PER_THRESHOLD = 60
N_REVERSED = 40
CODE_RE = re.compile(r"^\s*(S\d{3})\s+([ABNU])(c?)\s*([tm]\d+)?", re.I)


# ---------- books and dossiers ----------

def books() -> list[dict]:
    meta = {r["text_id"]: r for r in csv.DictReader(open(ROOT / "corpus" / "meta.csv", encoding="utf-8"))}
    out = [{"id": t, "title": meta[t]["title"], "author": meta[t]["author"], "year": meta[t]["year"],
            "group": "classic"} for t in CLASSICS]
    for b in json.load(open(ROOT / "work_calib" / "books.json", encoding="utf-8")):
        out.append({"id": b["id"], "title": b["title"], "author": b["author"], "year": b["year"], "group": "modern"})
    out += [dict(b, group="invented") for b in INVENTED]
    return out


def recall(q, work: Path) -> dict:
    tpl = prompt("S1")
    res, missing = {}, 0
    for b in books():
        p = render(tpl, {"TITLE": b["title"], "AUTHOR": b["author"], "YEAR": b["year"]})
        r = q.call(p)
        if r is None:
            missing += 1
            continue
        res[b["id"]] = r
    (work / "dossiers").mkdir(parents=True, exist_ok=True)
    if not missing:
        (work / "dossiers" / "memory.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return {"done": len(res), "missing": missing}


def _notes(claims: list[dict]) -> str:
    lines = []
    for i, c in enumerate(claims, 1):
        scope = "; ".join(x for x in (c.get("addressee", ""), c.get("scope", "")) if x and x != "not stated")
        lines.append(f"{i}. advice: {c['recommendation']} | applies to: {scope or 'not stated'} | "
                     f"conditions: {c.get('conditions', 'none')} | argued against: {c.get('argued_against', 'none')}")
    return "\n".join(lines)


def condense(q, work: Path) -> dict:
    key = {r["fragment_id"]: r["text_id"] for r in csv.DictReader(open(ROOT / "keys" / "fragments_key.csv"))}
    by = defaultdict(list)
    for c in load_jsonl(ROOT / "work_pilot1" / "claims.jsonl"):
        by[key[c["fragment_id"]]].append(c)
    tpl = prompt("S2")
    res, missing = {}, 0
    for t in CLASSICS:
        r = q.call(render(tpl, {"NOTES": _notes(by[t])}))
        if r is None:
            missing += 1
        else:
            res[t] = r
    if not missing:
        (work / "dossiers" / "text.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return {"done": len(res), "missing": missing, "notes_per_book": {t: len(by[t]) for t in CLASSICS}}


def entries(d: dict) -> list[dict]:
    return (d or {}).get("entries", []) or []


# ---------- statement pool ----------

def _real(x: str) -> bool:
    return bool(x) and x.strip().lower() not in {"none", "not stated", "n/a", ""}


def pool_items(work: Path) -> list[dict]:
    """Classics contribute text-dossier entries, modern books memory entries; at most 30 each."""
    text = json.loads((work / "dossiers" / "text.json").read_text())
    mem = json.loads((work / "dossiers" / "memory.json").read_text())
    items = []
    for b in books():
        if b["group"] == "classic":
            ents = entries(text.get(b["id"]))
        elif b["group"] == "modern":
            ents = entries(mem.get(b["id"]))
        else:
            continue
        for e in ents[:30]:
            items.append({"book": b["id"], "entry": e["id"], "kind": "advised", "text": e["advice"]})
            if _real(e.get("argued_against", "")):
                items.append({"book": b["id"], "entry": e["id"], "kind": "criticized", "text": e["argued_against"]})
    return items


def build_pool(work: Path, provider="sentence-transformers") -> dict:
    items = pool_items(work)
    z = tp.embed([i["text"] for i in items], provider=provider)
    groups, seen = [], {}
    for kind in ("advised", "criticized"):
        idx = [k for k, i in enumerate(items) if i["kind"] == kind]
        for lab, thr in THRESHOLDS.items():
            labels = tp.cluster(z[idx], thr)
            cl = defaultdict(list)
            for k, l in zip(idx, labels):
                cl[int(l)].append(k)
            cands = [m for m in cl.values() if len({items[k]["book"] for k in m}) >= 2]
            cands.sort(key=lambda m: (-len({items[k]["book"] for k in m}), -len(m)))
            for m in cands:
                sig = tuple(sorted(m))
                if sig in seen:
                    seen[sig]["thresholds"].append(lab)
                    continue
                g = {"kind": kind, "thresholds": [lab], "members": [items[k] for k in m],
                     "books": sorted({items[k]["book"] for k in m})}
                seen[sig] = g
                groups.append(g)
    # cap per threshold: rank by number of books, keep the top MAX_PER_THRESHOLD of each
    keep = set()
    for lab in THRESHOLDS:
        ranked = sorted([i for i, g in enumerate(groups) if lab in g["thresholds"]],
                        key=lambda i: (-len(groups[i]["books"]), -len(groups[i]["members"])))
        keep.update(ranked[:MAX_PER_THRESHOLD])
    groups = [g for i, g in enumerate(groups) if i in keep]
    for g in groups:
        g["thresholds"] = [t for t in g["thresholds"]]
    random.Random(SEED).shuffle(groups)
    for n, g in enumerate(groups, 1):
        g["group"] = f"G{n:03d}"
    (work / "pool").mkdir(exist_ok=True)
    save_jsonl(work / "pool" / "groups.jsonl", groups)
    per = {lab: sum(lab in g["thresholds"] for g in groups) for lab in THRESHOLDS}
    return {"items": len(items), "groups": len(groups), "per_threshold": per,
            "kinds": Counter(g["kind"] for g in groups)}


def _groups_text(gs: list[dict]) -> str:
    out = []
    for g in gs:
        out.append(f"{g['group']} (kind: {g['kind']})")
        out += [f"- {m['text']}" for m in g["members"][:12]]
    return "\n".join(out)


def statements(q, work: Path, chunk: int = 40) -> dict:
    groups = load_jsonl(work / "pool" / "groups.jsonl")
    tpl = prompt("S3")
    got, missing = {}, 0
    for i in range(0, len(groups), chunk):
        r = q.call(render(tpl, {"GROUPS": _groups_text(groups[i:i + chunk])}))
        if r is None:
            missing += 1
            continue
        for s in r.get("statements", []):
            got[s["group"]] = s["statement"].strip()
    if missing:
        return {"missing": missing}
    rows = []
    for g in groups:
        st = got.get(g["group"], "none")
        if st.lower().startswith("none") or not st:
            continue
        rows.append({"group": g["group"], "statement": st, "kind": g["kind"], "thresholds": g["thresholds"],
                     "books": g["books"], "entries": [[m["book"], m["entry"]] for m in g["members"]]})
    save_jsonl(work / "pool" / "statements.jsonl", rows)
    return {"statements": len(rows), "dropped_none": len(groups) - len(rows)}


def reverse(q, work: Path) -> dict:
    rows = load_jsonl(work / "pool" / "statements.jsonl")
    pick = random.Random(SEED).sample(range(len(rows)), min(N_REVERSED, len(rows)))
    lines = "\n".join(f"R{n:02d}. {rows[i]['statement']}" for n, i in enumerate(pick, 1))
    r = q.call(render(prompt("S5"), {"STATEMENTS": lines}))
    if r is None:
        return {"missing": 1}
    rev = {s["id"].strip().rstrip("."): s["statement"].strip() for s in r.get("statements", [])}
    out = [{"rid": f"R{n:02d}", "of": rows[i]["group"], "statement": rev.get(f"R{n:02d}", "")}
           for n, i in enumerate(pick, 1)]
    save_jsonl(work / "pool" / "reversed.jsonl", [x for x in out if x["statement"]])
    return {"reversed": sum(bool(x["statement"]) for x in out)}


def final_list(work: Path) -> list[dict]:
    """All coded statements with stable ids S001..; order seeded so reversals are not adjacent."""
    path = work / "pool" / "final.jsonl"
    if path.exists():
        return load_jsonl(path)
    rows = [{"src": r["group"], "statement": r["statement"], "kind": r["kind"], "thresholds": r["thresholds"],
             "books": r["books"], "reversal_of": None} for r in load_jsonl(work / "pool" / "statements.jsonl")]
    rows += [{"src": r["rid"], "statement": r["statement"], "kind": "reversed", "thresholds": [],
              "books": [], "reversal_of": r["of"]} for r in load_jsonl(work / "pool" / "reversed.jsonl")]
    random.Random(SEED + 1).shuffle(rows)
    for n, r in enumerate(rows, 1):
        r["sid"] = f"S{n:03d}"
    save_jsonl(path, rows)
    return rows


# ---------- coding rows ----------

def _null_dossier(seed: int, n: int = 20) -> list[dict]:
    rng = random.Random(seed)

    def word():
        return "".join(rng.choice("bcdfghklmnprstvz") + rng.choice("aeiou") for _ in range(rng.randint(1, 3)))
    return [{"id": f"t{i}", "advice": " ".join(word() for _ in range(rng.randint(6, 12))).capitalize() + "."}
            for i in range(1, n + 1)]


def rows(work: Path) -> list[dict]:
    """Coding rows: (row id, dossier entries). The key to books is written separately."""
    path = work / "rows_key.json"
    text = json.loads((work / "dossiers" / "text.json").read_text())
    mem = json.loads((work / "dossiers" / "memory.json").read_text())
    out = []
    for b in books():
        if b["group"] == "classic":
            out.append({"book": b["id"], "source": "text", "group": "classic", "entries": entries(text.get(b["id"]))})
        out.append({"book": b["id"], "source": "memory", "group": b["group"], "entries": entries(mem.get(b["id"]))})
    for k in (1, 2):
        out.append({"book": f"null{k}", "source": "null", "group": "null", "entries": _null_dossier(SEED + k)})
    random.Random(SEED + 2).shuffle(out)
    for n, r in enumerate(out, 1):
        r["row"] = f"D{n:02d}"
    path.write_text(json.dumps([{k: r[k] for k in ("row", "book", "source", "group")} for r in out], indent=1))
    return out


def _dossier_text(ents: list[dict]) -> str:
    if not ents:
        return "(no notes)"
    lines = []
    for e in ents:
        extra = []
        if _real(e.get("applies_to", "")):
            extra.append(f"applies to: {e['applies_to']}")
        if _real(e.get("conditions", "")):
            extra.append(f"conditions: {e['conditions']}")
        if _real(e.get("argued_against", "")):
            extra.append(f"argues against: {e['argued_against']}")
        lines.append(f"{e['id']}. {e['advice']}" + (f" ({'; '.join(extra)})" if extra else ""))
    return "\n".join(lines)


def coding_prompt(rater: str, row: dict, stmts: list[dict]) -> str:
    order = list(stmts)
    random.Random(f"{SEED}-{rater}-{row['row']}").shuffle(order)
    tpl = prompt("S4_stance_r1" if rater == "r1" else "S4_stance_r2")
    return render(tpl, {"DOSSIER": _dossier_text(row["entries"]),
                        "STATEMENTS": "\n".join(f"{s['sid']}. {s['statement']}" for s in order)})


def parse_codes(ans: dict, valid_ids: set[str], sids: set[str]) -> dict:
    out = {}
    for line in (ans or {}).get("codes", []):
        m = CODE_RE.match(str(line))
        if not m:
            continue
        sid, code, cond, ev = m.group(1).upper(), m.group(2).upper(), bool(m.group(3)), (m.group(4) or "").lower()
        if sid not in sids:
            continue
        flag = ""
        if code in "AB" and ev not in valid_ids:
            code, flag = "N", "no_evidence"
        out[sid] = {"code": code, "cond": cond and code in "AB", "ev": ev if code in "AB" else "", "flag": flag}
    return out


def code(q_by_rater: dict, work: Path) -> dict:
    stmts = final_list(work)
    sids = {s["sid"] for s in stmts}
    rs = rows(work)
    cells, missing = [], Counter()
    for rater, q in q_by_rater.items():
        for r in rs:
            ans = q.call(coding_prompt(rater, r, stmts))
            if ans is None:
                missing[rater] += 1
                continue
            parsed = parse_codes(ans, {e["id"].lower() for e in r["entries"]}, sids)
            for sid in sorted(sids):
                c = parsed.get(sid, {"code": "N", "cond": False, "ev": "", "flag": "missing_line"})
                cells.append({"rater": rater, "row": r["row"], "sid": sid, **c})
    if not missing:
        save_jsonl(work / "codes.jsonl", cells)
    return {"cells": len(cells), "missing": dict(missing)}


def tiebreak(q, work: Path) -> dict:
    """Haiku recodes cells where the 2 raters gave opposite signs, with the r1 prompt."""
    stmts = {s["sid"]: s for s in final_list(work)}
    rs = {r["row"]: r for r in rows(work)}
    cells = load_jsonl(work / "codes.jsonl")
    by = defaultdict(dict)
    for c in cells:
        by[(c["row"], c["sid"])][c["rater"]] = c["code"]
    disputes = defaultdict(list)
    for (row, sid), v in by.items():
        if {v.get("r1"), v.get("r2")} == {"A", "B"}:
            disputes[row].append(sid)
    out, missing = [], 0
    for row, sl in sorted(disputes.items()):
        r = rs[row]
        ans = q.call(coding_prompt("r1", r, [stmts[s] for s in sl]))
        if ans is None:
            missing += 1
            continue
        parsed = parse_codes(ans, {e["id"].lower() for e in r["entries"]}, set(sl))
        for s in sl:
            c = parsed.get(s, {"code": "N", "cond": False, "ev": "", "flag": "missing_line"})
            out.append({"rater": "tie", "row": row, "sid": s, **c})
    if not missing:
        save_jsonl(work / "codes_tie.jsonl", out)
    return {"disputes": sum(map(len, disputes.values())), "rows": len(disputes), "missing": missing}


# ---------- criteria ----------

SIGN = {"A": 1, "B": -1, "N": 0, "U": None}


def kripp_nominal(pairs: list[tuple]) -> float:
    """Krippendorff alpha, nominal, 2 coders, no missing values within a pair."""
    o = Counter()
    for a, b in pairs:
        o[(a, b)] += 1
        o[(b, a)] += 1
    n_c = Counter()
    for (a, _), v in o.items():
        n_c[a] += v
    n = sum(n_c.values())
    do = sum(v for (a, b), v in o.items() if a != b) / n
    de = sum(n_c[a] * n_c[b] for a in n_c for b in n_c if a != b) / (n * (n - 1))
    return 1 - do / de if de else float("nan")


def kripp_interval(pairs: list[tuple]) -> float:
    vals = [x for p in pairs for x in p]
    n = len(vals)
    do = sum((a - b) ** 2 for a, b in pairs) * 2 / n
    de = sum((a - b) ** 2 for a in vals for b in vals) / (n * (n - 1))
    return 1 - do / de if de else float("nan")


def consensus(work: Path) -> dict:
    """Final code per cell: agreed code; opposite signs -> tie-breaker; one signed, other N/U -> signed (weak)."""
    cells = load_jsonl(work / "codes.jsonl")
    tie = {(c["row"], c["sid"]): c for c in load_jsonl(work / "codes_tie.jsonl")}
    by = defaultdict(dict)
    for c in cells:
        by[(c["row"], c["sid"])][c["rater"]] = c
    out = {}
    for k, v in by.items():
        a, b = v["r1"]["code"], v["r2"]["code"]
        if a == b:
            f, how = a, "agree"
        elif {a, b} == {"A", "B"}:
            f, how = (tie[k]["code"] if k in tie else "U"), "tie"
        elif a in "AB" or b in "AB":
            f, how = (a if a in "AB" else b), "weak"
        else:
            f, how = "N" if "N" in (a, b) else "U", "agree0"
        out[k] = {"code": f, "how": how, "r1": a, "r2": b,
                  "cond": v["r1"].get("cond") or v["r2"].get("cond")}
    return out


def criteria(work: Path) -> dict:
    stmts = {s["sid"]: s for s in final_list(work)}
    key = {r["row"]: r for r in json.loads((work / "rows_key.json").read_text())}
    cells = load_jsonl(work / "codes.jsonl")
    fin = consensus(work)
    res = {}
    real = [k for k in fin if key[k[0]]["group"] in ("classic", "modern")]
    orig = [k for k in real if stmts[k[1]]["kind"] != "reversed"]

    # 1. negative share and two-sided statements
    addr = [k for k in orig if fin[k]["code"] in "AB"]
    neg = [k for k in addr if fin[k]["code"] == "B"]
    own = [k for k in neg if key[k[0]]["book"] in stmts[k[1]]["books"]]
    strict = [k for k in orig if fin[k]["how"] == "agree" and fin[k]["code"] in "AB"]
    two = [s for s in stmts if stmts[s]["kind"] != "reversed"
           and any(fin[(r, s)]["code"] == "A" for r in key if (r, s) in fin and key[r]["group"] in ("classic", "modern"))
           and any(fin[(r, s)]["code"] == "B" for r in key if (r, s) in fin and key[r]["group"] in ("classic", "modern"))]
    rej2 = [s for s in stmts if stmts[s]["kind"] != "reversed"
            and sum(fin[(r, s)]["code"] == "B" for r in key if (r, s) in fin and key[r]["group"] in ("classic", "modern")) >= 2]
    n_orig = sum(stmts[s]["kind"] != "reversed" for s in stmts)
    res["c1_negative"] = {
        "addressed_cells": len(addr), "negative": len(neg), "neg_share": round(len(neg) / max(len(addr), 1), 3),
        "neg_share_without_own_source": round((len(neg) - len(own)) / max(len(addr) - len(own), 1), 3),
        "neg_share_strict_agree": round(sum(fin[k]["code"] == "B" for k in strict) / max(len(strict), 1), 3),
        "statements_with_A_and_B": len(two), "statements_with_2plus_B": len(rej2),
        "share_2plus_B": round(len(rej2) / max(n_orig, 1), 3),
        "pass": len(neg) / max(len(addr), 1) >= 0.10 and len(two) >= 20}

    # coverage per row
    cov = defaultdict(list)
    for k in orig:
        cov[k[0]].append(fin[k]["code"] in "AB")
    res["coverage_median_share"] = round(float(np.median([np.mean(v) for v in cov.values()])), 3)

    # 3. agreement
    by = defaultdict(dict)
    for c in cells:
        by[(c["row"], c["sid"])][c["rater"]] = c["code"]
    pairs = [(v["r1"], v["r2"]) for k, v in by.items() if k in set(real)]
    sp = [(SIGN[a], SIGN[b]) for a, b in pairs if SIGN[a] is not None and SIGN[b] is not None]
    res["c3_alpha"] = {"nominal_ABNU": round(kripp_nominal(pairs), 3),
                       "interval_sign_U_missing": round(kripp_interval(sp), 3),
                       "pass": kripp_interval(sp) >= 0.7}
    res["rater_code_counts"] = {r: dict(Counter(c["code"] for c in cells if c["rater"] == r
                                                and key[c["row"]]["group"] in ("classic", "modern")))
                                for r in ("r1", "r2")}
    res["no_evidence_cells"] = dict(Counter(c["rater"] for c in cells if c["flag"] == "no_evidence"))

    # 4. reversal pairs (acquiescence)
    rev = [(s, next(x for x in stmts if stmts[x]["src"] == stmts[s]["reversal_of"]))
           for s in stmts if stmts[s]["kind"] == "reversed"]
    both = total = 0
    for rs, os_ in rev:
        for r in key:
            if key[r]["group"] not in ("classic", "modern"):
                continue
            a, b = fin[(r, rs)]["code"], fin[(r, os_)]["code"]
            if a in "AB" and b in "AB":
                total += 1
                both += a == "A" and b == "A"
    res["c4_reversal"] = {"pairs_both_addressed": total, "both_advised": both,
                          "share": round(both / max(total, 1), 3), "pass": both / max(total, 1) <= 0.10}

    # 5. invented and null
    ctl = [k for k in fin if key[k[0]]["group"] in ("invented", "null")]
    unc = [k for k in ctl if fin[k]["code"] in "NU"]
    per = {key[r]["book"] + "/" + key[r]["source"]: round(np.mean([fin[k]["code"] in "NU" for k in ctl if k[0] == r]), 3)
           for r in {k[0] for k in ctl}}
    res["c5_controls"] = {"uncoded_share": round(len(unc) / max(len(ctl), 1), 3), "per_row": per,
                          "pass": len(unc) / max(len(ctl), 1) >= 0.9}

    # 6. memory vs text for classics
    tx = {key[r]["book"]: r for r in key if key[r]["source"] == "text"}
    mm = {key[r]["book"]: r for r in key if key[r]["source"] == "memory" and key[r]["group"] == "classic"}
    agree = n = 0
    cov_t = cov_m = 0
    for b in tx:
        for s in stmts:
            if stmts[s]["kind"] == "reversed":
                continue
            a, m = fin[(tx[b], s)]["code"], fin[(mm[b], s)]["code"]
            cov_t += a in "AB"
            cov_m += m in "AB"
            if a in "AB" and m in "AB":
                n += 1
                agree += a == m
    res["c6_memory_vs_text"] = {"both_addressed": n, "sign_agreement": round(agree / max(n, 1), 3),
                                "addressed_text": cov_t, "addressed_memory": cov_m, "pass": agree / max(n, 1) >= 0.7}

    # first SVD component vs group and source
    rws = sorted({k[0] for k in orig})
    ss = sorted({k[1] for k in orig})
    m = np.array([[SIGN.get(fin[(r, s)]["code"]) or 0 for s in ss] for r in rws], dtype=float)
    u, sv, _ = np.linalg.svd(m - m.mean(0), full_matrices=False)
    pc1 = u[:, 0] * sv[0]
    lab = lambda f: np.array([f(key[r]) for r in rws], dtype=float)  # noqa: E731

    def r2(x):
        return round(float(np.corrcoef(pc1, x)[0, 1] ** 2), 3)
    res["pc1"] = {"var_share": round(float(sv[0] ** 2 / (sv ** 2).sum()), 3),
                  "r2_modern": r2(lab(lambda k: k["group"] == "modern")),
                  "r2_text_source": r2(lab(lambda k: k["source"] == "text")),
                  "r2_coverage": r2(np.abs(m).sum(1)), "r2_neg_count": r2((m < 0).sum(1))}
    # per threshold
    res["per_threshold"] = {}
    for lab_ in THRESHOLDS:
        ks = [k for k in addr if lab_ in stmts[k[1]]["thresholds"]]
        res["per_threshold"][lab_] = {"statements": sum(lab_ in s["thresholds"] for s in stmts.values()),
                                      "addressed": len(ks),
                                      "neg_share": round(sum(fin[k]["code"] == "B" for k in ks) / max(len(ks), 1), 3)}
    return res


def gold_sample(work: Path, n_b=15, n_a=10, n_n=10, n_d=5) -> list[dict]:
    """40 cells for blind hand coding: 15 predicted B, 10 A, 10 N, 5 disputes (real books only)."""
    key = {r["row"]: r for r in json.loads((work / "rows_key.json").read_text())}
    stmts = {s["sid"]: s for s in final_list(work)}
    fin = consensus(work)
    rng = random.Random(SEED + 3)
    real = sorted(k for k in fin if key[k[0]]["group"] in ("classic", "modern") and stmts[k[1]]["kind"] != "reversed")
    dis = [k for k in real if fin[k]["how"] in ("tie", "weak")]
    pick = rng.sample(dis, min(n_d, len(dis)))
    taken = set(pick)
    for code_, n in (("B", n_b), ("A", n_a), ("N", n_n)):
        pool = [k for k in real if fin[k]["code"] == code_ and fin[k]["how"] == "agree" and k not in taken]
        s = rng.sample(pool, min(n, len(pool)))
        pick += s
        taken.update(s)
    rng.shuffle(pick)
    return [{"cell": f"g{i:02d}", "row": r, "sid": s, "stratum": fin[(r, s)]["code"] if fin[(r, s)]["how"] == "agree"
             else "dispute"} for i, (r, s) in enumerate(pick, 1)]
