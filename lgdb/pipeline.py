"""Pipeline stages. Each stage reads files from the work folder and writes its own.

Blind stages see only prompts and the previous stage's output. The key file that
maps fragments back to texts is read only by code (never sent to a model).
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import factors as fa
from . import graph as gr
from . import topics as tp
from .llm import render

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
WORDING_WEIGHT = {"absolute": 1.0, "direct": 0.75, "hedged": 0.5, "tentative": 0.25}
UNRESTRICTED_SCOPE = {"all people", "not stated", ""}

CALIBRATION = [
    {"id": "CAL-smoking", "behavior": "Smoking tobacco",
     "definition": "Regularly smoking cigarettes or other tobacco products.", "expected": "lower_better"},
    {"id": "CAL-activity", "behavior": "Physical activity",
     "definition": "Moving the body with moderate or vigorous effort on most days.", "expected": "higher_better"},
    {"id": "CAL-sleep", "behavior": "Nightly sleep duration",
     "definition": "The number of hours a person usually sleeps per night.", "expected": "peak"},
]


def prompt(name: str) -> str:
    matches = sorted(PROMPTS.glob(f"{name}_*.txt"))  # "P5" must not pick up P5p_..., "P1" not P10_...
    if len(matches) != 1:
        raise FileNotFoundError(f"expected 1 prompt file for {name}, found {[m.name for m in matches]}")
    return matches[0].read_text(encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def save_jsonl(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_key(keys_dir: Path) -> dict[str, str]:
    with open(keys_dir / "fragments_key.csv", newline="", encoding="utf-8") as f:
        return {r["fragment_id"]: r["text_id"] for r in csv.DictReader(f)}


def claim_texts(claims: list[dict], key: dict[str, str]) -> dict[str, str]:
    return {c["claim_id"]: key[c["fragment_id"]] for c in claims}


# ---------- stage 1: extraction and recognition ----------

def extract(llm, work: Path) -> int:
    frags = load_jsonl(work / "fragments.jsonl")
    tpl = prompt("P2")
    outs = llm.map([render(tpl, {"PASSAGE_ID": f["fragment_id"], "TEXT": f["text"]}) for f in frags])
    claims = []
    for f, out in zip(frags, outs):
        for k, r in enumerate((out or {}).get("recommendations", []) or [], 1):
            if not isinstance(r, dict) or not r.get("recommendation"):
                continue
            r = dict(r)
            r["claim_id"], r["fragment_id"] = f"{f['fragment_id']}-{k}", f["fragment_id"]
            claims.append(r)
    save_jsonl(work / "claims.jsonl", claims)
    return len(claims)


def recognize(llm, work: Path) -> int:
    frags = load_jsonl(work / "fragments.jsonl")
    tpl = prompt("P3")
    outs = llm.map([render(tpl, {"TEXT": f["text"]}) for f in frags])
    save_jsonl(work / "recognition.jsonl",
               [{"fragment_id": f["fragment_id"], **(o or {})} for f, o in zip(frags, outs)])
    return len(frags)


# ---------- stage 2: topics ----------

def topics(cfg: dict, work: Path, key: dict[str, str]) -> dict:
    claims = load_jsonl(work / "claims.jsonl")
    ids = [c["claim_id"] for c in claims]
    text_of = claim_texts(claims, key)
    # Opposite instructions ("save money" vs "spend money") can embed far apart, which would
    # hide a split from the compatibility graph. Mixing in the polarity-free "domain" field
    # keeps them in 1 topic: cosine similarity becomes w * sim(domain) + (1 - w) * sim(recommendation).
    emb = cfg["embedding"]
    w = cfg["topics"].get("domain_weight", 0.5)
    z_rec = tp.embed([c["recommendation"] for c in claims], emb["provider"], emb["model"])
    z_dom = tp.embed([str(c.get("domain", "")) or c["recommendation"] for c in claims],
                     emb["provider"], emb["model"])
    z = np.hstack([np.sqrt(w) * z_dom, np.sqrt(1 - w) * z_rec])
    z /= np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-12, None)
    np.save(work / "embeddings.npy", z)
    out = {"resolutions": {}}
    for name, thr in cfg["topics"]["distance_thresholds"].items():
        labels = tp.cluster(z, thr)
        elig = tp.eligible(labels, ids, text_of, cfg["topics"]["min_claims"], cfg["topics"]["min_texts"])
        out["resolutions"][name] = {
            "labels": {cid: int(l) for cid, l in zip(ids, labels)},
            "eligible": {f"{name}-{lab}": members for lab, members in elig.items()},
            "stability_ari": tp.subsample_stability(z, thr),
        }
    (work / "topics.json").write_text(json.dumps(out))
    return {k: len(v["eligible"]) for k, v in out["resolutions"].items()}


def _listing(claims_by_id, ids, limit, seed):
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    return "\n".join(f"{i}: {claims_by_id[i]['recommendation']}" for i in ids[:limit])


def name_topics(llm, work: Path, resolution: str = "primary", limit: int = 60) -> int:
    claims = {c["claim_id"]: c for c in load_jsonl(work / "claims.jsonl")}
    elig = json.loads((work / "topics.json").read_text())["resolutions"][resolution]["eligible"]
    tpl = prompt("P4a")
    tids = sorted(elig)
    outs = llm.map([render(tpl, {"CLUSTER_ID": t, "RECOMMENDATIONS": _listing(claims, elig[t], limit, t)})
                    for t in tids])
    (work / "topic_names.json").write_text(json.dumps(dict(zip(tids, outs)), ensure_ascii=False))
    return len(tids)


# ---------- stage 3: pairs and relations ----------

def sample_pairs(cfg: dict, work: Path, gold_csv: Path | None = None, resolution: str = "primary") -> int:
    rng = random.Random(cfg["seed"])
    claims = {c["claim_id"]: c for c in load_jsonl(work / "claims.jsonl")}
    elig = json.loads((work / "topics.json").read_text())["resolutions"][resolution]["eligible"]
    seen, rows = set(), []

    def add(a, b, kind, topic=""):
        pair = tuple(sorted((a, b)))
        if a == b or pair in seen:
            return
        seen.add(pair)
        first, second = (a, b) if rng.random() < 0.5 else (b, a)
        rows.append({"pair_id": f"P{len(rows):06d}", "a": first, "b": second, "kind": kind, "topic": topic,
                     "rec_1": claims[first]["recommendation"], "rec_2": claims[second]["recommendation"]})

    k = cfg["pairs"]["partners_per_claim"]
    for t, members in elig.items():
        for a in members:
            for b in rng.sample(members, min(k + 1, len(members))):
                add(a, b, "within", t)
    pool = [(t, c) for t, members in elig.items() for c in members]
    for _ in range(cfg["pairs"]["cross_topic_pairs"]):
        (t1, a), (t2, b) = rng.sample(pool, 2)
        if t1 != t2:
            add(a, b, "cross")
    if gold_csv and gold_csv.exists():
        with open(gold_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({"pair_id": r["pair_id"], "a": "", "b": "", "kind": "gold", "topic": "",
                             "rec_1": r["rec_1"], "rec_2": r["rec_2"]})
    rng.shuffle(rows)
    save_jsonl(work / "pairs.jsonl", rows)
    return len(rows)


def relate(llm, work: Path, out_name: str = "relations.jsonl", fraction: float = 1.0, seed: int = 0) -> int:
    pairs = load_jsonl(work / "pairs.jsonl")
    if fraction < 1.0:
        pairs = random.Random(seed).sample(pairs, int(fraction * len(pairs)))
    tpl = prompt("P5")
    outs = llm.map([render(tpl, {"PAIR_ID": p["pair_id"], "REC_1": p["rec_1"], "REC_2": p["rec_2"]},
                           shuffle_seed=p["pair_id"]) for p in pairs])
    save_jsonl(work / out_name, [{**p, "label": (o or {}).get("label", ""),
                                  "condition": (o or {}).get("condition", "")} for p, o in zip(pairs, outs)])
    return len(pairs)


# ---------- stage 4: camps ----------

def camps(cfg: dict, work: Path, key: dict[str, str], conditional_as_negative: bool = False,
          relations: str = "relations.jsonl", out_name: str = "camps.json", resolution: str = "primary") -> dict:
    claims = load_jsonl(work / "claims.jsonl")
    text_of = claim_texts(claims, key)
    elig = json.loads((work / "topics.json").read_text())["resolutions"][resolution]["eligible"]
    rel = [r for r in load_jsonl(work / relations) if r["kind"] == "within"]
    by_topic = defaultdict(list)
    for r in rel:
        by_topic[r["topic"]].append((r["a"], r["b"], r["label"]))
    g = cfg["graph"]
    results = {}
    for t, members in sorted(elig.items()):
        sg = gr.SignedGraph.from_labels(members, by_topic.get(t, []), conditional_as_negative)
        res = gr.analyze_topic(sg, text_of, g["neg_rate_min"], g["frustration_max"], g["null_alpha"],
                               g["n_perm"], g["min_camp_claims"], g["min_camp_texts"], seed=cfg["seed"])
        results[t] = asdict(res)
    (work / out_name).write_text(json.dumps(results))
    return {"topics": len(results), "bipolar": sum(r["bipolar"] for r in results.values())}


def describe_camps(llm, work: Path, limit: int = 40) -> int:
    claims = {c["claim_id"]: c for c in load_jsonl(work / "claims.jsonl")}
    res = json.loads((work / "camps.json").read_text())
    tpl = prompt("P4b")
    jobs = []
    for t, r in sorted(res.items()):
        if not r["bipolar"]:
            continue
        for camp in (0, 1):
            ids = [c for c, v in r["camp_of"].items() if v == camp]
            jobs.append((f"{t}/{camp}", ids))
    outs = llm.map([render(tpl, {"SET_ID": sid, "RECOMMENDATIONS": _listing(claims, ids, limit, sid)})
                    for sid, ids in jobs])
    (work / "camp_names.json").write_text(json.dumps({sid: o for (sid, _), o in zip(jobs, outs)},
                                                     ensure_ascii=False))
    return len(jobs)


# ---------- stage 5: positions and factors ----------

def positions(work: Path, key: dict[str, str], camps_file: str = "camps.json") -> list[dict]:
    claims = {c["claim_id"]: c for c in load_jsonl(work / "claims.jsonl")}
    res = json.loads((work / camps_file).read_text())
    rows = []
    for t, r in res.items():
        per_text = defaultdict(list)
        for cid, camp in r["camp_of"].items():
            per_text[key[claims[cid]["fragment_id"]]].append((cid, camp))
        for text_id, items in per_text.items():
            w = {0: 0.0, 1: 0.0}
            limited = False
            for cid, camp in items:
                c = claims[cid]
                w[camp] += WORDING_WEIGHT.get(str(c.get("wording", "")).lower(), 0.5)
                cond = str(c.get("conditions", "none")).strip().lower()
                scope = str(c.get("scope", "not stated")).strip().lower()
                limited |= (cond not in ("none", "")) or (scope not in UNRESTRICTED_SCOPE)
            camps_used = {camp for _, camp in items}
            rows.append({"topic": t, "text_id": text_id, "bipolar": r["bipolar"], "n_claims": len(items),
                         "position": w[0] / (w[0] + w[1]) if (w[0] + w[1]) else np.nan,
                         "one_sided": len(items) >= 2 and len(camps_used) == 1 and not limited})
    with open(work / "positions.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["topic"])
        wr.writeheader()
        wr.writerows(rows)
    return rows


def position_matrix(rows: list[dict]):
    rows = [r for r in rows if str(r["bipolar"]) in ("True", "true", "1") or r["bipolar"] is True]
    texts = sorted({r["text_id"] for r in rows})
    tops = sorted({r["topic"] for r in rows})
    x = np.full((len(texts), len(tops)), np.nan)
    ti, ci = {t: i for i, t in enumerate(texts)}, {t: j for j, t in enumerate(tops)}
    for r in rows:
        x[ti[r["text_id"]], ci[r["topic"]]] = float(r["position"])
    return x, texts, tops


def factors(cfg: dict, work: Path) -> dict:
    with open(work / "positions.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    x, texts, tops = position_matrix(rows)
    f_cfg = cfg["factors"]
    res = fa.analyze(x, f_cfg["k_max"], f_cfg["n_boot"], f_cfg["congruence_min"], f_cfg["stable_share"],
                     f_cfg["ridge"], cfg["seed"])
    out = {"k": res.k, "cv": {str(k): v for k, v in res.cv_table.items()}, "topics": tops,
           "loadings": res.loadings.tolist(), "stability": res.stability.tolist(),
           "stable": res.stable.tolist()}
    (work / "factors.json").write_text(json.dumps(out))
    return {"k": res.k, "stable": int(np.sum(res.stable))}


def name_factors(llm, work: Path, top_n: int = 6) -> int:
    fac = json.loads((work / "factors.json").read_text())
    names = json.loads((work / "camp_names.json").read_text())
    tpl = prompt("P6")
    load = np.array(fac["loadings"])
    jobs = []
    for j in range(fac["k"]):
        order = np.argsort(-np.abs(load[:, j]))[:top_n]
        end1, end2 = [], []
        for i in order:
            t, l = fac["topics"][i], load[i, j]
            hi, lo = (0, 1) if l > 0 else (1, 0)  # positive loading: camp 0 sits at END 1
            for camp, bucket in ((hi, end1), (lo, end2)):
                d = names.get(f"{t}/{camp}") or {}
                bucket.append(f"{t}/{camp}: {d.get('rule', '')} ({abs(l):.2f})")
        jobs.append((f"F{j + 1}", "\n".join(end1), "\n".join(end2)))
    outs = llm.map([render(tpl, {"FACTOR_ID": fid, "END_1": e1, "END_2": e2}) for fid, e1, e2 in jobs])
    (work / "factor_names.json").write_text(json.dumps({fid: o for (fid, _, _), o in zip(jobs, outs)},
                                                       ensure_ascii=False))
    return len(jobs)


# ---------- stage 6: evidence (H5) ----------

def aggregate_evidence(coded: list[dict], meta_weight: float = 3.0, min_weight: float = 3.0) -> str:
    """Preregistered rule turning per-study codes into 1 label per behavior."""
    w = defaultdict(float)
    for c in coded:
        lab = str(c.get("label", "")).lower()
        if lab in ("", "unclear"):
            continue
        w[lab] += meta_weight if "meta" in str(c.get("design", "")).lower() else 1.0
    total = sum(w.values())
    if total < min_weight:
        return "insufficient"
    if (w["curved"] + w["subgroup"]) / total >= 1 / 3:
        return "peak" if w["curved"] >= w["subgroup"] else "varies"
    for lab, out in (("positive", "higher_better"), ("negative", "lower_better"), ("none", "none")):
        if w[lab] / total >= 2 / 3:
            return out
    return "insufficient"


def evidence(llm_search, llm, work: Path, seed: int = 0) -> dict:
    res = json.loads((work / "camps.json").read_text())
    names = json.loads((work / "camp_names.json").read_text())
    rng = random.Random(seed)
    targets = []
    for t, r in sorted(res.items()):
        if r["bipolar"]:
            a = rng.choice((0, 1))  # which camp is presented as "the behavior" is a coin flip
            d = names.get(f"{t}/{a}") or {}
            targets.append({"id": t, "camp_a": a, "behavior": d.get("name", t), "definition": d.get("rule", "")})
    targets += [dict(c) for c in CALIBRATION]
    tpl_a, tpl_b = prompt("P7a"), prompt("P7b")
    found = llm_search.map([render(tpl_a, {"BEHAVIOR": x["behavior"], "DEFINITION": x["definition"]})
                            for x in targets])
    jobs = []
    for x, out in zip(targets, found):
        for k, s in enumerate((out or {}).get("sources", []) or [], 1):
            jobs.append((x, {**s, "source_id": f"{x['id']}-S{k}"}))
    codes = llm.map([render(tpl_b, {"SOURCE_ID": s["source_id"], "BEHAVIOR": x["behavior"],
                                    "OUTCOME": s.get("outcome", ""), "RESULT": s.get("result", "")},
                            shuffle_seed=s["source_id"]) for x, s in jobs])
    coded = defaultdict(list)
    rows = []
    for (x, s), c in zip(jobs, codes):
        row = {**s, "target": x["id"], "label": (c or {}).get("label", ""), "groups": (c or {}).get("groups", "")}
        coded[x["id"]].append(row)
        rows.append(row)
    save_jsonl(work / "evidence_sources.jsonl", rows)
    summary = {x["id"]: {**x, "label": aggregate_evidence(coded[x["id"]]), "n_sources": len(coded[x["id"]])}
               for x in targets}
    (work / "evidence.json").write_text(json.dumps(summary, ensure_ascii=False))
    return {k: v["label"] for k, v in summary.items()}
