"""Preregistered hypothesis tests H1 to H5. Runs only on a frozen pipeline that passed the gates."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .pipeline import load_jsonl


def read_meta(corpus_dir: Path) -> dict[str, dict]:
    with open(corpus_dir / "meta.csv", newline="", encoding="utf-8") as f:
        return {r["text_id"]: r for r in csv.DictReader(f)}


def read_positions(work: Path) -> list[dict]:
    with open(work / "positions.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["bipolar"] = r["bipolar"] in ("True", "true", "1")
        r["one_sided"] = r["one_sided"] in ("True", "true", "1")
        r["n_claims"] = int(r["n_claims"])
        r["position"] = float(r["position"]) if r["position"] not in ("", "nan") else np.nan
    return rows


def verdict(value: float, support: float, refute: float, higher_is_support: bool = True) -> str:
    if np.isnan(value):
        return "not evaluable"
    if higher_is_support:
        return "supported" if value >= support else "refuted" if value < refute else "inconclusive"
    return "supported" if value <= support else "refuted" if value > refute else "inconclusive"


def bootstrap(texts: list[str], stat, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    texts = list(texts)
    vals = []
    for _ in range(n):
        sample = list(rng.choice(texts, size=len(texts), replace=True))
        v = stat(sample)
        if not np.isnan(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def _multiset(sample):
    counts = defaultdict(int)
    for t in sample:
        counts[t] += 1
    return counts


# ---------- H1 ----------

def h1(work: Path, key: dict[str, str], cfg: dict, allowed: set | None = None):
    claims = {c["claim_id"]: c for c in load_jsonl(work / "claims.jsonl")}
    elig = json.loads((work / "topics.json").read_text())["resolutions"]["primary"]["eligible"]
    bip = {t for t, r in json.loads((work / "camps.json").read_text()).items() if r["bipolar"]}
    per_text = defaultdict(lambda: [0, 0])  # text -> [claims in bipolar topics, claims in eligible topics]
    for t, members in elig.items():
        for cid in members:
            tx = key[claims[cid]["fragment_id"]]
            if allowed is not None and tx not in allowed:
                continue
            per_text[tx][1] += 1
            per_text[tx][0] += t in bip

    def stat(sample):
        m = _multiset(sample)
        num = sum(per_text[t][0] * k for t, k in m.items())
        den = sum(per_text[t][1] * k for t, k in m.items())
        return num / den if den else np.nan

    texts = list(per_text)
    value = stat(texts)
    th = cfg["thresholds"]["H1"]
    return {"value": value, "ci95": bootstrap(texts, stat, seed=cfg["seed"]),
            "n_topics": len(elig), "n_bipolar": len(bip),
            "verdict": verdict(value, th["support"], th["refute"])}


# ---------- H2 ----------

def h2(rows: list[dict], meta: dict, cfg: dict):
    th = cfg["thresholds"]["H2"]
    by_topic = defaultdict(list)
    for r in rows:
        if r["bipolar"] and not np.isnan(r["position"]) and r["position"] != 0.5:
            by_topic[r["topic"]].append(r)

    def ok(topic_rows):
        camp = defaultdict(set)
        for r in topic_rows:
            camp[0 if r["position"] > 0.5 else 1].add(r["text_id"])
        n0, n1 = len(camp[0]), len(camp[1])
        if n0 + n1 == 0:
            return False
        minority = camp[0] if n0 <= n1 else camp[1]
        cells = {(meta[t]["period"], meta[t]["region"], meta[t]["format"]) for t in minority}
        return (min(n0, n1) / (n0 + n1) >= th["minority_share"]
                and len(minority) >= th["minority_texts"] and len(cells) >= th["minority_cells"])

    def stat(sample):
        m = _multiset(sample)
        flags = []
        for topic_rows in by_topic.values():
            expanded = [r for r in topic_rows for _ in range(m.get(r["text_id"], 0))]
            if expanded:
                flags.append(ok(expanded))
        return float(np.mean(flags)) if flags else np.nan

    texts = sorted({r["text_id"] for rs in by_topic.values() for r in rs})
    value = stat(texts)
    return {"value": value, "ci95": bootstrap(texts, stat, seed=cfg["seed"]),
            "n_bipolar_topics": len(by_topic), "verdict": verdict(value, th["support"], th["refute"])}


# ---------- H3 and H3b ----------

def h3(rows: list[dict], cfg: dict):
    th = cfg["thresholds"]["H3"]
    elig = [r for r in rows if r["bipolar"] and r["n_claims"] >= 2]
    by_text = defaultdict(list)
    for r in elig:
        by_text[r["text_id"]].append(r["one_sided"])

    def stat(sample):
        vals = [v for t in sample for v in by_text[t]]
        return float(np.mean(vals)) if vals else np.nan

    texts = list(by_text)
    value = stat(texts)
    return {"value": value, "ci95": bootstrap(texts, stat, seed=cfg["seed"]), "n_pairs": len(elig),
            "verdict": verdict(value, th["support"], th["refute"])}


def h3b(rows: list[dict], meta: dict, cfg: dict):
    """One-sidedness: thesis-driven books minus collections. Permutation of the format
    label across texts within period x region (preregistered amendment, see notes)."""
    elig = [r for r in rows if r["bipolar"] and r["n_claims"] >= 2]
    by_text = defaultdict(list)
    for r in elig:
        by_text[r["text_id"]].append(r["one_sided"])
    texts = [t for t in by_text if meta[t]["format"] in ("thesis", "collection")]

    def diff(fmt):
        th = [v for t in texts if fmt[t] == "thesis" for v in by_text[t]]
        co = [v for t in texts if fmt[t] == "collection" for v in by_text[t]]
        return (np.mean(th) - np.mean(co)) if th and co else np.nan

    fmt = {t: meta[t]["format"] for t in texts}
    obs = diff(fmt)
    rng = np.random.default_rng(cfg["seed"])
    strata = defaultdict(list)
    for t in texts:
        strata[(meta[t]["period"], meta[t]["region"])].append(t)
    null = []
    for _ in range(cfg["thresholds"]["n_perm"]):
        perm = {}
        for members in strata.values():
            labels = [fmt[t] for t in members]
            rng.shuffle(labels)
            perm.update(dict(zip(members, labels)))
        null.append(diff(perm))
    null = np.array([v for v in null if not np.isnan(v)])
    p = (1 + np.sum(null >= obs)) / (1 + len(null)) if not np.isnan(obs) and len(null) else np.nan
    v = ("supported" if p < 0.05 and obs > 0 else "refuted" if obs < 0 else "inconclusive") \
        if not np.isnan(p) else "not evaluable"
    return {"diff_thesis_minus_collection": obs, "p_one_sided": p, "verdict": v}


# ---------- H4 ----------

def _r2(y, x):
    x = np.column_stack([np.ones(len(y)), x]) if x.size else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    tot = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum(resid ** 2) / tot if tot else 0.0


def _dummies(values):
    levels = sorted(set(values))[1:]
    return np.array([[v == l for l in levels] for v in values], dtype=float).reshape(len(values), len(levels))


def h4(rows: list[dict], meta: dict, cfg: dict):
    th = cfg["thresholds"]["H4"]
    rng = np.random.default_rng(cfg["seed"])
    by_topic = defaultdict(list)
    for r in rows:
        if r["bipolar"] and not np.isnan(r["position"]):
            by_topic[r["topic"]].append(r)
    results = {}
    for t, rs in by_topic.items():
        if len(rs) < th["min_texts"]:
            continue
        y = np.array([r["position"] for r in rs])
        per = [meta[r["text_id"]]["period"] for r in rs]
        reg = [meta[r["text_id"]]["region"] for r in rs]
        fmt = [meta[r["text_id"]]["format"] for r in rs]
        base = _dummies(fmt)

        def delta(p_, r_):
            full = np.hstack([base, _dummies(p_), _dummies(r_)])
            return _r2(y, full) - _r2(y, base)

        obs = delta(per, reg)
        null = []
        idx_by_fmt = defaultdict(list)
        for i, f in enumerate(fmt):
            idx_by_fmt[f].append(i)
        for _ in range(cfg["thresholds"]["n_perm"]):
            p2, r2 = list(per), list(reg)
            for idx in idx_by_fmt.values():
                order = rng.permutation(idx)
                for src, dst in zip(idx, order):
                    p2[dst], r2[dst] = per[src], reg[src]
            null.append(delta(p2, r2))
        p = (1 + np.sum(np.array(null) >= obs)) / (1 + len(null))
        results[t] = {"delta_r2": obs, "p": p, "n_texts": len(rs)}
    n = len(results)
    sig = sum(r["p"] < 0.05 for r in results.values())
    share = sig / n if n else np.nan
    binom_p = binomtest(sig, n, 0.05, alternative="greater").pvalue if n else np.nan
    if np.isnan(share):
        v = "not evaluable"
    elif share >= th["support"] and binom_p < 0.05:
        v = "supported"
    elif share <= th["refute"]:
        v = "refuted"
    else:
        v = "inconclusive"
    return {"share_significant": share, "binom_p": binom_p, "n_topics": n, "per_topic": results, "verdict": v}


# ---------- H5 ----------

def h5(work: Path, cfg: dict):
    ev = json.loads((work / "evidence.json").read_text())
    cal = {k: v for k, v in ev.items() if k.startswith("CAL-")}
    cal_ok = all(v["label"] == v.get("expected") for v in cal.values())
    labels = [v["label"] for k, v in ev.items() if not k.startswith("CAL-") and v["label"] != "insufficient"]
    if not cal_ok:
        return {"calibration": {k: (v["label"], v.get("expected")) for k, v in cal.items()},
                "verdict": "not evaluated (calibration failed)"}
    if not labels:
        return {"verdict": "not evaluable", "n": 0}
    inter = sum(l in ("peak", "varies") for l in labels) / len(labels)
    dom = sum(l in ("higher_better", "lower_better") for l in labels) / len(labels)
    th = cfg["thresholds"]["H5"]
    v = "supported" if inter >= th["support"] else "refuted" if dom >= th["refute_dominant"] else "inconclusive"
    return {"share_peak_or_varies": inter, "share_dominant": dom, "n": len(labels), "verdict": v}


def run_all(work: Path, corpus_dir: Path, key: dict[str, str], cfg: dict, exclude_texts: set | None = None):
    meta = read_meta(corpus_dir)
    # Real texts only: synthetic controls and excluded (for example pilot) texts are dropped.
    allowed = {t for t in meta if not (exclude_texts and t in exclude_texts)}
    rows = [r for r in read_positions(work) if r["text_id"] in allowed]
    out = {"H1": h1(work, key, cfg, allowed), "H2": h2(rows, meta, cfg), "H3": h3(rows, cfg),
           "H3b": h3b(rows, meta, cfg), "H4": h4(rows, meta, cfg)}
    if (work / "evidence.json").exists():
        out["H5"] = h5(work, cfg)
    return out
