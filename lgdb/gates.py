"""Quality gates. Hypothesis tests are refused until every gate passes.

A gate with missing input data reports pass = None, which counts as not passed.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity

from .graph import CONDITIONAL, NEGATIVE, POSITIVE
from .pipeline import load_jsonl

ORDINAL = {"tentative": 0, "hedged": 1, "direct": 2, "absolute": 3}


def sign_class(label: str) -> str:
    label = (label or "").strip().lower()
    if label in POSITIVE:
        return "yes"
    if label in NEGATIVE:
        return "no"
    if label in CONDITIONAL:
        return "conditional"
    return "unrelated"


def _csv(path: Path):
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def gate(value, threshold, higher_is_better=True, note=""):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {"value": None, "threshold": threshold, "pass": None, "note": note or "no data yet"}
    ok = value >= threshold if higher_is_better else value <= threshold
    return {"value": float(value), "threshold": threshold, "pass": bool(ok), "note": note}


def leakage(root: Path):
    data = json.loads((root / "audit" / "leakage_audit.json").read_text()) \
        if (root / "audit" / "leakage_audit.json").exists() else None
    if data is None:
        return gate(None, "all prompts pass or are exempt")
    current = [p.stem for p in (root / "prompts").glob("P*.txt") if not p.stem.startswith("P0")]
    bad = [p for p in current
           if data["prompts"].get(p, {}).get("verdict") not in ("pass", "exempt")]
    return {"value": len(bad), "threshold": 0, "pass": not bad,
            "note": ", ".join(bad) or "all current prompts pass or are exempt"}


def _audit_gates(prefix: str, g: dict, th_panel: dict) -> dict:
    return {f"{prefix}_audit": gate(g["audit_agreement"], th_panel["audit_agreement_min"],
                                    note=f"reviewer confirmed the unanimous panel on {g['audit_n']} hidden audit items"),
            f"{prefix}_pending": gate(g["pending"], 0, higher_is_better=False, note="review items not yet answered")}


def extraction(root: Path, work: Path, th: dict, th_panel: dict):
    """Gold from the model panel plus the researcher's review (see lgdb/panel.py)."""
    from . import panel
    human = root / "human"
    gc = panel.gold(work, human, "claims")
    lab = gc["labels"]
    out = {}
    out["supported_share"] = gate(np.mean([v["label"] == "yes" for v in lab.values()]) if lab else None,
                                  th["supported_min"], note=f"n={len(lab)}")
    # wording force is validated on planted synthetic texts (synthetic.wording_kappa), not here:
    # asking the gold panel for it leaked the hypothesis in the P0 audit
    out.update(_audit_gates("claims", gc, th_panel))
    gm = panel.gold(work, human, "missing")
    passages = load_jsonl(work / "panel_passages.jsonl") if (work / "panel_passages.jsonl").exists() else []
    n_ext = sum(p["n_extracted"] for p in passages)
    n_missed = sum(v["label"] == "yes" for v in gm["labels"].values())
    out["recall"] = gate(n_ext / (n_ext + n_missed) if passages and n_ext + n_missed else None, th["recall_min"],
                         note=f"{n_ext} extracted, {n_missed} confirmed missed, {len(passages)} passages")
    out.update(_audit_gates("missing", gm, th_panel))
    return out


def compatibility(root: Path, work: Path, th: dict, th_panel: dict):
    """Annotator (P5) against gold from the model panel (P5p) plus the researcher's review."""
    from . import panel
    gp = panel.gold(work, root / "human", "pairs")
    rel = {r["pair_id"]: panel.sign(r["label"]) for r in load_jsonl(work / "relations.jsonl")} \
        if (work / "relations.jsonl").exists() else {}
    both = [(v["label"], rel[i]) for i, v in gp["labels"].items() if i in rel and v["label"]]
    out = {"kappa": gate(cohen_kappa_score(*zip(*both)) if len(both) > 10 else None, th["kappa_min"],
                         note=f"n={len(both)}")}
    gy = [m for t, m in both if t == "compatible"]
    gn = [m for t, m in both if t == "incompatible"]
    out["false_no"] = gate(np.mean([m == "incompatible" for m in gy]) if gy else None, th["false_rate_max"], False,
                           note=f"n gold compatible={len(gy)}")
    out["false_yes"] = gate(np.mean([m == "compatible" for m in gn]) if gn else None, th["false_rate_max"], False,
                            note=f"n gold incompatible={len(gn)}")
    out.update(_audit_gates("pairs", gp, th_panel))
    return out


def synthetic(root: Path, work: Path, th: dict):
    spec_path = root / "keys" / "synthetic_spec.json"
    if not spec_path.exists():
        return {"camp_ari": gate(None, th["camp_ari_min"]), "no_false_bipolar": gate(None, 1),
                "wording_kappa": gate(None, th["wording_kappa_min"])}
    spec = {r["passage_id"]: r for r in json.loads(spec_path.read_text())}
    key = {r["fragment_id"]: r["text_id"] for r in _csv(root / "keys" / "fragments_key.csv")}
    claims = load_jsonl(work / "claims.jsonl")
    camps = json.loads((work / "camps.json").read_text())
    camp_of = {cid: (t, v) for t, r in camps.items() for cid, v in r["camp_of"].items()}
    topic_members = json.loads((work / "topics.json").read_text())["resolutions"]["primary"]["eligible"]
    topic_of = {cid: t for t, ms in topic_members.items() for cid in ms}

    by_passage = defaultdict(list)
    for c in claims:
        tid = key.get(c["fragment_id"], "")
        if tid.startswith("SYN::"):
            by_passage[tid[5:]].append(c)
    matched = []  # (spec_row, planted_rec, claim)
    for pid, row in spec.items():
        cand = by_passage.get(pid, [])
        if not cand:
            continue
        planted = [r["content"] for r in row["recommendations"]]
        vec = TfidfVectorizer().fit(planted + [c["recommendation"] for c in cand])
        sim = cosine_similarity(vec.transform(planted), vec.transform([c["recommendation"] for c in cand]))
        for i, rec in enumerate(row["recommendations"]):
            j = int(np.argmax(sim[i]))
            if sim[i, j] > 0.1:
                matched.append((row, rec, cand[j]))

    s1 = [(r["camp"], camp_of[c["claim_id"]][1]) for r, _, c in matched
          if r["structure"] == "two_camps" and c["claim_id"] in camp_of]
    ari = adjusted_rand_score(*zip(*s1)) if len(s1) >= 4 else None
    false_bip = 0
    for structure in ("unanimous", "complementary"):
        tops = Counter(topic_of.get(c["claim_id"]) for r, _, c in matched if r["structure"] == structure)
        tops.pop(None, None)
        if tops:
            false_bip += bool(camps[tops.most_common(1)[0][0]]["bipolar"])
    wk = [(ORDINAL[rec["wording"]], ORDINAL.get(str(c.get("wording", "")).lower(), -1)) for _, rec, c in matched]
    wk = [p for p in wk if p[1] >= 0]
    return {"camp_ari": gate(ari, th["camp_ari_min"], note=f"{len(s1)} matched claims"),
            "no_false_bipolar": gate(false_bip, 0, higher_is_better=False),
            "wording_kappa": gate(cohen_kappa_score(*zip(*wk), weights="quadratic") if wk else None,
                                  th["wording_kappa_min"])}


def negative_corpus(neg_work: Path, th: dict):
    path = neg_work / "camps.json"
    if not path.exists():
        return gate(None, th["bipolar_share_max"], False)
    res = json.loads(path.read_text())
    return gate(np.mean([r["bipolar"] for r in res.values()]) if res else 0.0, th["bipolar_share_max"], False)


def stability(work: Path, th: dict):
    m1 = {r["pair_id"]: sign_class(r["label"]) for r in load_jsonl(work / "relations.jsonl")}
    m2 = {r["pair_id"]: sign_class(r["label"]) for r in load_jsonl(work / "relations_m2.jsonl")}
    common = [p for p in m2 if p in m1]
    kappa = cohen_kappa_score([m1[p] for p in common], [m2[p] for p in common]) if len(common) > 20 else None
    topics = json.loads((work / "topics.json").read_text()) if (work / "topics.json").exists() else None
    ari = topics["resolutions"]["primary"]["stability_ari"] if topics else None
    return {"second_model_kappa": gate(kappa, th["second_model_kappa_min"], note=f"n={len(common)}"),
            "clustering_ari": gate(ari, th["clustering_ari_min"])}


def run_all(root: Path, work: Path, neg_work: Path, cfg: dict) -> dict:
    g = cfg["gates"]
    out = {"leakage": leakage(root), **{f"extraction.{k}": v for k, v in extraction(root, work, g["extraction"], g["panel"]).items()},
           **{f"compatibility.{k}": v for k, v in compatibility(root, work, g["compatibility"], g["panel"]).items()},
           **{f"synthetic.{k}": v for k, v in synthetic(root, work, g["synthetic"]).items()},
           "negative_corpus": negative_corpus(neg_work, g["negative_corpus"]),
           **{f"stability.{k}": v for k, v in stability(work, g["stability"]).items()}}
    out["ALL_PASSED"] = all(v["pass"] is True for k, v in out.items() if isinstance(v, dict))
    (work / "gates.json").write_text(json.dumps(out, indent=1))
    return out
