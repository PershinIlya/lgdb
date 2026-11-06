"""End-to-end dry run on a toy corpus with a fake model whose answers are known.

The toy world: 40 texts, 3 contested topics (money, work, people) and 1 consensus
topic (honesty). Texts with latent score +1 take camp A on money and work, texts
with -1 take camp B, and people is independent noise. The pipeline should find 3
bipolar topics, call honesty non-bipolar, and put money and work on 1 factor.
"""
import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np
import yaml

from lgdb import analysis, gates, pipeline as pl
from lgdb.llm import FakeLLM
from lgdb.prepare import prepare

ROOT = Path(__file__).resolve().parent.parent

RECS = {
    "money": {"A": ["Save money every month.", "Put money aside for later.", "Keep money in savings."],
              "B": ["Spend money on experiences now.", "Use money to enjoy today.", "Do not hoard money."]},
    "work": {"A": ["Work long hours to succeed.", "Put work before rest.", "Never stop working hard at work."],
             "B": ["Limit work hours.", "Stop work every evening.", "Protect your evenings from work."]},
    "people": {"A": ["Spend time with many people.", "Meet new people often.", "Seek people out."],
               "B": ["Spend time away from people.", "Avoid crowds of people.", "Keep distance from people."]},
    "honesty": {"A": ["Be honest always.", "Be honest in speech.", "Stay honest with friends."]},
}
WHERE = {rec: (topic, camp) for topic, camps in RECS.items() for camp, recs in camps.items() for rec in recs}


def fake(prompt: str):
    if "FIELDS FOR EACH RECOMMENDATION" in prompt and "PASSAGE" in prompt:
        body = prompt.rsplit("<<<\n", 1)[1].split("\n>>>", 1)[0]
        recs = [ln[2:].strip() for ln in body.splitlines() if ln.startswith("- ")]
        return {"recommendations": [{"recommendation": r, "domain": WHERE[r][0], "wording": "direct",
                                     "conditions": "none", "scope": "not stated"} for r in recs]}
    if "Do you recognize which text" in prompt:
        return {"recognized": False, "title": "", "author": "", "confidence": 0}
    if "Name the subject" in prompt:
        return {"subject": "toy", "description": "toy"}
    if "Write a 1-sentence rule" in prompt:
        first = re.search(r"\n[0-9a-f]+-\d+: (.*)", prompt)
        return {"coherent": True, "name": "set", "rule": first.group(1) if first else ""}
    if "every pair of entries gets a label" in prompt:
        r1 = re.search(r"\n1: (.*)", prompt).group(1).strip()
        r2 = re.search(r"\n2: (.*)", prompt).group(1).strip()
        (t1, c1), (t2, c2) = WHERE[r1], WHERE[r2]
        if t1 != t2:
            return {"label": "unrelated"}
        return {"label": "compatible" if c1 == c2 else "incompatible"}
    if "Decide how they relate" in prompt:  # P5p panel
        r1 = re.search(r"\nA: (.*)", prompt).group(1).strip()
        r2 = re.search(r"\nB: (.*)", prompt).group(1).strip()
        (t1, c1), (t2, c2) = WHERE[r1], WHERE[r2]
        return {"label": "unrelated" if t1 != t2 else "compatible" if c1 == c2 else "incompatible"}
    if "Code the entry against the passage." in prompt:  # P10
        return {"supported": "yes"}
    if "are missing from the list" in prompt:  # P11
        return {"missing": []}
    if "A statistical analysis of many texts" in prompt:
        return {"interpretable": True, "name": "toy factor"}
    if "use web search to find" in prompt:
        return {"sources": [{"design": "meta-analysis", "outcome": "well-being",
                             "result": "positive association"}] * 3}
    if "You are coding study results" in prompt:
        return {"label": "positive"}
    raise AssertionError("unexpected prompt")


def build_world(root: Path, n_texts: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    (root / "corpus" / "raw").mkdir(parents=True)
    shutil.copytree(ROOT / "prompts", root / "prompts")
    shutil.copytree(ROOT / "audit", root / "audit")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["embedding"] = {"provider": "tfidf", "model": ""}
    cfg["topics"]["distance_thresholds"] = {"primary": 0.9, "alt": 0.85}
    cfg["graph"]["n_perm"] = 50
    cfg["factors"].update({"k_max": 3, "n_boot": 40})
    cfg["thresholds"]["n_perm"] = 100
    (root / "config.yaml").write_text(yaml.safe_dump(cfg))
    periods, regions, formats = ["p1", "p2", "p3"], ["r1", "r2"], ["thesis", "collection"]
    with open(root / "corpus" / "meta.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text_id", "title", "author", "year", "period", "region", "format",
                    "translator", "translation_year", "split", "mask_terms"])
        for i in range(n_texts):
            s = 1 if i < n_texts // 2 else -1
            camp = {"money": "A" if s > 0 else "B", "work": "A" if s > 0 else "B",
                    "people": "A" if rng.random() < 0.5 else "B", "honesty": "A"}
            paras = []
            for frag in range(2):
                lines = ["Some story about life."]
                for topic in RECS:
                    pool = RECS[topic][camp[topic]]
                    lines += [f"- {pool[(frag + j) % len(pool)]}" for j in range(2)]
                paras.append("\n".join(lines) + "\n" + "filler " * 1500)
            (root / "corpus" / "raw" / f"T{i:02d}.txt").write_text("\n\n".join(paras))
            w.writerow([f"T{i:02d}", f"Title {i}", f"Author {i}", 1900 + i, periods[i % 3], regions[i % 2],
                        formats[(i // 2) % 2], "", "", "pilot", ""])
    return cfg


def test_end_to_end(tmp_path):
    cfg = build_world(tmp_path)
    work, keys = tmp_path / "work_pilot", tmp_path / "keys"
    n = prepare(tmp_path / "corpus", work, keys, "pilot", seed=cfg["seed"], **cfg["corpus"])
    assert n == 80
    llm = FakeLLM(fake, tmp_path / "cache")
    key = pl.load_key(keys)
    assert pl.extract(llm, work) == 80 * 8
    pl.recognize(llm, work)
    sizes = pl.topics(cfg, work, key)
    assert sizes["primary"] == 4, sizes
    pl.name_topics(llm, work)
    pl.sample_pairs(cfg, work)
    pl.relate(llm, work)
    pl.relate(llm, work, "relations_m2.jsonl", 0.2, 1)
    from lgdb import panel
    assert panel.sample(work, 60, 40, 5, seed=1)["pairs"] == 60
    plls = [FakeLLM(fake, tmp_path / f"panel{i}", f"fake-panel-{i}") for i in range(3)]
    panel.run_pairs(plls, work)
    panel.run_claims(plls, work)
    assert panel.run_missing(plls[0], plls[1:], work) == 0
    human = tmp_path / "human"
    counts = panel.review_sheets(work, human, 0.25, seed=1)
    assert 0 < counts["pairs"]["to_human"] < 60  # unanimous panel: only the hidden audit goes to the human
    with open(human / "review_pairs.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    col = [c for c in rows[0] if c.startswith("answer")][0]
    for r in rows:  # the reviewer answers from the truth of the toy world
        (t1, c1), (t2, c2) = WHERE[r["advice_A"]], WHERE[r["advice_B"]]
        r[col] = "unrelated" if t1 != t2 else "compatible" if c1 == c2 else "incompatible"
    with open(human / "review_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(human / "review_claims.csv", newline="") as f:
        crow = list(csv.DictReader(f))
    for r in crow:
        r[[c for c in r if c.startswith("supported")][0]] = "yes"
    with open(human / "review_claims.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(crow[0]))
        w.writeheader()
        w.writerows(crow)
    summary = pl.camps(cfg, work, key)
    assert summary == {"topics": 4, "bipolar": 3}, summary
    pl.describe_camps(llm, work)
    rows = pl.positions(work, key)
    assert all(r["one_sided"] for r in rows if r["bipolar"])

    fac = pl.factors(cfg, work)
    assert fac["k"] >= 1
    f = json.loads((work / "factors.json").read_text())
    load = np.abs(np.array(f["loadings"])[:, 0])
    names = {t: json.loads((work / "topic_names.json").read_text()) for t in f["topics"]}
    claims = {c["claim_id"]: c for c in pl.load_jsonl(work / "claims.jsonl")}
    elig = json.loads((work / "topics.json").read_text())["resolutions"]["primary"]["eligible"]
    topic_word = {t: next(w for w in ("money", "work", "people", "honest")
                          if w in claims[elig[t][0]]["recommendation"].lower()) for t in f["topics"]}
    top2 = {topic_word[f["topics"][i]] for i in np.argsort(-load)[:2]}
    assert top2 == {"money", "work"}, (top2, load)

    pl.name_factors(llm, work)
    pl.evidence(FakeLLM(fake, tmp_path / "cache2", "fake-search"), llm, work)
    res = analysis.run_all(work, tmp_path / "corpus", key, cfg)
    assert res["H1"]["value"] > 0.7 and res["H3"]["value"] == 1.0
    assert res["H5"]["verdict"].startswith("not evaluated")  # fake evidence fails calibration, as it should
    g = gates.run_all(tmp_path, work, tmp_path / "work_negative", cfg)
    assert g["ALL_PASSED"] is False  # human gold data missing, so gates cannot pass
    assert g["stability.second_model_kappa"]["pass"] is True
    for k in ("compatibility.kappa", "compatibility.pairs_audit", "compatibility.pairs_pending",
              "extraction.supported_share", "extraction.recall", "extraction.claims_audit"):
        assert g[k]["pass"] is True, (k, g[k])
