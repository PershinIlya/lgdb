"""Command line entry point.

Typical pilot sequence:
  python run.py synth                      # unblinded: planted controls
  python run.py prepare --split pilot
  python run.py blind --split pilot        # extract, recognize, topics, names, pairs, relations, camps
  python run.py panel --split pilot        # model panel labels; writes human/review_*.csv for the researcher
  python run.py gates --split pilot        # needs the filled review sheets and the negative-corpus run
  python run.py freeze                     # only after every gate passes
  python run.py hypotheses --split main
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from lgdb import analysis, freeze as fz, gates as gt, pipeline as pl, synthetic as sy
from lgdb.llm import LLM, render
from lgdb.prepare import prepare


def load_cfg(root: Path) -> dict:
    return yaml.safe_load((root / "config.yaml").read_text())


def make_llm(cfg: dict, root: Path, role: str = "annotator", search: bool = False) -> LLM:
    m = cfg["models"]
    return LLM(m[role], root / "cache", temperature=m["temperature"], workers=m["workers"],
               tools=[m["web_search_tool"]] if search else None, max_tokens=8192 if search else 4096)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command")
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="pilot")
    ap.add_argument("--work", default=None, help="work folder (default work_<split>)")
    ap.add_argument("--negative-work", default="work_negative")
    ap.add_argument("--exploratory", action="store_true")
    ap.add_argument("--exclude-split", default=None, help="drop texts of this split from hypothesis tests")
    a = ap.parse_args(argv)

    root = Path(a.root).resolve()
    cfg = load_cfg(root)
    work = root / (a.work or f"work_{a.split}")
    keys = root / "keys"
    work.mkdir(exist_ok=True)

    def say(x):
        print(json.dumps(x, indent=1, default=str))

    if a.command == "synth":
        say(sy.generate(make_llm(cfg, root, "generator"), root / "prompts", root / "corpus", keys))
    elif a.command == "prepare":
        say(prepare(root / "corpus", work, keys, a.split, root / "corpus" / "synthetic.jsonl",
                    seed=cfg["seed"], **cfg["corpus"]))
    elif a.command == "blind":
        llm = make_llm(cfg, root)
        key = pl.load_key(keys)
        say({"claims": pl.extract(llm, work), "recognized": pl.recognize(llm, work)})
        say(pl.topics(cfg, work, key))
        say({"named": pl.name_topics(llm, work)})
        say({"pairs": pl.sample_pairs(cfg, work, root / "human" / "pairs_gold.csv")})
        say({"related": pl.relate(llm, work)})
        say({"related_second": pl.relate(make_llm(cfg, root, "second_annotator"), work, "relations_m2.jsonl",
                                         cfg["pairs"]["second_model_fraction"], cfg["seed"])})
        say(pl.camps(cfg, work, key))
        say(pl.camps(cfg, work, key, conditional_as_negative=True, out_name="camps_conditional.json"))
        say({"camp_sets": pl.describe_camps(llm, work)})
        pl.positions(work, key)
    elif a.command == "panel":
        from lgdb import panel
        pc = cfg["gates"]["panel"]
        llms = [LLM(m, root / "cache", temperature=cfg["models"]["temperature"], workers=cfg["models"]["workers"])
                for m in cfg["models"]["panel"]]
        say(panel.sample(work, pc["n_pairs"], pc["n_claims"], pc["n_passages"], cfg["seed"]))
        say({"pairs": panel.run_pairs(llms, work), "claims": panel.run_claims(llms, work),
             "missing_candidates": panel.run_missing(llms[0], llms[1:], work)})
        say(panel.review_sheets(work, root / "human", pc["audit_share"], cfg["seed"]))
    elif a.command == "gates":
        say(gt.run_all(root, work, root / a.negative_work, cfg))
    elif a.command == "freeze":
        say(fz.freeze(root, cfg["models"]))
    elif a.command == "hypotheses":
        changed = fz.check(root)
        gates = json.loads((work / "gates.json").read_text()) if (work / "gates.json").exists() else {}
        if changed and not a.exploratory:
            sys.exit(f"Refusing: files changed since freeze: {changed}")
        if not gates.get("ALL_PASSED") and not a.exploratory:
            sys.exit("Refusing: quality gates have not all passed (see gates.json)")
        key = pl.load_key(keys)
        excl = None
        if a.exclude_split:
            meta = analysis.read_meta(root / "corpus")
            excl = {t for t, r in meta.items() if r["split"] == a.exclude_split}
        llm = make_llm(cfg, root)
        say(pl.factors(cfg, work))
        say({"factors_named": pl.name_factors(llm, work)})
        say(pl.evidence(make_llm(cfg, root, "search", search=True), llm, work, cfg["seed"]))
        res = analysis.run_all(work, root / "corpus", key, cfg, excl)
        res["mode"] = "exploratory" if a.exploratory else "confirmatory"
        (work / "hypotheses.json").write_text(json.dumps(res, indent=1, default=str))
        say(res)
    elif a.command == "corpus-candidates":
        llm = make_llm(cfg, root, "search", search=True)
        tpl = pl.prompt("P1")
        cells = [(p, r, f) for p in cfg["cells"]["periods"] for r in cfg["cells"]["regions"]
                 for f in ("collection", "thesis")]
        outs = llm.map([render(tpl, {"PERIOD": p, "REGION": r, "FORMAT": f}) for p, r, f in cells])
        (root / "corpus" / "candidates.json").write_text(json.dumps(outs, indent=1, ensure_ascii=False))
        say({"cells": len(cells)})
    elif a.command == "select-corpus":
        from lgdb import select as sel
        grid = [(p, r, f) for p in cfg["cells"]["periods"] for r in cfg["cells"]["regions"]
                for f in ("collection", "thesis")]
        rows = sel.select(root / "corpus" / "candidates.csv", root / "human" / "owned_texts.csv", cfg["seed"], grid,
                          root / "corpus" / "availability.csv")
        say(sel.write(rows, root / "corpus" / "selection.csv"))
    elif a.command == "build-meta":
        from lgdb import select as sel
        say(sel.build_meta(root / "corpus"))
    elif a.command == "fetch":
        from lgdb import fetch
        say(fetch.run(root / "corpus"))
    else:
        sys.exit(f"unknown command {a.command}")


if __name__ == "__main__":
    main()
