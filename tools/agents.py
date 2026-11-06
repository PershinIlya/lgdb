"""Run pipeline stages without an API key: agents answer queued prompts in batches.

  python tools/agents.py mini --n 10            # mark a seeded random subset of pilot texts as split pilot1
  python tools/agents.py stage extract          # run a stage; unanswered prompts go to the queue
  python tools/agents.py batches --size 8       # split the queue into batch files for agents
  python tools/agents.py ingest                 # load agent answers into the cache, clear the queue
Then run the same stage again: answered prompts come from the cache.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from lgdb import pipeline as pl  # noqa: E402
from lgdb.llm import QueueLLM  # noqa: E402

Q = ROOT / "work_pilot1" / "queue"


def llm(cfg, role="annotator"):
    m = cfg["models"][role]
    return QueueLLM(m, ROOT / "cache", Q / "queue.jsonl", temperature=cfg["models"]["temperature"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--size", type=int, default=8)
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    work, keys = ROOT / "work_pilot1", ROOT / "keys"

    if a.cmd == "mini":
        meta = list(csv.DictReader(open(ROOT / "corpus" / "meta.csv", encoding="utf-8")))
        pilot = sorted(r["text_id"] for r in meta if r["split"] == "pilot")
        pick = set(random.Random(cfg["seed"]).sample(pilot, min(a.n, len(pilot))))
        for r in meta:
            if r["text_id"] in pick:
                r["split"] = "pilot1"
        with open(ROOT / "corpus" / "meta.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(meta[0]))
            w.writeheader()
            w.writerows(meta)
        from lgdb.prepare import prepare
        n = prepare(ROOT / "corpus", work, keys, "pilot1", None, seed=cfg["seed"], **cfg["corpus"])
        print({"texts": sorted(pick), "fragments": n})
    elif a.cmd == "stage":
        q = llm(cfg)
        key = pl.load_key(keys)
        if a.arg == "extract":
            out = pl.extract(q, work)
        elif a.arg == "topics":
            out = pl.topics(cfg, work, key)
        elif a.arg == "name_topics":
            out = pl.name_topics(q, work)
        elif a.arg == "pairs":
            out = pl.sample_pairs(cfg, work)
        elif a.arg == "relate":
            out = pl.relate(q, work)
        elif a.arg == "camps":
            out = pl.camps(cfg, work, key)
        else:
            raise SystemExit(f"unknown stage {a.arg}")
        print({"result": out, "queued": q.queued})
    elif a.cmd == "batches":
        rows = [json.loads(x) for x in open(Q / "queue.jsonl", encoding="utf-8")]
        seen, uniq = set(), []
        for r in rows:
            if r["key"] not in seen:
                seen.add(r["key"])
                uniq.append(r)
        for i in range(0, len(uniq), a.size):
            with open(Q / f"batch_{i // a.size:03d}.jsonl", "w", encoding="utf-8") as f:
                for r in uniq[i:i + a.size]:
                    f.write(json.dumps({"key": r["key"], "prompt": r["prompt"]}) + "\n")
        print({"prompts": len(uniq), "batches": (len(uniq) + a.size - 1) // a.size})
    elif a.cmd == "ingest":
        q = llm(cfg)
        prompts = {}
        for f in Q.glob("batch_*.jsonl"):
            for x in open(f, encoding="utf-8"):
                r = json.loads(x)
                prompts[r["key"]] = r["prompt"]
        ok = bad = 0
        for f in Q.glob("answers_*.jsonl"):
            for x in open(f, encoding="utf-8"):
                if not x.strip():
                    continue
                r = json.loads(x)
                raw = r["answer"] if isinstance(r["answer"], str) else json.dumps(r["answer"])
                if r["key"] in prompts and q.ingest(r["key"], prompts[r["key"]], raw):
                    ok += 1
                else:
                    bad += 1
        done = Q / "done"
        done.mkdir(exist_ok=True)
        for f in list(Q.glob("batch_*.jsonl")) + list(Q.glob("answers_*.jsonl")) + [Q / "queue.jsonl"]:
            if f.exists():
                f.rename(done / f.name)
        print({"ingested": ok, "failed": bad})


if __name__ == "__main__":
    main()
