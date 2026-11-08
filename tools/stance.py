"""Stance-matrix pilot driver. No API: Explore agents answer batches of queued prompts.

  python tools/stance.py stage recall            # queue missing prompts (or finish the stage)
  python tools/stance.py batches opus --size 12  # write batch files for one model's queue
  python tools/stance.py harvest <agent_id> <batch_file> --label recall
  python tools/stance.py usage                   # token totals by label
Explore agents get no project instructions (so they stay blind) but cannot write files:
they return answers in their final report, and harvest reads it from the agent transcript.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lgdb import stance as st  # noqa: E402
from lgdb.llm import QueueLLM  # noqa: E402

WORK = ROOT / "work_stance"
Q = WORK / "queue"
BATCH_DIR = Path(os.environ.get("LGDB_BATCH_DIR", str(Q)))
MODELS = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5"}
STAGE_MODEL = {"recall": "opus", "condense": "sonnet", "statements": "opus", "reverse": "opus", "tie": "haiku"}
RATERS = {"r1": "sonnet", "r2": "opus"}
TASK_RE = re.compile(r"<<<KEY ([0-9a-f]{64})>>>\s*(.*?)\s*(?:<<<END>*|(?=<<<KEY)|\Z)", re.S)


def q(short: str) -> QueueLLM:
    return QueueLLM(MODELS[short], ROOT / "cache", Q / f"queue_{short}.jsonl", temperature=0.0)


def transcript(agent_id: str) -> Path:
    hits = glob.glob(os.path.expanduser(f"~/.claude/projects/*/*/subagents/agent-{agent_id}.jsonl"))
    if not hits:
        raise SystemExit(f"no transcript for {agent_id}")
    return Path(hits[0])


def read_agent(agent_id: str) -> tuple[str, dict]:
    """Final report text and token use (final context plus all output) from a transcript."""
    report, last, out_tok, model = "", None, 0, ""
    seen = set()
    for line in open(transcript(agent_id), encoding="utf-8"):
        r = json.loads(line)
        m = r.get("message") or {}
        if r.get("type") != "assistant":
            continue
        model = m.get("model", model)
        u = m.get("usage") or {}
        if m.get("id") not in seen:
            seen.add(m.get("id"))
            out_tok += u.get("output_tokens", 0)
            last = u
        for b in m.get("content") or []:
            if b.get("type") == "tool_use" and b.get("name") == "SubagentHandback":
                report = b["input"].get("message", "")
            elif b.get("type") == "text" and b.get("text"):
                report = report or ""
                if "<<<KEY" in b["text"]:
                    report = b["text"]
    ctx = sum((last or {}).get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    return report, {"model": model, "context_tokens": ctx, "output_tokens": out_tok, "total": ctx + out_tok}


def log_usage(label: str, agent_id: str, use: dict, note: str = "") -> None:
    WORK.mkdir(exist_ok=True)
    with open(WORK / "usage.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"label": label, "agent": agent_id, **use, "note": note}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--size", type=int, default=5)
    ap.add_argument("--label", default="")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    WORK.mkdir(exist_ok=True)
    Q.mkdir(exist_ok=True)

    if a.cmd == "stage":
        name = a.args[0]
        if name == "recall":
            out = st.recall(q("opus"), WORK)
        elif name == "condense":
            out = st.condense(q("sonnet"), WORK)
        elif name == "pool":
            out = st.build_pool(WORK)
        elif name == "statements":
            out = st.statements(q("opus"), WORK)
        elif name == "reverse":
            out = st.reverse(q("opus"), WORK)
        elif name == "code":
            out = st.code({r: q(m) for r, m in RATERS.items()}, WORK)
        elif name == "tie":
            out = st.tiebreak(q("haiku"), WORK)
        elif name == "criteria":
            out = st.criteria(WORK)
            (WORK / "criteria.json").write_text(json.dumps(out, indent=1))
        else:
            raise SystemExit(name)
        print(json.dumps(out, indent=1, default=str))
    elif a.cmd == "batches":
        short = a.args[0]
        qf = Q / f"queue_{short}.jsonl"
        rows, seen = [], set()
        for x in open(qf, encoding="utf-8"):
            r = json.loads(x)
            if r["key"] not in seen:
                seen.add(r["key"])
                rows.append(r)
        BATCH_DIR.mkdir(parents=True, exist_ok=True)
        label = a.label or short
        names = []
        for i in range(0, len(rows), a.size):
            p = BATCH_DIR / f"batch_{label}_{i // a.size:02d}.txt"
            with open(p, "w", encoding="utf-8") as f:
                for r in rows[i:i + a.size]:
                    f.write(f"##### TASK {r['key']}\n{r['prompt']}\n##### END TASK\n\n")
            names.append(str(p))
        qf.rename(Q / f"queued_{label}_{short}.jsonl")
        print(json.dumps({"prompts": len(rows), "batches": names}, indent=1))
    elif a.cmd == "harvest":
        agent_id, batch = a.args[0], Path(a.args[1])
        report, use = read_agent(agent_id)
        tasks = dict(re.findall(r"##### TASK ([0-9a-f]{64})\n(.*?)\n##### END TASK", batch.read_text(), re.S))
        short = next(s for s, full in MODELS.items() if full.split("-")[1] in use["model"])
        llm = q(short)
        ok, bad = 0, []
        for key, ans in TASK_RE.findall(report):
            if key in tasks and llm.ingest(key, tasks[key], ans):
                ok += 1
            else:
                bad.append(key[:8])
        missing = [k[:8] for k in tasks if k not in dict(TASK_RE.findall(report))]
        log_usage(a.label or batch.stem, agent_id, use, a.note)
        print(json.dumps({"model": use["model"], "ingested": ok, "bad": bad, "missing": missing,
                          "tokens": use["total"]}))
    elif a.cmd == "use":  # log usage of an agent without answers (leakage audit, probes)
        report, use = read_agent(a.args[0])
        log_usage(a.label, a.args[0], use, a.note)
        out = WORK / "audit_raw"
        out.mkdir(exist_ok=True)
        (out / f"{a.label}_{a.args[0][:8]}.txt").write_text(report)
        print(json.dumps(use))
    elif a.cmd == "usage":
        tot, by = 0, {}
        for x in open(WORK / "usage.jsonl", encoding="utf-8"):
            r = json.loads(x)
            lab = r["label"].split("_")[0]
            by[lab] = by.get(lab, 0) + r["total"]
            tot += r["total"]
        print(json.dumps({"by_stage": by, "total": tot}, indent=1))
    else:
        raise SystemExit(a.cmd)


if __name__ == "__main__":
    main()
