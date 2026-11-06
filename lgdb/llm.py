"""Thin, cached, reproducible wrapper around the Anthropic Messages API.

Every call is keyed by (model, params, rendered prompt) and stored on disk, so
re-running a stage never pays twice and the exact prompt each annotator saw is
kept next to its raw answer.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SHUFFLE_RE = re.compile(r"\{\{SHUFFLE\}\}\n(.*?)\n\{\{/SHUFFLE\}\}", re.S)


def render(template: str, values: dict, shuffle_seed: str | None = None) -> str:
    """Fill {PLACEHOLDERS} and shuffle lines inside {{SHUFFLE}} blocks.

    The shuffle is seeded per call (for example by pair id), so the order of
    answer options differs between calls but is reproducible.
    """
    def _shuffle(m: re.Match) -> str:
        lines = m.group(1).split("\n")
        rng = random.Random(shuffle_seed)
        rng.shuffle(lines)
        return "\n".join(lines)

    text = SHUFFLE_RE.sub(_shuffle, template)
    for k, v in values.items():
        text = text.replace("{" + k + "}", str(v))
    return text


def parse_json(text: str):
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start > 0:
        text = text[start:]
    return json.loads(text)


def final_text(content) -> str:
    """Text written after the last server tool result (the answer, not the narration)."""
    blocks = list(content)
    last_tool = max((i for i, b in enumerate(blocks)
                     if getattr(b, "type", "").endswith("tool_result")), default=-1)
    return "".join(b.text for b in blocks[last_tool + 1:] if getattr(b, "type", "") == "text")


class LLM:
    def __init__(self, model: str, cache_dir: str | Path, max_tokens: int = 4096,
                 temperature: float | None = 0.0, workers: int = 8, client=None,
                 tools: list | None = None, max_pauses: int = 5):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.workers = workers
        self.tools = tools
        self.max_pauses = max_pauses
        self.cache_dir = Path(cache_dir) / model
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = client
        self._lock = threading.Lock()

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def _key(self, prompt: str) -> str:
        blob = json.dumps([self.model, self.max_tokens, self.temperature, self.tools, prompt])
        return hashlib.sha256(blob.encode()).hexdigest()

    def _create(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        kwargs = dict(model=self.model, max_tokens=self.max_tokens)
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.tools:
            kwargs["tools"] = self.tools
        resp = self.client.messages.create(messages=messages, **kwargs)
        pauses = 0
        # Long server-side searches can pause the turn; send it back unchanged to continue.
        while getattr(resp, "stop_reason", None) == "pause_turn" and pauses < self.max_pauses:
            messages = messages + [{"role": "assistant", "content": resp.content}]
            resp = self.client.messages.create(messages=messages, **kwargs)
            pauses += 1
        return resp

    def call(self, prompt: str, retries: int = 2):
        path = self.cache_dir / f"{self._key(prompt)}.json"
        if path.exists():
            return json.loads(path.read_text())["parsed"]
        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = self._create(prompt)
                raw = final_text(resp.content)
                parsed = parse_json(raw)
                path.write_text(json.dumps({"prompt": prompt, "raw": raw, "parsed": parsed,
                                            "model": self.model, "time": time.time()}))
                return parsed
            except Exception as e:  # network errors, refusals, malformed JSON
                last_err = e
                time.sleep(2 ** attempt)
        with self._lock:
            (self.cache_dir / "failures.log").open("a").write(
                json.dumps({"key": self._key(prompt), "error": repr(last_err)}) + "\n")
        return None

    def map(self, prompts: list[str]):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self.call, prompts))


class FakeLLM(LLM):
    """Deterministic stand-in for tests: answers come from a Python function."""

    def __init__(self, fn, cache_dir, model="fake"):
        super().__init__(model, cache_dir, workers=1, client=object())
        self.fn = fn

    def call(self, prompt: str, retries: int = 0):
        return self.fn(prompt)


class QueueLLM(LLM):
    """No API: answers come from the disk cache; missing prompts are queued for agents.

    Run a pipeline stage once to fill work/queue/<model>.jsonl, let agents answer the queue
    (see tools/agents.py), ingest the answers into the cache, then run the stage again.
    """

    def __init__(self, model: str, cache_dir, queue_file, **kw):
        super().__init__(model, cache_dir, workers=1, client=object(), **kw)
        self.queue_file = Path(queue_file)
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        self.queued = 0

    def call(self, prompt: str, retries: int = 0):
        path = self.cache_dir / f"{self._key(prompt)}.json"
        if path.exists():
            return json.loads(path.read_text())["parsed"]
        with self._lock:
            with self.queue_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": self._key(prompt), "model": self.model, "prompt": prompt}) + "\n")
            self.queued += 1
        return None

    def ingest(self, key: str, prompt: str, raw: str) -> bool:
        try:
            parsed = parse_json(raw)
        except Exception:
            return False
        (self.cache_dir / f"{key}.json").write_text(json.dumps(
            {"prompt": prompt, "raw": raw, "parsed": parsed, "model": self.model + " (agent batch)",
             "time": time.time()}))
        return True
