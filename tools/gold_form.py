"""Blind gold cells for the stance pilot: build the hand-coding form, then score criterion 2.

  python tools/gold_form.py build             # work_stance/gold_key.json + human/stance_gold_form.html
  python tools/gold_form.py score answers.json # reject precision and recall vs the researcher's codes
The form shows only the notes (no title) and the statement; model codes are never in it.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lgdb import stance as st  # noqa: E402

WORK = ROOT / "work_stance"

PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stance gold cells</title>
<style>
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--card:#fff;--line:#e3e0d8;--accent:#2f5d8a}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#17171a;--fg:#ececea;--muted:#a3a29c;--card:#202024;--line:#33333a;--accent:#8fb6e0}}
:root[data-theme="dark"]{--bg:#17171a;--fg:#ececea;--muted:#a3a29c;--card:#202024;--line:#33333a;--accent:#8fb6e0}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:780px;margin:0 auto;padding:16px}
h1{font-size:1.3rem;margin:.5rem 0}
.help{color:var(--muted);font-size:.95rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:18px 0}
.stmt{font-weight:600;margin:.4rem 0 .8rem}
details{margin:.3rem 0 .8rem}
details pre{white-space:pre-wrap;font:14px/1.45 ui-monospace,monospace;background:var(--bg);padding:10px;border-radius:6px;max-height:340px;overflow:auto}
.codes label{display:inline-block;margin:4px 10px 4px 0;cursor:pointer}
input[type=text]{width:6em;padding:4px;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--fg)}
textarea{width:100%;box-sizing:border-box;min-height:2.2em;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--fg)}
.bar{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:8px 14px;font-size:1rem;cursor:pointer}
.n{color:var(--muted);font-size:.9rem}
</style></head><body><main>
<h1>Ручная проверка: 40 ячеек</h1>
<p class="help">В каждой карточке: заметки о том, что советует одна книга (название скрыто), и одно утверждение.
Ответьте, что эта книга сказала бы читателю об утверждении, только по заметкам.<br>
A = советует; B = советует против (критикует это или советует другой путь в той же ситуации); N = не затрагивает; U = по заметкам не понять.
Отметка «с оговорками», если позиция только для названных людей, времён или ситуаций. Для A и B укажите номер заметки (например t3 или m12).
Ответы сохраняются в этом браузере. В конце нажмите «Скачать ответы» и пришлите файл.</p>
<div class="bar"><button id="dl">Скачать ответы</button><span id="cnt" class="n"></span></div>
__CARDS__
</main>
<script>
const KEY="lgdb_stance_gold_v1";
let st={};try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){};count()}
function count(){const n=Object.values(st).filter(v=>v&&v.code).length;document.getElementById("cnt").textContent="Заполнено: "+n+" из __N__"}
document.querySelectorAll(".card").forEach(c=>{const id=c.dataset.cell;const v=st[id]||{};
 c.querySelectorAll("input[type=radio]").forEach(r=>{if(r.value===v.code)r.checked=true;r.onchange=()=>{st[id]={...st[id],code:r.value};save()}});
 const f=c.querySelector("input[type=checkbox]");f.checked=!!v.cond;f.onchange=()=>{st[id]={...st[id],cond:f.checked};save()};
 const e=c.querySelector("input[type=text]");e.value=v.ev||"";e.oninput=()=>{st[id]={...st[id],ev:e.value.trim()};save()};
 const t=c.querySelector("textarea");t.value=v.note||"";t.oninput=()=>{st[id]={...st[id],note:t.value};save()};});
document.getElementById("dl").onclick=()=>{const b=new Blob([JSON.stringify(st,null,1)],{type:"application/json"});
 const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="stance_gold_answers.json";a.click()};
count();
</script></body></html>"""

CARD = """<section class="card" data-cell="{cell}">
<div class="n">{cell}</div>
<div class="stmt">{stmt}</div>
<details open><summary>Заметки о книге</summary><pre>{dossier}</pre></details>
<div class="codes">
<label><input type="radio" name="{cell}" value="A"> A советует</label>
<label><input type="radio" name="{cell}" value="B"> B против</label>
<label><input type="radio" name="{cell}" value="N"> N не затрагивает</label>
<label><input type="radio" name="{cell}" value="U"> U не понять</label>
</div>
<label><input type="checkbox"> с оговорками</label> &nbsp; заметка: <input type="text" placeholder="t3">
<div class="n" style="margin-top:6px">Комментарий (по желанию)</div><textarea></textarea>
</section>"""


def build():
    cells = st.gold_sample(WORK)
    stmts = {s["sid"]: s for s in st.final_list(WORK)}
    rows = {r["row"]: r for r in st.rows(WORK)}
    (WORK / "gold_key.json").write_text(json.dumps(cells, indent=1))
    cards = [CARD.format(cell=c["cell"], stmt=html.escape(stmts[c["sid"]]["statement"]),
                         dossier=html.escape(st._dossier_text(rows[c["row"]]["entries"]))) for c in cells]
    out = ROOT / "human" / "stance_gold_form.html"
    out.write_text(PAGE.replace("__CARDS__", "\n".join(cards)).replace("__N__", str(len(cells))))
    print({"cells": len(cells), "form": str(out)})


def score(path: str):
    ans = json.load(open(path))
    key = {c["cell"]: c for c in json.loads((WORK / "gold_key.json").read_text())}
    fin = st.consensus(WORK)
    tp = fp = fn = 0
    for cell, c in key.items():
        h = (ans.get(cell) or {}).get("code")
        if not h:
            continue
        m = fin[(c["row"], c["sid"])]["code"]
        tp += m == "B" and h == "B"
        fp += m == "B" and h != "B"
        fn += m != "B" and h == "B"
    res = {"answered": sum(bool((ans.get(c) or {}).get("code")) for c in key),
           "reject_precision": round(tp / max(tp + fp, 1), 3), "reject_recall": round(tp / max(tp + fn, 1), 3)}
    res["pass"] = res["reject_precision"] >= 0.7 and res["reject_recall"] >= 0.6
    print(json.dumps(res))
    (WORK / "gold_score.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"build": build, "score": lambda: score(sys.argv[2])}[sys.argv[1]]()
