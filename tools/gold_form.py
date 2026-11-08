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

PAGE = """<title>Stance gold cells</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono&display=swap">
<style>
/* Layout: one reading column of cards, a sticky status bar on top */
:root{--bg:#f7f8f6;--fg:#1c2120;--muted:#5f6864;--card:#ffffff;--line:#dde2de;--accent:#2d6a5a;--accent-fg:#ffffff;--pick:#e3efe9;
--body:"Source Sans 3",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141816;--fg:#e6ebe8;--muted:#9aa49f;--card:#1c211f;--line:#2e3532;--accent:#7cc0aa;--accent-fg:#0e1311;--pick:#21332c;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#141816;--fg:#e6ebe8;--muted:#9aa49f;--card:#1c211f;--line:#2e3532;--accent:#7cc0aa;--accent-fg:#0e1311;--pick:#21332c;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:16px/1.5 var(--body);padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
body::before{content:"";position:fixed;inset:0 0 auto;height:env(safe-area-inset-top,0px);z-index:10;background:var(--bg)}
main{max-width:760px;margin:0 auto;padding-block:12px 40px}
h1{font-size:1.35rem;margin:.4rem 0;text-wrap:balance}
.help{color:var(--muted);font-size:.95rem;max-width:65ch}
.bar{position:sticky;top:env(safe-area-inset-top,0px);background:var(--bg);padding-block:8px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:5}
.n{color:var(--muted);font-size:.9rem;font-variant-numeric:tabular-nums}
.cards{display:grid;gap:18px;margin-top:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;display:grid;gap:10px;min-width:0}
.stmt{font-weight:600;font-size:1.05rem}
details pre{white-space:pre-wrap;font:13.5px/1.45 var(--mono);background:var(--bg);padding:10px;border-radius:6px;max-height:340px;overflow:auto;margin:6px 0 0}
.codes{display:flex;flex-wrap:wrap;gap:6px}
.codes label{border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
.codes label:has(input:checked){background:var(--pick);border-color:var(--accent)}
.row{display:flex;flex-wrap:wrap;gap:12px;align-items:center}
input[type=text]{width:6em;padding:4px 6px;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--fg);font:inherit}
textarea{width:100%;box-sizing:border-box;min-height:2.4em;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--fg);font:inherit}
button{background:var(--accent);color:var(--accent-fg);border:0;border-radius:6px;padding:7px 14px;font:inherit;cursor:pointer}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
</style>
<main>
<h1>Ручная проверка: 40 ячеек</h1>
<p class="help">В каждой карточке заметки о том, что советует одна книга (название скрыто), и одно утверждение.
Ответьте, что эта книга сказала бы читателю об утверждении, только по заметкам.<br>
A = советует. B = советует против (критикует это или советует другой путь в той же ситуации). N = не затрагивает. U = по заметкам не понять.<br>
«С оговорками»: позиция только для названных людей, времён или ситуаций. Для A и B укажите номер заметки, например t3 или m12.</p>
<div class="bar"><span id="cnt" class="n"></span><span id="sync" class="n"></span><button id="dl" hidden>Скачать ответы</button></div>
<div class="cards">
__CARDS__
</div>
</main>
<script>
const KEY="lgdb_stance_gold_v1";let st={};let db=null;const timers={};
try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
function count(){const n=Object.values(st).filter(v=>v&&v.code).length;document.getElementById("cnt").textContent="Заполнено: "+n+" из __N__"}
function paint(){document.querySelectorAll(".card").forEach(c=>{const id=c.dataset.cell;const v=st[id]||{};
 c.querySelectorAll("input[type=radio]").forEach(r=>{r.checked=r.value===v.code});
 c.querySelector("input[type=checkbox]").checked=!!v.cond;c.querySelector("input[type=text]").value=v.ev||"";c.querySelector("textarea").value=v.note||""});count()}
function sync(t){document.getElementById("sync").textContent=t}
function save(id){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){};count();
 if(!db)return;clearTimeout(timers[id]);timers[id]=setTimeout(()=>{sync("Сохраняю…");
 db.doc("gold/"+id).set(st[id]).then(()=>sync("Сохранено на сервере")).catch(e=>sync("Не сохранилось на сервере: "+(e&&e.code||"ошибка")+". Ответ остался в этом браузере."))},600)}
function upd(id,patch){st[id]={...(st[id]||{}),...patch};save(id)}
document.querySelectorAll(".card").forEach(c=>{const id=c.dataset.cell;
 c.querySelectorAll("input[type=radio]").forEach(r=>r.onchange=()=>upd(id,{code:r.value}));
 const f=c.querySelector("input[type=checkbox]");f.onchange=()=>upd(id,{cond:f.checked});
 const e=c.querySelector("input[type=text]");e.oninput=()=>upd(id,{ev:e.value.trim()});
 const t=c.querySelector("textarea");t.oninput=()=>upd(id,{note:t.value})});
paint();
(async()=>{let d=null;try{d=window.claude&&window.claude.use?await window.claude.use("db"):null}catch(e){}
 if(!d){sync("Ответы хранятся в этом браузере.");const b=document.getElementById("dl");b.hidden=false;
  b.onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(st,null,1)],{type:"application/json"}));a.download="stance_gold_answers.json";a.click()};return}
 db=d;sync("Загружаю…");const timer=setTimeout(()=>sync("Сервер пока не отвечает. Ответы сохраняются в браузере."),10000);
 try{const snap=await db.collection("gold").get();clearTimeout(timer);snap.docs.forEach(x=>{if(x.exists)st[x.id]={...(st[x.id]||{}),...x.data()}});
  paint();sync("Ответы сохраняются на сервере");
  for(const id in st){if(!snap.docs.some(x=>x.id===id)&&st[id]&&st[id].code)save(id)}}
 catch(e){clearTimeout(timer);sync("Сервер недоступен: ответы сохраняются в браузере.")}})();
</script>"""

CARD = """<section class="card" data-cell="{cell}">
<div class="n">{cell}</div>
<div class="stmt">{stmt}</div>
<details open><summary>Заметки о книге</summary><pre>{dossier}</pre></details>
<div class="codes">
<label><input type="radio" id="{cell}-A" name="{cell}" value="A"> A советует</label>
<label><input type="radio" id="{cell}-B" name="{cell}" value="B"> B против</label>
<label><input type="radio" id="{cell}-N" name="{cell}" value="N"> N не затрагивает</label>
<label><input type="radio" id="{cell}-U" name="{cell}" value="U"> U не понять</label>
</div>
<div class="row"><label><input type="checkbox" id="{cell}-c"> с оговорками</label><label>заметка: <input type="text" id="{cell}-ev" placeholder="t3"></label></div>
<label class="n" for="{cell}-note">Комментарий (по желанию)</label><textarea id="{cell}-note"></textarea>
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
    """answers: the downloaded JSON {cell: {...}}, or an ArtifactData list dump of collection gold."""
    raw = json.load(open(path))
    if isinstance(raw, dict) and "docs" in raw:
        raw = {d["id"]: d.get("data", d) for d in raw["docs"]}
    ans = raw
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
