"""Write corpus/sources.csv from cfg.json (ref/start/end/note/filter and new-row fields) and drops/<id>.txt."""
import csv, json
from pathlib import Path
HERE = Path(__file__).parent
CSV = HERE.parent.parent / 'corpus' / 'sources.csv'
cfg = json.loads((HERE / 'cfg.json').read_text(encoding='utf-8'))
rows = list(csv.DictReader(open(CSV, encoding='utf-8')))
fields = list(rows[0].keys())
if 'filter' not in fields:
    fields.append('filter')
ids = [r['text_id'] for r in rows]
for tid, c in cfg.items():
    if tid not in ids and 'title' in c:
        rows.append({'text_id': tid})
for r in rows:
    for f in fields:
        r.setdefault(f, '')
        if r[f] is None:
            r[f] = ''
    c = cfg.get(r['text_id'])
    if not c:
        continue
    for k in ('title', 'kind', 'ref', 'translator', 'year', 'start', 'end', 'note', 'filter'):
        if k in c:
            r[k] = c[k]
    dp = HERE / 'drops' / f"{r['text_id']}.txt"
    if dp.exists():
        pats = [l.strip() for l in dp.read_text(encoding='utf-8').split('\n') if l.strip()]
        r['drop'] = ' || '.join(pats)
with open(CSV, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print('wrote', len(rows), fields)
