"""Build the 3D map page from work_axes/analysis.json.

  python tools/axes_map.py     # writes human/stance_map.html
Books: memory rows of the mechanical sample plus the extra well-known titles (marked).
Each point sits at the consensus of 2 raters; its cloud shows how much the raters disagreed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lgdb import axes as ax  # noqa: E402
from lgdb import stance_full as sf  # noqa: E402
from lgdb.pipeline import load_jsonl  # noqa: E402

W = ROOT / "work_axes"


def data() -> dict:
    an = json.loads((W / "analysis.json").read_text())
    disp = json.loads((W / "display.json").read_text())
    ru = json.loads((W / "axes_ru.json").read_text())
    meta = {b["id"]: b for b in sf.books()}
    meta.update({b["id"]: b for b in ax.extra_books(W)})
    axes = []
    for a in an["axes"]:
        axes.append({"k": a["kid"], "q": ru[a["kid"]]["q"], "x": ru[a["kid"]]["x"], "y": ru[a["kid"]]["y"],
                     "placed": a["placed"], "keep": a["keep"], "r": a["rater_r"], "sx": a["side_x"], "sy": a["side_y"]})
    books = []
    for row, b in an["books"].items():
        if b["source"] != "memory":
            continue
        m = meta[b["book"]]
        pos = {k: [round(c["v"], 2), None if c["spread"] is None else c["spread"], int(c["weak"])] for k, c in b["pos"].items()}
        books.append({"id": b["book"], "ru": disp[b["book"]]["ru"], "t": m["title"], "a": m["author"],
                      "y": str(m.get("year", "")).split(",")[0], "p": m["period"], "r": m["region"],
                      "tier": disp[b["book"]]["tier"], "extra": int(b["group"] == "extra"), "pos": pos})
    # default view: the 3 kept axes on which the most well-known books can be drawn (placed on 2 of 3)
    import itertools
    kept = [a["k"] for a in axes if a["keep"]]
    def shown(t, tier1):
        return sum(1 for b in books if (b["tier"] == 1 or not tier1) and sum(k in b["pos"] for k in t) >= 2)
    default = max(itertools.combinations(kept, 3), key=lambda t: (shown(t, True), shown(t, False)))
    return {"axes": axes, "books": books, "default": list(default)}


PAGE = (ROOT / "tools" / "axes_map_template.html")


def main():
    d = data()
    html = PAGE.read_text().replace("__DATA__", json.dumps(d, ensure_ascii=False))
    (ROOT / "human" / "stance_map.html").write_text(html)
    print({"books": len(d["books"]), "axes": len(d["axes"]), "kept": sum(a["keep"] for a in d["axes"])})


if __name__ == "__main__":
    main()
