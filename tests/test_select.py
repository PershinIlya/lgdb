import csv

from lgdb import select as sel

COLS = ["period", "region", "cell_format", "rank", "title", "author", "date", "format_reported",
        "translation", "public_domain", "influence_verified"]
GRID = [("p", "r", "collection"), ("p", "r", "thesis")]


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)


def test_rules(tmp_path):
    rows = [("p", "r", "collection", i, f"C{i}", "a", "x", "collection", f"T {1900 + i}", "", "") for i in range(1, 9)]
    rows += [("p", "r", "collection", 9, "Mixed", "a", "x", "mixed", "T 1900", "", ""),
             ("p", "r", "thesis", 1, "C2", "a", "x", "thesis", "T 1900", "", ""),     # dup, ranks higher here
             ("p", "r", "thesis", 2, "Modern", "a", "1990", "thesis", "original English", "", ""),
             ("p", "r", "thesis", 3, "Old", "a", "1850", "thesis", "original English", "", ""),
             ("p", "r", "thesis", 4, "Alt", "a", "x", "thesis", "T 1960", "", "")]
    write(tmp_path / "c.csv", rows)
    with open(tmp_path / "av.csv", "w", newline="") as f:
        csv.writer(f).writerows([["title", "available", "translation", "reason"], ["Alt", "yes", "Other 1912", ""],
                                 ["C3", "no", "", "partial"]])
    out = {r["title"] + "|" + r["cell_format"]: r for r in
           sel.select(tmp_path / "c.csv", None, 1, GRID, tmp_path / "av.csv")}
    assert out["Mixed|collection"]["status"] == "format mismatch"
    assert out["C2|collection"]["status"].startswith("kept in")      # moved to the thesis cell
    assert out["C2|thesis"]["status"] == "selected"
    assert out["Modern|thesis"]["status"] == "no usable full text yet"
    assert out["Alt|thesis"]["status"] == "selected"                  # override found an old translation
    assert out["C3|collection"]["status"] == "no usable full text yet"  # override: nothing found
    picked = [r for r in out.values() if r["cell_format"] == "collection" and r["status"] == "selected"]
    assert [r["title"] for r in sorted(picked, key=lambda r: r["rank"])] == ["C1", "C4", "C5", "C6", "C7", "C8"]
    assert sum(r["split"] == "pilot" for r in picked) == 2
    thesis = [r for r in out.values() if r["cell_format"] == "thesis" and r["status"] == "selected"]
    assert len(thesis) == 3 and sum(r["split"] == "pilot" for r in thesis) == 1
    again = sel.select(tmp_path / "c.csv", None, 1, GRID, tmp_path / "av.csv")
    assert [r["split"] for r in again] == [r["split"] for r in out.values()]  # seeded, reproducible
