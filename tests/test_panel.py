import csv

from lgdb import panel
from lgdb.pipeline import save_jsonl


def test_audit_catches_a_wrong_panel(tmp_path):
    work, human = tmp_path / "work", tmp_path / "human"
    work.mkdir()
    rows = [{"item_id": f"P{i}", "rec_1": f"a{i}", "rec_2": f"b{i}",
             "votes": ["compatible"] * 3 if i < 90 else ["compatible", "incompatible", "compatible"]}
            for i in range(100)]
    save_jsonl(work / "panel_pairs.jsonl", rows)
    counts = panel.review_sheets(work, human, audit_share=0.5, seed=0)["pairs"]
    assert counts["to_human"] > 10  # 10 disputed plus about half of the 90 agreed
    sheet = list(csv.DictReader(open(human / "review_pairs.csv")))
    assert all("votes" not in r for r in sheet)  # the reviewer never sees the panel's answers
    col = [c for c in sheet[0] if c.startswith("answer")][0]
    for r in sheet:
        r[col] = "incompatible"  # the reviewer disagrees with the unanimous panel every time
    with open(human / "review_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    g = panel.gold(work, human, "pairs")
    assert g["pending"] == 0 and g["audit_n"] > 10 and g["audit_agreement"] == 0.0
    disputed = [i for i in g["labels"] if int(i[1:]) >= 90]
    assert all(g["labels"][i]["label"] == "incompatible" for i in disputed)  # human label wins on disputes
    again = panel.review_sheets(work, human, audit_share=0.5, seed=99)["pairs"]
    assert again == counts  # the audit is drawn once and not redrawn
