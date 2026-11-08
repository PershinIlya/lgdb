from lgdb import stance as st


def test_parse_codes_evidence_rule():
    ans = {"codes": ["S001 A t3", "S002 Bc m12", "S003 A", "S004 B t99", "S005 N", "S006 U", "junk"]}
    out = st.parse_codes(ans, {"t3", "m12"}, {f"S00{i}" for i in range(1, 8)})
    assert out["S001"] == {"code": "A", "cond": False, "ev": "t3", "flag": ""}
    assert out["S002"]["code"] == "B" and out["S002"]["cond"]
    assert out["S003"]["code"] == "N" and out["S003"]["flag"] == "no_evidence"
    assert out["S004"]["code"] == "N"
    assert out["S005"]["code"] == "N" and out["S006"]["code"] == "U"
    assert "S007" not in out


def test_kripp():
    same = [("A", "A"), ("B", "B"), ("N", "N")] * 5
    assert abs(st.kripp_nominal(same) - 1) < 1e-9
    assert abs(st.kripp_interval([(1, 1), (-1, -1), (0, 0)] * 5) - 1) < 1e-9
    assert st.kripp_nominal([("A", "B"), ("B", "A")] * 5) < 0


def test_null_dossier_deterministic():
    assert st._null_dossier(1) == st._null_dossier(1)
    assert len(st._null_dossier(1)) == 20
