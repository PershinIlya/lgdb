"""Dimensions of disagreement from the full stance matrix (plan v2, work_full/plan.md).

Cells: A=+1, B=-1, N/U=0 (missing treated as 0, so coverage is checked as a confound).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import stance as st
from . import stance_full as sf
from .pipeline import load_jsonl

SEED = 20261122
MIN_ADDR, MIN_MINORITY, MIN_BOOK_ADDR = 5, 0.10, 10
CONGRUENCE_MIN, ENDS_MIN, CONFOUND_R2_MAX, RATER_R_MIN = 0.85, 5, 0.3, 0.8
SIGN = {"A": 1.0, "B": -1.0}


def _matrix(codes: dict, rows: list[str], sids: list[str]) -> np.ndarray:
    return np.array([[SIGN.get(codes.get((r, s), "N"), 0.0) for s in sids] for r in rows])


def keep_statements(m: np.ndarray) -> np.ndarray:
    a, b = (m > 0).sum(0), (m < 0).sum(0)
    tot = a + b
    return (tot >= MIN_ADDR) & (np.minimum(a, b) >= MIN_MINORITY * np.maximum(tot, 1))


def svd(m: np.ndarray, k: int):
    c = m - m.mean(0)
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    return u[:, :k] * s[:k], vt[:k].T, s


def parallel_k(m: np.ndarray, n=200, seed=SEED) -> tuple[int, list, list]:
    rng = np.random.default_rng(seed)
    _, _, s = svd(m, 1)
    null = []
    for _ in range(n):
        p = np.column_stack([rng.permutation(col) for col in m.T])
        null.append(svd(p, 1)[2][: len(s)])
    q95 = np.percentile(np.array(null), 95, axis=0)
    k = 0
    while k < len(s) and s[k] > q95[k]:
        k += 1
    return k, s[:10].round(3).tolist(), q95[:10].round(3).tolist()


def congruence(a, b):
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def bootstrap_congruence(m, load, k, n=200, seed=SEED):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, k))
    for i in range(n):
        idx = rng.integers(0, len(m), len(m))
        _, lb, _ = svd(m[idx], max(k + 2, 3))
        for d in range(k):
            out[i, d] = max(congruence(load[:, d], lb[:, j]) for j in range(lb.shape[1]))
    return out.mean(0)


def r2(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def eta2(scores, labels):
    labels = np.asarray(labels)
    g = [scores[labels == v] for v in np.unique(labels)]
    tot = ((scores - scores.mean()) ** 2).sum()
    return float(sum(len(x) * (x.mean() - scores.mean()) ** 2 for x in g) / tot) if tot else 0.0


def eta2_perm(scores, labels, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    obs = eta2(scores, labels)
    null = [eta2(scores, rng.permutation(labels)) for _ in range(n)]
    return round(obs, 3), round(float((np.sum(np.array(null) >= obs) + 1) / (n + 1)), 4)


def extremity(z):
    """Share of books beyond 1 SD; bimodality coefficient (skew^2+1)/kurtosis (0.555 = uniform)."""
    z = (z - z.mean()) / (z.std() + 1e-12)
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())
    return float((abs(z) > 1).mean()), (skew ** 2 + 1) / kurt


def _pipeline(m, k):
    keep = keep_statements(m)
    mk = m[:, keep]
    ok = (mk != 0).sum(1) >= MIN_BOOK_ADDR
    sc, _, _ = svd(mk[ok], k)
    return sc


def null_extremity(m, k, n=200, seed=SEED):
    """Permute each statement's nonzero codes among the books that address it; same selection and SVD."""
    rng = np.random.default_rng(seed)
    tails, bcs = [], []
    for _ in range(n):
        p = m.copy()
        for j in range(m.shape[1]):
            nz = np.flatnonzero(m[:, j])
            p[nz, j] = rng.permutation(m[nz, j])
        sc = _pipeline(p, k)
        ex = [extremity(sc[:, d]) for d in range(min(k, sc.shape[1]))]
        tails.append([e[0] for e in ex])
        bcs.append([e[1] for e in ex])
    return np.array(tails), np.array(bcs)


def analyze(work: Path) -> dict:
    key = {r["row"]: r for r in json.loads((work / "rows_key.json").read_text())}
    stmts = {s["sid"]: s for s in sf.final_list(work)}
    fin = st.consensus(work)
    codes = {k: v["code"] for k, v in fin.items()}
    raw = defaultdict(dict)
    for c in load_jsonl(work / "codes.jsonl"):
        raw[c["rater"]][(c["row"], c["sid"])] = c["code"]
    mem_rows = sorted(r for r in key if key[r]["group"] == "real" and key[r]["source"] == "memory")
    sids = sorted(stmts)
    m_all = _matrix(codes, mem_rows, sids)
    keep = keep_statements(m_all)
    ksids = [s for s, k in zip(sids, keep) if k]
    m = m_all[:, keep]
    okb = (m != 0).sum(1) >= MIN_BOOK_ADDR
    rows = [r for r, o in zip(mem_rows, okb) if o]
    m = m[okb]
    res = {"books_memory": len(mem_rows), "statements": len(sids), "kept_statements": len(ksids),
           "dropped_near_unanimous_or_rare": len(sids) - len(ksids), "books_placed": len(rows),
           "kept_by_kind": dict(Counter(stmts[s]["kind"] for s in ksids))}
    # bipolarity of statements over all books
    a, b = (m_all > 0).sum(0), (m_all < 0).sum(0)
    addressed = (a + b) >= MIN_ADDR
    res["statements_two_sided_share"] = round(float(((np.minimum(a, b) >= 0.2 * (a + b)) & addressed).sum()
                                                    / max(addressed.sum(), 1)), 3)
    res["neg_share_addressed_cells"] = round(float(b.sum() / max((a + b).sum(), 1)), 3)
    k, sv, q95 = parallel_k(m)
    res["parallel_analysis"] = {"k": k, "singular_values": sv, "null_q95": q95}
    kk = max(k, 1)
    scores, load, s = svd(m, kk)
    cong = bootstrap_congruence(m, load, kk)
    cov = (m != 0).sum(1)
    nb = (m < 0).sum(1)
    basis = json.loads((work / "dossiers" / "memory.json").read_text())
    sec = [basis[key[r]["book"]].get("knowledge_basis") == "secondary" for r in rows]
    # rater-only scores on the same loadings
    m1 = _matrix(raw["r1"], rows, ksids)
    m2 = _matrix(raw["r2"], rows, ksids)
    p1, p2 = (m1 - m.mean(0)) @ load, (m2 - m.mean(0)) @ load
    # text rows of pilot classics projected on the same loadings
    txt = {key[r]["book"]: r for r in key if key[r]["source"] == "text"}
    memb = {key[r]["book"]: i for i, r in enumerate(rows)}
    pairs = [(memb[bk], tr) for bk, tr in txt.items() if bk in memb]
    if pairs:
        mt = _matrix(codes, [tr for _, tr in pairs], ksids)
        pt = (mt - m.mean(0)) @ load
    tails_null, bc_null = null_extremity(m_all, kk)
    dims = []
    for d in range(kk):
        z = (scores[:, d] - scores[:, d].mean()) / scores[:, d].std()
        hi, lo = z > 1, z < -1
        per = lambda f: np.array([key[r][f] for r in rows])  # noqa: E731
        tail, bc = extremity(scores[:, d])
        ld = load[:, d]
        order = np.argsort(ld)
        dim = {
            "dim": d + 1, "var_share": round(float(s[d] ** 2 / (s ** 2).sum()), 3),
            "bootstrap_congruence": round(float(cong[d]), 3),
            "books_high": int(hi.sum()), "books_low": int(lo.sum()),
            "periods_high": sorted(set(per("period")[hi])), "periods_low": sorted(set(per("period")[lo])),
            "regions_high": sorted(set(per("region")[hi])), "regions_low": sorted(set(per("region")[lo])),
            "r2_coverage": round(r2(scores[:, d], cov), 3), "r2_neg_count": round(r2(scores[:, d], nb), 3),
            "r2_secondary_knowledge": round(r2(scores[:, d], sec), 3),
            "rater_r": round(float(np.corrcoef(p1[:, d], p2[:, d])[0, 1]), 3),
            "text_vs_memory_r": (round(float(np.corrcoef([scores[i, d] for i, _ in pairs], pt[:, d])[0, 1]), 3)
                                 if pairs and len(pairs) > 2 else None),
            "tail_share": round(tail, 3), "tail_share_null_q95": round(float(np.percentile(tails_null[:, d], 95)), 3),
            "tail_p": round(float((np.sum(tails_null[:, d] >= tail) + 1) / (len(tails_null) + 1)), 4),
            "bimodality": round(bc, 3), "bimodality_null_q95": round(float(np.percentile(bc_null[:, d], 95)), 3),
            "bimodality_p": round(float((np.sum(bc_null[:, d] >= bc) + 1) / (len(bc_null) + 1)), 4),
            "eta2_period": eta2_perm(scores[:, d], per("period")),
            "eta2_region": eta2_perm(scores[:, d], per("region")),
            "pole_high": [stmts[ksids[j]]["statement"] for j in order[::-1][:12]],
            "pole_low": [stmts[ksids[j]]["statement"] for j in order[:12]],
            "books": sorted([{"row": r, "book": key[r]["book"], "z": round(float(z[i]), 3)} for i, r in enumerate(rows)],
                            key=lambda x: x["z"]),
        }
        dim["keep"] = (dim["bootstrap_congruence"] >= CONGRUENCE_MIN and dim["books_high"] >= ENDS_MIN
                       and dim["books_low"] >= ENDS_MIN and max(dim["r2_coverage"], dim["r2_neg_count"],
                                                                dim["r2_secondary_knowledge"]) < CONFOUND_R2_MAX
                       and dim["rater_r"] >= RATER_R_MIN)
        dim["bipolar"] = dim["books_high"] >= ENDS_MIN and dim["books_low"] >= ENDS_MIN
        dims.append(dim)
    res["dimensions"] = dims
    res["kept_statement_ids"] = ksids
    return res
