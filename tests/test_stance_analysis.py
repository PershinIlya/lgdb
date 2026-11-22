import numpy as np

from lgdb import stance_analysis as sa


def planted(n=120, p=150, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 2))
    w = rng.normal(size=(p, 2)) * (rng.random((p, 1)) < 0.6)
    lat = pos @ w.T
    m = np.sign(lat) * (np.abs(lat) > 0.8)
    m = m * (rng.random((n, p)) < 0.35)  # sparse coverage
    return m


def test_parallel_finds_planted_dims():
    m = planted()
    keep = sa.keep_statements(m)
    k, _, _ = sa.parallel_k(m[:, keep], n=50)
    assert k == 2


def test_null_breaks_structure():
    m = planted()
    rng = np.random.default_rng(1)
    p = m.copy()
    for j in range(m.shape[1]):
        nz = np.flatnonzero(m[:, j])
        p[nz, j] = rng.permutation(m[nz, j])
    k, _, _ = sa.parallel_k(p[:, sa.keep_statements(p)], n=50)
    assert k <= 1


def test_eta2():
    s = np.array([0, 0, 1, 1.0])
    assert abs(sa.eta2(s, ["a", "a", "b", "b"]) - 1) < 1e-9
