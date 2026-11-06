import numpy as np

from lgdb import factors as fa
from lgdb import graph as gr


def _graph(n_per_camp=10, partners=6, flip=0.0, all_positive=False, seed=0):
    rng = np.random.default_rng(seed)
    nodes = [f"c{i}" for i in range(2 * n_per_camp)]
    camp = {n: (i >= n_per_camp) for i, n in enumerate(nodes)}
    text_of = {n: f"t{i % 7}" for i, n in enumerate(nodes)}
    pairs = []
    for a in nodes:
        for b in rng.choice(nodes, partners, replace=False):
            if a == b:
                continue
            same = camp[a] == camp[b]
            ok = same or all_positive
            if rng.random() < flip:
                ok = not ok
            pairs.append((a, b, "compatible" if ok else "incompatible"))
    return gr.SignedGraph.from_labels(nodes, pairs), text_of, camp


def test_perfect_two_camps_is_bipolar_and_recovered():
    g, text_of, camp = _graph()
    res = gr.analyze_topic(g, text_of, n_perm=200)
    assert res.bipolar
    assert res.frustration_index == 0
    # recovered camps match planted ones up to label swap
    agree = np.mean([res.camp_of[n] == int(camp[n]) for n in res.camp_of])
    assert agree in (0.0, 1.0)


def test_consensus_topic_is_not_bipolar():
    g, text_of, _ = _graph(all_positive=True)
    assert not gr.analyze_topic(g, text_of, n_perm=200).bipolar


def test_random_conflicts_are_not_bipolar():
    # edge signs carry no camp structure: frustration should not beat the null
    rng = np.random.default_rng(3)
    nodes = [f"c{i}" for i in range(30)]
    pairs = [(a, b, "incompatible" if rng.random() < 0.4 else "compatible")
             for a in nodes for b in rng.choice(nodes, 6, replace=False) if a != b]
    g = gr.SignedGraph.from_labels(nodes, pairs)
    res = gr.analyze_topic(g, {n: f"t{i % 9}" for i, n in enumerate(nodes)}, n_perm=200)
    assert not res.bipolar


def test_factorization_recovers_one_planted_axis_and_rejects_noise():
    rng = np.random.default_rng(0)
    n, m = 120, 12
    u = rng.normal(size=(n, 1))
    v = rng.normal(size=(m, 1))
    x = 1 / (1 + np.exp(-(u @ v.T + 0.3 * rng.normal(size=(n, m)))))
    x[rng.random((n, m)) < 0.6] = np.nan  # sparse, like real texts touching few topics
    k, _ = fa.choose_k(x, k_max=4)
    assert k == 1
    noise = rng.random((n, m))
    noise[rng.random((n, m)) < 0.6] = np.nan
    k0, _ = fa.choose_k(noise, k_max=4)
    assert k0 == 0


def test_factor_stability_is_high_for_real_structure():
    rng = np.random.default_rng(1)
    n, m = 150, 10
    u = rng.normal(size=(n, 1))
    v = rng.normal(size=(m, 1))
    x = u @ v.T + 0.3 * rng.normal(size=(n, m))
    x[rng.random((n, m)) < 0.5] = np.nan
    res = fa.analyze(x, k_max=3, n_boot=50)
    assert res.k >= 1 and res.stable[0]
