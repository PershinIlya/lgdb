"""Signed compatibility graph within a topic.

Nodes are claims. An edge carries a sign: +1 when 2 recommendations can be
followed together, -1 when they cannot. The best split into 2 camps minimizes
frustration: positive edges across camps plus negative edges inside a camp.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# P5 labels mapped to edge signs. "conditional" is excluded in the primary
# analysis and counted as negative in the preregistered sensitivity analysis.
POSITIVE = {"duplicate", "narrower", "compatible"}
NEGATIVE = {"incompatible"}
CONDITIONAL = {"conditional"}
IGNORED = {"unrelated"}


def edge_sign(label: str, conditional_as_negative: bool = False) -> int:
    label = (label or "").strip().lower()
    if label in POSITIVE:
        return 1
    if label in NEGATIVE:
        return -1
    if label in CONDITIONAL:
        return -1 if conditional_as_negative else 0
    return 0


@dataclass
class SignedGraph:
    nodes: list[str]
    edges: list[tuple[int, int, int]]  # (i, j, sign), i < j
    index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.index = {n: k for k, n in enumerate(self.nodes)}

    @classmethod
    def from_labels(cls, nodes, labeled_pairs, conditional_as_negative=False):
        """labeled_pairs: iterable of (claim_a, claim_b, label)."""
        nodes = list(nodes)
        idx = {n: k for k, n in enumerate(nodes)}
        best: dict[tuple[int, int], int] = {}
        for a, b, label in labeled_pairs:
            if a not in idx or b not in idx or a == b:
                continue
            s = edge_sign(label, conditional_as_negative)
            if s == 0:
                continue
            i, j = sorted((idx[a], idx[b]))
            best[(i, j)] = s  # last label wins if a pair was labeled twice
        return cls(nodes, [(i, j, s) for (i, j), s in best.items()])

    @property
    def n_pos(self) -> int:
        return sum(1 for e in self.edges if e[2] > 0)

    @property
    def n_neg(self) -> int:
        return sum(1 for e in self.edges if e[2] < 0)

    def adjacency(self, signs: np.ndarray | None = None) -> np.ndarray:
        n = len(self.nodes)
        w = np.zeros((n, n))
        for k, (i, j, s) in enumerate(self.edges):
            v = s if signs is None else signs[k]
            w[i, j] = w[j, i] = v
        return w


def frustration(w: np.ndarray, spins: np.ndarray) -> float:
    """Count of violated edges for a +-1 assignment (each edge counted once)."""
    prod = w * np.outer(spins, spins)
    return float(np.sum(prod < 0) / 2)


def _local_search(w: np.ndarray, spins: np.ndarray, max_sweeps: int = 100) -> np.ndarray:
    spins = spins.copy()
    for _ in range(max_sweeps):
        field_ = w @ spins
        bad = np.where(spins * field_ < 0)[0]
        if bad.size == 0:
            break
        # flip the node with the most negative local agreement first
        k = bad[np.argmin(spins[bad] * field_[bad])]
        spins[k] = -spins[k]
    return spins


def best_partition(w: np.ndarray, restarts: int = 20, rng: np.random.Generator | None = None):
    """Heuristic minimum-frustration 2-partition: spectral start plus random restarts."""
    rng = rng or np.random.default_rng(0)
    n = w.shape[0]
    if n == 0:
        return np.array([]), 0.0
    starts = []
    if np.any(w):
        vals, vecs = np.linalg.eigh(w)
        lead = np.sign(vecs[:, -1])
        lead[lead == 0] = 1
        starts.append(lead)
    starts.append(np.ones(n))
    for _ in range(restarts):
        starts.append(rng.choice([-1.0, 1.0], size=n))
    best_spins, best_f = None, np.inf
    for s in starts:
        s = _local_search(w, s.astype(float))
        f = frustration(w, s)
        if f < best_f:
            best_spins, best_f = s, f
    return best_spins, best_f


@dataclass
class TopicResult:
    n_claims: int
    n_pos: int
    n_neg: int
    neg_rate: float
    frustration_index: float
    null_p: float
    camp_of: dict[str, int]
    camp_sizes: dict[int, int]
    camp_texts: dict[int, int]
    bipolar: bool


def analyze_topic(
    graph: SignedGraph,
    text_of: dict[str, str],
    neg_rate_min: float = 0.15,
    frustration_max: float = 0.2,
    null_alpha: float = 0.05,
    n_perm: int = 1000,
    min_camp_claims: int = 3,
    min_camp_texts: int = 2,
    restarts: int = 20,
    null_restarts: int = 5,
    seed: int = 0,
) -> TopicResult:
    rng = np.random.default_rng(seed)
    n_signed = graph.n_pos + graph.n_neg
    w = graph.adjacency()
    spins, f = best_partition(w, restarts=restarts, rng=rng)
    fi = f / n_signed if n_signed else 1.0
    neg_rate = graph.n_neg / n_signed if n_signed else 0.0

    # Permutation null: shuffle edge signs over the same edges.
    signs = np.array([s for _, _, s in graph.edges], dtype=float)
    null = []
    if n_signed and graph.n_neg and graph.n_pos:
        for _ in range(n_perm):
            perm = rng.permutation(signs)
            _, fn = best_partition(graph.adjacency(perm), restarts=null_restarts, rng=rng)
            null.append(fn / n_signed)
        null_p = (1 + sum(v <= fi for v in null)) / (1 + len(null))
    else:
        null_p = 1.0

    camp_of = {}
    if spins is not None and len(spins):
        # Only nodes that have at least 1 signed edge get a camp.
        has_edge = np.zeros(len(graph.nodes), dtype=bool)
        for i, j, _ in graph.edges:
            has_edge[i] = has_edge[j] = True
        for k, node in enumerate(graph.nodes):
            if has_edge[k]:
                camp_of[node] = 0 if spins[k] > 0 else 1
    sizes = {c: sum(1 for v in camp_of.values() if v == c) for c in (0, 1)}
    texts = {c: len({text_of[n] for n, v in camp_of.items() if v == c}) for c in (0, 1)}

    bipolar = (
        neg_rate >= neg_rate_min
        and fi <= frustration_max
        and null_p < null_alpha
        and all(sizes[c] >= min_camp_claims for c in (0, 1))
        and all(texts[c] >= min_camp_texts for c in (0, 1))
    )
    return TopicResult(
        n_claims=len(graph.nodes),
        n_pos=graph.n_pos,
        n_neg=graph.n_neg,
        neg_rate=neg_rate,
        frustration_index=fi,
        null_p=null_p,
        camp_of=camp_of,
        camp_sizes=sizes,
        camp_texts=texts,
        bipolar=bool(bipolar),
    )
