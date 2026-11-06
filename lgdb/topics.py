"""Group claims into topics by meaning of the recommendation text.

Embeddings capture subject matter, so "never give up" and "know when to quit"
land in the same topic. Whether a topic splits into camps is decided later by
the compatibility graph, not here.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score


def embed(texts: list[str], provider: str = "sentence-transformers",
          model_name: str = "all-mpnet-base-v2") -> np.ndarray:
    if provider == "tfidf":  # offline fallback for tests and dry runs
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        x = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1).fit_transform(texts)
        if x.shape[1] <= 256:  # small vocabulary: reducing dimensions would merge distinct words
            z = x.toarray()
        else:
            z = TruncatedSVD(128, random_state=0).fit_transform(x)
    elif provider == "sentence-transformers":
        from sentence_transformers import SentenceTransformer
        z = SentenceTransformer(model_name).encode(texts, batch_size=64, show_progress_bar=True)
    elif provider == "voyage":
        import voyageai
        client = voyageai.Client()
        z = np.vstack([client.embed(texts[i:i + 128], model=model_name).embeddings
                       for i in range(0, len(texts), 128)])
    else:
        raise ValueError(provider)
    z = np.asarray(z, dtype=float)
    return z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-12, None)


def cluster(z: np.ndarray, distance_threshold: float) -> np.ndarray:
    model = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                    distance_threshold=distance_threshold)
    return model.fit_predict(z)


def eligible(labels: np.ndarray, claim_ids: list[str], text_of: dict[str, str],
             min_claims: int = 15, min_texts: int = 5) -> dict[int, list[str]]:
    groups = defaultdict(list)
    for lab, cid in zip(labels, claim_ids):
        groups[int(lab)].append(cid)
    return {lab: ids for lab, ids in groups.items()
            if len(ids) >= min_claims and len({text_of[c] for c in ids}) >= min_texts}


def subsample_stability(z: np.ndarray, distance_threshold: float, frac: float = 0.9,
                        repeats: int = 5, seed: int = 0) -> float:
    """Mean adjusted Rand index between the full clustering and reruns on subsamples."""
    full = cluster(z, distance_threshold)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        idx = np.sort(rng.choice(len(z), size=int(frac * len(z)), replace=False))
        sub = cluster(z[idx], distance_threshold)
        scores.append(adjusted_rand_score(full[idx], sub))
    return float(np.mean(scores))
