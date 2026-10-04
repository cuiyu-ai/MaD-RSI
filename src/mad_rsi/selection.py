"""Greedy disagreement and response-space coverage selection."""
import math
import random
from .serialization import fingerprint


def select_queries(matrix, budget, seed=0, method="disagreement_coverage", reserve=True,
                   alpha=1.0, beta=1.0, sigma=.5):
    if method not in {"random", "difficulty", "disagreement", "disagreement_coverage"}:
        raise ValueError("unknown selector")
    if type(budget) is not int or any(type(v) not in (int, float) or not math.isfinite(v) for v in (alpha, beta, sigma)):
        raise ValueError("selection parameters must be finite and budget integer")
    sizes = {len(row["vector"]) for row in matrix.values()}
    if len(sizes) != 1 or 0 in sizes:
        raise ValueError("inconsistent response vector dimensions")
    if any(not math.isfinite(v) for row in matrix.values() for v in [row["mean"], row["disagreement"], *row["vector"]]):
        raise ValueError("nonfinite matrix value")
    ids = sorted(matrix)
    if not 1 <= budget <= len(ids) or min(alpha, beta) < 0 or sigma <= 0:
        raise ValueError("invalid selection parameters")
    rng = random.Random(seed)
    selected, reasons = [], {}
    if reserve:
        pools = [("success_guard", [q for q in ids if matrix[q]["mean"] == 1]),
                 ("all_fail_diagnostic", [q for q in ids if matrix[q]["mean"] == 0]),
                 ("exploration", ids)]
        for label, pool in pools:
            available = [q for q in pool if q not in selected]
            if available and len(selected) < budget:
                chosen = rng.choice(available)
                selected.append(chosen)
                reasons[chosen] = label
    similarity = {(a, b): math.exp(-sum((u - v) ** 2 for u, v in zip(matrix[a]["vector"], matrix[b]["vector"])) / (2 * sigma ** 2))
                  for a in ids for b in ids}
    covered = {j: max((similarity[j, q] for q in selected), default=0) for j in ids}
    while len(selected) < budget:
        remaining = [q for q in ids if q not in selected]
        if method == "random":
            chosen = rng.choice(remaining)
        else:
            def score(q):
                if method == "difficulty":
                    return 1 - matrix[q]["mean"]
                gain = alpha * matrix[q]["disagreement"]
                if method == "disagreement_coverage":
                    gain += beta * sum(max(0, similarity[j, q] - covered[j]) for j in ids) / len(ids)
                return gain
            chosen = max(remaining, key=lambda q: (score(q), q))
        selected.append(chosen)
        reasons[chosen] = method
        covered = {j: max(covered[j], similarity[j, chosen]) for j in ids}
    return {"query_ids": selected, "reasons": reasons, "method": method, "seed": seed,
            "matrix_hash": fingerprint(matrix), "budget": budget}
