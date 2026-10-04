"""Optional posterior-uncertainty acquisition proxy, not predicted harness gain."""
import math
import random
from statistics import mean
from .validation import family_weights
from .matrix import response_tensor

def choose_probe(records, ids, models, temperatures, policy, seed=0):
    """Dirac acquisition policy, or uniform random control, over (query, model, T)."""
    response_tensor(records, ids, models, temperatures)
    weights = family_weights(models)
    candidates = []
    for q in sorted(ids):
        for model in models:
            for t in temperatures:
                cell = [r for r in records if r["task_id"] == q and r["model_id"] == model["id"] and r["temperature"] == t]
                if not cell:
                    raise ValueError("active acquisition requires the initial grid")
                if any(r["split"] != "evolve" for r in cell):
                    raise ValueError("active policy cannot inspect non-evolve outcomes")
                successes = sum(r["success"] for r in cell)
                a, b = 1 + successes, 1 + len(cell) - successes
                posterior_variance = a * b / ((a + b) ** 2 * (a + b + 1))
                expected_reduction = posterior_variance / (a + b + 1)
                tokens = [r.get("tokens") for r in cell]
                if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in tokens):
                    raise ValueError("cost-aware probing requires known positive finite token costs")
                cost = mean(tokens)
                candidates.append({"query_id": q, "model_id": model["id"], "temperature": float(t),
                                   "trial_start": len(cell), "posterior_variance": posterior_variance,
                                   "expected_variance_reduction": expected_reduction, "estimated_tokens": cost,
                                   "score": weights[model["id"]] * expected_reduction / cost})
    if policy == "random":
        action = random.Random(seed).choice(candidates)
    elif policy == "uncertainty_per_cost":
        action = max(candidates, key=lambda c: (c["score"], c["query_id"], c["model_id"], c["temperature"]))
    else:
        raise ValueError("active policy must be random or uncertainty_per_cost")
    return {**action, "policy": policy, "proxy": "Beta(1,1) expected posterior variance reduction per token; not harness gain"}
