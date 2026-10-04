"""Temperature-conditioned response statistics; estimates are not causal effects."""
import math
from collections import defaultdict
from statistics import mean
from .validation import family_weights, validate_records

def entropy(p):
    return 0.0 if p in (0.0, 1.0) else -p * math.log(p) - (1 - p) * math.log(1 - p)


def response_matrix(records, ids, models, k):
    validate_records(records, ids, models, k, "evolve")
    weights = family_weights(models)
    cells = defaultdict(list)
    for r in records:
        cells[r["task_id"], r["model_id"]].append(int(r["success"]))
    result = {}
    for q in ids:
        p = {m["id"]: mean(cells[q, m["id"]]) for m in models}
        avg = sum(weights[m] * v for m, v in p.items())
        disagreement = max(0.0, entropy(avg) - sum(weights[m] * entropy(v) for m, v in p.items()))
        result[q] = {"p": p, "mean": avg, "disagreement": disagreement,
                     "vector": [math.sqrt(weights[m["id"]]) * p[m["id"]] for m in models],
                     "observations_per_model": k,
                     "estimate_warning": "finite-sample plug-in MI; not training utility"}
    return result

def response_tensor(records, ids, models, temperatures, min_repeats=2):
    if type(min_repeats) is not int or min_repeats < 2:
        raise ValueError("at least two repeats required")
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("unique nonempty query ids required")
    family_weights(models)
    if (not temperatures or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in temperatures)
            or len(set(temperatures)) != len(temperatures)):
        raise ValueError("temperatures must be unique finite nonnegative numbers")
    temperatures = sorted(float(t) for t in temperatures)
    mids = {m["id"] for m in models}
    expected = {(q, m, t) for q in ids for m in mids for t in temperatures}
    cells = defaultdict(list)
    if len({r["harness_hash"] for r in records}) != 1:
        raise ValueError("probe evidence must use one harness version")
    for r in records:
        key = (r["task_id"], r["model_id"], r.get("temperature"))
        if key not in expected:
            raise ValueError("unexpected task, model or temperature in response tensor")
        cells[key].append(r)
    if set(cells) != expected:
        raise ValueError("incomplete model-query-temperature grid")
    lookup = {m["id"]: m for m in models}
    for (q, mid, temp), rows in cells.items():
        if len(rows) < min_repeats:
            raise ValueError("insufficient within-temperature repeats")
        validate_records(rows, [q], [{**lookup[mid], "temperature": temp}], len(rows), "evolve")
        if len({r["input_hash"] for r in rows}) != 1:
            raise ValueError("mixed task inputs in one probe cell")
    weights = family_weights(models)
    result = {}
    for q in ids:
        if len({r["input_hash"] for r in records if r["task_id"] == q}) != 1:
            raise ValueError("models/temperatures did not receive the same frozen task")
        p = {(mid, t): mean(int(r["success"]) for r in cells[q, mid, t]) for mid in mids for t in temperatures}
        model_avg = {mid: mean(p[mid, t] for t in temperatures) for mid in mids}
        temp_avg = {t: sum(weights[mid] * p[mid, t] for mid in mids) for t in temperatures}
        avg = sum(weights[mid] * model_avg[mid] for mid in mids)
        noise = sum(weights[mid] * mean(entropy(p[mid, t]) for t in temperatures) for mid in mids)
        h_given_model = sum(weights[mid] * entropy(model_avg[mid]) for mid in mids)
        h_given_temp = mean(entropy(temp_avg[t]) for t in temperatures)
        diagnostics = {
            "model_given_temperature": max(0, h_given_temp - noise),
            "temperature_given_model": max(0, h_given_model - noise),
            "within_condition_entropy": noise,
            "model_marginal": max(0, entropy(avg) - h_given_model),
            "temperature_marginal": max(0, entropy(avg) - h_given_temp),
            "joint_information": max(0, entropy(avg) - noise),
            "total_entropy": entropy(avg),
        }
        result[q] = {
            "p": model_avg, "mean": avg,
            "disagreement": diagnostics["model_given_temperature"],
            "vector": [math.sqrt(weights[m["id"]] / len(temperatures)) * p[m["id"], t]
                       for m in models for t in temperatures],
            "conditions": {mid: {str(t): {"p": p[mid, t], "n": len(cells[q, mid, t])}
                                 for t in temperatures} for mid in sorted(mids)},
            "temperature_effect": {mid: p[mid, temperatures[-1]] - p[mid, temperatures[0]] for mid in sorted(mids)},
            "diagnostics": diagnostics,
            "estimate_warning": "adaptive plug-in estimates; no inferential coverage or harness utility guarantee",
        }
    return result
