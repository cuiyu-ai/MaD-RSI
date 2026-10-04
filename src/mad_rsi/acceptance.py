"""Optional paired, family-balanced confirmation gate; native gates remain pluggable."""
import math
import random
from statistics import mean
from .validation import validate_records


def paired_group_decision(inc, cand, models, tasks, k, cfg):
    if cfg.get("bound", "hoeffding") not in {"hoeffding", "bootstrap"}:
        raise ValueError("unknown confidence bound")
    alpha_value = cfg.get("alpha", .05)
    if type(alpha_value) not in (int, float) or not 0 < alpha_value < 1:
        raise ValueError("alpha must be in (0,1)")
    for key, default in (("candidates", 2), ("rounds", 2), ("bootstrap_samples", 4000), ("min_confirmation_groups", 8)):
        if type(cfg.get(key, default)) is not int or cfg.get(key, default) < 1:
            raise ValueError("invalid " + key)
    for key, default in (("min_gain", 0.0), ("regression_tolerance", .05), ("max_cost_ratio", 1.5)):
        value = cfg.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid " + key)
    if any(m.get("family") == "__average__" for m in models):
        raise ValueError("reserved family name")
    if not tasks or any(not isinstance(t.get("group"), str) or not t["group"] for t in tasks):
        raise ValueError("independent context groups required")
    ids = [x["id"] for x in tasks]
    validate_records(inc, ids, models, k, "confirmation")
    validate_records(cand, ids, models, k, "confirmation")
    index = lambda rs: {(r["task_id"], r["model_id"], r["trial"]): r for r in rs}
    left, right = index(inc), index(cand)
    for key in left:
        if left[key]["input_hash"] != right[key]["input_hash"] or left[key]["protocol_hash"] != right[key]["protocol_hash"]:
            raise ValueError("unpaired inputs or incompatible evaluation protocols")
    fams = sorted({m["family"] for m in models})
    groups = sorted({x["group"] for x in tasks})
    if len(groups) < cfg.get("min_confirmation_groups", 8):
        return {"accepted": False, "reasons": ["insufficient independent confirmation groups"], "n_groups": len(groups)}
    differences = {f: [] for f in fams}
    for f in fams:
        mids = [m["id"] for m in models if m["family"] == f]
        for group in groups:
            qids = [x["id"] for x in tasks if x["group"] == group]
            differences[f].append(mean(right[q, m, r]["reward"] - left[q, m, r]["reward"]
                                       for q in qids for m in mids for r in range(k)))
    average = [mean(differences[f][j] for f in fams) for j in range(len(groups))]
    sequences = {**differences, "__average__": average}
    alpha = cfg.get("alpha", .05) / ((len(fams) + 1) * cfg.get("candidates", 2) * cfg.get("rounds", 2))
    method = cfg.get("bound", "hoeffding")
    stats = {}
    for label, values in sequences.items():
        if method == "hoeffding":
            radius = math.sqrt(2 * math.log(1 / alpha) / len(values))
            lower = mean(values) - radius
        else:
            rng = random.Random(cfg.get("seed", 0))
            samples = sorted(mean(values[rng.randrange(len(values))] for _ in values)
                             for _ in range(cfg.get("bootstrap_samples", 4000)))
            lower = samples[max(0, int(alpha * len(samples)) - 1)]
        stats[label] = {"gain": mean(values), "lower": lower, "n_groups": len(values)}
    reasons = []
    if stats["__average__"]["lower"] <= cfg.get("min_gain", 0.0):
        reasons.append("average gain lower bound below threshold")
    for f in fams:
        if stats[f]["lower"] < -cfg.get("regression_tolerance", .05):
            reasons.append(f"family regression guard: {f}")
    costs = []
    for rs in (inc, cand):
        if any(not isinstance(r.get("tokens"), (int, float)) or r["tokens"] <= 0 for r in rs):
            reasons.append("unknown policy token cost")
            costs.append(None)
        else:
            costs.append(mean(mean(r["tokens"] for r in rs if r["family"] == f) for f in fams))
    ratio = costs[1] / costs[0] if all(c is not None for c in costs) else None
    if ratio is not None and ratio > cfg.get("max_cost_ratio", 1.5):
        reasons.append("group inference cost ratio exceeded")
    return {"accepted": not reasons, "reasons": reasons, "families": stats,
            "cost_ratio": ratio, "bound": method, "per_comparison_alpha": alpha,
            "n_groups": len(groups), "estimand": "equal context-group and equal family mean",
            "guarantee": "Hoeffding conditional on independent groups and fresh shard; bootstrap is approximate"}
