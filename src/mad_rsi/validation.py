"""Model metadata and complete observation-grid validation."""
import math
from collections import defaultdict


def family_weights(models):
    families = defaultdict(list)
    if any(not isinstance(m.get("id"), str) or not m["id"] or not isinstance(m.get("family"), str) or not m["family"] for m in models):
        raise ValueError("model ids and families must be nonempty strings")
    if not models or len({m["id"] for m in models}) != len(models):
        raise ValueError("models must be nonempty with unique ids")
    for m in models:
        if not m.get("family"):
            raise ValueError("each model needs an explicit family")
        families[m["family"]].append(m["id"])
    return {m: 1 / (len(families) * len(ms)) for ms in families.values() for m in ms}


def validate_records(records, ids, models, k, split=None, trial_start=0):
    if type(k) is not int or k < 1 or type(trial_start) is not int or trial_start < 0:
        raise ValueError("invalid trial range")
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(q, str) or not q for q in ids):
        raise ValueError("task ids must be nonempty unique strings")
    family_weights(models)
    expected = {(q, m["id"], r) for q in ids for m in models for r in range(trial_start, trial_start + k)}
    observed = set()
    families = {m["id"]: m["family"] for m in models}
    temperatures = {m["id"]: float(m.get("temperature", 0)) for m in models}
    versions = set()
    for rec in records:
        key = (rec["task_id"], rec["model_id"], rec["trial"])
        if key not in expected or key in observed:
            raise ValueError("unexpected or duplicate observation")
        observed.add(key)
        if rec["family"] != families[rec["model_id"]]:
            raise ValueError("model family mismatch")
        if rec.get("temperature", 0.0) != temperatures[rec["model_id"]]:
            raise ValueError("sampling temperature mismatch")
        if rec.get("status") != "ok":
            raise ValueError("infrastructure/missing observation is not a model failure")
        if split and rec.get("split") != split:
            raise ValueError("forbidden split in feedback")
        reward = rec["reward"]
        if type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("invalid reward")
        if not isinstance(rec.get("harness_hash"), str) or not rec["harness_hash"]:
            raise ValueError("missing harness identity")
        if not isinstance(rec.get("input_hash"), str) or not rec["input_hash"]:
            raise ValueError("missing input identity")
        if not isinstance(rec.get("success"), bool):
            raise ValueError("task success must be an explicit boolean")
        tokens = rec.get("tokens")
        if tokens is not None and (type(tokens) not in (int, float) or not math.isfinite(tokens) or tokens <= 0):
            raise ValueError("invalid token accounting")
        if type(rec.get("trial")) is not int:
            raise ValueError("trial must be an integer")
        versions.add(rec["harness_hash"])
    if observed != expected or len(versions) != 1:
        raise ValueError("incomplete matrix or mixed harness versions")
