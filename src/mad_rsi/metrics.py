"""Aligned model-group coverage. M counts models, not samples."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .serialization import fingerprint, read_json, write_json


def model_coverage(
    records: Sequence[Mapping[str, Any]], *, model_ids: Sequence[str],
    task_ids: Sequence[str], trials: Sequence[int],
) -> dict[str, Any]:
    """Compute any-model and all-model success on a complete, frozen grid.

    Require one harness, one protocol and one sampling condition per model.
    Infrastructure errors and missing outcomes are rejected, never silently
    counted as incorrect or removed. The caller defines task-specific success.
    """
    for name, values in (("model_ids", model_ids), ("task_ids", task_ids)):
        if not values or any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
            raise ValueError(name + " must contain unique nonempty strings")
    if not trials or any(type(t) is not int or t < 0 for t in trials) or len(set(trials)) != len(trials):
        raise ValueError("trials must contain unique nonnegative integers")
    expected = {(q, m, t) for q in task_ids for m in model_ids for t in trials}
    observed = {}
    inputs: dict[str, set[str]] = defaultdict(set)
    harnesses, protocols = set(), set()
    sampling: dict[str, set[str]] = defaultdict(set)
    for row in records:
        if type(row.get("trial")) is not int:
            raise ValueError("trial must be an integer")
        key = (row.get("task_id"), row.get("model_id"), row["trial"])
        if key not in expected or key in observed:
            raise ValueError("duplicate or unexpected observation")
        if row.get("status") != "ok" or type(row.get("success")) is not bool:
            raise ValueError("explicit scored outcome required; infrastructure error is not failure")
        for field in ("input_hash", "harness_hash", "protocol_hash"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError("missing " + field)
        # Callers using richer controls should supply the full sampling mapping.
        condition = row.get("sampling", {"temperature": row.get("temperature", 0.0)})
        sampling[row["model_id"]].add(fingerprint(condition))
        observed[key] = row["success"]
        inputs[row["task_id"]].add(row["input_hash"])
        harnesses.add(row["harness_hash"])
        protocols.add(row["protocol_hash"])
    if set(observed) != expected:
        raise ValueError("incomplete model/task/trial grid")
    if len(harnesses) != 1 or len(protocols) != 1:
        raise ValueError("mixed harnesses or evaluation protocols")
    if any(len(v) != 1 for v in inputs.values()) or any(len(v) != 1 for v in sampling.values()):
        raise ValueError("input or sampling mismatch")
    per_trial = []
    for trial in sorted(trials):
        groups = [[observed[q, m, trial] for m in model_ids] for q in task_ids]
        any_count = sum(any(group) for group in groups)
        all_count = sum(all(group) for group in groups)
        per_trial.append({"trial": trial, "tasks": len(task_ids), "any_count": any_count,
                          "all_count": all_count, "pass_at_m": any_count / len(task_ids),
                          "pass_all_m": all_count / len(task_ids)})
    total = len(task_ids) * len(trials)
    return {"m": len(model_ids), "model_ids": list(model_ids), "tasks": len(task_ids),
            "repeats": len(trials), "paired_groups": total,
            "pass_at_m": sum(x["any_count"] for x in per_trial) / total,
            "pass_all_m": sum(x["all_count"] for x in per_trial) / total,
            "per_trial": per_trial, "harness_hash": next(iter(harnesses)),
            "protocol_hash": next(iter(protocols)),
            "sampling_fingerprint": fingerprint({m: next(iter(sampling[m])) for m in sorted(sampling)}),
            "input_fingerprint": fingerprint({q: next(iter(inputs[q])) for q in sorted(inputs)}),
            "interpretation": "Aligned repeat mean; no best-of-run or best-of-model selection."}


def compare_harnesses(arms: Mapping[str, Sequence[Mapping[str, Any]]], **grid: Any) -> dict[str, Any]:
    """Compare separate harnesses under an identical explicit evaluation grid."""
    if not arms:
        raise ValueError("at least one arm required")
    results = {label: model_coverage(rows, **grid) for label, rows in arms.items()}
    if len({x["input_fingerprint"] for x in results.values()}) != 1:
        raise ValueError("arms evaluated different task inputs")
    if len({x["protocol_hash"] for x in results.values()}) != 1:
        raise ValueError("arms evaluated under different protocols")
    if len({x["sampling_fingerprint"] for x in results.values()}) != 1:
        raise ValueError("arms use different sampling configurations")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute pass@M and pass^M from aligned outcomes")
    parser.add_argument("input", help="JSON containing arms, model_ids, task_ids, and trials")
    parser.add_argument("--output", help="Optional output JSON path")
    args = parser.parse_args()
    data = read_json(args.input)
    result = compare_harnesses(data["arms"], model_ids=data["model_ids"],
                               task_ids=data["task_ids"], trials=data["trials"])
    if args.output:
        write_json(args.output, result)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
