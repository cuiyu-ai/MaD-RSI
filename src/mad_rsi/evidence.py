"""Same-query trace consolidation with verifiable evidence references."""
import copy
from statistics import mean
from .serialization import canonical, fingerprint


def build_bundle(q, records, models, include_success=True, scope="group", max_chars=20000):
    if scope not in {"group", "single"} or type(max_chars) is not int or max_chars < 1:
        raise ValueError("invalid scope or feedback budget")
    if type(include_success) is not bool:
        raise ValueError("include_success must be boolean")
    rows = [r for r in records if r["task_id"] == q]
    expected_models = {m["id"] for m in models}
    if any(r.get("model_id") not in expected_models or r.get("status") != "ok" for r in rows):
        raise ValueError("unexpected model or unscored evidence")
    evidence_ids = [r.get("evidence_id") for r in rows]
    if any(not isinstance(e, str) or not e for e in evidence_ids) or len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("evidence IDs must be nonempty and unique")
    if len({r.get("input_hash") for r in rows}) != 1 or any(not r.get("input_hash") for r in rows):
        raise ValueError("mixed or missing query inputs")
    if not rows or any(r["split"] != "evolve" for r in rows):
        raise ValueError("only evolve evidence is allowed")
    if len({r["harness_hash"] for r in rows}) != 1:
        raise ValueError("stale evidence mixed into query bundle")
    profile = {}
    for m in models:
        mr = [r for r in rows if r["model_id"] == m["id"]]
        if not mr:
            raise ValueError("missing model in query evidence")
        profile[m["id"]] = {"family": m["family"], "successes": sum(r["success"] for r in mr), "trials": len(mr)}
        profile[m["id"]]["by_temperature"] = {
            str(float(temp)): {"successes": sum(r["success"] for r in mr if r.get("temperature", 0.0) == temp),
                               "trials": sum(r.get("temperature", 0.0) == temp for r in mr)}
            for temp in sorted({r.get("temperature", 0.0) for r in mr})}
        profile[m["id"]]["condition_weighted_success_rate"] = mean(
            cell["successes"] / cell["trials"] for cell in profile[m["id"]]["by_temperature"].values())
    eligible = [r for r in rows if (include_success or not r["success"])
                and (scope == "group" or r["model_id"] == models[0]["id"])]
    # Deduplicate identical visible evidence, never deduplicate the outcome denominator.
    clusters = {}
    for r in eligible:
        key = fingerprint([r["success"], r["evidence_text"]])
        cluster = clusters.setdefault(key, {"success": r["success"], "text": r["evidence_text"], "sources": []})
        cluster["sources"].append({**{k: r[k] for k in ("evidence_id", "model_id", "family", "trial")},
                                   "temperature": r.get("temperature", 0.0)})
    representatives = list(clusters.values())
    # Explicit bounded excerpts, fair across distinct traces; full originals remain on disk.
    per_cap = max(0, (max_chars - len(canonical(profile)) - 800) // max(1, len(representatives)))
    for c in representatives:
        c["original_chars"] = len(c["text"])
        c["text"] = c["text"][:per_cap]
        c["truncated"] = c["original_chars"] > per_cap
    result = {"task_id": q, "harness_hash": rows[0]["harness_hash"], "profile": profile,
              "n_queries": 1, "n_models": len(profile), "n_families": len({m["family"] for m in models}),
              "n_trials": len(rows), "evidence": representatives,
              "warning": "Observed similarity is not a causal mechanism; preserve disagreements and counterevidence."}
    # Never silently exceed the budget or drop provenance to fit it.
    if len(canonical(result)) > max_chars:
        overhead = len(canonical(result)) - sum(len(c["text"]) for c in representatives)
        remaining = max_chars - overhead - 100
        if remaining < 0:
            raise ValueError("feedback budget too small for provenance; increase feedback_chars_per_query")
        cap = remaining // max(1, len(representatives))
        for c in representatives:
            c["text"] = c["text"][:cap]
            c["truncated"] = len(c["text"]) < c["original_chars"]
    if len(canonical(result)) > max_chars:
        raise ValueError("serialized evidence exceeds budget")
    return result


def validate_digest(digest, bundle):
    allowed = {s["evidence_id"] for e in bundle["evidence"] for s in e["sources"]}
    if not isinstance(digest, dict):
        raise ValueError("query digest must be an object")
    for field in ("shared_failures", "distinct_failures", "success_contrasts", "counterevidence"):
        if not isinstance(digest.get(field), list):
            raise ValueError(f"missing digest list {field}")
        for entry in digest[field]:
            if not isinstance(entry, dict) or not isinstance(entry.get("claim"), str):
                raise ValueError("each claim must have text and evidence_ids")
            refs = entry.get("evidence_ids")
            if not isinstance(refs, list) or not refs or any(r not in allowed for r in refs):
                raise ValueError("fabricated or missing evidence reference")
    if not isinstance(digest.get("uncertainty"), list) or any(not isinstance(x, str) for x in digest["uncertainty"]):
        raise ValueError("missing uncertainty list")
    if "data_quality" in digest and digest["data_quality"] not in {"supported", "suspect", "unassessed"}:
        raise ValueError("invalid data_quality")
    return {**digest, "task_id": bundle["task_id"], "n_queries": 1,
            "profile": bundle["profile"], "bundle_hash": fingerprint(bundle)}


def validate_analysis_report(report, allowed_queries):
    """Recompute task counts and reject invented/out-of-split task references."""
    if report.get("error"):
        raise ValueError("RRSI analyst did not complete a valid report")
    result = copy.deepcopy(report)
    allowed = set(allowed_queries)
    for field in ("failure_modes", "capability_gaps", "success_habits"):
        if not isinstance(result.get(field), list):
            raise ValueError("analyst report missing " + field)
        for item in result[field]:
            ids = set(item.get("affected_tasks", []))
            evidence = item.get("representative_evidence", [])
            refs = {e["task_id"] for e in evidence if isinstance(e, dict) and "task_id" in e}
            if not (ids | refs) <= allowed:
                raise ValueError("analyst cited a query outside the selected evolve evidence")
            if field != "success_habits" and not (ids | refs):
                raise ValueError("failure/capability claim requires task references")
            if ids or refs:
                item["n_tasks"] = len(ids | refs)
                item["affected_tasks"] = sorted(ids | refs)
            else:
                # Original success_habits schema has no task IDs: do not invent a count.
                item["n_tasks"] = 0
                item["count_status"] = "unverified; no task references supplied"
    return result
