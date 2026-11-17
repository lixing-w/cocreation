"""Coverage from realized structured annotations, never guessed from spoken text."""
from collections import Counter
from copy import deepcopy


def histogram(values):
    return dict(sorted(Counter(str(value) for value in values).items()))


def generation_snapshot(realizer):
    return dict(getattr(realizer, "metrics", {}),
                rewrite_attempts=getattr(realizer, "rewrite_attempts", 0),
                verifier_attempts=getattr(getattr(realizer, "verifier", None), "attempt_count", 0))


def collect_stats(result, observation, *, elapsed_seconds, generation):
    turns = [t for t in result["turns"] if t.get("content")]
    users = [t for t in turns if t["role"] == "user"]
    produced = {p for t in turns for p in t["output_media"]}
    media = [m for m in result["placeholders"] if m["placeholder"] in produced]
    labels = result["behavior_labels"]  # Exact canonical names from planner.BEHAVIORS.
    counts = Counter(b for t in users for b in set(t["intent"]["behavior_ids"]))
    logical = {a for m in media for a in m["asset_ids"]}
    selections = [s for s in result["selections"] if s["after_message"] <= len(turns)]
    scopes = [t["intent"]["scope"] for t in users if "scope" in t["intent"]]
    versions = Counter(a for m in media if m["purpose"] == "asset" for a in m["asset_ids"])
    generation = deepcopy(generation)
    return dict(
        schema_version=1,
        number_of_turns=len(turns), planned_number_of_turns=len(result["turns"]),
        number_of_attempts=generation.get("rewrite_attempts", 0),
        number_of_assets=len(media), number_of_logical_assets=len(logical),
        planned_number_of_assets=len(result["placeholders"]),
        time_elapsed_seconds=round(elapsed_seconds, 3),
        user_behaviors_covered=[labels[str(b)] for b in sorted(counts)],
        user_behavior_ids_covered=sorted(counts),
        user_behavior_turn_counts={labels[str(b)]: counts[b] for b in sorted(counts)},
        number_of_user_behaviors_covered=len(counts),
        number_of_defined_user_behaviors=len(labels),
        user_behavior_coverage_fraction=round(len(counts) / len(labels), 6),
        turns_by_role=histogram(t["role"] for t in turns),
        user_turns_by_module=histogram(t["intent"]["module"] for t in users),
        turns_by_action=histogram(t["intent"]["action"] for t in turns),
        logical_assets_by_type=histogram(a["kind"] for a in observation["assets"] if a["id"] in logical),
        media_by_kind=histogram(m["kind"] for m in media),
        media_by_purpose=histogram(m["purpose"] for m in media),
        reference_depth_counts=histogram(m["reference_depth"] for m in media),
        max_reference_depth=max((m["reference_depth"] for m in media), default=0),
        reference_edges=sum(len(m["reference_materials"]) for m in media),
        reference_versions_per_asset=dict(sorted(versions.items())),
        number_of_target_shots=len(result["target_order"]),
        number_of_shot_versions=sum(m["purpose"] == "shot" for m in media),
        number_of_trial_clips=sum(m["purpose"] == "exploration" and m["kind"] == "video" for m in media),
        number_of_visual_edits=sum(bool(m.get("edit_parent")) for m in media),
        edited_attribute_counts=histogram(m["variation"]["type"] for m in media if m.get("variation")),
        requested_scope_counts=histogram(scopes),
        conflict_type_counts=histogram(t["intent"]["conflict_type"] for t in users if "conflict_type" in t["intent"]),
        cross_turn_rules=dict(
            introduced=sum(r['started_at'] <= len(turns) for r in result.get('rules', [])),
            active=sum(r['started_at'] <= len(turns) and (r['ended_at'] is None or r['ended_at'] > len(turns)) for r in result.get('rules', [])),
            scopes=histogram(r['scope'] for r in result.get('rules', []) if r['started_at'] <= len(turns)),
            decisions=histogram(t['intent']['rule_event']['outcome'] for t in users if t['intent'].get('rule_event', {}).get('type') == 'resolve'),
            cleanup_cancellations=sum(t['intent'].get('rule_event', {}).get('cleanup', False) for t in turns),
        ),
        selection_decisions=histogram(s["decision"] for s in selections),
        selection_visibility=histogram("explicit" if s["explicit"] else "implicit" for s in selections),
        number_of_rollbacks=sum(s["rollback"] for s in selections),
        number_of_turns_with_media=sum(bool(t["intent"].get("display_media", []) or t["output_media"]) for t in turns),
        number_of_multi_reference_turns=sum(len(t["input_media"]) > 1 for t in turns),
        personas={role: p["id"] for role, p in result["personas"].items()},
        planning_seed=result["seed"], generation=generation,
    )
