"""Actual media construction graph, independent of chronological dialogue inputs."""
from collections import deque


def assign_reference_materials(plan):
    media = {row["placeholder"]: row for row in plan["placeholders"]}
    root = plan["final"]["media"]
    media[root]["source_video"] = plan["source_video"]
    media[root]["reference_materials"] = []
    for token, row in media.items():
        if token == root or row["purpose"] == "asset":
            continue
        # Formal shots are extracted/edited from the truth, including earlier
        # versions with provisional designs. Explorations and intermediate cuts
        # are composed from their visible constituent assets/clips instead.
        row["reference_materials"] = ([root] if row["purpose"] == "shot"
                                      else list(dict.fromkeys(row["inputs"])))
    for asset, target in plan["target_assets"].items():
        versions = {p: row for p, row in media.items()
                    if row["purpose"] == "asset" and row["asset_ids"] == [asset]}
        canonical = plan["final"]["assets"][asset]
        neighbors = {p: [] for p in versions}
        for p, row in versions.items():
            parent = row["edit_parent"]
            if parent in versions:
                neighbors[p].append(parent)
                neighbors[parent].append(p)
        # Orient each edit component from its target-compatible versions, so
        # the actual construction can run backwards relative to the dialogue.
        remaining = set(versions)
        while remaining:
            start = min(remaining)
            component, todo = set(), [start]
            while todo:
                p = todo.pop()
                if p in component:
                    continue
                component.add(p)
                todo.extend(neighbors[p])
            remaining -= component
            anchors = sorted(p for p in component
                             if versions[p]["attributes"] == target["attributes"])
            if anchors:
                for p in anchors:
                    versions[p]["reference_materials"] = [root]
            else:
                # An independently proposed candidate has no edit path to the
                # target version. Derive its closest version from the canonical
                # asset, then follow its existing edit edges outwards.
                anchor = min(component, key=lambda p: (
                    sum(value != target["attributes"][key]
                        for key, value in versions[p]["attributes"].items()), p))
                anchors = [anchor]
                versions[anchor]["reference_materials"] = [canonical]
            visited, queue = set(anchors), deque(anchors)
            while queue:
                parent = queue.popleft()
                for p in sorted(neighbors[parent]):
                    if p not in visited:
                        versions[p]["reference_materials"] = [parent]
                        visited.add(p)
                        queue.append(p)
    depths, active = {}, set()
    def depth(p):
        if p in active:
            raise ValueError("Cyclic construction references")
        if p not in depths:
            active.add(p)
            refs = media[p]["reference_materials"]
            depths[p] = 0 if p == root else 1 + max(depth(ref) for ref in refs)
            active.remove(p)
        return depths[p]
    for p, row in media.items():
        row["reference_depth"] = depth(p)


def validate_reference_materials(plan):
    media = {row["placeholder"]: row for row in plan["placeholders"]}
    root = plan["final"]["media"]
    active, depths = set(), {}
    def visit(p):
        if p in active:
            raise ValueError("Cyclic construction references")
        if p in depths:
            return depths[p]
        row = media[p]
        refs = row.get("reference_materials")
        if (not isinstance(refs, list) or any(not isinstance(ref, str) or ref not in media for ref in refs)
                or len(refs) != len(set(refs))):
            raise ValueError("Invalid reference_materials")
        if (p == root) != (not refs):
            raise ValueError("Only the target video may be a construction root")
        active.add(p)
        expected = 0 if p == root else 1 + max(visit(ref) for ref in refs)
        active.remove(p)
        if type(row.get("reference_depth")) is not int or row["reference_depth"] != expected:
            raise ValueError("Incorrect reference_depth")
        depths[p] = expected
        return expected
    if media[root].get("source_video") != plan["source_video"]:
        raise ValueError("Construction root must be the source target video")
    if root != plan["placeholders"][-1]["placeholder"]:
        raise ValueError("Target video must be the last generated media")
    for p in media:
        visit(p)
    # Also enforce the reference selection policy, not just a well-formed DAG.
    from copy import deepcopy
    expected = deepcopy(plan)
    assign_reference_materials(expected)
    for row in expected["placeholders"]:
        if row["reference_materials"] != media[row["placeholder"]]["reference_materials"]:
            raise ValueError("Construction references do not follow the asset/shot policy")
    return True
