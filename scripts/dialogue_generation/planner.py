"""Seeded, terminal-anchored dialogue planning. No models or media I/O here."""
from __future__ import annotations

from copy import deepcopy
import math
import random
from scripts.dialogue_generation.conflicts import CONFLICTS
from scripts.dialogue_generation.cross_turn_rule import CrossTurnRules, normalize_rule, validate_rule_timeline, matches
from scripts.dialogue_generation.references import assign_reference_materials, validate_reference_materials
from scripts.dialogue_generation.generation_requirements import resolve_requirements
from scripts.dialogue_generation.asset_variations import (
    POOLS, CATALOG, AssetSpec, asset_type, alternate, relative_variation,
    alternatives, validate_attributes, edit_request, changed_attributes, fields, register_attributes,
)


BEHAVIORS = {
    1: "Version rollback", 2: "Goal clarification", 3: "Ambiguous reference", 4: "Multiple candidates in one turn",
    5: "Separate preparation and assembly", 6: "Trial shots and assembly", 7: "Revision scope clarification",
    8: "Reference correction", 9: "Task suspension and resumption", 10: "Multiple-reference composition",
    11: "Context-dependent shot numbers", 12: "Relative edits and previews", 13: "Persistent rules and exceptions",
    14: "Conflicting requirements", 15: "Adoption decisions",
}
SINGLE_MODULES = ("direct", "candidates", "trial_edit", "clarify", "relative_preview")
CROSS_MODULES = ("task_switch", "temporary_rollback")
MODULES = SINGLE_MODULES + CROSS_MODULES


def apply_numbering(order, operation):
    result = list(order)
    positions = operation.get("positions", [])
    if (len(positions) != 2 or any(type(p) is not int or not 1 <= p <= len(result) for p in positions)
            or positions[0] == positions[1]):
        raise ValueError("Invalid numbered shot operation")
    a, b = (p - 1 for p in positions)
    if operation["type"] == "swap":
        result[a], result[b] = result[b], result[a]
    elif operation["type"] == "move_after":
        source, anchor = result[a], result[b]
        result.remove(source)
        result.insert(result.index(anchor) + 1, source)
    else:
        raise ValueError("Unknown numbered shot operation")
    return result


def observation(value):
    """Normalize only the small upstream contract; ignore extra VLM metadata."""
    data = deepcopy(value)
    if not isinstance(data, dict):
        raise ValueError("Observation must be a JSON object")
    version = data.get("attribute_schema_version", CATALOG["schema_version"])
    if type(version) is not int or version != CATALOG["schema_version"]:
        raise ValueError(f"Unsupported attribute_schema_version {version!r}; expected {CATALOG['schema_version']} (English enum values)")
    assets, shots = data.get("assets"), data.get("shots")
    if not isinstance(assets, list) or not assets:
        raise ValueError("assets must be a nonempty list")
    if not isinstance(shots, list) or not 1 <= len(shots) <= 4:
        raise ValueError("Target must contain 1–4 shots")
    ids = set()
    for asset in assets:
        if not isinstance(asset, dict) or not isinstance(asset.get("id"), str) or not asset["id"].strip():
            raise ValueError("Every asset needs a nonempty id")
        if asset["id"] in ids:
            raise ValueError("Duplicate asset id: " + asset["id"])
        ids.add(asset["id"])
        asset_type(asset)
        if not isinstance(asset.get("description"), str) or not asset["description"].strip():
            raise ValueError("Every asset needs a description")
        frames = asset.get("frames")
        if (not isinstance(frames, list) or not 1 <= len(frames) <= 2
                or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in frames)):
            raise ValueError("Asset frames must contain one or two nonnegative seconds")
        asset.setdefault("name", asset["id"])
        if not isinstance(asset["name"], str) or not asset["name"].strip():
            raise ValueError("Every asset needs a nonempty name")
        register_attributes(asset, asset.get("attributes"))
        validate_attributes(asset)
        asset["asset_type"] = asset_type(asset)
    shot_ids, used = set(), set()
    for shot in shots:
        if not isinstance(shot, dict) or not isinstance(shot.get("id"), str) or not shot["id"].strip():
            raise ValueError("Every shot needs an id")
        if shot["id"] in shot_ids:
            raise ValueError("Duplicate shot id: " + shot["id"])
        shot_ids.add(shot["id"])
        refs = shot.get("assets")
        if not isinstance(refs, list) or not refs or any(not isinstance(a, str) or a not in ids for a in refs):
            raise ValueError("Every shot must reference known assets")
        if len(refs) != len(set(refs)):
            raise ValueError("Duplicate asset within a shot")
        if not isinstance(shot.get("description"), str) or not shot["description"].strip():
            raise ValueError("Every shot needs a description")
        used.update(refs)
    if ids != used:
        raise ValueError("Every managed asset must be used by a target shot")
    order = data.setdefault("target_order", [s["id"] for s in shots])
    if not isinstance(order, list) or len(order) != len(shot_ids) or set(order) != shot_ids:
        raise ValueError("target_order must be a permutation of shot IDs")
    constraints = data.setdefault("constraints", [])
    if not isinstance(constraints, list):
        raise ValueError("constraints must be a list")
    data['constraints'] = [normalize_rule(rule, {a['id']: a for a in assets}) for rule in constraints]
    return data


class Planner:
    def __init__(self, value, *, seed=0, reworks=1, p_compile=0.65,
                 modules=None, extra_behaviors=True, cross_turn_rules=None):
        self.source = observation(value)
        if type(reworks) is not int or not 0 <= reworks <= 2:
            raise ValueError("reworks must be 0, 1 or 2")
        if not 0 <= p_compile <= 1:
            raise ValueError("p_compile must be between 0 and 1")
        if modules is not None and (not modules or set(modules) - set(MODULES)):
            raise ValueError("Unknown/empty module selection")
        self.seed, self.rng = seed, random.Random(seed)
        # Surface omissions/preview styles must not perturb action scheduling.
        self.adoption_rng = random.Random(f"adoption:{seed}")
        self.variation_rng = random.Random(f"variation:{seed}")
        self.numbering_rng = random.Random(f"numbering:{seed}")
        self.conflict_rng = random.Random(f"conflict:{seed}")
        self.selections = []
        self.p_compile, self.extra = p_compile, extra_behaviors
        self.modules = tuple(modules or MODULES)
        self.assets = {a["id"]: a for a in self.source["assets"]}
        self.shots = {s["id"]: s for s in self.source["shots"]}
        self.ready, self.versions = {}, {a: [] for a in self.assets}
        self.rendered, self.bindings, self.overrides = {}, {}, {s: {} for s in self.shots}
        self.render_order, self.dirty = [], set()
        self.media, self.visible = {}, set()
        self.turns, self.events = [], []
        self.counts = {"image": 0, "video": 0}
        self.comp_order, self.comp_media, self.comp_snapshot = [], None, {}
        self.conflict_done = False
        self.rules = CrossTurnRules(self, enabled=extra_behaviors if cross_turn_rules is None else cross_turn_rules)
        self._rules_started = False
        candidates = [a for a, row in self.assets.items() if alternatives(row, locked=self.variation_locks(a))]
        # Every object now has color, even when unknown. Prefer editable objects
        # rather than testing merely whether that key exists.
        self.rng.shuffle(candidates)
        candidates.sort(key=lambda a: self.assets[a]["asset_type"] != "object")
        self.pending_reworks = set(candidates[:min(reworks, len(candidates))])
        self.initial_specs = {a: alternate(row, AssetSpec(row), self.rng, locked=self.variation_locks(a))
                              if a in self.pending_reworks else AssetSpec(row)
                              for a, row in self.assets.items()}
        self.rework_plan = [dict(asset_id=a, field=self.initial_specs[a].change["field"],
                                 initial_value=self.initial_specs[a].change["after"],
                                 target_value=self.initial_specs[a].change["before"])
                            for a in self.assets if a in self.pending_reworks]
        self.module = "setup"

    def label(self, asset):
        return self.assets[asset]["name"]

    def reference_label(self, asset):
        """Distinguish continuing the current subject from returning to an older one."""
        current = self.turns[-1]["intent"]["asset_ids"] if self.turns else []
        return ("same " if asset in current else "previous ") + self.label(asset)

    def shot_label(self, shot):
        return self.shots[shot]["description"]

    def variation_locks(self, asset):
        from scripts.dialogue_generation.appearance_policy import COLOR_FIELDS
        row = dict(kind='image', purpose='asset', asset_ids=[asset])
        rules = list(self.rules.active.values()) if self._rules_started or self.rules.active else self.source['constraints']
        mode = None
        locked=set()
        for rule in rules:
            if matches(rule,row):
                locked.update(rule['effect'].get('asset_attributes',{}).get(asset,{}))
                if rule['effect'].get('color_mode'):mode=rule['effect']['color_mode']
        return locked | COLOR_FIELDS if mode=='black_and_white' else locked

    def refresh_initial_spec(self, asset):
        """Rules sampled before first delivery can make a reserved hue edit invisible."""
        if self.versions[asset] or asset not in self.pending_reworks:
            return
        current=self.initial_specs[asset]
        if current.change and current.change['field'] in self.variation_locks(asset):
            row=self.assets[asset]
            choices=alternatives(row,locked=self.variation_locks(asset))
            if not choices:
                self.pending_reworks.remove(asset);self.initial_specs[asset]=AssetSpec(row)
                self.rework_plan=[r for r in self.rework_plan if r['asset_id']!=asset]
                return
            current=alternate(row,AssetSpec(row),self.rng,locked=self.variation_locks(asset))
            self.initial_specs[asset]=current
            for entry in self.rework_plan:
                if entry['asset_id']==asset:
                    entry.update(field=current.change['field'],initial_value=current.change['after'],target_value=current.change['before'])

    def alternate(self, asset, base=None):
        row = self.assets[asset]
        return alternate(row, base or AssetSpec(row), self.rng, locked=self.variation_locks(asset))

    def compatible(self, module, asset):
        if module in ('direct', 'clarify'):
            return True
        return bool(alternatives(self.assets[asset], self.initial_specs[asset],
                    relative=module == 'relative_preview', locked=self.variation_locks(asset)))

    def add_media(self, kind, description, *, assets=(), shots=(), inputs=(),
                  parent=None, goal_match=False, purpose="asset", bindings=None, rule_role=None,
                  generation_requirements=None):
        if parent is not None and parent not in self.visible:
            raise ValueError("Edit source has not been shown")
        depth = 0 if parent is None else self.media[parent]["edit_depth"] + 1
        if depth > 2:
            raise ValueError("At most two successive edits per initial proposal")
        if not assets and purpose in ('compilation', 'final_export'):
            assets = list(dict.fromkeys(a for source in inputs for a in self.media[source]['asset_ids']))
        self.counts[kind] += 1
        token = f"[[{kind}_{self.counts[kind]:04d}]]"
        self.media[token] = dict(placeholder=token, kind=kind, description=str(description), base_description=str(description),
                                asset_ids=list(assets), shot_ids=list(shots),
                                inputs=list(inputs), edit_parent=parent, edit_depth=depth,
                                goal_match=goal_match, purpose=purpose,
                                source_frames={a: self.assets[a]["frames"] for a in assets})
        self.media[token]['generation_requirements'] = resolve_requirements(kind, purpose,
            [self.assets[aid]['kind'] for aid in assets],
            parent=self.media[parent]['generation_requirements'] if parent and kind == 'image' and self.media[parent]['kind'] == 'image' else None,
            requested=generation_requirements)
        if rule_role is not None:
            self.media[token]['rule_role'] = rule_role
        state, applied = self.rules.snapshot(self.media[token])
        self.media[token].update(render_state=state, applied_rules=applied,
            reference_render_states={p: deepcopy(self.media[p]['render_state'] or self.media[p].get('reference_render_states', {}))
                                     for p in inputs if self.media[p]['render_state'] or self.media[p].get('reference_render_states')})
        if bindings is not None:
            self.media[token]["bindings"] = dict(bindings)
        return token

    def emit(self, role, action, description, *, assets=(), shots=(), inputs=(),
             outputs=(), references=(), behaviors=(), decision=None, scope=None,
             generation_requirements=None):
        if any(p not in self.visible for p in (*inputs, *references)):
            raise ValueError("Turn refers to media from the future")
        if any(p in self.visible or p not in self.media for p in outputs):
            raise ValueError("Media must be introduced exactly once")
        if role == "user" and outputs:
            raise ValueError("The synthetic user does not supply hidden target media")
        reference_policy = {
            "ambiguous_reference": "ambiguous_subject",
            "ambiguous_shot_number": "ambiguous_shot_number",
            "request_reworks": "per_edit",
        }.get(action, "clear")
        intent = dict(action=action, module=self.module, description=description,
                      reference_policy=reference_policy,
                      behavior_ids=sorted(set(behaviors)), asset_ids=list(assets), shot_ids=list(shots),
                      asset_types={a: self.assets[a]["asset_type"] for a in assets})
        if decision is not None:
            intent["decision"] = decision
        if scope is not None:
            intent["scope"] = scope
        if generation_requirements is not None:
            intent['generation_requirements'] = deepcopy(generation_requirements)
        # Historical pointers are supervision. Actual delivery inputs are
        # finalized by the forward graph; only display_media is re-attached.
        if inputs:
            intent["context_media"] = list(inputs)
        if action in {"request_asset", "specify_goal", "request_trial", "begin_task", "request_candidates"}:
            intent["first_creation"] = all(not self.versions[a] for a in assets)
        if references:
            intent["display_media"] = list(references)
        self.turns.append(dict(id=len(self.turns) + 1, role=role, content=None,
                               intent=intent, input_media=list(dict.fromkeys((list(inputs) if outputs else []) + list(references))),
                               output_media=list(outputs)))
        for p in outputs:
            self.media[p]["introduced_at"] = len(self.turns)
        self.visible.update(outputs)

    def asset_media(self, asset, *, description=None, parent=None, kind="image", desired=True,
                    generation_requirements=None):
        spec = description or self.initial_specs[asset]
        if not isinstance(spec, AssetSpec):
            raise ValueError("Asset versions require structured attributes")
        if generation_requirements is None and self.turns:
            turn = self.turns[-1]
            if turn['role'] == 'user' and asset in turn['intent']['asset_ids']:
                generation_requirements = turn['intent'].get('generation_requirements')
        goal = desired and asset not in self.pending_reworks
        token = self.add_media(kind, spec, assets=[asset], inputs=[parent] if parent else [],
                               parent=parent, goal_match=goal, generation_requirements=generation_requirements)
        self.versions[asset].append(token)
        self.media[token]["asset_type"] = self.assets[asset]["asset_type"]
        self.media[token]["attributes"] = dict(spec.attributes)
        if spec.change:
            self.media[token]["variation"] = dict(type=spec.change["field"], phase="alternative",
                                                  before=spec.change["before"], after=spec.change["after"])
        if parent:
            self.media[token]["attribute_changes"] = changed_attributes(self.media[parent]["attributes"], spec.attributes)
        return token

    def adopt(self, asset, media, *, temporary=False, rollback=False, choice_required=False, contrast=None):
        explicit = temporary or rollback or choice_required or self.adoption_rng.random() >= 0.5
        self.selections.append(dict(asset_id=asset, target_media=media,
                                    decision="temporary" if temporary else "accepted",
                                    explicit=explicit, after_message=len(self.turns),
                                    choice_required=choice_required, rollback=rollback))
        self.ready[asset] = media
        if not explicit:
            return
        description = ("Go back to the earlier version: " + self.media[media]["description"] if rollback else
                       "Use this version for now while we explore." if temporary else "Let's use this version.")
        if choice_required:
            position = self.turns[-1]["output_media"].index(media) + 1
            description = ("For now, use" if temporary else "Choose") + f" option {position}: " + self.media[media]["description"]
        if contrast:
            description = f"Go back to the earlier {self.label(asset)} with {contrast['label']}: {contrast['value']}."
        self.emit("user", "rollback" if rollback else "adopt",
                  description,
                  assets=[asset], inputs=[media], behaviors=[15] + ([1] if rollback else []),
                  decision="temporary" if temporary else "accepted")
        if contrast:
            self.turns[-1]["intent"]["required_attribute"] = contrast
        self.emit("assistant", "confirm_selection", "Acknowledge the selected version.",
                  assets=[asset], inputs=[media], behaviors=[15] + ([1] if rollback else []))

    def single(self, asset, module):
        self.module = module
        self.refresh_initial_spec(asset)
        name, desired = self.label(asset), self.initial_specs[asset]
        if module == "direct":
            self.emit("user", "request_asset", f"Request the assistant to make a reference image of the {name}. {desired}.",
                      assets=[asset], behaviors=[5])
            token = self.asset_media(asset)
            self.emit("assistant", "deliver_asset", "Present the reference image.",
                      assets=[asset], outputs=[token], behaviors=[5])
            self.adopt(asset, token, temporary=self.rng.random() < 0.3)
        elif module == "candidates":
            kind = self.rng.choice(("image", "video"))
            alternate = self.alternate(asset, desired)
            desired = AssetSpec(self.assets[asset], desired.attributes, focus=[alternate.change["field"]])
            specs = [desired, alternate]
            self.rng.shuffle(specs)
            numbered = []
            for i, spec in enumerate(specs, 1):
                numbered.append(f"Option {i}: {spec}")
            self.emit("user", "request_candidates",
                      f"Request the assistant to make two {'reference images' if kind == 'image' else 'short clips'} of the {name} to compare."
                      + " " + "; ".join(numbered) + ".",
                      assets=[asset], behaviors=[4])
            outputs = [self.asset_media(asset, description=s, kind=kind,
                                       desired=s.attributes == desired.attributes) for s in specs]
            self.emit("assistant", "deliver_candidates", "Present both options and leave the choice to the user.",
                      assets=[asset], outputs=outputs, behaviors=[4])
            chosen = outputs[next(i for i, s in enumerate(specs) if s.attributes == desired.attributes)]
            self.adopt(asset, chosen, temporary=self.rng.random() < 0.5, choice_required=True)
        elif module == "trial_edit":
            trial = self.alternate(asset, desired)
            self.emit("user", "request_trial", f"Request the assistant to make a test image of the {name}. {trial}.", assets=[asset])
            before = self.asset_media(asset, description=trial, desired=False)
            self.emit("assistant", "deliver_trial", "Show the test image.", assets=[asset], outputs=[before])
            self.emit("user", "reject_and_edit", f"Set that one aside. {edit_request(self.assets[asset], trial, desired)}.",
                      assets=[asset], inputs=[before], behaviors=[15], decision="rejected")
            self.turns[-1]["intent"]["attribute_changes"] = changed_attributes(trial.attributes, desired.attributes)
            after = self.asset_media(asset, parent=before)
            self.emit("assistant", "deliver_edit", "Show the revised reference.", assets=[asset],
                      inputs=[before], outputs=[after])
            self.adopt(asset, after)
        elif module == "clarify":
            self.emit("user", "vague_request", f"I'd like to figure out the look of the {name}.",
                      assets=[asset], behaviors=[2])
            self.emit("assistant", "clarify_goal", f"What would you like the {name} to look like?",
                      assets=[asset], behaviors=[2])
            self.emit("user", "specify_goal", f"Request the assistant to make a reference image with this design: {desired}.",
                      assets=[asset], behaviors=[2])
            token = self.asset_media(asset)
            self.emit("assistant", "deliver_asset", "Show the reference image based on the clarified request.",
                      assets=[asset], outputs=[token], behaviors=[2])
            self.adopt(asset, token)
        elif module == "relative_preview":
            # The relative change is explicit and checkable, rather than asking
            # Qwen to invent what 'a bit lighter' should mean.
            variation, earlier, request, question = self.relative_variation(asset, desired)
            self.emit("user", "request_trial", f"Request the assistant to make an initial reference image of the {name}. {earlier}.", assets=[asset])
            before = self.asset_media(asset, description=earlier, desired=False)
            self.media[before]["variation"]["phase"] = "before"
            self.emit("assistant", "deliver_trial", "Show the image and wait for feedback.", assets=[asset], outputs=[before])
            self.emit("user", "relative_request", request,
                      assets=[asset], inputs=[before], behaviors=[12])
            self.turns[-1]["intent"]["variation"] = variation
            after = self.asset_media(asset, parent=before,
                                     description=AssetSpec(self.assets[asset], desired.attributes, focus=[variation]))
            self.media[after]["variation"] = dict(type=variation, phase="preview",
                                                  before=earlier.attributes[variation], after=desired.attributes[variation])
            self.emit("assistant", "preview_for_confirmation", question,
                      assets=[asset], inputs=[before], outputs=[after], behaviors=[12])
            self.turns[-1]["intent"]["variation"] = variation
            self.adopt(asset, after, temporary=True)
        else:
            raise ValueError(module)

    def relative_variation(self, asset, desired):
        return relative_variation(self.assets[asset], desired, self.variation_rng, locked=self.variation_locks(asset))

    def cross(self, first, second, module):
        self.refresh_initial_spec(first);self.refresh_initial_spec(second)
        self.module = module
        if module == "task_switch":
            draft = self.alternate(first, self.initial_specs[first])
            self.emit("user", "begin_task", f"Request the assistant to make an initial test image of the {self.label(first)}. {draft}.", assets=[first])
            old = self.asset_media(first, description=draft, desired=False)
            self.emit("assistant", "deliver_trial", "Show the first test image, still a draft.", assets=[first], outputs=[old])
            self.emit("user", "suspend_task", f"Put the {self.label(first)} aside for now and switch to the {self.label(second)}.",
                      assets=[first, second], behaviors=[5, 9])
            self.emit("assistant", "save_task", "Confirm that the draft will be kept while switching tasks.", assets=[first, second], behaviors=[9])
            self.single(second, self.rng.choice([m for m in SINGLE_MODULES if self.compatible(m, second)]))
            self.module = module
            self.emit("user", "resume_task", f"Back to the {self.label(first)}. In the earlier test image: "
                      + edit_request(self.assets[first], draft, self.initial_specs[first]) + ".",
                      assets=[first], inputs=[old], behaviors=[9])
            self.turns[-1]["intent"]["attribute_changes"] = changed_attributes(draft.attributes, self.initial_specs[first].attributes)
            after = self.asset_media(first, parent=old)
            self.emit("assistant", "deliver_resumed_task", "Confirm that work on the earlier draft resumed and the edit is ready.",
                      assets=[first], inputs=[old], outputs=[after], behaviors=[9])
            self.adopt(first, after)
        elif module == "temporary_rollback":
            base_request = len(self.turns)
            self.single(first, "direct")
            base = self.ready[first]
            self.module = module
            self.emit("user", "switch_for_trial", f"Let's prep the {self.label(second)} first. We'll put it with the {self.label(first)} in a test shot later.",
                      assets=[first, second], behaviors=[5, 9])
            self.emit("assistant", "confirm_switch", f"What should the {self.label(second)} look like?",
                      assets=[second], behaviors=[9])
            self.single(second, "direct")
            self.module = module
            alternate = self.alternate(first, self.initial_specs[first])
            field = alternate.change["field"]
            before, after = alternate.change["before"], alternate.change["after"]
            label = CATALOG["categories"][self.assets[first]["kind"]]["attributes"][field]["label"].lower()
            original = dict(field=field, label=label, value=before, alternative=after)
            revision = dict(field=field, label=label, value=after, alternative=before)
            base_spec = AssetSpec(self.assets[first], self.media[base]["attributes"], focus=[field])
            self.media[base]["description"] = str(base_spec)
            self.media[base]["base_description"] = str(base_spec)
            self.turns[base_request]["intent"].update(
                description=f"Request the assistant to make a reference image of the {self.label(first)}. {base_spec}.",
                required_attribute=original,
            )
            self.emit("user", "request_variant",
                      f"Try the {self.reference_label(first)} with a different {label}: {after}.",
                      assets=[first], inputs=[base], behaviors=[15])
            self.turns[-1]["intent"].update(required_attribute=revision, attribute_changes={field: dict(before=before, after=after)})
            alternative = self.asset_media(first, description=alternate,
                                           parent=base, desired=False)
            self.emit("assistant", "deliver_variant",
                      f"Here's the test version with {label}: {after}.", assets=[first],
                      inputs=[base], outputs=[alternative])
            self.turns[-1]["intent"]["required_attribute"] = revision
            refs = [alternative, self.ready[second]]
            self.emit("user", "temporary_adoption_trial", f"Use the {after} version of the {self.label(first)} we just made for now. Put it together with the {self.label(second)} in a short test clip showing a plausible action or interaction between them.",
                      assets=[first, second], inputs=refs, behaviors=[10, 15], decision="temporary")
            self.turns[-1]["intent"]["required_attribute"] = revision
            trial = self.add_media("video", f"Short trial of a plausible action or relationship between {self.media[alternative]['description']} and the {self.label(second)}. Keep the selected designs consistent throughout the action.",
                                   assets=[first, second], inputs=refs, purpose="exploration")
            self.emit("assistant", "deliver_exploration", "Present the clip as a test.",
                      assets=[first, second], inputs=refs, outputs=[trial], behaviors=[15])
            self.adopt(first, base, rollback=True, contrast=original)
        else:
            raise ValueError(module)

    def ambiguity(self, asset, target):
        other = [p for p in self.visible if p != target and self.media[p]["purpose"] == "asset"]
        if not other:
            return
        # Sort set-backed candidates before sampling to keep cross-process seeds stable.
        wrong = self.rng.choice(sorted(other))
        self.emit("user", "ambiguous_reference", "I'd like to change that thing in one of the earlier images.",
                  assets=[asset], behaviors=[3, 8])
        self.turns[-1]["intent"]["target_media"] = target  # Oracle label, not a visible attachment.
        appearance = self.version_label(wrong, target)
        self.emit("assistant", "clarify_reference", "Are you referring to this version: "
                  + appearance + "?",
                  assets=self.media[wrong]["asset_ids"], references=[wrong], behaviors=[3, 8])
        self.emit("user", "correct_reference", "No, I mean the earlier " + self.version_label(target, wrong) + ".",
                  assets=[asset], inputs=[target], behaviors=[8])
        shown_asset = self.media[wrong]["asset_ids"][0]
        self.turns[-1]["intent"].update(
            description=f"No, you showed the {appearance}. I meant the earlier {self.version_label(target, wrong)}.",
            reference_correction=dict(shown_media=wrong, target_media=target,
                                      shown_name=self.label(shown_asset), target_name=self.label(asset)))
        self.emit("assistant", "confirm_reference", "Acknowledge that the user meant the earlier " + self.label(asset) + "; ask what they want to change about it. This is referent clarification, not version adoption.",
                  assets=[asset], inputs=[target], behaviors=[8])

    def revision_target(self, asset, old):
        from scripts.dialogue_generation.appearance_policy import COLOR_FIELDS
        before=AssetSpec(self.assets[asset],self.media[old]['attributes'])
        target=AssetSpec(self.assets[asset])
        locks=set(self.variation_locks(asset))
        uses=[sid for sid in self.rendered if asset in self.shots[sid]['assets']]
        if uses and all(self.rules.snapshot(dict(kind='video',purpose='shot',asset_ids=self.shots[sid]['assets']))[0].get('color_mode')=='black_and_white' for sid in uses):
            locks.update(COLOR_FIELDS)
        changes=changed_attributes(before.attributes,target.attributes)
        if changes and set(changes)<=locks:
            if not alternatives(self.assets[asset],before,locked=locks):return None
            return alternate(self.assets[asset],before,self.rng,locked=locks)
        return target

    def revision_preview(self, asset, old, target=None):
        before = AssetSpec(self.assets[asset], self.media[old]['attributes'])
        target = target or AssetSpec(self.assets[asset])
        changed = changed_attributes(before.attributes, target.attributes)
        new = self.asset_media(asset, description=AssetSpec(self.assets[asset], target.attributes, focus=list(changed)), parent=old)
        self.media[new]['goal_match'] = target.attributes == self.assets[asset]['attributes']
        return new

    def accept_revision(self, edit):
        asset, new = edit['asset_id'], edit['new_media']
        targets = [edit['target_shot']] if edit['scope'] == 'shot' else edit['rendered_uses']
        self.apply_version(asset, new, targets)
        if edit['scope'] == 'all_uses':
            self.ready[asset] = new
        self.pending_reworks.remove(asset)
        edit['applied_shots'] = list(targets)

    def deliver_revision(self, edits, *, images=()):
        clips = self.render_due(automatic=True)
        cut = self.refresh_compilation(automatic=True)
        outputs = list(images) + clips + ([cut] if cut else [])
        self.module = 'batch_revision' if len(edits) > 1 else 'later_revision'
        self.emit('assistant', 'deliver_reworked_clips',
                  'The affected video clips are updated within the confirmed scope. Present every updated clip together.'
                  + (' The existing cut also contains these replacements in the same order.' if cut else ''),
                  assets=[e['asset_id'] for e in edits], shots=list(dict.fromkeys(s for e in edits for s in e['applied_shots'])),
                  outputs=outputs, behaviors=[7, 10])
        self.turns[-1]['intent'].update(resolved_edits=deepcopy(edits), updated_clips=clips,
                                       preview_media=list(images), updated_compilation=cut)

    def rework(self, asset, target=None):
        self.module = 'later_revision'
        old = self.ready[asset]
        target = target or self.revision_target(asset,old)
        if target is None:
            self.pending_reworks.remove(asset)
            return
        if self.extra and self.rng.random() < 0.7:
            self.ambiguity(asset, old)
        uses = [s for s in self.shots if asset in self.shots[s]['assets']]
        rendered = [s for s in uses if s in self.rendered]
        self.events.append(dict(module=self.module, asset_ids=[asset], rendered_uses=rendered))
        first = self.rng.choice(rendered)
        def reference(shot):
            if shot in self.comp_order:
                return f'shot {self.comp_order.index(shot) + 1} in the current cut'
            return 'the existing clip, “' + self.shot_label(shot) + '”'
        before = AssetSpec(self.assets[asset], self.media[old]['attributes'])
        self.emit('user', 'request_rework', f'In {reference(first)}: ' + edit_request(self.assets[asset], before, target) + '.',
                  assets=[asset], shots=[first], inputs=[self.rendered[first]], behaviors=[7])
        self.turns[-1]['intent']['attribute_changes'] = changed_attributes(before.attributes, target.attributes)
        new = self.revision_preview(asset, old, target)
        scope = 'all_uses' if len(uses) == 1 else 'shot'
        images = [new]
        if len(rendered) > 1:
            others = ', '.join(reference(s) for s in rendered if s != first)
            self.emit('assistant', 'clarify_update_scope',
                f'Here is the revised {self.label(asset)} reference preview. The videos have not been changed yet. '
                f'The same {self.label(asset)} also appears in {others}. Update just the mentioned shot, or all existing shots using it?',
                assets=[asset], shots=rendered, inputs=[old], outputs=[new], behaviors=[7])
            self.turns[-1]['intent']['preview_media'] = [new]
            scope = 'shot' if self.rng.random() < .5 else 'all_uses'
            self.emit('user', 'choose_update_scope', 'Change just the shot I mentioned.' if scope == 'shot' else 'Update all the existing shots that use it.',
                assets=[asset], shots=[first] if scope == 'shot' else rendered, behaviors=[7], scope=scope)
            images = []
        edit = dict(index=1, asset_id=asset, target_media=old, target_shot=first, uses=uses,
                    rendered_uses=rendered, scope=scope, new_media=new,
                    changes=changed_attributes(before.attributes, target.attributes), target_attributes=dict(target.attributes))
        self.accept_revision(edit)
        self.deliver_revision([edit], images=images)

    def apply_version(self, asset, version, shots):
        for shot in shots:
            self.overrides[shot][asset] = version
            if shot in self.rendered and self.bindings[shot][asset] != version:
                self.dirty.add(shot)

    def rework_remaining(self):
        """Resolve unclear identities first; show known asset previews with scope questions."""
        assets = [a for a in self.assets if a in self.pending_reworks]
        targets={a:self.revision_target(a,self.ready[a]) for a in assets}
        for asset in assets:
            if targets[asset] is None:self.pending_reworks.remove(asset)
        assets=[a for a in assets if targets[a] is not None]
        if len(assets) < 2:
            for asset in assets: self.rework(asset,targets[asset])
            return
        self.module = 'batch_revision'
        edits, requests, questions, answers, references, queries = [], [], [], [], [], []
        for index, asset in enumerate(assets, 1):
            old = self.ready[asset]
            uses = [s for s in self.shots if asset in self.shots[s]['assets']]
            rendered = [s for s in uses if s in self.rendered]
            shot = self.rng.choice(rendered)
            where = (f'shot {self.comp_order.index(shot)+1} in the current cut' if shot in self.comp_order else
                     f'shot {self.render_order.index(shot)+1} in generation order')
            target = targets[asset]; before = AssetSpec(self.assets[asset], self.media[old]['attributes'])
            changes = changed_attributes(before.attributes, target.attributes)
            others = sorted(p for p in self.visible if p != old and self.media[p]['purpose']=='asset'
                            and (self.media[p]['asset_ids'] != [asset] or self.media[p]['attributes'] != before.attributes))
            ambiguous = bool(self.extra and others and self.rng.random()<.7)
            scope_required = len(rendered)>1
            scope = ('shot' if self.rng.random()<.5 else 'all_uses') if scope_required else ('all_uses' if len(uses)==1 else 'shot')
            edit = dict(index=index,asset_id=asset,target_media=old,target_shot=shot,uses=uses,
                        rendered_uses=rendered,changes=changes,target_attributes=dict(target.attributes),ambiguous=ambiguous,scope_required=scope_required,
                        reference_basis='current_compilation' if shot in self.comp_order else 'generation_order',
                        reference_position=self.comp_order.index(shot)+1 if shot in self.comp_order else self.render_order.index(shot)+1)
            edits.append(edit)
            if ambiguous:
                change_text=', '.join(fields(self.assets[asset])[k]['label']+' to '+v['after'] for k,v in changes.items())
                requests.append(f'Edit {index}: In {where}, change that thing from the earlier image: '+change_text)
                candidates=[old,self.rng.choice(others)]; self.rng.shuffle(candidates)
                questions.append(f'Edit {index}: did you mean {self.version_label(candidates[0],old)} or {self.version_label(candidates[1],old)}?')
                references.extend(candidates);queries.append(dict(edit_index=index,kind='reference',candidate_media=candidates))
                answers.append(f'Edit {index}: I mean {self.version_label(old,next(p for p in candidates if p != old))}')
            else:
                requests.append(f'Edit {index}: In {where}, '+edit_request(self.assets[asset],before,target))
            if scope_required:
                questions.append(f'Edit {index}: update just the mentioned shot, or all existing shots that use it?')
                queries.append(dict(edit_index=index,kind='scope'))
                answers.append(f'Edit {index}: '+('just the mentioned shot' if scope=='shot' else 'all existing shots using it'))
            edit['resolved_scope']=scope
        scopes={e['index']:e.pop('resolved_scope') for e in edits}
        self.emit('user','request_reworks','Make these edits together: '+'; '.join(requests)+'.',
                  assets=assets,shots=[e['target_shot'] for e in edits],behaviors=[7]+([3] if any(e['ambiguous'] for e in edits) else []))
        self.turns[-1]['intent']['edits']=deepcopy(edits)
        previewable=not any(e['ambiguous'] for e in edits)
        previews={e['index']:self.revision_preview(e['asset_id'],e['target_media'],targets[e['asset_id']]) for e in edits} if previewable else {}
        if questions:
            self.emit('assistant','clarify_reworks',
                ('Here are the revised reference previews; the videos are unchanged. ' if previewable else '')+' '.join(questions),
                assets=assets,references=list(dict.fromkeys(references)),outputs=list(previews.values()),behaviors=[7]+([3,8] if references else []))
            self.turns[-1]['intent'].update(edits=deepcopy(edits),clarifications=queries,preview_media=list(previews.values()))
            self.emit('user','resolve_reworks','; '.join(answers)+'.',assets=assets,behaviors=[7]+([8] if references else []))
            self.turns[-1]['intent']['resolved_edits']=[dict(e,scope=scopes[e['index']]) for e in edits]
        resolved=[]; images=[]
        for edit in edits:
            new=previews.get(edit['index']) or self.revision_preview(edit['asset_id'],edit['target_media'],targets[edit['asset_id']])
            if not questions or not previewable: images.append(new)
            item=dict(edit,scope=scopes[edit['index']],new_media=new)
            self.accept_revision(item);resolved.append(item)
        self.deliver_revision(resolved,images=images)
        self.events.append(dict(module='batch_revision',asset_ids=assets,edits=deepcopy(resolved)))

    def version_label(self, token, other):
        """Short visible labels; distinguish historical versions of one asset."""
        row, alternative = self.media[token], self.media[other]
        asset = row["asset_ids"][0]
        if row["asset_ids"] != alternative["asset_ids"]:
            return self.label(asset)
        differing = [key for key, value in row["attributes"].items() if value != alternative["attributes"][key]]
        if not differing:
            return self.label(asset)
        key = differing[0]
        label = CATALOG["categories"][self.assets[asset]["kind"]]["attributes"][key]["label"]
        return f"{self.label(asset)} ({label.lower()}: {row['attributes'][key]})"

    def render_due(self, *, automatic=False):
        outputs = []
        for shot, row in self.shots.items():
            if automatic and shot not in self.rendered:
                continue
            if any(a not in self.ready for a in row["assets"]):
                continue
            if shot in self.rendered and shot not in self.dirty:
                continue
            self.module = "render_shot" if shot not in self.rendered else "update_shot"
            new_shot = shot not in self.rendered
            binding = {a: self.overrides[shot].get(a, self.ready[a]) for a in row["assets"]}
            refs = list(binding.values())
            appearance_update = (dict(before=self.media[self.rendered[shot]]['render_state'], after=self.rules.snapshot(dict(kind='video', purpose='shot', asset_ids=row['assets']))[0])
                                 if not new_shot and self.media[self.rendered[shot]]['render_state'] else None)
            if appearance_update and appearance_update['before'] == appearance_update['after']:
                appearance_update = None
            descriptions = ", ".join(str(AssetSpec(self.assets[a], self.media[binding[a]]["attributes"], max_fields=2))
                                      for a in row["assets"])
            if not automatic and self.extra and not self.conflict_done and self.rng.random() < 0.3:
                self.conflict_done = True
                kinds = {self.assets[a]["kind"] for a in row["assets"]}
                eligible = [key for key, spec in CONFLICTS.items()
                            if not spec.get("requires_any") or kinds.intersection(spec["requires_any"])]
                conflict_type = self.conflict_rng.choice(eligible)
                conflict = CONFLICTS[conflict_type]
                self.emit("user", "incompatible_framing" if conflict_type == "framing" else "incompatible_requirements", ("Request a NEW video shot with these requirements: " if new_shot else "Edit this existing video shot with these requirements: ") + conflict["request"],
                          shots=[shot], behaviors=[14])
                self.turns[-1]["intent"].update(conflict_type=conflict_type, generation_mode="new_video" if new_shot else "edit_video")
                self.emit("assistant", "clarify_conflict", conflict["question"],
                          shots=[shot], behaviors=[14])
                self.turns[-1]["intent"].update(conflict_type=conflict_type, generation_mode="new_video" if new_shot else "edit_video")
                resolution = {
                    "shot_structure": "Drop the hard cuts. Make a single uninterrupted take",
                    "camera_motion": "Drop that fixed-camera/orbit combination. Use this camera movement instead",
                    "framing": "Use this framing instead",
                    "depth_of_field": "Drop those conflicting background requirements. Make this shot instead",
                    "aspect_ratio": "Use one consistent format. Make this shot",
                }[conflict_type]
                self.emit("user", "resolve_conflict", f"{resolution}: {row['description']}.",
                          assets=row["assets"], shots=[shot], inputs=refs, behaviors=[10, 14])
                self.turns[-1]["intent"].update(conflict_type=conflict_type, generation_mode="new_video" if new_shot else "edit_video")
            elif not automatic:
                updates = ({a: dict(before=self.bindings[shot][a], after=binding[a])
                            for a in row["assets"] if self.bindings[shot][a] != binding[a]} if not new_shot else {})
                revised = " and ".join(self.label(a) for a in updates)
                self.emit("user", "request_shot" if new_shot else "request_shot_update",
                          (f"Request the assistant to make this video shot using the {descriptions} we made earlier: {row['description']}." if new_shot else
                           f"Request the assistant to update this existing clip using the revised {revised} reference: {row['description']}."),
                          assets=row["assets"], shots=[shot], inputs=refs, behaviors=[5, 10])
                if not new_shot:
                    self.turns[-1]["intent"]["reference_updates"] = updates
                if appearance_update:
                    self.turns[-1]['intent']['render_update'] = appearance_update
                    self.turns[-1]['intent']['description'] = f'Update the existing video shot to the current agreed rendering requirements and references. Preserve its action and composition: {row["description"]}.'
            if not automatic:
                self.turns[-1]["intent"].update(shot_description=row["description"], generation_mode="new_video" if new_shot else "edit_video")
            target = all(self.media[p]["goal_match"] for p in refs)
            spec = row["description"] + ". References used: " + "; ".join(self.media[p]["description"] for p in refs)
            token = self.add_media("video", spec, assets=row["assets"], shots=[shot], inputs=refs,
                                   goal_match=target, purpose="shot", bindings=binding)
            outputs.append(token)
            if not automatic:
                self.emit("assistant", "deliver_shot", "Present the shot without adding other changes.",
                          assets=row["assets"], shots=[shot], inputs=refs, outputs=[token], behaviors=[10])
            self.rendered[shot], self.bindings[shot] = token, binding
            self.dirty.discard(shot)
            if new_shot:
                self.render_order.append(shot)
            if not automatic and self.comp_media and shot in self.comp_order:
                self.refresh_compilation()
            if not automatic and new_shot and len(self.rendered) >= 2 and self.rng.random() < self.p_compile:
                self.compile_preview()

        return outputs

    def compilation(self, order, action, *, specified=True, final=False):
        self.module = "final_order" if final else "compilation"
        refs = [self.rendered[s] for s in order]
        contents = " → ".join(self.shot_label(s) for s in order)
        if final and self.extra and self.comp_media and set(self.comp_order) == set(order):
            original = [s for s in self.render_order if s in self.comp_order]
            pairs = [(i, self.comp_order.index(wanted)) for i, wanted in enumerate(order)
                     if self.comp_order[i] != wanted]
            pairs = [(i, j) for i, j in pairs
                     if self.comp_order[i] != original[i] or self.comp_order[j] != original[j]]
            if pairs:
                i, j = pairs[0]
                # Both operations improve the target ordering. The action is
                # explicit; only the numbering convention needs clarification.
                move = i > 0 and j > i and self.numbering_rng.random() < 0.5
                # Moving names the preceding anchor (i - 1), rather than
                # destination i. Keep a swap if neither named position is
                # ambiguous under the two numbering conventions.
                if move and all(self.comp_order[k] == original[k] for k in (j, i - 1)):
                    move = False
                operation = dict(type="move_after" if move else "swap",
                                 positions=[j + 1, i] if move else [i + 1, j + 1])
                a, b = operation["positions"]
                instruction = (f"Move shot {a} after shot {b}." if move else
                               f"Swap shot {a} and shot {b}.")
                ambiguous = next(k for k in (a - 1, b - 1) if self.comp_order[k] != original[k])
                self.emit("user", "ambiguous_shot_number", instruction,
                          shots=list(self.comp_order), behaviors=[11])
                self.turns[-1]["intent"]["numbering_operation"] = operation
                self.emit("assistant", "clarify_shot_number",
                          f"Do you mean shot {ambiguous + 1} in generation order, “{self.shot_label(original[ambiguous])},”"
                          f" or in the current cut, “{self.shot_label(self.comp_order[ambiguous])}”?",
                          shots=list(self.comp_order), references=[self.comp_media], behaviors=[11])
                self.emit("user", "resolve_shot_number", "Use the numbering in the current cut. " + instruction,
                          shots=list(self.comp_order), behaviors=[11])
                self.turns[-1]["intent"].update(numbering_operation=operation, numbering_basis="current")
                updated = apply_numbering(self.comp_order, operation)
                self.deliver_compilation(updated, final=updated == list(order), operation=operation)
                if updated == list(order):
                    return
        self.emit("user", action,
                  ("Reorder the existing shots like this: " + contents if final else
                   "Put the clips together for a preview in this order: " + contents if specified else
                   "Add the new shot at the end of the last preview and keep the earlier order." if self.comp_media else
                   "Can you put those shots together so I can see how they play?"),
                  shots=order, inputs=refs, behaviors=[6] + ([11] if final else []))
        self.deliver_compilation(order, final=final)

    def deliver_compilation(self, order, *, final=False, operation=None):
        refs = [self.rendered[s] for s in order]
        contents = " → ".join(self.shot_label(s) for s in order)
        token = self.add_media("video", "Edited video in this shot order: " + contents, shots=order,
                               inputs=refs, goal_match=final, purpose="compilation")
        self.media[token]["shot_versions"] = dict(zip(order, refs))
        if operation:
            self.media[token]["numbering_operation"] = operation
            self.media[token]["previous_compilation"] = self.comp_media
        self.emit("assistant", "deliver_compilation", "Present the clips in the requested order without changing the individual shots.",
                  shots=order, inputs=refs, outputs=[token], behaviors=[6])
        self.comp_order, self.comp_media = list(order), token
        self.comp_snapshot = dict(zip(order, refs))

    def compile_preview(self):
        order = list(self.comp_order) + [s for s in self.render_order if s not in self.comp_order]
        if order == self.comp_order:
            return
        specified = self.rng.random() < 0.6
        if specified:
            self.rng.shuffle(order)
            if order == [s for s in self.render_order if s in order]:
                order = order[1:] + order[:1]
        self.compilation(order, "request_compilation", specified=specified)

    def refresh_compilation(self, *, automatic=False):
        changed = [s for s in self.comp_order if self.comp_snapshot[s] != self.rendered[s]]
        if not changed:
            return
        self.module = "refresh_compilation"
        order = list(self.comp_order)
        refs = [self.rendered[s] for s in order]
        if not automatic:
            self.emit("user", "replace_compilation_shots", "Replace the updated shots in the last preview. Keep everything else where it is.",
                      shots=changed, inputs=[self.comp_media] + [self.rendered[s] for s in changed], behaviors=[7])
        token = self.add_media("video", "Preview with replacement shots in the same order: " + " → ".join(self.shot_label(s) for s in order),
                               shots=order, inputs=refs, purpose="compilation")
        self.media[token]["shot_versions"] = dict(zip(order, refs))
        if not automatic:
            self.emit("assistant", "deliver_refreshed_compilation", "Show the preview with the agreed replacements and original order. The earlier cut remains available.",
                      shots=order, inputs=refs, outputs=[token], behaviors=[7])
        self.comp_media, self.comp_snapshot = token, dict(zip(order, refs))
        return token

    def run(self):
        self._rules_started = True
        for rule in self.source['constraints']:
            self.rules.start(rule)
        while len(self.ready) < len(self.assets):
            self.rules.advance()
            missing = [a for a in self.assets if a not in self.ready]
            compatible = {m: [a for a in missing if self.compatible(m, a)] for m in self.modules}
            legal = [m for m in self.modules if compatible[m] and (m in SINGLE_MODULES or len(missing) >= 2)]
            # A cross-only request still needs a final one-asset completion.
            module = self.rng.choice(legal or ["direct"])
            first = self.rng.choice(compatible[module] if legal else missing)
            selected = [first]
            if module in CROSS_MODULES:
                second = self.rng.choice([a for a in missing if a != first])
                selected.append(second)
                self.cross(first, second, module)
            else:
                self.single(first, module)
            self.events.append(dict(module=module, asset_ids=selected))
            self.rules.advance()
            self.render_due()
            possible = [a for a in self.assets if a in self.pending_reworks and a in self.ready
                        and any(a in self.shots[s]["assets"] for s in self.rendered)]
            if possible and self.rng.random() < 0.6:
                self.rework(self.rng.choice(possible))
                self.render_due()
        # Planned closure, rather than a random late request to magically repair
        # an arbitrary conversation. Every reserved rework has a known exit.
        self.rework_remaining()
        self.render_due()
        # Final acceptance keeps the user's current rules and realized designs.
        self.rules.finish()
        goal = self.source["target_order"]
        if len(goal) == 1:
            final = self.rendered[goal[0]]
        elif self.comp_order != goal:
            self.compilation(goal, "reorder_shots" if self.comp_media else "request_compilation", final=True)
            final = self.comp_media
        else:
            final = self.comp_media
            self.media[final]["goal_match"] = True
        if final != next(reversed(self.media)):
            self.module = "final_export"
            self.emit("user", "request_final_export", "Export the current cut as the final video, keeping its shots and order unchanged.",
                      shots=goal, inputs=[final])
            exported = self.add_media("video", self.media[final]["description"], shots=goal,
                                      inputs=[final], goal_match=True, purpose="final_export")
            self.emit("assistant", "deliver_final_export", "Here's the final video with the same shots and order.",
                      shots=goal, inputs=[final], outputs=[exported])
            final = exported
        self.module = "accept_project"
        self.emit("user", "accept_project", "Let's make this the final cut.", shots=goal,
                  inputs=[final], behaviors=[15], decision="accepted")
        self.emit("assistant", "confirm_final", "Confirm the final cut and its reference versions are saved.", shots=goal, inputs=[final], behaviors=[15])
        # Add this after planning: some modules refine an earlier request's
        # description while constructing the version contrast.
        self.turns[0]["intent"]["description"] = "A fresh conversation starts. " + self.turns[0]["intent"]["description"]
        self.rules.decorate()
        result = dict(schema_version=5, rules=deepcopy(self.rules.records), attribute_schema_version=CATALOG["schema_version"],
                      rework_plan=self.rework_plan,
                      target_assets={a: dict(id=a, name=row['name'], kind=row["kind"], description=row["description"],
                                              attributes=row["attributes"],
                                              **({"attribute_extensions": deepcopy(row["attribute_extensions"])} if row.get("attribute_extensions") else {})) for a, row in self.assets.items()},
                      seed=self.seed, source_video=self.source.get("video", ""),
                      media_generated=False, target_order=goal, events=self.events,
                      selections=self.selections,
                      turns=self.turns, placeholders=list(self.media.values()),
                      behavior_coverage=sorted({b for t in self.turns for b in t["intent"]["behavior_ids"]}),
                      behavior_labels={str(k): v for k, v in BEHAVIORS.items()},
                      final=dict(media=final, order=goal, assets=dict(self.ready),
                                 shots={s: dict(media=self.rendered[s], bindings=self.bindings[s]) for s in self.shots}))
        assign_reference_materials(result)
        validate_plan(result)
        return result


def validate_plan(plan):
    """Check the generated data contract without asking an LLM to judge it."""
    media = {m["placeholder"]: m for m in plan["placeholders"]}
    if len(media) != len(plan["placeholders"]):
        raise ValueError("Duplicate placeholder")
    seen = set()
    latest_shots = {}
    pending_numbering = resolved_numbering = None
    pending_batch = None
    for turn in plan["turns"]:
        intent = turn["intent"]
        action = intent["action"]
        if action == "request_reworks":
            edits = intent["edits"]
            if pending_batch or [e["index"] for e in edits] != list(range(1, len(edits) + 1)) or len({e["asset_id"] for e in edits}) != len(edits):
                raise ValueError("Invalid batch edit request")
            for edit in edits:
                if edit["target_media"] not in seen:
                    raise ValueError("Future/unknown batch referent")
                parent = media[edit["target_media"]]
                target = plan["target_assets"][edit["asset_id"]]
                expected_changes = {key: dict(before=parent["attributes"][key], after=value)
                                    for key, value in edit.get("target_attributes", target["attributes"]).items() if value != parent["attributes"][key]}
                if not expected_changes or edit["changes"] != expected_changes or edit["target_shot"] not in edit["rendered_uses"]:
                    raise ValueError("Incorrect batch edit target")
            required = {(e["index"], kind) for e in edits for kind, enabled in
                        (("reference", e["ambiguous"]), ("scope", e["scope_required"])) if enabled}
            pending_batch = dict(edits=edits, required=required, asked=False, resolved={} if required else
                                 {e["index"]: "all_uses" for e in edits})
        if action == "clarify_reworks":
            if not pending_batch or pending_batch["asked"]:
                raise ValueError("Unexpected batch clarification")
            questions = intent["clarifications"]
            if {(q["edit_index"], q["kind"]) for q in questions} != pending_batch["required"]:
                raise ValueError("Batch clarification omitted an ambiguity or scope")
            for query in questions:
                if query["kind"] == "reference" and any(p not in seen for p in query["candidate_media"]):
                    raise ValueError("Future/unknown batch clarification reference")
            pending_batch["asked"] = True
        if action == "resolve_reworks":
            if not pending_batch or not pending_batch["asked"]:
                raise ValueError("Batch resolution without clarification")
            answers = intent["resolved_edits"]
            if len(answers) != len(pending_batch["edits"]):
                raise ValueError("Batch resolution omitted an edit")
            for edit, answer in zip(pending_batch["edits"], answers):
                if any(answer.get(k) != v for k, v in edit.items()) or answer.get("scope") not in ("shot", "all_uses"):
                    raise ValueError("Batch resolution changed its edit target")
            pending_batch["resolved"] = {e["index"]: e["scope"] for e in answers}
        if action == 'clarify_reworks' and turn['output_media']:
            if any(e['ambiguous'] for e in pending_batch['edits']):
                raise ValueError('Asset preview generated before its reference was resolved')
            if len(turn['output_media']) != len(pending_batch['edits']):
                raise ValueError('Batch preview omitted an edit')
            for edit, token in zip(pending_batch['edits'],turn['output_media']):
                if media[token]['purpose']!='asset' or media[token]['edit_parent']!=edit['target_media']:
                    raise ValueError('Batch preview changed its edit target')
        if action == 'deliver_reworked_clips':
            delivered = intent['resolved_edits']
            if pending_batch:
                if not pending_batch['resolved']:
                    raise ValueError('Batch generated before all clarifications were resolved')
                if len(delivered)!=len(pending_batch['edits']):
                    raise ValueError('Batch delivery omitted an edit')
                for edit, answer in zip(pending_batch['edits'],delivered):
                    if any(answer.get(k)!=v for k,v in edit.items()) or answer['scope']!=pending_batch['resolved'][edit['index']]:
                        raise ValueError('Batch delivery changed its target or confirmed scope')
                pending_batch=None
            expected_shots=set()
            for edit in delivered:
                new=media[edit['new_media']]
                if new['edit_parent']!=edit['target_media'] or new['asset_ids']!=[edit['asset_id']]:
                    raise ValueError('Revision preview has the wrong parent or subject')
                wanted=[edit['target_shot']] if edit['scope']=='shot' else edit['rendered_uses']
                if edit['applied_shots']!=wanted:
                    raise ValueError('Revision changed its confirmed shot scope')
                expected_shots.update(wanted)
            clips=intent['updated_clips']
            if set(s for token in clips for s in media[token]['shot_ids'])!=expected_shots or len(clips)!=len(expected_shots):
                raise ValueError('Revision delivery omitted or duplicated an affected clip')
            for token in clips:
                row=media[token];sid=row['shot_ids'][0]
                for edit in delivered:
                    if sid in edit['applied_shots'] and row['bindings'][edit['asset_id']]!=edit['new_media']:
                        raise ValueError('Updated clip does not use its confirmed revision')
        if intent["action"] == "ambiguous_shot_number" and "numbering_operation" in intent:
            pending_numbering = intent["numbering_operation"]
        if intent["action"] == "resolve_shot_number":
            if pending_numbering != intent.get("numbering_operation") or intent.get("numbering_basis") != "current":
                raise ValueError("Numbering clarification does not resolve the pending operation")
            resolved_numbering, pending_numbering = pending_numbering, None
        pointers = (turn["input_media"] + turn["intent"].get("display_media", [])
                    + turn["intent"].get("context_media", []))
        correction = intent.get("reference_correction")
        if correction:
            pointers += [correction["shown_media"], correction["target_media"]]
        if any(p not in seen for p in pointers):
            raise ValueError("Future/unknown media reference")
        if action == "request_shot_update":
            shot_ids = intent["shot_ids"]
            if len(shot_ids) != 1 or shot_ids[0] not in latest_shots:
                raise ValueError("Shot update has no existing clip")
            before = latest_shots[shot_ids[0]]["bindings"]
            after = {a: token for token in intent["context_media"] for a in media[token]["asset_ids"]}
            expected = {a: dict(before=before[a], after=after[a]) for a in before if before[a] != after[a]}
            render_update = intent.get('render_update')
            if render_update:
                before_state = latest_shots[shot_ids[0]].get('render_state', {})
                if render_update.get('before') != before_state or render_update.get('after') == before_state or not isinstance(render_update.get('after'), dict):
                    raise ValueError('Incorrect render-state restoration')
            if (not expected and not render_update) or intent.get("reference_updates") != expected:
                raise ValueError("Shot update does not identify its revised references")
        if turn["intent"].get("target_media") is not None and turn["intent"]["target_media"] not in seen:
            raise ValueError("Future/unknown oracle referent")
        for p in turn["output_media"]:
            if pending_batch and not pending_batch["resolved"] and action != "clarify_reworks":
                raise ValueError("Media generated before all batch clarifications were resolved")
            if pending_numbering is not None:
                raise ValueError("Media generated before numbering clarification")
            if p in seen or p not in media:
                raise ValueError("Invalid media introduction")
            row = media[p]
            if 'generation_requirements' in row:
                requirements = row['generation_requirements']
                checked = resolve_requirements(row['kind'], row['purpose'],
                    [plan['target_assets'][aid]['kind'] for aid in row['asset_ids']], requested=requirements)
                if requirements != checked:
                    raise ValueError('Invalid per-generation presentation requirements')
            if row.get("purpose") == "asset" and "target_assets" in plan:
                asset = plan["target_assets"][row["asset_ids"][0]]
                validate_attributes(asset, row.get("attributes"))
                if row["goal_match"] and row["attributes"] != asset["attributes"]:
                    raise ValueError("Asset marked terminal-compatible has different attributes")
                if row["edit_parent"]:
                    parent_attrs = media[row["edit_parent"]]["attributes"]
                    changes = changed_attributes(parent_attrs, row["attributes"])
                    if changes != row.get("attribute_changes"):
                        raise ValueError("Incorrect structured attribute edit")
            if "variation" in row and row["variation"]["type"] not in fields(plan["target_assets"][row["asset_ids"][0]]):
                raise ValueError("Variation is incompatible with the asset type")
            if "numbering_operation" in row:
                previous = row.get("previous_compilation")
                if previous not in seen or row["numbering_operation"] != resolved_numbering:
                    raise ValueError("Numbered edit has not been confirmed")
                if apply_numbering(media[previous]["shot_ids"], resolved_numbering) != row["shot_ids"]:
                    raise ValueError("Numbered edit produced the wrong shot order")
                resolved_numbering = None
            if any(source not in seen for source in row["inputs"]):
                raise ValueError("Media provenance uses a future input")
            if not 0 <= row["edit_depth"] <= 2:
                raise ValueError("Edit chain is too deep")
            if row["edit_parent"] is not None:
                if row["edit_depth"] != media[row["edit_parent"]]["edit_depth"] + 1:
                    raise ValueError("Incorrect edit ancestry")
            if row["purpose"] == "shot":
                latest_shots[row["shot_ids"][0]] = row
            seen.add(p)
    if seen != set(media):
        raise ValueError("Unintroduced media")
    if pending_numbering is not None or resolved_numbering is not None:
        raise ValueError("Unfinished numbering operation")
    if pending_batch is not None:
        raise ValueError("Unfinished batch edits")
    final = plan["final"]
    if final["order"] != plan["target_order"]:
        raise ValueError("Final shot order is wrong")
    for sid, shot in final['shots'].items():
        row=media[shot['media']]
        if row['purpose']!='shot' or row['shot_ids']!=[sid] or row['bindings']!=shot['bindings']:
            raise ValueError('Final shot binding is inconsistent')
        for aid, token in shot['bindings'].items():
            if media[token]['purpose']!='asset' or media[token]['asset_ids']!=[aid]:
                raise ValueError('Final asset binding has the wrong identity')
    artifact = media[final["media"]]
    if artifact["purpose"] == "final_export":
        artifact = media[artifact["inputs"][0]]
    if len(final["order"]) > 1:
        if artifact["shot_ids"] != final["order"]:
            raise ValueError("Final compilation order is stale")
        if artifact["shot_versions"] != {s: final["shots"][s]["media"] for s in final["order"]}:
            raise ValueError("Final compilation contains stale shot versions")
    validate_rule_timeline(plan)
    validate_reference_materials(plan)
    return True


def build_plan(value, **options):
    return Planner(value, **options).run()
