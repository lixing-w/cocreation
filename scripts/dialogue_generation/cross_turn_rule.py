"""Independent, overlapping user rules; no model calls or media I/O."""
from contextlib import contextmanager
from copy import deepcopy
import random

from scripts.dialogue_generation.asset_variations import CATEGORIES, AssetSpec, validate_attributes, fields

ROLES = ('asset_image', 'asset_video', 'trial_video', 'shot', 'compilation', 'final_export')
EFFECTS = {'color_mode': ('black_and_white', 'original_colors'),
           'season': ('spring', 'summer', 'fall', 'winter')}
ROLE_NAMES = {'asset_image': 'standalone asset images', 'asset_video': 'asset reference clips',
              'trial_video': 'trial clips', 'shot': 'formal shots',
              'compilation': 'assembled cuts', 'final_export': 'final exports'}


def media_role(row):
    if row.get('rule_role'):
        return row['rule_role']
    if row['purpose'] == 'asset':
        return 'asset_image' if row['kind'] == 'image' else 'asset_video'
    if row['purpose'] == 'exploration':
        return 'asset_image' if row['kind'] == 'image' else 'trial_video'
    return row['purpose']


def normalize_rule(value, assets):
    """Accept old all/shots inputs, but store only global/partial selectors."""
    if not isinstance(value, dict):
        raise ValueError('A constraint must be an object')
    scope = value.get('scope')
    selector = deepcopy(value.get('selector', {}))
    if scope == 'all':
        scope = 'global'
    elif scope == 'shots':
        scope, selector = 'partial', {'roles': ['shot', 'compilation', 'final_export']}
    if scope not in ('global', 'partial') or not isinstance(selector, dict):
        raise ValueError('Constraint scope must be global or partial')
    if set(selector) - {'roles', 'asset_ids'}:
        raise ValueError('Unknown constraint selector field')
    if scope == 'global' and selector:
        raise ValueError('Global constraints apply to all generated media; omit selector')
    if scope == 'partial' and not selector:
        raise ValueError('Partial constraints need a selector')
    for key, choices in (('roles', ROLES), ('asset_ids', assets)):
        if key in selector:
            rows = selector[key]
            if (not isinstance(rows, list) or not rows or any(not isinstance(v, str) or v not in choices for v in rows)
                    or len(set(rows)) != len(rows)):
                raise ValueError('Invalid constraint selector ' + key)
    effect = deepcopy(value.get('effect', {}))
    if not isinstance(effect, dict):
        raise ValueError('Constraint effect must be an object')
    if 'asset_attributes' in value:
        effect['asset_attributes'] = deepcopy(value['asset_attributes'])
    if not isinstance(effect, dict) or not effect or set(effect) - {*EFFECTS, 'asset_attributes'}:
        raise ValueError('A constraint needs a supported structured effect')
    for key in EFFECTS:
        if key in effect and effect[key] not in EFFECTS[key]:
            raise ValueError('Invalid constraint effect ' + key)
    if 'asset_attributes' in effect:
        locks = effect['asset_attributes']
        if not isinstance(locks, dict) or not locks:
            raise ValueError('asset_attributes must map known assets to nonempty attribute maps')
        for aid, attrs in locks.items():
            if aid not in assets or not isinstance(attrs, dict) or not attrs:
                raise ValueError('Unknown constraint asset or empty attribute map')
            validate_attributes(assets[aid], {**assets[aid]['attributes'], **attrs})
    text = value.get('text', '')
    if not isinstance(text, str):
        raise ValueError('Constraint text must be a string')
    return dict(scope=scope, selector=selector, effect=effect, text=text)


def matches(rule, row):
    selector = rule['selector']
    return (('roles' not in selector or media_role(row) in selector['roles'])
            and ('asset_ids' not in selector or bool(set(row['asset_ids']) & set(selector['asset_ids']))))


def effect_fields(effect):
    return {**{k: v for k, v in effect.items() if k != 'asset_attributes'},
            **{f'asset:{aid}:{key}': value for aid, attrs in effect.get('asset_attributes', {}).items()
               for key, value in attrs.items()}}


def overlaps(a, b):
    for key, universe in (('roles', ROLES), ('asset_ids', None)):
        av, bv = a['selector'].get(key), b['selector'].get(key)
        if av is not None and bv is not None and not set(av) & set(bv):
            return False
    return bool(set(effect_fields(a['effect'])) & set(effect_fields(b['effect'])))


def merge_effect(state, effect):
    for key, value in effect.items():
        if key == 'asset_attributes':
            for aid, attrs in value.items():
                state.setdefault(key, {}).setdefault(aid, {}).update(attrs)
        else:
            state[key] = value


def scope_text(rule, names=None):
    if rule['scope'] == 'global':
        return 'all generated images and videos, including assets, trials, intermediate results and finished footage'
    selector = rule['selector']
    result = ', '.join(ROLE_NAMES[v] for v in selector.get('roles', ())) or 'generated media'
    if selector.get('asset_ids'):
        result += ' involving ' + ', '.join((names or {}).get(a, a) for a in selector['asset_ids'])
    return result


def effect_text(effect, names=None):
    parts = []
    if 'color_mode' in effect:
        parts.append('black-and-white rendering' if effect['color_mode'] == 'black_and_white' else 'original-color rendering')
    if 'season' in effect:
        parts.append(effect['season'] + ' as the background season')
    for aid, attrs in effect.get('asset_attributes', {}).items():
        parts.extend(f"{key.replace('_', ' ')} of {(names or {}).get(aid, aid)}: {value}" for key, value in attrs.items())
    return ', '.join(parts)


def render_signature(row, target_assets):
    """Only visual differences from the target, not redundant rule declarations."""
    state, signature = row.get('render_state', {}), {}
    if state.get('color_mode') == 'black_and_white':
        signature['color_mode'] = 'black_and_white'
    if 'season' in state:
        environments = [target_assets[a] for a in row['asset_ids'] if 'season' in target_assets[a]['attributes']]
        if not environments or any(a['attributes']['season'] != state['season'] for a in environments):
            signature['season'] = state['season']
    for aid, attrs in state.get('asset_attributes', {}).items():
        if aid in row['asset_ids']:
            for key, value in attrs.items():
                if target_assets[aid]['attributes'][key] != value:
                    signature[f'{aid}.{key}'] = value
    return signature


def visible_effect(effect, row):
    """Asset-specific instructions concern only subjects present in this output."""
    result = {key: deepcopy(value) for key,value in effect.items() if key != 'asset_attributes'}
    attrs = {aid: deepcopy(values) for aid,values in effect.get('asset_attributes', {}).items() if aid in row['asset_ids']}
    if attrs:
        result['asset_attributes'] = attrs
    return result


def describe_media(row, target_assets):
    """One output instruction; design/reference appearance is never inherited as a rule."""
    state = visible_effect(row.get('render_state', {}), row)
    base = row['base_description']
    if row.get('purpose') == 'asset' and row.get('attributes'):
        aid = row['asset_ids'][0]
        attrs = deepcopy(row['attributes'])
        attrs.update(state.get('asset_attributes', {}).get(aid, {}))
        if 'season' in state and 'season' in attrs:
            attrs['season'] = state['season']
        row['effective_attributes'] = attrs
        validate_attributes(target_assets[aid], attrs)
        if state:
            base = str(AssetSpec(target_assets[aid], attrs))
    source_states = row.get('reference_render_states', {})
    if not state:
        if source_states and row['purpose'] in ('compilation', 'final_export'):
            return base + '. Preserve each input clip\'s existing appearance; assembling or exporting does not undo earlier treatments.'
        if source_states:
            return base + '. Render this new output without the earlier reference-only treatments, using the underlying design colors and original scene conditions.'
        return base
    effects = effect_text(state, {a: r.get('name', a) for a, r in target_assets.items()})
    return (f'Content and design reference: {base}. Output requirements: {effects}. '
            'Apply these requirements to the entire output frame; they replace conflicting scene or rendering conditions in the reference. '
            'Preserve the subject identities, other design details, action and composition. Do not inherit obsolete rendering treatments from the reference inputs.')


class CrossTurnRules:
    """A side layer sampled only at completed work boundaries, never inside a clarification."""
    def __init__(self, planner, *, enabled=True):
        self.p, self.enabled = planner, enabled
        self.rng = random.Random(f'cross_turn_rule:{planner.seed}')
        self.records, self.active = [], {}
        self.exceptions = {}
        self.closed = False
        self.names = {a: r['name'] for a, r in planner.assets.items()}

    def emit(self, role, action, text, *, event=None, requested_role=None, **kwargs):
        p = self.p
        old = p.module
        p.module = 'cross_turn_rule'
        p.emit(role, action, text, behaviors=[13] + ([14] if 'conflict' in action else []), **kwargs)
        if event:
            p.turns[-1]['intent']['rule_event'] = deepcopy(event)
        if requested_role:
            p.turns[-1]['intent']['requested_media_role'] = requested_role
        p.module = old

    def start(self, spec):
        rule = dict(deepcopy(spec), id=f'R{len(self.records)+1}', started_at=None, ended_at=None)
        conflicts = [r for r in self.active.values() if overlaps(r, rule)
                     and any(effect_fields(r['effect'])[k] != v for k, v in effect_fields(rule['effect']).items()
                             if k in effect_fields(r['effect']))]
        if conflicts:
            self.emit('user', 'propose_rule', f"From now on, use {effect_text(rule['effect'], self.names)} for {scope_text(rule, self.names)}.")
            old_text = '; '.join(effect_text(r['effect'], self.names) + ' for ' + scope_text(r, self.names) for r in conflicts)
            self.emit('assistant', 'clarify_rule_conflict', f'This conflicts with the existing rule: {old_text}. Should I keep it or cancel it and use the new rule?',
                      event=dict(type='challenge', ids=[r['id'] for r in conflicts], proposed=rule['effect']))
            keep = self.rng.random() < .5
            self.emit('user', 'resolve_rule_conflict', 'Keep the existing rule; drop my proposed rule.' if keep else 'Cancel those earlier rules entirely and use the new rule with its stated scope.',
                      event=dict(type='resolve', ids=[r['id'] for r in conflicts], outcome='keep' if keep else 'override'))
            if keep:
                self.emit('assistant', 'acknowledge_rule_decision', 'The earlier rule remains in force; I will not apply the proposed change.')
                return None
            return self.replace(conflicts, {k: rule[k] for k in ('scope', 'selector', 'effect', 'text')})
        self.records.append(rule)
        self.emit('user', 'set_cross_turn_rule', f"From now on, use {effect_text(rule['effect'], self.names)} for {scope_text(rule, self.names)}.", scope=rule['scope'])
        confirmation = f"Remember {effect_text(rule['effect'], self.names)} for {scope_text(rule, self.names)}. "
        if rule['scope'] == 'partial':
            confirmation += 'Other output categories are not restricted by this particular rule.'
            if self.active:
                confirmation += ' Previously agreed rules remain in effect.'
        else:
            confirmation += 'This includes asset references, trials, intermediate results and final footage.'
        self.emit('assistant', 'confirm_cross_turn_rule', confirmation,
                  event=dict(type='start', rule={k: deepcopy(v) for k, v in rule.items() if k not in ('started_at', 'ended_at')}), scope=rule['scope'])
        rule['started_at'] = len(self.p.turns)
        self.active[rule['id']] = rule
        return rule

    def replace(self, old_rules, spec):
        rule = dict(deepcopy(spec), id=f'R{len(self.records)+1}', started_at=None, ended_at=None)
        self.records.append(rule)
        self.emit('assistant', 'confirm_rule_resolution',
                  f"Replace the earlier rule with {effect_text(rule['effect'], self.names)} for {scope_text(rule, self.names)}. Other rules remain unchanged.",
                  event=dict(type='replace', ids=[r['id'] for r in old_rules], rule={k: deepcopy(v) for k,v in rule.items() if k not in ('started_at','ended_at')}))
        for old in old_rules:
            old['ended_at'] = len(self.p.turns)
            del self.active[old['id']]
        rule['started_at'] = len(self.p.turns)
        self.active[rule['id']] = rule
        return rule

    def replace_group(self, rules, effects):
        replacements = [dict(deepcopy(old), id=f'R{len(self.records)+i+1}',
                             effect=deepcopy(effects[old['id']]), started_at=None, ended_at=None)
                        for i, old in enumerate(rules)]
        self.records.extend(replacements)
        description = '; '.join(effect_text(r['effect'], self.names) + ' for ' + scope_text(r, self.names) for r in replacements)
        self.emit('assistant', 'confirm_rule_resolution', 'Update the affected rules: ' + description + '. Preserve each scope and all other requirements.',
                  event=dict(type='replace', ids=[r['id'] for r in rules],
                             rules=[{k: deepcopy(v) for k,v in r.items() if k not in ('started_at','ended_at')} for r in replacements]))
        for old, new in zip(rules, replacements):
            old['ended_at'] = new['started_at'] = len(self.p.turns)
            del self.active[old['id']]
            self.active[new['id']] = new

    def end(self, rule, *, confirmed=False, cleanup=False):
        if rule['id'] not in self.active:
            raise ValueError('Cannot cancel an inactive rule')
        text = f"Cancel the rule requiring {effect_text(rule['effect'], self.names)} for {scope_text(rule, self.names)}. Leave other rules and existing media unchanged."
        if not confirmed:
            self.emit('user', 'cancel_cross_turn_rule', text)
        self.emit('assistant', 'confirm_rule_cancel', text.replace('Cancel the rule requiring', 'The rule is canceled:') + ' Future outputs follow the remaining requirements.',
                  event=dict(type='end', id=rule['id'], cleanup=cleanup))
        rule['ended_at'] = len(self.p.turns)
        del self.active[rule['id']]

    def snapshot(self, row):
        state, applied = {}, []
        for rule in self.active.values():
            if matches(rule, row):
                effect = self.exceptions.get(rule['id'], rule['effect'])
                merge_effect(state, effect)
                applied.append(dict(id=rule['id'], effect=deepcopy(effect)))
        return state, applied

    @contextmanager
    def exception(self, overrides):
        if self.exceptions:
            raise ValueError('An exception must finish before another starts')
        self.exceptions = deepcopy(overrides)
        try:
            yield
        finally:
            self.exceptions = {}

    def changed_effect(self, rule):
        effect = deepcopy(rule['effect'])
        key = self.rng.choice(list(effect))
        if key in EFFECTS:
            effect[key] = self.rng.choice([v for v in EFFECTS[key] if v != effect[key]])
        else:
            aid = self.rng.choice(list(effect[key]))
            field = self.rng.choice(list(effect[key][aid]))
            row = self.p.assets[aid]
            choices = fields(row)[field]['values']
            allowed = []
            for value in choices:
                if value in ('unknown', 'not applicable', effect[key][aid][field]):
                    continue
                try:
                    validate_attributes(row, {**row['attributes'], **effect[key][aid], field: value})
                except ValueError:
                    continue
                allowed.append(value)
            if not allowed:
                return None
            effect[key][aid][field] = self.rng.choice(allowed)
        return effect

    def preview_context(self, rule, changed=None):
        if not self.p.ready:
            return None
        roles = rule['selector'].get('roles', ['asset_image', 'trial_video'])
        # Test a concrete generation category, never call a formal-shot request a generic preview.
        if 'shot' in roles:
            roles = ['shot']
        role = self.rng.choice(roles)
        affected = rule['effect'].get('asset_attributes', {})
        if changed and affected:
            affected = {a: attrs for a, attrs in changed.get('asset_attributes', {}).items()
                        if attrs != rule['effect']['asset_attributes'].get(a)}
        aids = [a for a in self.p.ready if (not rule['selector'].get('asset_ids') or a in rule['selector']['asset_ids'])
                and (not affected or a in affected)]
        if not aids:
            return None
        aid = self.rng.choice(aids)
        source = self.p.ready[aid]
        if role == 'compilation':
            if self.p.comp_media is None:
                return None
            source = self.p.comp_media
        elif role == 'final_export':
            source = self.p.comp_media or next(iter(self.p.rendered.values()), None)
            if source is None:
                return None
        return dict(kind='image' if role == 'asset_image' else 'video', purpose='exploration', rule_role=role, asset_ids=[aid], source=source)

    def challenge(self, rule, *, outcome=None):
        effect = self.changed_effect(rule)
        context = self.preview_context(rule, effect)
        if context is None or effect is None:
            return
        p = self.p
        aid = context['asset_ids'][0]
        source = context['source']
        subject = {'asset_image': 'a standalone asset image', 'asset_video': 'an asset reference clip',
                   'trial_video': 'a trial clip', 'shot': 'a formal video shot',
                   'compilation': 'an assembled cut', 'final_export': 'a final export'}[context['rule_role']]
        difference = {k: v for k, v in effect.items() if k != 'asset_attributes' and v != rule['effect'].get(k)}
        changed_attrs = {a: {k: v for k, v in attrs.items() if v != rule['effect'].get('asset_attributes', {}).get(a, {}).get(k)}
                         for a, attrs in effect.get('asset_attributes', {}).items()}
        changed_attrs = {a: attrs for a, attrs in changed_attrs.items() if attrs}
        if changed_attrs:
            difference['asset_attributes'] = changed_attrs
        changed_fields = effect_fields(difference)
        conflicts = [r for r in self.active.values() if matches(r, context)
                     and any(k in effect_fields(r['effect']) and effect_fields(r['effect'])[k] != value for k,value in changed_fields.items())]
        effects = {}
        for current in conflicts:
            updated = deepcopy(current['effect'])
            for key, value in difference.items():
                if key == 'asset_attributes':
                    for asset, attrs in value.items():
                        for field, chosen in attrs.items():
                            if field in updated.get(key, {}).get(asset, {}):
                                updated[key][asset][field] = chosen
                elif key in updated:
                    updated[key] = value
            effects[current['id']] = updated
        self.emit('user', 'request_rule_conflict', f"Make {subject} of the {p.label(aid)} using {effect_text(difference, self.names)}.", assets=[aid], inputs=[source], requested_role=context['rule_role'])
        reminder = '; '.join(effect_text(r['effect'], self.names) + ' for ' + scope_text(r, self.names) for r in conflicts)
        event = dict(type='challenge', ids=[r['id'] for r in conflicts], proposed=effect)
        if len(conflicts) > 1:
            event['effects'] = effects
        self.emit('assistant', 'clarify_rule_conflict', f"The existing rule requires {reminder}. Should I keep it, make an exception for this request only, or replace it going forward?", assets=[aid], event=event)
        outcome = outcome or self.rng.choice(('keep', 'once', 'override'))
        old_requirement = effect_text(rule['effect'], self.names)
        new_requirement = effect_text(effect, self.names)
        coverage = scope_text(rule, self.names)
        wording = {
            'keep': f'Keep {old_requirement} for {coverage}. Withdraw my requested change and make this result under that existing rule.',
            'once': f'Use {new_requirement} for this generation request only. For later outputs, keep {old_requirement} for {coverage}.',
            'override': f'Replace the earlier rule: from now on, use {new_requirement} for {coverage}. Keep other rules unchanged.'
        }[outcome]
        event = dict(type='resolve', ids=[r['id'] for r in conflicts], outcome=outcome, effect=effect)
        if len(conflicts) > 1:
            event['effects'] = effects
            wording = {
                'keep': 'Keep all those earlier rules. Withdraw my requested change and make the result under the existing requirements.',
                'once': f'For this request only, use {effect_text(difference, self.names)}. Keep every earlier rule for subsequent outputs.',
                'override': f'Use {effect_text(difference, self.names)} going forward in each affected rule. Preserve each original scope and every other requirement.'
            }[outcome]
        self.emit('user', 'resolve_rule_conflict', wording, assets=[aid], event=event)
        if outcome == 'override':
            spec = {k: deepcopy(rule[k]) for k in ('scope', 'selector', 'effect', 'text')}
            spec['effect'] = effect
            if len(conflicts) > 1:
                self.replace_group(conflicts, effects)
            else:
                self.replace([rule], spec)
            self.emit('user', 'request_rule_followup', f'Now make {subject} under the replacement rule.', assets=[aid], inputs=[source], requested_role=context['rule_role'])
        overrides = effects if outcome == 'once' else {}
        with self.exception(overrides):
            # This whole request is one transaction; multiple outputs would share the exception.
            token = p.add_media(context['kind'], f"{subject.capitalize()} of the {p.label(aid)}, Show a short plausible action, interaction, natural motion or revealing camera movement suitable for the referenced identity. Reference design: {p.media[source]['base_description']}",
                                assets=[aid], inputs=[source], purpose='exploration', rule_role=context['rule_role'])
            self.emit('assistant', 'deliver_rule_preview', 'Here is the preview under the agreed rule decision.', assets=[aid], inputs=[source], outputs=[token])
        # Demonstrate that a one-request exception really expires.
        if outcome == 'once':
            self.emit('user', 'request_rule_followup', f'Make another {subject.removeprefix("a ").removeprefix("an ")} under the continuing rules.', assets=[aid], inputs=[source], requested_role=context['rule_role'])
            token = p.add_media(context['kind'], f"Another {subject} of the {p.label(aid)}. Reference design: {p.media[source]['base_description']}",
                                assets=[aid], inputs=[source], purpose='exploration', rule_role=context['rule_role'])
            self.emit('assistant', 'deliver_rule_preview', 'Here is the next preview under the continuing rules.', assets=[aid], inputs=[source], outputs=[token])

    def advance(self):
        if not self.enabled or self.closed:
            return
        draw = self.rng.random()
        if draw < .40 and len(self.active) < 3 and len(self.records) < 6:
            spec = dict(scope=self.rng.choice(('global', 'partial')), selector={}, text='')
            if spec['scope'] == 'partial':
                spec['selector'] = dict(roles=[self.rng.choice(('asset_image', 'trial_video', 'shot'))])
            key = self.rng.choice(list(EFFECTS))
            spec['effect'] = {key: self.rng.choice(EFFECTS[key][:-1] if key == 'color_mode' else EFFECTS[key])}
            # Redundant/contradictory new defaults are not silently stacked.
            if not any(overlaps(r, spec) for r in self.active.values()):
                self.start(spec)
        elif draw < .78 and self.active and self.p.ready:
            self.challenge(self.rng.choice(list(self.active.values())))
        elif draw < .92 and self.active:
            self.end(self.rng.choice(list(self.active.values())))

    def finish(self):
        self.closed = True
        # Accepted global constraints persist into the final cut. Explicit
        # user cancellations during the conversation remain prospective.

    def decorate(self):
        for row in self.p.media.values():
            row['description'] = describe_media(row, self.p.assets)
        records = {r['id']: r for r in self.records}
        requests = {'request_asset','request_candidates','request_trial','begin_task','specify_goal','request_variant',
                    'relative_request','resume_task','reject_and_edit','request_shot','request_shot_update',
                    'request_compilation','reorder_shots','replace_compilation_shots','restore_rule_asset','request_rule_followup'}
        for index, turn in enumerate(self.p.turns):
            if not turn['output_media']:
                continue
            contexts = []
            for token in turn['output_media']:
                row = self.p.media[token]
                parts = []
                for applied in row['applied_rules']:
                    rule = records[applied['id']]
                    relevant = visible_effect(applied['effect'], row)
                    if not relevant:
                        continue
                    text = effect_text(relevant, self.names)
                    if applied['effect'] != rule['effect']:
                        text += ' for this request only; subsequent requests return to ' + effect_text(rule['effect'], self.names)
                    parts.append(text)
                context = ROLE_NAMES.get(media_role(row), media_role(row)) + ': ' + ('; '.join(parts) if parts else 'no additional rule treatment applies')
                if context not in contexts:
                    contexts.append(context)
            turn['intent']['media_rule_context'] = contexts
            states=[self.p.media[t]['render_state'] for t in turn['output_media']]
            rendering={'color_mode':'black_and_white'} if all(s.get('color_mode')=='black_and_white' for s in states) else {}
            turn['intent']['output_render_state']=rendering
            # Attach the exact operation scope to its request, never an unrelated adoption.
            for previous in reversed(self.p.turns[:index]):
                if previous['output_media']:
                    break
                if previous['role'] == 'user' and previous['intent']['action'] in requests:
                    previous['intent']['media_rule_context'] = contexts
                    previous['intent']['output_render_state'] = rendering
                    break



def validate_rule_timeline(plan):
    """Replay confirmations; verify snapshots, exceptions, cancellations and terminal closure."""
    if 'rules' not in plan:
        return True
    media = {m['placeholder']: m for m in plan['placeholders']}
    active, records, pending, resolution, once = {}, {}, None, None, {}
    for index, turn in enumerate(plan['turns']):
        event = turn['intent'].get('rule_event', {})
        kind = event.get('type')
        event_actions = {'challenge': 'clarify_rule_conflict', 'resolve': 'resolve_rule_conflict',
                         'start': 'confirm_cross_turn_rule', 'end': 'confirm_rule_cancel', 'replace': 'confirm_rule_resolution'}
        if kind and (kind not in event_actions or turn['intent']['action'] != event_actions[kind]):
            raise ValueError('Rule event must be attached to its actual dialogue action')
        if turn['intent']['action'] in event_actions.values() and not kind:
            raise ValueError('Missing structured rule event')
        if kind == 'challenge':
            if pending or any(r not in active for r in event['ids']):
                raise ValueError('Rule conflict references an inactive rule or unresolved conflict')
            pending = deepcopy(event)
        elif kind == 'resolve':
            if not pending or event['ids'] != pending['ids'] or event['outcome'] not in ('keep', 'once', 'override'):
                raise ValueError('Rule resolution requires the preceding rule reminder')
            if event.get('effect', pending['proposed']) != pending['proposed']:
                raise ValueError('Rule resolution changed the proposed effect')
            if event.get('effects') != pending.get('effects'):
                raise ValueError('Rule resolution changed the affected rules')
            resolution = dict(deepcopy(event), effect=deepcopy(pending['proposed']))
            if event['outcome'] == 'once':
                once = deepcopy(event.get('effects') or {rid: event['effect'] for rid in event['ids']})
            pending = None
        elif kind == 'replace':
            if not resolution or resolution['outcome'] != 'override' or event['ids'] != resolution['ids']:
                raise ValueError('Rule replacement lacks explicit user authorization')
            replacements = event.get('rules') or [event.get('rule')]
            if event.get('rules') and len(replacements) != len(event['ids']):
                raise ValueError('Rule replacement omitted an affected rule')
            for rule in replacements:
                normalize_rule(rule, plan['target_assets'])
            for rid in event['ids']:
                if rid not in active:
                    raise ValueError('Replacement references an inactive rule')
                records[rid]['ended_at'] = turn['id']
            for index_in_group, rule in enumerate(replacements):
                expected_effect = (resolution['effects'][event['ids'][index_in_group]] if resolution.get('effects') else resolution['effect'])
                if rule['effect'] != expected_effect:
                    raise ValueError('Replacement changed the authorized effect')
                if event.get('rules'):
                    before = active[event['ids'][index_in_group]]
                    if rule['scope'] != before['scope'] or rule['selector'] != before['selector']:
                        raise ValueError('Group replacement changed a rule scope')
                if rule['id'] in records:
                    raise ValueError('Duplicate rule id')
                active[rule['id']] = deepcopy(rule)
                records[rule['id']] = dict(deepcopy(rule), started_at=turn['id'], ended_at=None)
            for rid in event['ids']:
                del active[rid]
            resolution = None
        elif kind == 'start':
            normalize_rule(event['rule'], plan['target_assets'])
            if index == 0 or plan['turns'][index-1]['role'] != 'user' or plan['turns'][index-1]['intent']['action'] != 'set_cross_turn_rule':
                raise ValueError('A rule start requires a user rule request')
            rule = deepcopy(event['rule'])
            if rule['id'] in records:
                raise ValueError('Duplicate rule id')
            active[rule['id']] = rule
            records[rule['id']] = dict(rule, started_at=turn['id'], ended_at=None)
        elif kind == 'end':
            if index == 0 or plan['turns'][index-1]['role'] != 'user' or plan['turns'][index-1]['intent']['action'] != 'cancel_cross_turn_rule':
                raise ValueError('Rule cancellation requires an explicit user request')
            if event['id'] not in active:
                raise ValueError('Cannot cancel an inactive rule')
            records[event['id']]['ended_at'] = turn['id']
            del active[event['id']]
        for token in turn['output_media']:
            if pending:
                raise ValueError('Media generated before rule conflict resolution')
            row = media[token]
            state, applied = {}, []
            for rule in active.values():
                if matches(rule, row):
                    effect = once.get(rule['id'], rule['effect'])
                    merge_effect(state, effect)
                    applied.append(dict(id=rule['id'], effect=deepcopy(effect)))
            if row.get('render_state') != state or row.get('applied_rules') != applied:
                raise ValueError('Media does not follow the rules active at its creation')
            expected_sources = {p: deepcopy(media[p]['render_state'] or media[p].get('reference_render_states', {}))
                                for p in row['inputs'] if media[p]['render_state'] or media[p].get('reference_render_states')}
            if row.get('reference_render_states') != expected_sources:
                raise ValueError('Incorrect reference appearance snapshot')
            expected_row = deepcopy(row)
            expected = describe_media(expected_row, plan['target_assets'])
            if row.get('effective_attributes') != expected_row.get('effective_attributes'):
                raise ValueError('Effective asset attributes disagree with rendering rules')
            if row['description'] != expected:
                raise ValueError('Placeholder description disagrees with its effective rules')
        if turn['output_media']:
            once = {}
    if pending or once or (resolution and resolution['outcome']=='override') or list(records.values()) != plan['rules']:
        raise ValueError('Unfinished rule lifecycle or inconsistent rule history')
    return True
