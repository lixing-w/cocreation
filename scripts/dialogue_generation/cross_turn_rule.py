"""Independent, overlapping user rules; no model calls or media I/O."""
from contextlib import contextmanager
from copy import deepcopy
import random

from scripts.dialogue_generation.asset_variations import CATEGORIES, AssetSpec, validate_attributes, fields, alternatives, TIMES

KINDS = ('image', 'video')
GENERATION_REQUESTS = {'request_asset', 'request_candidates', 'request_trial', 'specify_goal',
                      'begin_task', 'request_variant', 'reject_and_edit', 'relative_request',
                      'resume_task', 'temporary_adoption_trial', 'request_shot', 'request_shot_update'}
EFFECTS = {'season': ('spring', 'summer', 'fall', 'winter'),
           'aspect_ratio': ('portrait', 'landscape', 'square_assets_landscape_video', 'square_assets_portrait_video'),
           'time_of_day': TIMES}
SCENE_FIELDS = ('season', 'time_of_day')


ASPECT_NAMES = {
    'portrait': 'portrait 9:16',
    'landscape': 'landscape 16:9',
    'square_assets_landscape_video': 'square 1:1 images and landscape 16:9 videos',
    'square_assets_portrait_video': 'square 1:1 images and portrait 9:16 videos',
    '9:16': 'portrait 9:16', '16:9': 'landscape 16:9', '1:1': 'square 1:1',
}


def output_aspect(preset, kind):
    if preset in ('1:1', '9:16', '16:9'):
        return preset
    if preset == 'portrait':
        return '9:16'
    if preset == 'landscape':
        return '16:9'
    if preset in ('square_assets_landscape_video', 'square_assets_portrait_video'):
        return '1:1' if kind == 'image' else ('16:9' if preset.endswith('landscape_video') else '9:16')
    raise ValueError('Unknown aspect ratio preset: ' + str(preset))


def different_for_kind(before, after, kind):
    def projected(effect):
        result = deepcopy(effect)
        if 'aspect_ratio' in result:
            result['aspect_ratio'] = output_aspect(result['aspect_ratio'], kind)
        return result
    return projected(before) != projected(after)


def normalize_rule(value, assets):
    """Validate global rules or selectors over image/video and subject ids."""
    if not isinstance(value, dict):
        raise ValueError('A constraint must be an object')
    scope = value.get('scope')
    selector = deepcopy(value.get('selector', {}))
    if scope == 'all':
        scope = 'global'
    if scope not in ('global', 'partial') or not isinstance(selector, dict):
        raise ValueError('Constraint scope must be global or partial')
    if set(selector) - {'kinds', 'asset_ids'}:
        raise ValueError('Unknown constraint selector field')
    if scope == 'global' and selector:
        raise ValueError('Global constraints apply to all generated media; omit selector')
    if scope == 'partial' and not selector:
        raise ValueError('Partial constraints need a selector')
    for key, choices in (('kinds', KINDS), ('asset_ids', assets)):
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
    return (('kinds' not in selector or row['kind'] in selector['kinds'])
            and ('asset_ids' not in selector or bool(set(row['asset_ids']) & set(selector['asset_ids']))))


def effect_fields(effect):
    return {**{k: v for k, v in effect.items() if k != 'asset_attributes'},
            **{f'asset:{aid}:{key}': value for aid, attrs in effect.get('asset_attributes', {}).items()
               for key, value in attrs.items()}}


def overlaps(a, b):
    for key, universe in (('kinds', KINDS), ('asset_ids', None)):
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
        return 'all generated images and videos'
    selector = rule['selector']
    result = ' and '.join(v + 's' for v in selector.get('kinds', ())) or 'generated media'
    if selector.get('asset_ids'):
        result += ' involving ' + ', '.join((names or {}).get(a, a) for a in selector['asset_ids'])
    return result


def effect_text(effect, names=None):
    parts = []
    if 'season' in effect:
        parts.append(effect['season'] + ' as the background season')
    if 'time_of_day' in effect:
        parts.append(effect['time_of_day'] + ' as the time of day')
    if 'aspect_ratio' in effect:
        parts.append(ASPECT_NAMES[effect['aspect_ratio']])
    for aid, attrs in effect.get('asset_attributes', {}).items():
        parts.extend(f"{key.replace('_', ' ')} of {(names or {}).get(aid, aid)}: {value}" for key, value in attrs.items())
    return ', '.join(parts)


def render_signature(row, target_assets):
    """Only visual differences from the target, not redundant rule declarations."""
    state, signature = row.get('render_state', {}), {}
    if 'aspect_ratio' in state:
        signature['aspect_ratio'] = output_aspect(state['aspect_ratio'], row['kind'])
    for field in SCENE_FIELDS:
        if field in state:
            values = [target_assets[a]['attributes'][field] for a in row['asset_ids']
                      if field in target_assets[a]['attributes']]
            if not values or any(value != state[field] for value in values):
                signature[field] = state[field]
    for aid, attrs in state.get('asset_attributes', {}).items():
        if aid in row['asset_ids']:
            for key, value in attrs.items():
                if target_assets[aid]['attributes'][key] != value:
                    signature[f'{aid}.{key}'] = value
    return signature


def visible_effect(effect, row):
    """Asset-specific instructions concern only subjects present in this output."""
    result = {key: deepcopy(value) for key,value in effect.items() if key != 'asset_attributes'}
    if 'aspect_ratio' in result:
        result['aspect_ratio'] = output_aspect(result['aspect_ratio'], row['kind'])
    attrs = {aid: deepcopy(values) for aid,values in effect.get('asset_attributes', {}).items() if aid in row['asset_ids']}
    if attrs:
        result['asset_attributes'] = attrs
    return result


def describe_media(row, target_assets):
    """One output instruction; design/reference appearance is never inherited as a rule."""
    state = visible_effect(row.get('render_state', {}), row)
    base = row['base_description']
    if row.get('operation') == 'create_reference' and row.get('attributes'):
        aid = row['asset_ids'][0]
        attrs = deepcopy(row['attributes'])
        attrs.update(state.get('asset_attributes', {}).get(aid, {}))
        attrs.update({field: state[field] for field in SCENE_FIELDS if field in state and field in attrs})
        row['effective_attributes'] = attrs
        validate_attributes(target_assets[aid], attrs)
        if state:
            base = str(AssetSpec(target_assets[aid], attrs))
    source_states = row.get('reference_render_states', {})
    if not state:
        if source_states and row['operation'] in ('assemble', 'export'):
            return base + '. Preserve each input clip\'s existing appearance; assembling or exporting does not undo earlier treatments.'
        if source_states:
            return base + '. Render this new output without the earlier reference-only treatments, using the current subject attributes and scene conditions.'
        return base
    effects = effect_text(state, {a: r.get('name', a) for a, r in target_assets.items()})
    lighting = (' Show the requested time of day through lighting or visible sky when the framing allows.'
                if 'time_of_day' in state else '')
    return (f'Content and design reference: {base}. Output requirements: {effects}. '
            'Apply these requirements to the entire output frame; they replace conflicting scene or rendering conditions in the reference. '
            'Preserve the subject identities, other design details, action and composition. Do not inherit obsolete rendering treatments from the reference inputs.'
            + lighting)


class CrossTurnRules:
    """A side layer sampled only at completed work boundaries, never inside a clarification."""
    def __init__(self, planner, *, enabled=True, legacy_sampling=False, sample_time_of_day=True):
        self.p, self.enabled = planner, enabled
        self.legacy_sampling = legacy_sampling
        self.sample_time_of_day = sample_time_of_day and not legacy_sampling
        self.rng = random.Random(f'cross_turn_rule:{planner.seed}')
        self.records, self.active = [], {}
        self.exceptions = {}
        self.exception_kind = None
        self.closed = False
        self.pending = None
        self.confirmation = None
        self.overlay_once = False
        self.names = {a: r['name'] for a, r in planner.assets.items()}

    def emit(self, role, action, text, *, event=None, requested_kind=None, **kwargs):
        p = self.p
        old = p.module
        p.module = 'cross_turn_rule'
        p.emit(role, action, text, behaviors=[13] + ([14] if 'conflict' in action else []), **kwargs)
        if event:
            p.turns[-1]['intent']['rule_event'] = deepcopy(event)
        if requested_kind:
            p.turns[-1]['intent']['requested_media_kind'] = requested_kind
        p.module = old

    def start(self, spec):
        spec = normalize_rule(spec, self.p.assets)
        rule = dict(deepcopy(spec), id=f'R{len(self.records)+1}', started_at=None, ended_at=None)
        conflicts = [r for r in self.active.values() if overlaps(r, rule)
                     and any(effect_fields(r['effect'])[k] != v for k, v in effect_fields(rule['effect']).items()
                             if k in effect_fields(r['effect']))]
        if conflicts:
            self.emit('user', 'propose_rule', f"From now on, use {effect_text(rule['effect'], self.names)} for {scope_text(rule, self.names)}.")
            old_text = '; '.join(effect_text(r['effect'], self.names) + ' for ' + scope_text(r, self.names) for r in conflicts)
            self.emit('assistant', 'clarify_rule_conflict', f'This conflicts with the existing rule: {old_text}. Should I keep it or replace it going forward?',
                      event=dict(type='challenge', ids=[r['id'] for r in conflicts], proposed=rule['effect']))
            keep = self.rng.random() < .5
            self.emit('user', 'resolve_rule_conflict', 'Keep the existing rule; drop my proposed rule.' if keep else 'Replace those earlier settings going forward with the new rule and its stated scope.',
                      event=dict(type='resolve', ids=[r['id'] for r in conflicts], outcome='keep' if keep else 'override'))
            if keep:
                self.emit('assistant', 'acknowledge_rule_decision', 'The earlier rule remains in force; I will not apply the proposed change.')
                return None
            return self.replace(conflicts, {k: rule[k] for k in ('scope', 'selector', 'effect', 'text')})
        self.records.append(rule)
        self.emit('user', 'set_cross_turn_rule', f"From now on, use {effect_text(rule['effect'], self.names)} for {scope_text(rule, self.names)}.", scope=rule['scope'])
        confirmation = f"Remember {effect_text(rule['effect'], self.names)} for {scope_text(rule, self.names)}. "
        if rule['scope'] == 'partial':
            confirmation += 'This applies only to the specified media kinds and subjects.'
            if self.active:
                confirmation += ' Previously agreed rules remain in effect.'
        else:
            confirmation += 'I will keep it for subsequent images and videos.'
        self.emit('assistant', 'confirm_cross_turn_rule', confirmation,
                  event=dict(type='start', rule={k: deepcopy(v) for k, v in rule.items() if k not in ('started_at', 'ended_at')}), scope=rule['scope'])
        rule['started_at'] = len(self.p.turns)
        self.active[rule['id']] = rule
        return rule

    def replace(self, old_rules, spec):
        spec = normalize_rule(spec, self.p.assets)
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
        if 'aspect_ratio' in rule['effect']:
            raise ValueError('Aspect ratio rules cannot be canceled; use an explicit replacement or one-request exception')
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
                effect = self.exceptions.get(rule['id'], rule['effect']) if self.exception_kind in (None, row['kind']) else rule['effect']
                merge_effect(state, effect)
                applied.append(dict(id=rule['id'], effect=deepcopy(effect)))
        return state, applied

    @contextmanager
    def exception(self, overrides, *, kind=None):
        if kind is not None and kind not in KINDS:
            raise ValueError('Exception media kind must be image or video')
        if self.exceptions:
            raise ValueError('An exception must finish before another starts')
        self.exceptions = deepcopy(overrides)
        self.exception_kind = kind
        try:
            yield
        finally:
            self.exceptions = {}
            self.exception_kind = None

    def changed_effect(self, rule, *, kind=None, once=False):
        effect = deepcopy(rule['effect'])
        key = self.rng.choice(list(effect))
        if key == 'aspect_ratio':
            kind = kind or 'image'
            choices = ('1:1', '9:16', '16:9') if once and kind == 'image' else (('9:16', '16:9') if once else EFFECTS[key])
            choices = [v for v in choices if output_aspect(v, kind) != output_aspect(effect[key], kind)]
            if not choices:
                return None
            effect[key] = self.rng.choice(choices)
        elif key in EFFECTS:
            effect[key] = self.rng.choice([v for v in EFFECTS[key] if v != effect[key]])
        else:
            candidates = []
            for aid, locked_attrs in effect[key].items():
                row = self.p.assets[aid]
                base = AssetSpec(row, {**row['attributes'], **locked_attrs})
                for field, values in alternatives(row, base).items():
                    if field in locked_attrs:
                        candidates.append((aid, field, values))
            if not candidates:
                return None
            aid, field, values = self.rng.choice(candidates)
            effect[key][aid][field] = self.rng.choice(values)
        return effect

    def preview_context(self, rule, changed=None, *, kind=None):
        if not self.p.ready:
            return None
        kinds = rule['selector'].get('kinds', list(KINDS))
        if kind is not None and kind not in kinds:
            return None
        sampled_kind = self.rng.choice(kinds) if kind is None else kind
        # A rule demonstration has no requested motion. Prefer a still when its
        # scope allows it; video-only rules and explicitly requested videos stay video.
        kind = kind or ('image' if 'image' in kinds else sampled_kind)
        affected = rule['effect'].get('asset_attributes', {})
        if changed and affected:
            affected = {a: attrs for a, attrs in changed.get('asset_attributes', {}).items() if attrs != rule['effect']['asset_attributes'].get(a)}
        aids = [a for a in self.p.ready if (not rule['selector'].get('asset_ids') or a in rule['selector']['asset_ids'])
                and (not affected or a in affected)]
        if not aids:
            return None
        aid = self.rng.choice(aids)
        return dict(kind=kind, operation='generate', asset_ids=[aid], source=self.p.ready[aid])

    def one_time(self, rule, context, effect):
        """Queue a rule overlay for a real request; never create demonstration media."""
        kind = context['kind']
        if not matches(rule, context) or not different_for_kind(rule['effect'], effect, kind):
            raise ValueError('An exception must visibly differ for the requested media kind')
        self.pending = dict(rule_id=rule['id'], outcome='once', kind=kind,
                            effect=deepcopy(effect), direct=True)

    def challenge(self, rule, *, outcome=None, kind=None):
        """The next applicable ordinary generation supplies the content and subject."""
        self.pending = dict(rule_id=rule['id'], outcome=outcome or self.rng.choice(('keep', 'once', 'override')),
                            kind=kind, direct=False)

    def before_generation(self, kind, assets, operation):
        p = self.p
        if not self.pending or operation in ('assemble', 'export') or not p.turns:
            return
        request = p.turns[-1]
        if request['role'] != 'user' or request['intent']['action'] not in GENERATION_REQUESTS:
            return
        pending = self.pending
        rule = self.active.get(pending['rule_id'])
        context = dict(kind=kind, asset_ids=list(assets))
        if rule is None:
            self.pending = None
            return
        if pending['kind'] not in (None, kind) or not matches(rule, context):
            return
        effect = pending.get('effect') or self.changed_effect(rule, kind=kind, once=pending['outcome']=='once')
        if effect is None or not different_for_kind(rule['effect'], effect, kind):
            self.pending = None
            return
        if not different_for_kind(visible_effect(rule['effect'], context), visible_effect(effect, context), kind):
            return  # A subject-specific rule waits for a request involving that subject.
        self.pending = None
        intent = request['intent']
        intent['rule_overlay'] = True
        intent['requested_media_kind'] = kind
        intent['behavior_ids'] = sorted(set(intent['behavior_ids']) | {13})
        new_requirement = effect_text(effect, self.names)
        intent['rule_request'] = dict(kind=kind, requirement=new_requirement,
            requirements=[effect_text({key: value}, self.names) for key, value in effect.items()],
            once=pending['direct'] or (pending['outcome']=='once' and 'aspect_ratio' in rule['effect']),
            applies_to=scope_text(rule, self.names))
        if pending['direct'] or (pending['outcome']=='once' and 'aspect_ratio' in rule['effect']):
            intent.update(rule_overlay=True, rule_event=dict(type='exception', ids=[rule['id']],
                          effect=deepcopy(effect), kind=kind))
            self.exceptions, self.exception_kind, self.overlay_once = {rule['id']: effect}, kind, True
            return
        intent['behavior_ids'] = sorted(set(intent['behavior_ids']) | {14})
        difference = {k:v for k,v in effect.items() if v != rule['effect'].get(k)}
        conflicts = [r for r in self.active.values() if matches(r, context)
                     and any(k in effect_fields(r['effect']) and effect_fields(r['effect'])[k] != value
                             for k,value in effect_fields(difference).items())]
        effects = {}
        for current in conflicts:
            updated = deepcopy(current['effect'])
            for key,value in difference.items():
                if key == 'asset_attributes':
                    for asset,attrs in value.items():
                        for field,chosen in attrs.items():
                            if field in updated.get(key, {}).get(asset, {}):
                                updated[key][asset][field] = chosen
                elif key in updated:
                    updated[key] = value
            effects[current['id']] = updated
        reminder = '; '.join(effect_text(r['effect'], self.names) + ' for ' + scope_text(r, self.names) for r in conflicts)
        event = dict(type='challenge', ids=[r['id'] for r in conflicts], proposed=effect, kind=kind)
        if len(conflicts)>1:
            event['effects'] = effects
        self.emit('assistant', 'clarify_rule_conflict',
                  'I understand the requested content. The existing rule requires ' + reminder
                  + '. Should I keep it, make an exception for this request only, or replace it going forward?',
                  assets=assets, event=event, requested_kind=kind)
        outcome = pending['outcome']
        old_requirement = effect_text(rule['effect'], self.names)
        coverage = scope_text(rule, self.names)
        wording = {
            'keep': f'Keep {old_requirement} for {coverage}; withdraw only my conflicting setting. Proceed with the requested content.',
            'once': f'Use {new_requirement} for this request only, and {old_requirement} for later outputs. Proceed with the requested content.',
            'override': f'From now on use {new_requirement} for {coverage}. Keep other rules and the requested content unchanged.',
        }[outcome]
        event = dict(type='resolve', ids=[r['id'] for r in conflicts], outcome=outcome, effect=effect, kind=kind)
        if len(conflicts)>1:
            event['effects'] = effects
        self.emit('user', 'resolve_rule_conflict', wording, assets=assets, event=event, requested_kind=kind)
        if outcome == 'once':
            self.exceptions, self.exception_kind, self.overlay_once = effects, kind, True
        elif outcome == 'override':
            if len(conflicts)>1:
                self.replace_group(conflicts, effects)
            else:
                spec = {k:deepcopy(rule[k]) for k in ('scope', 'selector', 'effect', 'text')}
                spec['effect'] = effect
                self.replace([rule], spec)
            # The ordinary delivery absorbs this acknowledgement and its event.
            self.confirmation = p.turns[-1]

    def take_confirmation(self, role, outputs):
        if role == 'assistant' and outputs and self.confirmation is not None:
            if self.p.turns[-1] is not self.confirmation:
                raise ValueError('Rule acknowledgement must accompany the pending delivery')
            turn = self.p.turns.pop()
            self.confirmation = None
            return turn
        return None

    def after_delivery(self):
        if self.overlay_once:
            self.exceptions, self.exception_kind, self.overlay_once = {}, None, False

    def _sample_new_rule(self):
        spec = dict(scope=self.rng.choice(('global', 'partial')), selector={}, text='')
        if spec['scope'] == 'partial':
            spec['selector'] = dict(kinds=[self.rng.choice(KINDS)])
        keys = tuple(EFFECTS) if self.sample_time_of_day else ('season', 'aspect_ratio')
        key = self.rng.choice(keys)
        spec['effect'] = {key: self.rng.choice(EFFECTS[key])}
        if key == 'aspect_ratio':
            spec.update(scope='global', selector={})
        if not any(overlaps(rule, spec) for rule in self.active.values()):
            self.start(spec)

    def _advance_legacy(self, can_start):
        """Reproduce the original draw order when replaying a saved plan."""
        draw = self.rng.random()
        if draw < .40 and can_start:
            self._sample_new_rule()
        elif draw < .78 and self.active and self.p.ready:
            self.challenge(self.rng.choice(list(self.active.values())))
        elif draw < .92:
            cancellable = [r for r in self.active.values() if 'aspect_ratio' not in r['effect']]
            if cancellable:
                self.end(self.rng.choice(cancellable))

    def advance(self):
        if not self.enabled or self.closed:
            return
        can_start = len(self.active) < 3 and len(self.records) < 6
        if self.legacy_sampling:
            self._advance_legacy(can_start)
            return

        attempt_start = self.rng.random() < .20
        attempt_challenge = self.rng.random() < .30
        attempt_cancel = self.rng.random() < .20
        previous_rules = list(self.active.values())

        if attempt_start and can_start:
            self._sample_new_rule()
        if attempt_cancel and previous_rules:
            canceled = self.rng.choice(previous_rules)
            if 'aspect_ratio' not in canceled['effect']:
                self.end(canceled)
                previous_rules.remove(canceled)
                if self.pending and self.pending['rule_id'] == canceled['id']:
                    self.pending = None
        # Exclude both rules introduced on this check and rules just canceled.
        if attempt_challenge and previous_rules and self.p.ready:
            self.challenge(self.rng.choice(previous_rules))

    def finish(self):
        self.closed = True
        self.pending = None
        # Accepted global constraints persist into the final cut. Explicit
        # user cancellations during the conversation remain prospective.

    def decorate(self):
        for row in self.p.media.values():
            row['description'] = describe_media(row, self.p.assets)
        records = {r['id']: r for r in self.records}
        requests = {'request_rule_exception','request_asset','request_candidates','request_trial','begin_task','specify_goal','request_variant',
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
                context = row['kind'] + ': ' + ('; '.join(parts) if parts else 'no additional rule treatment applies')
                if context not in contexts:
                    contexts.append(context)
            turn['intent']['media_rule_context'] = contexts
            # Attach the exact operation scope to its request, never an unrelated adoption.
            for previous in reversed(self.p.turns[:index]):
                if previous['output_media']:
                    break
                if previous['role'] == 'user' and previous['intent']['action'] in requests:
                    previous['intent']['media_rule_context'] = contexts
                    break



def validate_rule_timeline(plan):
    """Replay confirmations; verify snapshots, exceptions, cancellations and terminal closure."""
    if 'rules' not in plan:
        return True
    media = {m['placeholder']: m for m in plan['placeholders']}
    active, records, pending, resolution, once = {}, {}, None, None, {}
    once_kind = None
    for index, turn in enumerate(plan['turns']):
        event = turn['intent'].get('rule_event', {})
        kind = event.get('type')
        event_actions = {'challenge': 'clarify_rule_conflict', 'resolve': 'resolve_rule_conflict',
                         'exception': 'request_rule_exception', 'start': 'confirm_cross_turn_rule', 'end': 'confirm_rule_cancel', 'replace': 'confirm_rule_resolution'}
        overlay = turn['intent'].get('rule_overlay') and ((kind == 'exception' and turn['role'] == 'user'
                   and turn['intent']['action'] in GENERATION_REQUESTS)
                   or (kind == 'replace' and turn['role'] == 'assistant' and turn['output_media']))
        if kind and (kind not in event_actions or (turn['intent']['action'] != event_actions[kind] and not overlay)):
            raise ValueError('Rule event must be attached to its actual dialogue action')
        if turn['intent']['action'] in event_actions.values() and not kind:
            raise ValueError('Missing structured rule event')
        if kind == 'exception':
            if pending or once or turn['role'] != 'user' or any(rid not in active for rid in event['ids']):
                raise ValueError('Invalid one-request exception')
            once_kind = event['kind']
            if once_kind not in KINDS or not all(different_for_kind(active[rid]['effect'], event['effect'], once_kind) for rid in event['ids']):
                raise ValueError('Exception repeats the existing output attributes')
            once = {rid: deepcopy(event['effect']) for rid in event['ids']}
        elif kind == 'challenge':
            if pending or any(r not in active for r in event['ids']):
                raise ValueError('Rule conflict references an inactive rule or unresolved conflict')
            if event.get('kind') in KINDS and not all(different_for_kind(active[rid]['effect'], event['proposed'], event['kind']) for rid in event['ids']):
                raise ValueError('Conflict does not change the requested media kind')
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
                once_kind = event.get('kind')
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
            if 'aspect_ratio' in active[event['id']]['effect']:
                raise ValueError('Aspect ratio rules cannot be canceled')
            records[event['id']]['ended_at'] = turn['id']
            del active[event['id']]
        for token in turn['output_media']:
            if pending:
                raise ValueError('Media generated before rule conflict resolution')
            row = media[token]
            state, applied = {}, []
            for rule in active.values():
                if matches(rule, row):
                    effect = once.get(rule['id'], rule['effect']) if once_kind in (None, row['kind']) else rule['effect']
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
            once_kind = None
    if pending or once or (resolution and resolution['outcome']=='override') or list(records.values()) != plan['rules']:
        raise ValueError('Unfinished rule lifecycle or inconsistent rule history')
    return True
