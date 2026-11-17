"""Executable placeholder construction contracts, independent of dialogue chronology."""
from copy import deepcopy
import hashlib
import json
import re

from scripts.dialogue_generation.cross_turn_rule import visible_effect
from scripts.dialogue_generation.generation_requirements import assign_generation_requirements, image_layout

VERSION = 5
FLUX_MODEL_ID = 'black-forest-labs/FLUX.2-klein-4B'


def asset_state(row, aid, targets):
    attrs = deepcopy(row.get('effective_attributes', row.get('attributes', targets[aid]['attributes'])))
    state = visible_effect(row.get('render_state', {}), row)
    attrs.update(state.get('asset_attributes', {}).get(aid, {}))
    if 'season' in state and 'season' in attrs:
        attrs['season'] = state['season']
    return attrs


def changes_between(before, after, prefix=''):
    changes = []
    for key in sorted(before.keys() | after.keys()):
        old, new = before.get(key), after.get(key)
        field = prefix + key
        if isinstance(old, dict) or isinstance(new, dict):
            changes.extend(changes_between(old or {}, new or {}, field + '.'))
        elif old != new:
            changes.append(dict(field=field, before=old, after=new))
    return changes


def treatment(row):
    state = visible_effect(row.get('render_state', {}), row)
    return {k: v for k, v in state.items() if k != 'asset_attributes'} | {
        'color_mode': state.get('color_mode', 'original_colors'),
        'season': state.get('season', 'no added seasonal effects')}


def clip_asset_states(row, aid, media, targets):
    """Follow realized clip bindings, including treatments already applied to it."""
    if row['purpose'] == 'asset':
        return [asset_state(row, aid, targets)]
    bound = row.get('bindings', {}).get(aid)
    children = [bound] if bound else [p for p in row['inputs'] if aid in media[p]['asset_ids']]
    states = [state for p in children for state in clip_asset_states(media[p], aid, media, targets)]
    states = states or [deepcopy(targets[aid]['attributes'])]
    for attrs in states:
        attrs.update(row.get('render_state', {}).get('asset_attributes', {}).get(aid, {}))
        if 'season' in row.get('render_state', {}) and 'season' in attrs:
            attrs['season'] = row['render_state']['season']
    return states


def clip_treatments(row, media):
    if row['purpose'] not in ('compilation', 'final_export'):
        return [treatment(row)]
    states = [state for p in row['inputs'] for state in clip_treatments(media[p], media)]
    for state in states:
        state.update({k: v for k, v in row.get('render_state', {}).items() if k != 'asset_attributes'})
    return states or [treatment(row)]


def action_context(row, observation):
    """Target context grounds inferred trial relationships and unchanged scene content."""
    aids = set(row['asset_ids'])
    shots = sorted(observation['shots'], key=lambda s: (-len(aids & set(s['assets'])), len(set(s['assets'])-aids)))
    return [dict(description=s['description'], assets=s['assets']) for s in shots
            if aids & set(s['assets'])][:1]


def _clean_text(text):
    # Older runs contain the retired static-trial restriction. Keep the archive
    # intact, but never propagate that obsolete default into expert instructions.
    text = re.sub(r'(?i)static test clip with a locked camera:\s*', 'Short trial: ', text)
    text = re.sub(r'(?i)static composition,\s*no character action\.?|no character action\.?', '', text)
    return re.sub(r'\[\[(?:image|video)_\d+\]\]', 'the referenced media', text)


SEASONAL_ACCENTS = {
    'spring': 'a few scattered blossoms and green leaves around the subject base',
    'summer': 'one small lush tree at the edge of the setup and warm sunlight',
    'autumn': 'fallen autumn leaves arranged around the subject base',
    'fall': 'fallen autumn leaves arranged around the subject base',
    'winter': 'visible snow around the subject base and resting on the subject surface',
}


def clean_current_context(text, states, targets):
    """Drop obsolete attribute phrases; the selected generated versions are authoritative."""
    stale = set()
    for aid, attrs in states.items():
        for key, before in targets[aid]['attributes'].items():
            if attrs.get(key, before) != before and before not in ('unknown', 'not applicable'):
                stale.add(before)
    for phrase in sorted(stale, key=len, reverse=True):
        text = re.sub(r'(?i)(?<!\w)' + re.escape(phrase) + r'(?!\w)', '', text)
    return re.sub(r' +', ' ', text).strip()


def constituent_clips(token, media):
    row=media[token]
    if row['purpose']=='compilation':
        return [clip for sid in row['shot_ids'] for clip in constituent_clips(row['shot_versions'][sid],media)]
    if row['purpose']=='final_export':
        return constituent_clips(row['inputs'][0],media)
    return [dict(placeholder=token, **({'shot_id':row['shot_ids'][0]} if len(row['shot_ids'])==1 else {}))]


def build_tasks(plan, observation):
    """Realize the conversation forwards. The observed video is textual guidance only."""
    assign_generation_requirements(plan)
    media = {m['placeholder']: m for m in plan['placeholders']}
    targets = plan['target_assets']
    shots = {s['id']: s for s in observation['shots']}
    tasks = {}
    for token, row in media.items():
        refs = row['reference_materials']
        task = dict(version=VERSION, operation='', backend=None, prompt=None,
            references=[], dependencies=list(refs),
            desired_description=_clean_text(row['description']), changes=[],
            asset_ids=row['asset_ids'], shot_ids=row['shot_ids'], asset_names={a:targets[a]['name'] for a in row['asset_ids']},
            render_state=visible_effect(row.get('render_state', {}), row),
            generation_requirements=deepcopy(row['generation_requirements']))
        if row['kind'] == 'image':
            task['layout'] = image_layout(row, targets)
        # One slot per actual causal input. A video-to-image edge retrieves only
        # the requested identity, from this already-generated clip.
        for ref in refs:
            source = media[ref]
            slot = dict(placeholder=ref, kind=source['kind'], role='identity/design reference')
            common = [aid for aid in row['asset_ids'] if aid in source['asset_ids']]
            if common:
                slot['asset_ids'] = common
                slot['subjects'] = [targets[aid]['name'] for aid in common]
                if source['kind'] == 'image' and source['purpose'] == 'asset':
                    slot['appearance'] = [dict(subject=targets[aid]['name'],
                        attributes=asset_state(source, aid, targets)) for aid in common]
            if row['kind'] == 'image' and source['kind'] == 'video':
                slot.update(kind='image', crop_query='; '.join(targets[aid]['name'] for aid in common)
                    or _clean_text(row['description']))
            task['references'].append(slot)
        if row['purpose'] == 'asset':
            aid = row['asset_ids'][0]
            task['subject'] = targets[aid]['name']
            task['desired_description'] = targets[aid]['description']
            task['desired_attributes'] = asset_state(row, aid, targets)
            task['backend'] = 'flux' if row['kind'] == 'image' else 'vdn_h3'
            parent = media.get(row.get('edit_parent')) or (media[refs[0]] if refs else None)
            if parent:
                task['operation'] = 'local_edit' if row['kind'] == 'image' else ('video_edit' if parent['kind'] == 'video' else 'reference_generation')
                task['references'][0]['role'] = 'base to edit' if row['kind'] == 'image' else 'source video to edit'
                task['changes'] = changes_between(asset_state(parent, aid, targets), asset_state(row, aid, targets))
                task['changes'] += changes_between(treatment(parent), treatment(row), 'render.')
                task['changes'] += changes_between(parent.get('generation_requirements', {}), row['generation_requirements'], 'output.')
                if row['kind'] == 'image' and parent['kind'] == 'video':
                    task['operation'] = 'reference_generation'
                    task['references'][0]['role'] = 'identity to present as an independent asset image'
                elif not task['changes'] and parent['kind'] == row['kind']:
                    task.update(operation='copy', backend=None)
            else:
                task['operation'] = 'reference_generation' if refs else 'text_generation'
        elif row['purpose'] in ('compilation', 'final_export'):
            task['references'] = []
            task['clips'] = ([dict(placeholder=row['shot_versions'][sid], shot_id=sid) for sid in row['shot_ids']]
                if row['purpose'] == 'compilation' else [dict(placeholder=ref) for ref in refs])
            # Reordering itself is always a deterministic concatenation. Only an
            # additional explicit appearance change calls the video expert.
            for clip in task['clips']:
                source = media[clip['placeholder']]
                for field, value in row.get('render_state', {}).items():
                    if field == 'asset_attributes':
                        for aid, attrs in value.items():
                            if aid in source['asset_ids']:
                                for key, after in attrs.items():
                                    before = list(dict.fromkeys(state.get(key) for state in clip_asset_states(source, aid, media, targets)))
                                    if any(v != after for v in before):
                                        task['changes'].append(dict(field=targets[aid]['name']+'.'+key, before=before, after=after))
                    else:
                        before = list(dict.fromkeys(state.get(field) for state in clip_treatments(source, media)))
                        if any(v != value for v in before):
                            task['changes'].append(dict(field='render.'+field, before=before, after=value))
            task['operation'] = 'concatenate' if len(task['clips']) > 1 else 'copy'
            if task['changes']:
                task['clips'] = [part for clip in task['clips'] for part in constituent_clips(clip['placeholder'],media)]
                task['dependencies'] = list(dict.fromkeys(task['dependencies']+[c['placeholder'] for c in task['clips']]))
                task['clip_edits'] = []
                for clip in task['clips']:
                    source = media[clip['placeholder']]
                    relevant = [c for c in task['changes'] if c['field'].startswith('render.') or
                        any(c['field'].startswith(targets[aid]['name']+'.') for aid in source['asset_ids'])]
                    if relevant:
                        child = dict(version=VERSION, operation='video_edit', backend='vdn_h3', prompt=None,
                            references=[dict(placeholder=clip['placeholder'],kind='video',role='source clip to edit',label='<Video 1>')],
                            dependencies=[clip['placeholder']], asset_ids=source['asset_ids'],shot_ids=source['shot_ids'],asset_names={a:targets[a]['name'] for a in source['asset_ids']},
                            changes=deepcopy(relevant),render_state=deepcopy(task['render_state']),generation_requirements={},
                            desired_description='Edit this individual existing clip only.',seconds=4.,output_seconds=4.)
                        if len(source['shot_ids'])==1:
                            child['source_action']=shots[source['shot_ids'][0]]['description']
                        task['clip_edits'].append(dict(placeholder=clip['placeholder'],generation=child))
                task['changes'] = []
        else:
            task['backend'] = 'flux' if row['kind'] == 'image' else 'vdn_h3'
            task['operation'] = 'reference_generation' if refs else 'text_generation'
            if row['purpose'] == 'shot':
                sid = row['shot_ids'][0]
                task['shot_id'] = sid
                task['desired_description'] = shots[sid]['description']
                task['source_action'] = shots[sid]['description']
                prior = next((media[ref] for ref in refs if media[ref]['purpose'] == 'shot'), None)
                if prior:
                    task['operation'] = 'video_edit'
                    task['references'][0]['role'] = 'source video to edit'
                    for aid, bound in row['bindings'].items():
                        before = asset_state(media[prior['bindings'][aid]], aid, targets)
                        task['changes'] += changes_between(before, asset_state(media[bound], aid, targets), targets[aid]['name']+'.')
                    task['changes'] += changes_between(treatment(prior), treatment(row), 'render.')
                task['subject_bindings'] = [dict(subject=targets[aid]['name'], reference=bound)
                    for aid, bound in row.get('bindings', {}).items()]
            else:
                if row['kind'] == 'video':
                    # Trial prose used to concatenate a guide caption and a
                    # conflicting variation. Let current refs supply appearance.
                    task['desired_description'] = 'Short action trial featuring the supplied subjects.'
                    task['action_context'] = action_context(row, observation)
                else:
                    task['desired_description'] = 'Independent asset image of ' + ', '.join(targets[a]['name'] for a in row['asset_ids'])
        states = {aid: asset_state(media[ref], aid, targets) for ref in refs
                  for aid in media[ref]['asset_ids'] if media[ref]['purpose'] == 'asset'}
        if row['purpose'] == 'asset':
            states[row['asset_ids'][0]] = task['desired_attributes']
        for key in ('desired_description', 'source_action'):
            if task.get(key):
                task[key] = clean_current_context(task[key], states, targets)
        for context in task.get('action_context', []):
            context['description'] = clean_current_context(context['description'], states, targets)
        if row['kind'] == 'image':
            season = task['render_state'].get('season')
            task['presentation'] = dict(style='minimalistic studio product photograph' if task['layout'] in ('product', 'portrait') else 'coherent environment or scene image',
                backdrop='plain neutral studio background' if task['layout'] in ('product', 'portrait') else 'retain the requested environment',
                season=season or 'none', seasonal_accents=SEASONAL_ACCENTS.get(season, 'none'))
        if task['backend'] == 'vdn_h3':
            # Guide shot lengths do not determine synthetic clip length.
            task.update(seconds=4., output_seconds=4.)
        from tools.placeholder_prompter import writer_system, writer_input
        for request in [task] + [part['generation'] for part in task.get('clip_edits', [])]:
            if request['backend']:
                counters = {'image': 0, 'video': 0}
                for ref in request['references']:
                    counters[ref['kind']] += 1
                    label = 'Image' if request['backend']=='flux' else ('Picture' if ref['kind']=='image' else 'Video')
                    ref['label'] = f'<{label} {counters[ref["kind"]]}>'
                request['model_id'] = FLUX_MODEL_ID if request['backend']=='flux' else 'OpenVDN/vdn-minimax-h3-edge'
                request['writer_system_hash'] = hashlib.sha256(writer_system(request).encode()).hexdigest()
                request['writer_input_hash'] = hashlib.sha256(writer_input(request).encode()).hexdigest()
        task['fingerprint'] = hashlib.sha256(json.dumps(task, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        tasks[token] = task
    return tasks


def construction_order(tasks):
    order, active, done = [], set(), set()
    def visit(token):
        if token in active:
            raise ValueError('Cyclic generation dependencies')
        if token not in tasks:
            raise ValueError('Unknown generation dependency: ' + token)
        if token in done:
            return
        active.add(token)
        for ref in tasks[token]['dependencies']:
            visit(ref)
        active.remove(token)
        done.add(token)
        order.append(token)
    for token in tasks:
        visit(token)
    return order
