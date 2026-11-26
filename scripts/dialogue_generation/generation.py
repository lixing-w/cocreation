"""Executable placeholder construction contracts, independent of dialogue chronology."""
from copy import deepcopy
import hashlib
import json
import re

from scripts.dialogue_generation.cross_turn_rule import SCENE_FIELDS, visible_effect
from scripts.dialogue_generation.generation_requirements import assign_generation_requirements, image_layout

VERSION = 11
FLUX_MODEL_ID = 'black-forest-labs/FLUX.2-klein-4B'
H3_IMAGE_MODEL_ID = 'MiniMaxAI/MiniMax-H3-Image'


def asset_state(row, aid, targets):
    attrs = deepcopy(row.get('effective_attributes', row.get('attributes', targets[aid]['attributes'])))
    state = visible_effect(row.get('render_state', {}), row)
    attrs.update(state.get('asset_attributes', {}).get(aid, {}))
    attrs.update({field: state[field] for field in SCENE_FIELDS if field in state and field in attrs})
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
    return {k: v for k, v in state.items() if k not in ('asset_attributes', 'aspect_ratio')} | {
        'season': state.get('season', 'no added seasonal effects')}


def clip_asset_states(row, aid, media, targets):
    """Follow realized clip bindings, including treatments already applied to it."""
    if row['operation'] == 'create_reference':
        return [asset_state(row, aid, targets)]
    bound = row.get('bindings', {}).get(aid)
    children = [bound] if bound else [p for p in row['inputs'] if aid in media[p]['asset_ids']]
    states = [state for p in children for state in clip_asset_states(media[p], aid, media, targets)]
    states = states or [deepcopy(targets[aid]['attributes'])]
    for attrs in states:
        attrs.update(row.get('render_state', {}).get('asset_attributes', {}).get(aid, {}))
        state = row.get('render_state', {})
        attrs.update({field: state[field] for field in SCENE_FIELDS if field in state and field in attrs})
    return states


def clip_treatments(row, media):
    if row['operation'] not in ('assemble', 'export'):
        return [treatment(row)]
    states = [state for p in row['inputs'] for state in clip_treatments(media[p], media)]
    for state in states:
        state.update({k: v for k, v in row.get('render_state', {}).items() if k != 'asset_attributes'})
    return states or [treatment(row)]


def _clean_text(text):
    # Older runs contain the retired static-trial restriction. Keep the archive
    # intact, but never propagate that obsolete default into expert instructions.
    text = re.sub(r'(?i)static test clip with a locked camera:\s*', 'Short trial: ', text)
    text = re.sub(r'(?i)static composition,\s*no character action\.?|no character action\.?', '', text)
    return re.sub(r'\[\[(?:image|video)_\d+\]\]', 'the referenced media', text)


def relationship_context(asset_ids, targets):
    """Keep identity-caption links that a short shot action can leave implicit."""
    result = []
    for aid in asset_ids:
        caption = targets[aid]['description']
        related = [targets[other]['name'] for other in asset_ids if other != aid
                   and re.search(r'(?i)\b' + re.escape(targets[other]['name']) + r'\b', caption)]
        if related:
            result.append(dict(subject=targets[aid]['name'], related_subjects=related,
                               relationship_source=caption))
    return result


SEASONAL_ACCENTS = {
    'spring': 'a few scattered blossoms and green leaves around the subject base',
    'summer': 'one small lush tree at the edge of the setup and warm sunlight',
    'autumn': 'fallen autumn leaves arranged around the subject base',
    'fall': 'fallen autumn leaves arranged around the subject base',
    'winter': 'visible snow around the subject base and resting on the subject surface',
}



def constituent_clips(token, media):
    row=media[token]
    if row['operation']=='assemble':
        return [clip for sid in row['shot_ids'] for clip in constituent_clips(row['shot_versions'][sid],media)]
    if row['operation']=='export':
        return constituent_clips(row['inputs'][0],media)
    return [dict(placeholder=token, **({'shot_id':row['shot_ids'][0]} if len(row['shot_ids'])==1 else {}))]


def assembly_clip_changes(source, render_state, media, targets):
    """Diff each actual clip; another clip's stale treatment is not its edit."""
    changes = []
    for field, value in render_state.items():
        if field == 'asset_attributes':
            for aid, attrs in value.items():
                if aid in source['asset_ids']:
                    for key, after in attrs.items():
                        before = list(dict.fromkeys(state.get(key) for state in clip_asset_states(source, aid, media, targets)))
                        if any(v != after for v in before):
                            changes.append(dict(field=targets[aid]['name']+'.'+key, before=before, after=after))
        elif field != 'aspect_ratio':
            before = list(dict.fromkeys(state.get(field) for state in clip_treatments(source, media)))
            if any(v != value for v in before):
                changes.append(dict(field='render.'+field, before=before, after=value))
    return changes


def build_tasks(plan, observation):
    """Realize the conversation forwards. The observed video is textual guidance only."""
    assign_generation_requirements(plan)
    media = {m['placeholder']: m for m in plan['placeholders']}
    targets = plan['target_assets']
    shots = {s['id']: s for s in observation['shots']}
    tasks = {}
    for token, row in media.items():
        refs = row['reference_materials']
        task = dict(version=VERSION, kind=row['kind'], operation='', backend=None, prompt=None,
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
                if source['kind'] == 'image' and source['operation'] == 'create_reference':
                    slot['appearance'] = [dict(subject=targets[aid]['name'],
                        attributes=asset_state(source, aid, targets)) for aid in common]
            if row['kind'] == 'image' and source['kind'] == 'video':
                slot.update(kind='image', crop_query='; '.join(targets[aid]['name'] for aid in common)
                    or _clean_text(row['description']))
            task['references'].append(slot)
        if row['operation'] == 'create_reference':
            aid = row['asset_ids'][0]
            task['subject'] = targets[aid]['name']
            task['desired_description'] = targets[aid]['description']
            task['desired_attributes'] = asset_state(row, aid, targets)
            task['backend'] = 'h3_image' if row['kind'] == 'image' else 'minimax_h3'
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
        elif row['operation'] in ('assemble', 'export'):
            task['references'] = []
            task['clips'] = ([dict(placeholder=row['shot_versions'][sid], shot_id=sid) for sid in row['shot_ids']]
                if row['operation'] == 'assemble' else [dict(placeholder=ref) for ref in refs])
            # Assembly preserves existing clip content. Rules introduced after a
            # source was generated cannot silently turn an ordering request into
            # an appearance edit. Explicit clip revisions are separate tasks.
            task['operation'] = 'concatenate' if len(task['clips']) > 1 else 'copy'
            task['assembly_version'] = 2
            task['render_state'] = {key:value for key,value in task['render_state'].items() if key=='aspect_ratio'}
        else:
            task['backend'] = 'h3_image' if row['kind'] == 'image' else 'minimax_h3'
            task['operation'] = 'reference_generation' if refs else 'text_generation'
            if row['operation'] == 'generate_clip':
                sid = row['shot_ids'][0]
                task['shot_id'] = sid
                task['desired_description'] = shots[sid]['description']
                task['source_action'] = shots[sid]['description']
                prior = next((media[ref] for ref in refs if media[ref]['operation'] == 'generate_clip'), None)
                if prior:
                    task['operation'] = 'video_edit'
                    task['references'][0]['role'] = 'source video to edit'
                    for aid, bound in row['bindings'].items():
                        before = asset_state(media[prior['bindings'][aid]], aid, targets)
                        task['changes'] += changes_between(before, asset_state(media[bound], aid, targets), targets[aid]['name']+'.')
                    task['changes'] += changes_between(treatment(prior), treatment(row), 'render.')
                task['subject_bindings'] = [dict(subject=targets[aid]['name'], reference=bound)
                    for aid, bound in row.get('bindings', {}).items()]
                task['subject_relationship_context'] = relationship_context(row['asset_ids'], targets)
            else:
                task['desired_description'] = 'Independent ' + row['kind'] + ' featuring ' + ', '.join(targets[a]['name'] for a in row['asset_ids'])
                task['scene_invention'] = dict(subjects=[targets[a]['name'] for a in row['asset_ids']],
                    instruction='Invent one coherent scene using only these supplied subjects, their current attributes and references. '
                    'For a video choose a simple feasible action; for an image choose a coherent still composition. '
                    'Use a minimal unobtrusive backdrop if no environment is supplied. Do not add an unprovided person, animal, '
                    'vehicle or salient prop, or import a setting or event from another scene or example.')
        states = {aid: asset_state(media[ref], aid, targets) for ref in refs
                  for aid in media[ref]['asset_ids'] if media[ref]['operation'] == 'create_reference'}
        if row['operation'] == 'create_reference':
            states[row['asset_ids'][0]] = task['desired_attributes']
        for aid, bound in row.get('bindings', {}).items():
            states[aid] = asset_state(media[bound], aid, targets)
        states.update(deepcopy(row.get('related_subject_states', {})))
        if task.get('scene_invention'):
            states = {aid: attrs for aid, attrs in states.items() if aid in row['asset_ids']}
        for aid, attrs in states.items():
            attrs.update(task['render_state'].get('asset_attributes', {}).get(aid, {}))
            state = task['render_state']
            attrs.update({field: state[field] for field in SCENE_FIELDS if field in state and field in attrs})
        task['attribute_overrides'] = [dict(subject=targets[aid]['name'], attributes=deepcopy(attrs)) for aid, attrs in states.items()]
        if row['kind'] == 'image':
            season = task['render_state'].get('season')
            task['presentation'] = dict(style='minimalistic studio product photograph' if task['layout'] in ('product', 'portrait') else 'coherent environment or scene image',
                backdrop='plain neutral studio background' if task['layout'] in ('product', 'portrait') else 'retain the requested environment',
                color_mode='natural color photograph',
                season=season or 'none', seasonal_accents=SEASONAL_ACCENTS.get(season, 'none'))
            if 'time_of_day' in task['render_state']:
                task['presentation']['time_of_day'] = task['render_state']['time_of_day']
        if task['backend'] == 'minimax_h3':
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
                request['model_id'] = {'flux': FLUX_MODEL_ID, 'h3_image': H3_IMAGE_MODEL_ID,
                                       'minimax_h3': 'Comfy-Org/MiniMax-H3'}[request['backend']]
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
