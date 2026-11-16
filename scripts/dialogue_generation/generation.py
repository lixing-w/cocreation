"""Executable placeholder construction contracts, independent of dialogue chronology."""
from copy import deepcopy
import hashlib
import json
import re

from scripts.dialogue_generation.cross_turn_rule import visible_effect

VERSION = 2


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
        'color_mode': state.get('color_mode', 'original_colors')}


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
            if aids & set(s['assets'])][:2]


def _clean_text(text):
    # Older runs contain the retired static-trial restriction. Keep the archive
    # intact, but never propagate that obsolete default into expert instructions.
    text = re.sub(r'(?i)static test clip with a locked camera:\s*', 'Short trial: ', text)
    text = re.sub(r'(?i)static composition,\s*no character action\.?|no character action\.?', '', text)
    return re.sub(r'\[\[(?:image|video)_\d+\]\]', 'the referenced media', text)


def build_tasks(plan, observation):
    media = {m['placeholder']: m for m in plan['placeholders']}
    root, targets = plan['final']['media'], plan['target_assets']
    assets = {a['id']: a for a in observation['assets']}
    shots = {s['id']: s for s in observation['shots']}
    tasks = {}
    for token, row in media.items():
        task = dict(version=VERSION, operation='', backend=None, prompt=None,
                    references=[], dependencies=list(row['reference_materials']),
                    desired_description=_clean_text(row['description']), changes=[],
                    asset_ids=row['asset_ids'], shot_ids=row['shot_ids'],
                    render_state=visible_effect(row.get('render_state', {}), row))
        refs = row['reference_materials']
        if token == root:
            task.update(operation='source', source_video=plan['source_video'], dependencies=[])
        elif row['purpose'] == 'asset' and row['kind'] == 'image':
            aid = row['asset_ids'][0]
            source = media[refs[0]]
            task.update(backend='flux', model_id='black-forest-labs/FLUX.2-dev')
            if refs == [root]:
                task['operation'] = 'asset_reconstruction'
                task['layout'] = assets[aid].get('reference_layout',
                    'three_view' if targets[aid]['kind'] in ('person', 'object') else 'environment')
                task['desired_attributes'] = asset_state(row, aid, targets)
                task['references'] = [dict(placeholder=root, kind='image',
                    role='identity from the target-video frame', description=targets[aid]['description'],
                    attributes=targets[aid]['attributes'], frame_seconds=assets[aid]['frames'][0])]
            else:
                task['operation'] = 'local_edit'
                task['changes'] = changes_between(asset_state(source, aid, targets), asset_state(row, aid, targets), aid + '.')
                task['changes'] += changes_between(treatment(source), treatment(row), 'render.')
                task['references'] = [dict(placeholder=refs[0], kind=source['kind'], role='base to edit',
                                            description=_clean_text(source['description']))]
                if not task['changes']:
                    task.update(operation='extract_frame' if source['kind'] == 'video' else 'copy', backend=None)
                    if source['kind'] == 'video':
                        task['frame_seconds'] = 0.0
        elif row['purpose'] == 'shot' or (row['purpose'] in ('exploration', 'asset') and row['kind'] == 'video'
                and any(set(row['asset_ids']) <= set(s['assets']) for s in observation['shots'])):
            trial = row['purpose'] != 'shot'
            candidates = [s for s in observation['shots'] if set(row['asset_ids']) <= set(s['assets'])]
            selected = min(candidates, key=lambda s: (len(s['assets']), observation['shots'].index(s))) if trial else shots[row['shot_ids'][0]]
            sid = selected['id']
            task['desired_description'] = selected['description']
            task['trial'] = trial
            if root not in task['dependencies']:
                task['dependencies'].insert(0, root)
            # Hidden target source establishes action, composition, relationships
            # and unchanged identities, even when they are not named in dialogue.
            bindings = dict(row.get('bindings', {}))
            if trial:
                for ref in row['inputs']:
                    for aid in media[ref]['asset_ids']:
                        if aid in row['asset_ids']:
                            bindings[aid] = ref
            if row['purpose'] == 'asset':
                bindings = {row['asset_ids'][0]: refs[0]}
            task['scene_assets'] = selected['assets']
            task['render_state'] = visible_effect(row.get('render_state', {}), {'asset_ids': selected['assets']})
            for aid in task['render_state'].get('asset_attributes', {}):
                bindings.setdefault(aid, root)

            task['shot_id'] = sid
            task['references'] = [dict(placeholder=root, kind='video', role='source target-video shot to edit',
                                        description=shots[sid]['description'], shot_id=sid)]
            for aid, bound in bindings.items():
                desired = asset_state(row if row['purpose'] == 'asset' else media[bound], aid, targets)
                desired.update(task['render_state'].get('asset_attributes', {}).get(aid, {}))
                if 'season' in row.get('render_state', {}) and 'season' in desired:
                    desired['season'] = row['render_state']['season']
                delta = changes_between(targets[aid]['attributes'], desired, aid + '.')
                task['changes'] += delta
                if delta and bound != root:
                    task['references'].append(dict(placeholder=bound, kind=media[bound]['kind'],
                        role='desired design for ' + targets[aid]['name'], description=_clean_text(media[bound]['description'])))
                    task['dependencies'].append(bound)
            task['changes'] += changes_between({'color_mode': 'original_colors'}, treatment(row), 'render.')
            task['operation'] = 'video_edit' if task['changes'] else 'extract_shot'
            task['backend'] = 'vdn_h3' if task['changes'] else None
        elif row['purpose'] in ('compilation', 'final_export'):
            task['clips'] = [dict(placeholder=ref, shot_id=sid) for sid, ref in row.get('shot_versions', {}).items()]
            if row['purpose'] == 'compilation':
                # shot_ids is the authoritative order, never dict insertion or a model's guess.
                task['clips'] = [dict(placeholder=row['shot_versions'][sid], shot_id=sid) for sid in row['shot_ids']]
            else:
                task['clips'] = [dict(placeholder=ref) for ref in refs]
            for clip in task['clips']:
                source_clip = media[clip['placeholder']]
                # An absent assembly rule preserves each input's actual appearance.
                # Explicit rules that already hold must not turn a reorder into H3.
                active = {k: v for k, v in row.get('render_state', {}).items() if k != 'asset_attributes'}
                before_states = clip_treatments(source_clip, media)
                for field, value in active.items():
                    old = list(dict.fromkeys(state.get(field) for state in before_states))
                    if any(v != value for v in old):
                        task['changes'].append(dict(field='render.'+field, before=old[0] if len(old)==1 else old,
                                                    after=value, clip=clip['placeholder']))
                for aid, attrs in row.get('render_state', {}).get('asset_attributes', {}).items():
                    if aid in media[clip['placeholder']]['asset_ids']:
                        for field, value in attrs.items():
                            old = list(dict.fromkeys(state.get(field) for state in clip_asset_states(source_clip, aid, media, targets)))
                            if any(v != value for v in old):
                                task['changes'].append(dict(field=aid+'.'+field, before=old[0] if len(old)==1 else old,
                                                            after=value, clip=clip['placeholder']))
            task['operation'] = 'video_edit' if task['changes'] else ('concatenate' if len(task['clips']) > 1 else 'copy')
            if task['changes']:
                task['backend'] = 'vdn_h3'
                task['preprocess'] = 'concatenate'
                task['references'] = [dict(kind='video', role='assembled source video to edit',
                    description='Assembly in this exact order: ' + '; '.join(_clean_text(media[c['placeholder']]['description']) for c in task['clips']))]
        else:
            task.update(operation='reference_generation', backend='flux' if row['kind'] == 'image' else 'vdn_h3')
            task['references'] = [dict(placeholder=ref, kind=media[ref]['kind'], role='target-video action/composition context' if ref == root else 'visible identity/design reference',
                                      description=_clean_text(media[ref]['description'])) for ref in refs]
            if row['kind'] == 'video':
                task['action_context'] = action_context(row, observation)
        task['dependencies'] = list(dict.fromkeys(task['dependencies']))
        if task['backend'] == 'flux':
            for ref in task['references']:
                if ref['kind'] == 'video':
                    ref.update(kind='image', frame_seconds=0.0, role=ref['role'] + ' from a representative video frame')
        if task['backend']:
            if task['backend'] == 'vdn_h3':
                # The expert cannot generate <5 seconds: produce at least 5s and
                # deterministically trim to the observed shot duration afterward.
                shot = shots.get(task.get('shot_id'))
                duration = shot.get('end', 0)-shot.get('start', 0) if shot else 5.0
                if task.get('preprocess'):
                    duration = sum(shots[sid].get('end', 5)-shots[sid].get('start', 0) for sid in row['shot_ids'])
                if duration <= 0:
                    duration = 5.0
                if duration > 15:
                    raise ValueError('H3 edit exceeds 15 seconds; split the assembly into shots before editing')
                task['seconds'] = max(5.0, duration)
                task['output_seconds'] = duration
            counters = {'image': 0, 'video': 0}
            for ref in task['references']:
                counters[ref['kind']] += 1
                label_kind = 'Image' if task['backend'] == 'flux' else ('Picture' if ref['kind'] == 'image' else 'Video')
                ref['label'] = f'<{label_kind} {counters[ref["kind"]]}>'
            task['model_id'] = 'black-forest-labs/FLUX.2-dev' if task['backend'] == 'flux' else 'OpenVDN/vdn-minimax-h3-edge'
        if task['backend']:
            from tools.flux_prompter_prompts import ASSET_SYSTEM
            from tools.h3_prompter_prompts import VIDEO_SYSTEM
            task['writer_system_hash'] = hashlib.sha256((ASSET_SYSTEM if task['backend'] == 'flux' else VIDEO_SYSTEM).encode()).hexdigest()
        # Fingerprint includes reference descriptions/states, not just the leaf prompt.
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
