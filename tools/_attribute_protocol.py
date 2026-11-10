"""Pure nested-attribute adapters, bounded selection and provenance bookkeeping."""
import copy
import re

from tools.extract_attribute_nodes import validate_attributes
from tools.extract_attribute_nodes_prompts import CATEGORY_RULES, OBJECT_FIELDS, TOP_LEVEL_CATEGORIES


def atomize(observation, provenance=None):
    """Stable IDs for an immutable observation; paths index its unmodified JSON.

    A record is one scalar, one string-list item or one relation/viewpoint/text
    object. Container descriptions count toward K. UIDs/links never count alone.
    Object UIDs are globally namespaced; source strings remain verbatim in text.
    """
    artifact = observation['artifact_id']
    validate_attributes(observation, artifact, observation['kind'])
    records = []
    provenance = provenance or {}
    for category in TOP_LEVEL_CATEGORIES:
        if category not in observation['attributes']:
            continue
        value = observation['attributes'][category]
        if category not in OBJECT_FIELDS:
            records.append(dict(category=category, text=value, path=['attributes', category]))
            continue
        for index, entry in enumerate(value):
            owner = artifact + '_' + entry['uid']
            base = ['attributes', category, index]
            if category not in ('identity', 'environment'):
                record = dict(category=category, text=entry['description'], owner_uid=owner,
                              path=base + ['description'])
                if 'links' in entry:
                    record['links'] = [artifact + '_' + link for link in entry['links']]
                records.append(record)
                continue
            for field in OBJECT_FIELDS[category]:
                if field == 'uid' or field not in entry:
                    continue
                val = entry[field]
                for i, text in enumerate(val if isinstance(val, list) else [val]):
                    path = base + [field] + ([i] if isinstance(val, list) else [])
                    records.append(dict(category=category+'.'+field, text=text,
                                        owner_uid=owner, path=path))
    local_uids = {entry['uid']: artifact+'_'+entry['uid']
                  for category, entries in observation['attributes'].items()
                  if category in OBJECT_FIELDS for entry in entries}
    for index, record in enumerate(records):
        record.update(id=f'{artifact}_a{index:04d}', carrier_artifact_id=artifact,
                      uid_map=local_uids)
        previous = provenance.get(record['id'])
        record['source_artifact_id'] = previous['source_artifact_id'] if previous else artifact
        record['source_attribute_uid'] = previous['source_attribute_uid'] if previous else record['id']
    return records


def _bound_text(record):
    text = record['text']
    # Only global transition accounts explicitly reference object UIDs. Do not
    # replace ordinary nouns in descriptions (a UID may literally be 'garden').
    mapping = record.get('uid_map', {}) if record['category'] in ('environment_change', 'viewpoint_change') else {}
    if mapping:
        pattern = r'(?<![A-Za-z0-9_.-])(' + '|'.join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r')(?![A-Za-z0-9_-]|\.[A-Za-z0-9_-])'
        text = re.sub(pattern, lambda m: mapping[m.group(0)], text)
    return text


def assemble(records, artifact_id, kind):
    """Rebuild the agreed observation schema; no silent dependency repair."""
    attrs, objects = {}, {}
    for record in records:
        category = record['category']
        text = _bound_text(record)
        if '.' in category:
            parent, field = category.split('.')
            key = (parent, record['owner_uid'])
            if key not in objects:
                objects[key] = {'uid': record['owner_uid']}
                attrs.setdefault(parent, []).append(objects[key])
            entry = objects[key]
            if OBJECT_FIELDS[parent][field] == 'list[str]':
                entry.setdefault(field, []).append(text)
            elif field in entry:
                raise ValueError('Duplicate scalar field: '+category)
            else:
                entry[field] = text
        elif category in OBJECT_FIELDS:
            entry = {'uid': record['owner_uid'], 'description': text}
            if 'links' in record:
                entry['links'] = list(record['links'])
            attrs.setdefault(category, []).append(entry)
        elif category in attrs:
            raise ValueError('Duplicate global field: '+category)
        else:
            attrs[category] = text
    result = dict(artifact_id=artifact_id, kind=kind, attributes=attrs)
    validate_attributes(result, artifact_id, kind)
    return result


def dependencies(pool):
    """Conservative structural closure. Change timelines retain all source states.

    Impossible temporal transitions have a missing sentinel dependency. Relations
    retain endpoint descriptions; optional descriptive properties need their owner.
    """
    descriptions = {r.get('owner_uid'): r['id'] for r in pool
                    if r['category'] in ('identity.description', 'environment.description')}
    changes = {'identity.appearance_change': 'identity.appearance',
               'identity.expression_change': 'identity.expression',
               'identity.pose_change': 'identity.body_pose',
               'environment_change': 'environment.description', 'viewpoint_change': 'viewpoint'}
    result = {}
    for record in pool:
        category = record['category']
        required = set()
        if '.' in category and not category.endswith('.description'):
            required.add(descriptions.get(record['owner_uid'], '!missing'))
        for link in record.get('links', []):
            required.add(descriptions.get(link, '!missing'))
        if category in changes:
            states = [r['id'] for r in pool if r['category'] == changes[category]
                      and r['carrier_artifact_id'] == record['carrier_artifact_id']
                      and ('.' not in category or r.get('owner_uid') == record.get('owner_uid'))]
            required.update(states if len(states) >= 2 else ['!missing'])
        result[record['id']] = required
    return result



def canonicalize_shared(pool):
    """One representative per proven origin, from the earliest candidate parent.

    Never infer equivalence from captions. Shared owner descriptions allow child
    bindings to use that canonical owner; relation links follow the same mapping.
    Dependency aliases retain the original timeline's states across deduplication.
    """
    original_dependencies = dependencies(pool)
    chosen, aliases, owners = {}, {}, {}
    for record in sorted(pool, key=lambda r: (int(r['carrier_artifact_id'].rsplit('_', 1)[1]), r['id'])):
        key = (record['category'], record['source_artifact_id'], record['source_attribute_uid'])
        representative = chosen.setdefault(key, record)
        aliases[record['id']] = representative['id']
        if record['category'] in ('identity.description', 'environment.description', 'viewpoint'):
            owners[record['owner_uid']] = representative['owner_uid']
    result = []
    for original in pool:
        if aliases[original['id']] != original['id']:
            continue
        record = copy.deepcopy(original)
        if 'owner_uid' in record:
            record['owner_uid'] = owners.get(record['owner_uid'], record['owner_uid'])
        if 'links' in record:
            record['links'] = [owners.get(uid, uid) for uid in record['links']]
        record['uid_map'] = {key: owners.get(value, value) for key, value in record.get('uid_map', {}).items()}
        record['dependency_ids'] = sorted({aliases.get(uid, uid)
            for uid in original_dependencies[record['id']]})
        result.append(record)
    return result


def sample_selection(pool, k, kind, rng, max_tries=128, *, parent_count=None):
    """Bounded dependency-aware random sampling, not uniform over legal subsets.

    At least K or None. Stop at the first legal closure reaching K; dependency
    atoms may take the total above K. Global scalar
    conflicts from different sources are rejected by assemble. Image sampling
    excludes flow-required fields, but semantic frame compatibility needs the VLM.
    """
    if parent_count is not None and (type(parent_count) is not int or parent_count < 1):
        raise ValueError('parent_count must be a positive integer')
    limit = min(parent_count if parent_count is not None else 32, 4 if kind == 'image' else 32)
    pool = canonicalize_shared(pool)
    candidates = [r for r in pool if kind != 'image' or CATEGORY_RULES[r['category']]['temporal'] != 'Yes']
    if not 0 < k <= len(candidates):
        return None
    by_id = {r['id']: r for r in candidates}
    deps = dependencies(pool)
    deps.update({r['id']: set(r['dependency_ids']) for r in pool})
    def closure(uid, found):
        if uid in found:
            return True
        if uid not in by_id:
            return False
        found.add(uid)
        return all(closure(dep, found) for dep in sorted(deps[uid]))
    for _ in range(max_tries):
        chosen = set()
        shuffled = rng.sample(candidates, len(candidates))
        for record in shuffled:
            proposal = set(chosen)
            if not closure(record['id'], proposal):
                continue
            rows = [r for r in candidates if r['id'] in proposal]
            if len({r['carrier_artifact_id'] for r in rows}) > limit:
                continue
            try:
                assemble(rows, 'selection', kind)
            except ValueError:
                continue
            chosen = proposal
            if len(chosen) >= k:
                return [{key: copy.deepcopy(value) for key, value in row.items()
                         if key != 'dependency_ids'} for row in rows]
    return None


def compact_rows(records):
    """Expose only IDs, descriptions and bindings; no paths/provenance machinery."""
    return [{key: record[key] for key in ('id', 'category', 'text', 'owner_uid', 'links')
             if key in record} for record in records]


def apply_variations(selected, edits, vary_ids):
    """Validate exact edit coverage and immutable bindings; reject no-op edits."""
    if not isinstance(edits, list) or len(vary_ids) != len(set(vary_ids)):
        raise ValueError('Expected edits array and unique variation IDs')
    originals = {r['id']: r for r in selected}
    if not set(vary_ids) <= set(originals):
        raise ValueError('Unknown variation ID')
    changes = {}
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {'id', 'text'}:
            raise ValueError('Each edit needs exactly id and text')
        uid, text = edit['id'], edit['text']
        if not isinstance(uid, str) or uid not in vary_ids or uid in changes:
            raise ValueError('Unexpected/duplicate edit ID')
        if (not isinstance(text, str) or not 0 < len(text.split()) <= 15 or
                text.strip().lower() in ('none', 'null', 'unknown')):
            raise ValueError('Edited text must have 1–15 words')
        if ' '.join(text.lower().split()) == ' '.join(originals[uid]['text'].lower().split()):
            raise ValueError('No-op variation')
        changes[uid] = text
    if set(changes) != set(vary_ids):
        raise ValueError('Missing variations')
    result = copy.deepcopy(selected)
    for row in result:
        row['operation'] = 'variate' if row['id'] in changes else 'keep'
        if row['id'] in changes:
            row['text'] = changes[row['id']]
    return result


def expert_rows(records):
    """Adapt atoms to existing flat gen-expert contract without invented source support."""
    rows = []
    for record in records:
        category = record['category']
        row = dict(id=record['id'], category=category, text=_bound_text(record),
                   artifact_ids=[] if record.get('operation') == 'variate' else [record['carrier_artifact_id']])
        if category in ('identity.description', 'environment.description'):
            # The description is also the structural anchor, not an extra counted atom.
            row.update(id=record['owner_uid'], category=category.split('.')[0])
        elif '.' in category:
            row['parent_id'] = record['owner_uid']
        elif 'owner_uid' in record:
            row['id'] = record['owner_uid']
        if 'links' in record:
            row['links'] = record['links']
        rows.append(row)
    return rows


def validate_matches(matches, observed, selected):
    """Reject invented IDs, incompatible categories and inconsistent object bindings."""
    if not isinstance(matches, list):
        raise ValueError('Expected matches array')
    new = {r['id']: r for r in observed}
    old = {r['id']: r for r in selected}
    seen, bindings, reverse = set(), {}, {}
    def bind(new_owner, old_owner, carrier):
        key, reverse_key = (new_owner, carrier), (old_owner, carrier)
        if key in bindings and bindings[key] != old_owner:
            raise ValueError('One new object matched conflicting old objects in the same media')
        if reverse_key in reverse and reverse[reverse_key] != new_owner:
            raise ValueError('One old object matched conflicting new objects')
        bindings[key], reverse[reverse_key] = old_owner, new_owner
    for match in matches:
        if not isinstance(match, dict) or set(match) != {'observed_id', 'source_id'}:
            raise ValueError('Match needs exactly observed_id/source_id')
        a, b = match['observed_id'], match['source_id']
        if not isinstance(a, str) or not isinstance(b, str) or a not in new or b not in old:
            raise ValueError('Match references unknown or unselected attribute')
        if (a, b) in seen:
            raise ValueError('Duplicate match')
        seen.add((a, b))
        left, right = new[a], old[b]
        if left['category'] != right['category']:
            raise ValueError('Matched categories must agree')
        carrier = right['carrier_artifact_id']
        if 'owner_uid' in left:
            bind(left['owner_uid'], right['owner_uid'], carrier)
        if 'links' in left:
            for new_owner, old_owner in zip(left['links'], right['links']):
                bind(new_owner, old_owner, carrier)
    return matches


def attach_provenance(observed, matches, selected):
    """Only verified matches inherit earliest sources; no historical reappearance search."""
    validate_matches(matches, observed, selected)
    old = {r['id']: r for r in selected}
    def number(value):
        return int(value.rsplit('_', 1)[1])
    provenance = {}
    for record in observed:
        inherited = [old[m['source_id']] for m in matches if m['observed_id'] == record['id']]
        earliest = min(inherited, key=lambda r: (number(r['source_artifact_id']), r['source_attribute_uid'])) if inherited else record
        provenance[record['id']] = dict(path=record['path'],
            source_artifact_id=earliest['source_artifact_id'],
            source_attribute_uid=earliest['source_attribute_uid'],
            inherited_from=[dict(artifact_id=r['carrier_artifact_id'], attribute_uid=r['id']) for r in inherited])
    return provenance
