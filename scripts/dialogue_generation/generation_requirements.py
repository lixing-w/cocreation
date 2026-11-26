"""Optional per-image composition and pose requirements."""
from copy import deepcopy


def resolve_requirements(kind, operation, asset_kinds, *, parent=None, requested=None):
    result = deepcopy(parent) if parent is not None else {}
    requested = {} if requested is None else requested
    if not isinstance(requested, dict) or set(requested) - {'composition', 'pose'}:
        raise ValueError('Generation requirements support composition and pose')
    result.update(requested)
    for key, value in result.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(key + ' must be nonempty text')
    return result


def assign_generation_requirements(plan):
    media = {}
    for row in plan['placeholders']:
        parent = media.get(row.get('edit_parent'), {})
        row['generation_requirements'] = resolve_requirements(row['kind'], row['operation'],
            [plan['target_assets'][aid]['kind'] for aid in row['asset_ids']],
            parent=parent.get('generation_requirements') if row['kind'] == 'image' and parent.get('kind') == 'image' else None,
            requested=row.get('generation_requirements'))
        media[row['placeholder']] = row
    return plan


def image_layout(row, targets):
    if len(row['asset_ids']) == 1:
        kind = targets[row['asset_ids'][0]]['kind']
        return 'environment' if kind.endswith('environment') else ('product' if kind == 'object' else 'portrait')
    return 'scene'
