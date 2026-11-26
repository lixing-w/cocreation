"""Project frozen product counts and attached settings into separate writer inputs."""
from collections import Counter
from copy import deepcopy


PRODUCT_REQUESTS = {
    'request_asset', 'request_candidates', 'request_trial', 'specify_goal', 'begin_task',
    'request_variant', 'reject_and_edit', 'relative_request', 'resume_task',
    'temporary_adoption_trial', 'request_shot', 'request_shot_update',
    'request_rule_exception', 'request_rule_followup', 'restore_rule_asset',
    'request_preview', 'request_compilation', 'reorder_shots',
    'replace_compilation_shots', 'request_final_export',
    'incompatible_requirements', 'incompatible_framing', 'resolve_conflict',
}


def requested_products(turn, media):
    """Use planned delivery rows, never aspect-ratio prose, to count products."""
    intent = turn['intent']
    if 'requested_outputs' in intent:
        return deepcopy(intent['requested_outputs'])
    kind = intent.get('requested_media_kind') or intent.get('rule_event', {}).get('kind')
    if intent['action'] not in PRODUCT_REQUESTS and not (
            intent['action'] in {'clarify_rule_conflict', 'resolve_rule_conflict'} and kind):
        return []
    future = [row for row in media.values()
              if row.get('introduced_at', 0) > turn['id']]
    if future:
        delivery = min(row['introduced_at'] for row in future)
        counts = Counter(row['kind'] for row in future if row['introduced_at'] == delivery)
        return [dict(kind=medium, count=count) for medium, count in counts.items()]
    # Small/injected contexts may omit introduction IDs; the explicit category
    # still distinguishes a single request from the two-candidate action.
    if kind:
        return [dict(kind=kind, count=2 if intent['action'] == 'request_candidates' else 1)]
    return []


def content_description(intent):
    """New plans separate settings; strip only exact old planner suffixes on reuse."""
    if 'content_description' in intent:
        return intent['content_description']
    description = intent['description']
    setting = intent.get('rule_request')
    if setting:
        kind, requirement = setting['kind'], setting['requirement']
        suffixes = (
            f' For this requested {kind}, also use {requirement}.',
            f' For this {kind} only, also use {requirement}; keep the existing rule for later requests.',
        )
        for suffix in suffixes:
            if description.endswith(suffix):
                return description[:-len(suffix)]
    return description


def additional_requirements(intent):
    setting = intent.get('rule_request')
    if not setting:
        return []
    return [dict(
        requirement=requirement,
        duration='this request only' if setting['once'] else 'from now on, including this request and later applicable outputs',
        applies_to=('the requested products in this request only' if setting['once'] else
                    setting.get('applies_to', 'the stated image/video categories and subjects')),
    ) for requirement in setting.get('requirements', [setting['requirement']])]


def product_lines(products):
    lines = ['Products requested by the user (authoritative type and exact count):']
    for product in products:
        count, kind = product['count'], product['kind']
        noun = ('still image' if count == 1 else 'separate still images') if kind == 'image' else (
            'video' if count == 1 else 'separate videos')
        lines.append(f'- {count} {noun}.')
    lines += [
        'For a production request, make each requested product type and count explicit. '
        'For a clarification or rule decision, preserve these same pending products while performing that turn\'s specified action.',
        'A setting for images and videos does not request both media types. '
        'Changing or withdrawing a setting does not add, cancel, replace or change the type or count of these products.',
    ]
    return lines


def requirement_lines(requirements):
    lines = ['Additional requirements attached to this request (settings, not extra products):']
    if not requirements:
        return lines + ['- None. Express any separately specified rule decision; do not invent another requirement.']
    for index, requirement in enumerate(requirements, 1):
        lines += [f'{index}. Requirement: {requirement["requirement"]}.',
                  '   Takes effect: ' + requirement['duration'] + '.',
                  '   Applies to: ' + requirement['applies_to'] + '.']
    lines.append('State the requirement, its starting point/duration and its scope separately from the product request. '
                 'A continuing rule must sound like "From now on, use ...", rather than "Also provide ...". '
                 'A one-request exception applies only this time. Keep intentional conflicts for the planned clarification.')
    return lines
