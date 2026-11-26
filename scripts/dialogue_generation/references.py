"""Causal media dependencies: every reference already exists in the dialogue."""
from copy import deepcopy


def assign_reference_materials(plan):
    media, latest_shots = {}, {}
    deliveries = {token: turn for turn in plan['turns'] for token in turn['output_media']}
    for row in plan['placeholders']:
        refs = list(row['inputs'])
        if row['operation'] == 'create_reference' and not refs:
            turn = deliveries[row['placeholder']]
            # Rule cleanup regenerates the agreed identity from its last visible
            # version, even when the planner resets its bounded edit ancestry.
            refs = [ref for ref in turn['intent'].get('context_media', [])
                    if ref in media and set(row['asset_ids']) <= set(media[ref]['asset_ids'])]
        if row['operation'] == 'generate_clip':
            sid = row['shot_ids'][0]
            if sid in latest_shots:
                previous = media[latest_shots[sid]]
                refs = [latest_shots[sid]] + [token for aid, token in row.get('bindings', {}).items()
                    if previous.get('bindings', {}).get(aid) != token]
            latest_shots[sid] = row['placeholder']
        row.pop('source_video', None)
        row['reference_materials'] = list(dict.fromkeys(refs))
        if any(ref not in media for ref in row['reference_materials']):
            raise ValueError('Construction references must precede their output')
        row['reference_depth'] = max((media[ref]['reference_depth'] + 1 for ref in refs), default=0)
        media[row['placeholder']] = row
    for turn in plan['turns']:
        if turn['output_media']:
            turn['input_media'] = list(dict.fromkeys(ref for token in turn['output_media']
                for ref in media[token]['reference_materials'] if ref not in turn['output_media']))
        else:
            turn['input_media'] = list(turn['intent'].get('display_media', []))
    return plan


def validate_reference_materials(plan):
    expected = deepcopy(plan)
    assign_reference_materials(expected)
    for actual, wanted in zip(plan['placeholders'], expected['placeholders']):
        for key in ('reference_materials', 'reference_depth'):
            if actual.get(key) != wanted[key]:
                raise ValueError('Incorrect forward construction ' + key)
        if 'source_video' in actual:
            raise ValueError('Guide video must not be a construction source')
    for actual, wanted in zip(plan['turns'], expected['turns']):
        if actual['input_media'] != wanted['input_media']:
            raise ValueError('Delivery input_media must equal actual construction inputs')
    return True
