"""Small, reproducible policies for real-prompt batch runs (no media/model I/O)."""
from copy import deepcopy
import random


CASE_FIELDS = ('id', 'source_format', 'input_file', 'seed', 'reworks',
               'video_width', 'video_height')


def real_prompt_cases(cases, *, policy='random_0_1', previous_tsv=None):
    """Pick 0/1 reserved asset reworks for new cases and preserve recorded rework choices.

    The dataset's reworks value is legacy batch configuration, not source content.
    Keep the dataset untouched and record the realized choices separately.
    """
    if policy not in ('random_0_1', 'manifest', '0', '1', '2'):
        raise ValueError('REAL_PROMPT_REWORKS_POLICY must be random_0_1, manifest, 0, 1 or 2')
    selected = deepcopy(cases)
    if len({c['id'] for c in selected}) != len(selected):
        raise ValueError('Duplicate real-prompt case id')
    previous = {}
    if previous_tsv is not None:
        for line in previous_tsv.splitlines():
            values = line.split('\t')
            if len(values) == len(CASE_FIELDS) + 1:
                del values[5]  # Legacy launch-time aspect rule.
            if len(values) != len(CASE_FIELDS) or values[0] in previous:
                raise ValueError('Invalid existing cases.tsv')
            previous[values[0]] = dict(zip(CASE_FIELDS, values))
        if set(previous) != {c['id'] for c in selected}:
            raise ValueError('Resume requires the original case set')
    choices = []
    for case in selected:
        original = case['reworks']
        if previous_tsv is not None:
            saved = previous[case['id']]
            if any(str(case[key]) != saved[key] for key in CASE_FIELDS if key != 'reworks'):
                raise ValueError('Resume case metadata changed: ' + case['id'])
            chosen = int(saved['reworks'])
        elif policy == 'random_0_1':
            if type(case['seed']) is not int:
                raise ValueError('Case seed must be an integer')
            chosen = random.Random(f"real-prompt-reworks-v1:{case['seed']}:{case['id']}").randint(0, 1)
        elif policy == 'manifest':
            chosen = original
        else:
            chosen = int(policy)
        if type(chosen) is not int or chosen not in (0, 1, 2):
            raise ValueError('Invalid reserved reworks for ' + case['id'])
        case['reworks'] = chosen
        choices.append(dict(id=case['id'], seed=case['seed'],
                            manifest_reworks=original, selected_reworks=chosen))
    report = dict(version=1, policy=policy,
                  source='existing_cases_tsv' if previous_tsv is not None else 'new_run',
                  unit='distinct assets reserved for post-shot rework, not video edit depth',
                  cases=choices)
    tsv = '\n'.join('\t'.join(str(c[k]) for k in CASE_FIELDS) for c in selected) + '\n'
    return tsv, report
