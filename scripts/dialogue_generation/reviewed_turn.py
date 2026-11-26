"""Review each output of an assistant delivery before producing the next one."""
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import tempfile
import time
import uuid

from cocreation import Reference
from scripts.dialogue_generation.generation_retry import attempt_placeholder, retry_edit_task, retry_source_path
from scripts.dialogue_generation.media_queue import AssetEvent, AssetRejected, generation_time, review_call
from scripts.dialogue_generation.output_geometry import aspect_ratio_rule
from tools.media_review import SCORES, save
from tools.turn_media_review import POLICY_VERSION

MAX_ATTEMPTS = 3


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024**2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def delivery_order(members, tasks):
    pending = list(members)
    order = []
    while pending:
        ready = next((token for token in pending
                      if not set(tasks[token]['dependencies']).intersection(pending)), None)
        if ready is None:
            raise ValueError('Cyclic dependencies inside an assistant delivery')
        order.append(ready)
        pending.remove(ready)
    return order


def materialize_turn(members, **options):
    for _ in iter_reviewed_turn(members, **options):
        pass


def iter_reviewed_turn(members, *, directory, tasks, records, resolved, context, fingerprint,
                       generate, execute, session, checkpoint, export, seed, progress=True,
                       defer_retries=False, async_reviews=False):
    directory = Path(directory)
    leader = members[0]
    previous = {token: deepcopy(records.get(token, {})) for token in members}
    order = delivery_order(members, tasks)
    positions = {token: index for index, token in enumerate(members, 1)}
    started = time.monotonic()

    def valid_complete(token):
        row = previous[token]
        path = directory / row.get('path', '')
        return (row.get('status') == 'complete' and row.get('fingerprint') == fingerprint(token)
                and path.is_file() and file_hash(path) == row.get('sha256'))

    if all(valid_complete(token) for token in order):
        resolved.update({token: directory / previous[token]['path'] for token in members})
        if progress:
            print('Turn delivery cached: ' + ', '.join(members), flush=True)
        for token in order[:-1]:
            yield None
        return

    run_id = uuid.uuid4().hex
    turn_dir = directory / 'turn_reviews' / leader.strip('[]') / run_id
    turn_dir.mkdir(parents=True)
    work = directory / '_work/materialize'
    work.mkdir(parents=True, exist_ok=True)
    reviews = deepcopy(previous[leader].get('turn_reviews', []))
    superseded = deepcopy(previous[leader].get('superseded_turn_reviews', []))
    if reviews and reviews[-1]['report'].get('policy_version') != POLICY_VERSION:
        superseded.extend(reviews)
        reviews.clear()

    active_token = None

    def persist(*, render=True):
        # The first declared output is the audit owner, even if a dependency
        # causes a different output to be produced first.
        owner = records.setdefault(leader, {})
        owner['turn_reviews'] = reviews
        owner['turn_review_active'] = active_token
        if superseded:
            owner['superseded_turn_reviews'] = superseded
        checkpoint()
        if render:
            export()

    def snapshot(token, candidate):
        return dict(placeholder=token,
                    attempt=deepcopy({key: value for key, value in candidate.items()
                                      if key not in ('critique', 'critique_history', 'next_prompt', 'retry_edit_prompt')}),
                    operation_record={key: deepcopy(records[token][key])
                                      for key in ('tool_call', 'operation_reasoner', 'kind')
                                      if key in records[token]})

    def archive(token, task, record, prompt, number):
        job = Path(tempfile.mkdtemp(prefix=token.strip('[]') + '-', dir=work))
        output = job / ('output.png' if task['kind'] == 'image' else 'output.mp4')
        one = {}
        generative = bool(task.get('backend'))
        try:
            if generative:
                versions = {earlier: attempt_placeholder(earlier, one.get('attempt', 1))
                            for earlier, (one, _) in accepted.items()}
                generate(token, dict(task, prompt=prompt, reference_versions=versions), job, output,
                         (record['seed'] + number - 1) % 2**63, one)
                location = (directory / 'generation_attempts' / token.strip('[]') /
                            ('turn-' + run_id) / f'attempt-{number:02d}')
            else:
                execution_started = time.monotonic()
                execute(token, task, job, output, record)
                one['generation_seconds'] = time.monotonic() - execution_started
                location = turn_dir / 'deterministic' / job.name
            location.mkdir(parents=True)
            saved = location / output.name
            shutil.copy2(output, saved)
            for index, ref in enumerate(one.get('references', [])):
                source = directory / ref['path']
                target = location / 'inputs' / (str(index) + source.suffix)
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(source, target)
                ref['path'] = str(target.relative_to(directory))
            one.update(kind=task['kind'], path=str(saved.relative_to(directory)), sha256=file_hash(saved))
            if generative:
                one.update(attempt=number, placeholder=attempt_placeholder(token, number),
                           prompt=one['actual_prompt'], input_fingerprint=record['fingerprint'])
                if number == 1 and task.get('reasoner'):
                    if task['reasoner']['generation_prompt'] != one['prompt']:
                        raise RuntimeError('First prompt changed after reasoner: ' + token)
                    one['reasoner'] = task['reasoner']
                record['attempts'].append(one)
                record['generation_seconds'] = sum(a['generation_seconds'] for a in record['attempts'])
                save(location / 'attempt.json', one)
            else:
                record['turn_candidate'] = one
            record['sha256'] = one['sha256']
            persist()
            return one
        finally:
            shutil.rmtree(job)

    def output_for(token, candidate, *, retry_available, earlier_report=None):
        task = tasks[token]
        rules = task.get('latest_rules', task.get('sequence_context', {}).get('latest_rules', []))
        refs = [Reference('source_' + str(index), directory / ref['path'], ref['kind'])
                for index, ref in enumerate(candidate.get('references', []))]
        output = dict(placeholder=attempt_placeholder(token, candidate.get('attempt', 1)),
                      output_position=positions[token], kind=task['kind'],
                      media=directory / candidate['path'], references=refs, latest_rules=rules,
                      reference_slots=candidate.get('references', []),
                      aspect_ratio_rule=aspect_ratio_rule(task.get('render_state', {}), task['kind']),
                      operation=task['operation'], retry_available=retry_available)
        if candidate.get('prompt'):
            output['generation_prompt'] = candidate['prompt']
        if records[token].get('tool_call'):
            output['operation_call'] = records[token]['tool_call']
        if task['operation'] == 'concatenate':
            output['assembly_clips'] = [clip['placeholder'] for clip in task['clips']]
        if earlier_report is not None:
            output['accepted_critique'] = earlier_report['thinking']
        return output

    accepted = {}
    for step, token in enumerate(order, 1):
        active_token = token
        task = tasks[token]
        old = previous[token]
        expected = fingerprint(token)
        prefix = [dict(placeholder=earlier, sha256=one['sha256'])
                  for earlier, (one, _) in accepted.items()]
        same_input = old.get('fingerprint') == expected
        same_stage = (same_input and old.get('turn_review_policy') == POLICY_VERSION
                      and old.get('turn_review_prefix') == prefix)
        if not same_stage:
            pending = set(order[step - 1:])
            discarded = [row for row in reviews if row['active_placeholder'] in pending]
            if discarded:
                reviews[:] = [row for row in reviews if row['active_placeholder'] not in pending]
                superseded.extend(discarded)

        record = deepcopy(old) if same_input else {}
        record.pop('turn_reviews', None)
        record.pop('superseded_turn_reviews', None)
        if old and not same_input:
            record['superseded_attempts'] = deepcopy(old.get('superseded_attempts', []) + old.get('attempts', []))
        suffix = '.png' if task['kind'] == 'image' else '.mp4'
        destination = directory / 'media' / (token.strip('[]') + suffix)
        relative_path = str(destination.relative_to(directory))
        if destination.exists() and old.get('path') != relative_path:
            raise FileExistsError('Refusing to overwrite an untracked media file: ' + str(destination))
        record.update(status='pending_review', operation=task['operation'], kind=task['kind'],
                      backend=task.get('backend'), path=relative_path, fingerprint=expected,
                      seed=(seed + int(hashlib.sha256(token.encode()).hexdigest()[:15], 16)) % 2**63,
                      turn_review_leader=leader, turn_review_policy=POLICY_VERSION, turn_review_prefix=prefix)
        record.pop('review_status', None)
        record.pop('accepted_placeholder', None)
        records[token] = record

        attempts = record.setdefault('attempts', [])
        if attempts and not same_stage:
            matching = [index for index, attempt in enumerate(attempts)
                        if attempt.get('input_fingerprint', expected) == expected]
            if not matching:
                raise RuntimeError('Archived turn output has no matching inputs: ' + token)
            chosen = matching[-1] if old.get('status') == 'complete' else matching[0]
            if chosen + 1 < len(attempts):
                record.setdefault('superseded_attempts', []).extend(attempts[chosen + 1:])
                del attempts[chosen + 1:]
            one = attempts[-1]
            if one.get('critique'):
                one.setdefault('critique_history', []).append(one.pop('critique'))
            one.pop('next_prompt', None)
            one.pop('retry_edit_prompt', None)

        candidate = attempts[-1] if attempts else record.get('turn_candidate')
        if candidate is not None:
            path = directory / candidate['path']
            if not path.is_file() or file_hash(path) != candidate['sha256']:
                raise RuntimeError('Archived turn output is missing or corrupt: ' + token)
            record['sha256'] = candidate['sha256']
            if attempts:
                record['generation_seconds'] = sum(a['generation_seconds'] for a in attempts)
        elif not task.get('backend') and same_input and valid_complete(token):
            candidate = dict(kind=task['kind'], path=old['path'], sha256=old['sha256'])
            record.update(turn_candidate=candidate, sha256=candidate['sha256'])
        completed_steps = {row['members'][-1]['attempt']['path']: row['report']
                           for row in reviews if row['active_placeholder'] == token}
        persist(render=False)

        new_generation = candidate is None
        while True:
            number = len(attempts) or 1
            if candidate is None:
                candidate = archive(token, task, record, task.get('prompt'), number)
            report = completed_steps.get(candidate['path'])
            reviewed_now = report is None
            retry = bool(task.get('backend') and number < MAX_ATTEMPTS)
            stage = dict(current_output_position=positions[token], step=step, output_count=len(members),
                         final_stage=step == len(members))
            review_dir = turn_dir / f'output-{positions[token]:02d}' / f'attempt-{number:02d}'
            if report is None:
                outputs = [output_for(earlier, one, retry_available=False, earlier_report=prior)
                           for earlier, (one, prior) in accepted.items()]
                outputs.append(output_for(token, candidate, retry_available=retry))
                report = yield from review_call(session, token, review_dir, 'verify_turn',
                    dict(context, delivery_stage=stage), outputs, deferred=async_reviews,
                    seed=(record['seed'] + number * 100) % 2**63, output_dir=review_dir)
                report.update(policy_version=POLICY_VERSION, delivery_stage=stage)
                check, = report['output_checks']
                if check['output_position'] != positions[token]:
                    raise ValueError('Review must score only the current delivery output')
                if candidate.get('attempt'):
                    candidate['critique'] = dict(report, **{key: check[key] for key in (*SCORES, 'decision', 'reason')},
                        turn_review=str((review_dir / 'report.json').relative_to(directory)))
                    save((directory / candidate['path']).parent / 'attempt.json', candidate)
                members_snapshot = [snapshot(earlier, one) for earlier, (one, _) in accepted.items()]
                members_snapshot.append(snapshot(token, candidate))
                reviews.append(dict(round=len(reviews) + 1, active_placeholder=token,
                    members=members_snapshot, changed=[token], report=report,
                    report_path=str((review_dir / 'report.json').relative_to(directory))))
                persist()
            seconds = generation_time(candidate) if new_generation else 0.0
            event = AssetEvent(token, candidate.get('executed_operation', task['operation']), candidate['path'], seconds,
                               'accepted' if report['decision'] == 'accepted' else 'rejected')
            if report['decision'] == 'accepted':
                accepted[token] = (candidate, report)
                resolved[token] = directory / candidate['path']
                if step != len(order):
                    yield event if new_generation else None
                break
            if not retry:
                record.update(status='failed', review_status='rejected')
                persist()
                raise AssetRejected(event, 'Turn delivery rejected with no retry left: ' + token)
            if defer_retries and reviewed_now:
                yield event
            edit = retry_edit_task(task, token, attempts[0], candidate)
            next_prompt = candidate.get('retry_edit_prompt')
            if not next_prompt:
                repair_dir = review_dir / 'fixer'
                rules = task.get('latest_rules', task.get('sequence_context', {}).get('latest_rules', []))
                repair_context = dict(context, output_position=positions[token],
                                      output_count=len(members), latest_rules=rules)
                next_prompt = yield from review_call(session, token, repair_dir, 'fix',
                    edit, repair_context, candidate['prompt'], report['output_checks'][0]['reason'],
                    media=retry_source_path(directory, edit), deferred=async_reviews,
                    seed=(record['seed'] + number * 1000) % 2**63, output_dir=repair_dir)
                candidate['retry_edit_prompt'] = next_prompt
                save((directory / candidate['path']).parent / 'attempt.json', candidate)
                persist(render=False)
            candidate = archive(token, edit, record, next_prompt, number + 1)
            new_generation = True

    # Publish the turn together after every output has been accepted.
    for token, (candidate, _) in accepted.items():
        record = records[token]
        destination = directory / record['path']
        source = directory / candidate['path']
        if source.resolve() != destination.resolve():
            temporary = destination.with_suffix(destination.suffix + '.turn.tmp')
            shutil.copy2(source, temporary)
            temporary.replace(destination)
        record.update({key: value for key, value in candidate.items()
                       if key not in ('path', 'sha256', 'attempt', 'placeholder', 'critique', 'critique_history', 'next_prompt', 'retry_edit_prompt')})
        if candidate.get('attempt'):
            record['accepted_placeholder'] = attempt_placeholder(token, candidate['attempt'])
        record.update(status='complete', review_status='accepted', sha256=candidate['sha256'],
                      size_bytes=destination.stat().st_size, runtime_cleaned=True,
                      elapsed_seconds=round(time.monotonic() - started, 6))
        resolved[token] = destination
    active_token = None
    persist()
    if progress:
        print('Turn delivery accepted: ' + ', '.join(members), flush=True)
    if new_generation:
        return AssetEvent(token, event.operation, records[token]['path'], event.seconds)
