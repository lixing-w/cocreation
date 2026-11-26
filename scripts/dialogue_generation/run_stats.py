"""Read a dialogue run's timing records and Claude Code receipts as JSON."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import shutil
import subprocess


TOKEN_FIELDS = ('cache_read_input_tokens', 'cache_creation_input_tokens', 'input_tokens', 'output_tokens')
PROMPT_TOOLS = {'H3-Image prompter', 'H3 video prompter', 'Production opening', 'Reasoners'}
TERMINAL_PHASES = {'complete', 'complete_with_partial', 'complete_with_failures', 'cancelled', 'interrupted', 'failed'}


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def read_events(path):
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # A live job may be in the middle of writing its last event.
    return events


def job_stats(run, state, log_path):
    job_id = str(state.get('job_id') or '')
    job = dict(id=job_id or None, state=state.get('phase'), node=state.get('node'),
               elapsed_seconds=None, elapsed_source=None)
    if job['state'] in TERMINAL_PHASES:
        job.update(elapsed_seconds=state.get('elapsed_seconds'), elapsed_source='pipeline_state')
    if job_id.isdecimal() and shutil.which('sacct'):
        fields = ('id', 'state', 'exit_code', 'elapsed_seconds', 'start', 'end', 'node')
        try:
            result = subprocess.run(['sacct', '-X', '-j', job_id, '-n', '-P', '-o',
                'JobIDRaw,State,ExitCode,ElapsedRaw,Start,End,NodeList'], capture_output=True, text=True, timeout=8)
            for row in result.stdout.splitlines():
                values = row.split('|')
                if len(values) == len(fields) and values[0] == job_id and values[3].isdigit():
                    job.update(zip(fields, values))
                    job.update(elapsed_seconds=int(values[3]), elapsed_source='sacct')
                    break
        except (OSError, subprocess.TimeoutExpired):
            pass
    if log_path is None and job_id.isdecimal():
        repo = Path(__file__).resolve().parents[2]
        candidates = [run/'slurm_logs', repo/'slurm_logs', Path.cwd()/'slurm_logs']
        log_path = next((log for folder in candidates for log in sorted(folder.glob(f'*-{job_id}.out'))), None)
    loads, waves = [], Counter()
    if log_path is not None and log_path.is_file():
        job['stdout_log'] = str(log_path)
        for line in log_path.read_text(encoding='utf-8', errors='replace').splitlines():
            if match := re.match(r'\[sglang\] reloaded (\S+): .* ([\d.]+)s$', line):
                loads.append(dict(lane=match[1], seconds=float(match[2])))
            if match := re.match(r'\[generation\]\[T(\d+)\]\[', line):
                outcome = next((status for status in ('rejected', 'exhausted') if f'({status})' in line), 'other')
                waves[int(match[1]), outcome] += 1
    return job, loads, [dict(wave=wave, status=status, count=count) for (wave, status), count in sorted(waves.items())]


def claude_tool(parts, request):
    if parts[0] == 'observation' and len(parts) > 1:
        stage = parts[1]
        if stage == 'inventory':
            return 'Prompt Observer', 'Inventory'
        if stage == 'description_separation':
            return 'Prompt Observer', 'Description verification' if 'review' in parts else 'Description projection'
        if stage.startswith('asset_'):
            return 'Prompt Observer', 'Attribute verification' if stage.endswith('_review') else 'Attribute extraction'
    fields = (request.get('response_schema') or {}).get('properties', {})
    messages = request.get('messages') or [{}]
    system = messages[0].get('content', '')
    system = system if isinstance(system, str) else ''
    if 'output_checks' in fields:
        return 'Sequential media critique' if 'delivery_stage' in str(messages[1:]) else 'Joint media critique', None
    if 'rule_checks' in fields and 'request_fidelity' in fields:
        return 'Single media critique', None
    if 'edit_prompt' in fields or ('edits' in fields and system.startswith('You are a generation prompt FIXER')):
        return 'Prompt Fixer', None
    if 'production_brief' in fields or (set(fields) == {'think'} and 'reasoner' in system.lower()):
        return 'Reasoners', None
    if 'content_fidelity' in fields:
        return 'Dialogue verifier', None
    for prefix, name in [
        ('Before writing, analyze conflicts', 'Dialogue rewriter'),
        ('Speak as the creative assistant. In one short', 'Production opening'),
        ('Speak as the creative assistant. Write one short natural first-person acknowledgment', 'Production opening'),
        ('# Text-to-image prompt writing guide', 'H3-Image prompter'),
        ('# Reference-image edit prompt writing guide', 'H3-Image prompter'),
        ('Write only the finished English H3 generation prompt', 'H3 video prompter'),
    ]:
        if system.startswith(prefix):
            return name, None
    return 'Other Claude calls', None


def call_usage(response):
    usage = response['usage']
    ttl = usage.get('cache_creation') or {}
    return {**{key: usage.get(key) or 0 for key in TOKEN_FIELDS},
            'ephemeral_5m_input_tokens': ttl.get('ephemeral_5m_input_tokens') or 0,
            'ephemeral_1h_input_tokens': ttl.get('ephemeral_1h_input_tokens') or 0,
            'thinking_tokens': (usage.get('output_tokens_details') or {}).get('thinking_tokens') or 0,
            'duration_ms': response.get('duration_ms') or 0,
            'duration_api_ms': response.get('duration_api_ms') or 0,
            'estimated_cost_usd': response.get('total_cost_usd') or 0,
            'error_results': int(bool(response.get('is_error')))}


def aggregate(counter, usage):
    counter.update(usage)
    counter['calls'] += 1
    counter['hits'] += usage['cache_read_input_tokens'] > 0


def summarize(counter):
    total = sum(counter[key] for key in TOKEN_FIELDS[:3])
    return dict(counter, calls=counter['calls'], hits=counter['hits'], total_input_tokens=total,
                request_hit_rate=counter['hits']/counter['calls'] if counter['calls'] else None,
                token_read_rate=counter['cache_read_input_tokens']/total if total else None)


def collect_calls(case_path, seen, coverage):
    for request_path in sorted(case_path.rglob('request.json')):
        request = read_json(request_path, {})
        model = request.get('model', '')
        if not isinstance(model, str) or 'messages' not in request:
            continue
        if not (model.startswith('claude-') or model in {'sonnet', 'haiku', 'opus'}):
            continue
        response_path = request_path.with_name('response.json')
        response = read_json(response_path, {})
        if response.get('type') != 'result' or not response.get('usage'):
            coverage['requests_without_usage'] += 1
            continue
        unique = response.get('uuid')
        if unique:
            if unique in seen:
                coverage['duplicate_results'] += 1
                continue
            seen.add(unique)
        tool, subtool = claude_tool(response_path.relative_to(case_path).parts, request)
        elapsed = max(0, (response_path.stat().st_mtime_ns-request_path.stat().st_mtime_ns)/1e9)
        yield dict(tool=tool, observer_stage=subtool, model=model, effort=request.get('effort'),
                   session_id=response.get('session_id'), stop_reason=response.get('stop_reason'),
                   ttft_ms=response.get('ttft_ms'), model_usage=response.get('modelUsage'),
                   archive_path=str(response_path.relative_to(case_path)),
                   request_to_result_seconds=elapsed, usage=call_usage(response))


def case_stats(case, path, result, calls):
    dialogue_path = path/'dialogue' if (path/'dialogue').is_dir() else path
    dialogue = read_json(dialogue_path/'dialogue.json', {})
    official = read_json(dialogue_path/'sequence.json', {})
    media = read_json(dialogue_path/'media_manifest.json', {}).get('media', {})
    vlm = read_json(path/'vlm_state.json', {})
    phase = vlm.get('phase_seconds', {})
    timing = read_json(path/'media_timings.json', {})
    prompt_phase = dialogue.get('prompt_rewriting', {}).get('elapsed_seconds')
    dialogue_phase = phase.get('dialogue_and_generation_prompts')
    counts = Counter(call['tool'] for call in calls)
    total_usage = Counter()
    for call in calls:
        aggregate(total_usage, call['usage'])
    if prompt_phase is None and dialogue_phase is not None and not counts['Reasoners']:
        if dialogue.get('realization', {}).get('status') == 'failed':
            prompt_phase = 0
    can_split_prompts = prompt_phase == 0 or any(tool in counts for tool in PROMPT_TOOLS)
    seconds = dict(observation=phase.get('observing'),
        dialogue_rewrite=max(0, dialogue_phase-prompt_phase) if None not in (dialogue_phase, prompt_phase) else None,
        prompting=sum(call['request_to_result_seconds'] for call in calls
                      if call['tool'] in PROMPT_TOOLS - {'Reasoners'}) if can_split_prompts else None,
        reasoning=sum(call['request_to_result_seconds'] for call in calls
                      if call['tool'] == 'Reasoners') if can_split_prompts else None,
        critique=timing.get('review', 0.0) if timing else None,
        image=timing.get('image', 0.0) if timing else None,
        video=timing.get('video_generation', 0.0)+timing.get('video_edit', 0.0) if timing else None)
    completed = Counter(row.get('kind') or ('image' if token.startswith('[[image_') else 'video')
                        for token, row in media.items() if row.get('status') == 'complete' and row.get('path'))
    status = result.get('status') or official.get('status')
    if status is None and vlm.get('phase') == 'complete':
        status = 'vlm_complete'
    elif status is None:
        status = vlm.get('phase')
    failed_reviews = []
    if status in ('failed', 'partial'):
        for report_path in sorted(dialogue_path.glob('turn_reviews/**/report.json')):
            report = read_json(report_path, {})
            if report.get('decision') == 'rejected':
                failed_reviews.append(dict(path=str(report_path.relative_to(dialogue_path)), reason=report.get('reason'),
                                           delivery_stage=report.get('delivery_stage')))
    realized = dialogue.get('realization', {}).get('completed_turns')
    return dict(id=case['id'], seed=case.get('seed'), source_format=case.get('source_format'),
        status=status, failed_stage=result.get('stage') if status == 'failed' else None,
        error=result.get('error'), failed_turn_reviews=failed_reviews,
        pruning=result.get('pruning') or dialogue.get('materialization', {}).get('pruning'),
        label_turns=len(official['turns']) if 'turns' in official else realized,
        realized_turns=realized, images=completed['image'], videos=completed['video'], seconds=seconds,
        vlm_phase=vlm.get('phase'), vlm_seconds=vlm.get('elapsed_seconds'), prompt_reasoner_wall=prompt_phase,
        media_active_seconds=timing.get('media_active'), cache=summarize(total_usage), claude_calls=dict(counts))


def collect(run, *, slurm_log=None, include_calls=False):
    events = read_events(run/'pipeline_events.jsonl')
    state = read_json(run/'pipeline_state.json', events[-1] if events else {})
    vlm = read_json(run/'vlm_summary.json', {})
    vlm_batches = [read_json(path) for path in sorted((run/'vlm-batches').glob('*.json'))]
    results = read_json(run/'results.json', {})
    cases = read_json(run/'inputs.json', {}).get('cases')
    if not cases:
        ids = vlm.get('cases') or list(results) or sorted(path.name for path in run.iterdir()
                                                        if (path/'vlm_state.json').is_file())
        cases = [dict(id=value) for value in ids]
    single = not cases and ((run/'dialogue.json').is_file() or (run/'dialogue').is_dir())
    if single:
        cases = [dict(id=run.name)]
    if not cases:
        raise ValueError('No dialogue case records found in this run directory')
    job, loads, waves = job_stats(run, state, slurm_log)
    seen, coverage = set(), Counter(requests_without_usage=0, duplicate_results=0)
    tools, models, observer = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    totals, rows = Counter(), []
    for case in cases:
        cid = case['id']
        if Path(cid).name != cid or cid in ('.', '..'):
            raise ValueError(f'Invalid sequence ID {cid!r}')
        path = run if single else run/cid
        calls = list(collect_calls(path, seen, coverage))
        for call in calls:
            for group in (totals, tools[call['tool']], models[call['model']]):
                aggregate(group, call['usage'])
            if call['observer_stage']:
                aggregate(observer[call['observer_stage']], call['usage'])
        row = case_stats(case, path, results.get(cid, {}), calls)
        if include_calls:
            row['receipts'] = calls
        rows.append(row)
    done = sum(row['status'] == 'complete' for row in rows)
    wall = job['elapsed_seconds']
    throughput = dict(submitted=len(rows), completed=done, partial=sum(row['status'] == 'partial' for row in rows),
        wall_seconds_per_submitted=wall/len(rows) if wall is not None else None,
        wall_seconds_per_completed=wall/done if wall is not None and done else None)
    cache = dict(total=summarize(totals), by_tool={key:summarize(value) for key, value in sorted(tools.items())},
                 by_model={key:summarize(value) for key, value in sorted(models.items())},
                 observer_detail={key:summarize(value) for key, value in sorted(observer.items())},
                 receipt_coverage=dict(coverage))
    return dict(run_dir=str(run), job=job, pipeline_state=state, events=events, vlm=vlm, batch=throughput,
                vlm_batches=vlm_batches, cases=rows, sglang_reloads=loads, generation_log=waves, cache=cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--slurm-log', type=Path, help='Explicit Slurm stdout path, if it is stored outside the repo')
    parser.add_argument('--include-calls', action='store_true', help='Include each Claude receipt’s metrics and archive path')
    args = parser.parse_args()
    try:
        run = args.run_dir.expanduser().resolve(strict=True)
        if not run.is_dir():
            raise ValueError('Run path must be a directory')
        stats = collect(run, slurm_log=args.slurm_log, include_calls=args.include_calls)
    except (OSError, ValueError) as error:
        parser.exit(1, f'run stats: {error}\n')
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
