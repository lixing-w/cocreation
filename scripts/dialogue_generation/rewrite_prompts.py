"""Rewrite every generated placeholder after dialogue completion, or backfill a run."""
import argparse
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.dialogue_models import BACKENDS, DEFAULT_BACKEND, SONNET_EFFORTS

from scripts.dialogue_generation.generation import build_tasks, construction_order, VERSION
from scripts.dialogue_generation.catalog import placeholder_catalog
from tools.placeholder_prompter import rewrite_placeholder
from tools.dialogue_realizer import QwenRealizer


def rewrite_run(result, observation, output_dir, *, writer=None, progress=True, force=False, **model_options):
    from scripts.dialogue_generation.generate import save
    if result.get('realization', {}).get('status') != 'complete':
        raise ValueError('Finish the main dialogue before rewriting generation prompts')
    from scripts.dialogue_generation.references import assign_reference_materials
    assign_reference_materials(result)
    tasks = build_tasks(result, observation)
    order = construction_order(tasks)
    output = Path(output_dir)
    rows = {r['placeholder']: r for r in result['placeholders']}
    from scripts.dialogue_generation.formal_sequence import generation_context
    for token, task in tasks.items():
        if task.get('backend') or task.get('assembly_version'):
            task['sequence_context'] = generation_context(result, token,
                operation=task['operation'] if task.get('assembly_version') else None)
        if task.get('assembly_version'):
            task['operation_reasoner_version']=2
            task['sequence_context']['source_formats']=[dict(placeholder=clip['placeholder'],
                aspect_ratio=tasks[clip['placeholder']].get('render_state',{}).get('aspect_ratio'))
                for clip in task['clips']]
        for part in task.get('clip_edits',[]):
            from scripts.dialogue_generation.planner_rules import media_rules, rule_applicability
            child=part['generation']
            child['sequence_context'] = dict(generation_context(result, token),
                latest_rules=media_rules(result,token,kind=child['kind'],asset_ids=child['asset_ids']),
                rule_applicability=rule_applicability(result,token,kind=child['kind'],asset_ids=child['asset_ids']))
        for request_task in [task]+[part['generation'] for part in task.get('clip_edits',[])]:
            if request_task.get('backend'):
                request_task['latest_rules']=request_task['sequence_context']['latest_rules']
                request_task['reasoner_version']=4
    rework_groups={}
    rework_for={}
    for turn in result['turns']:
        workflow=turn.get('intent',{}).get('asset_update_workflow')
        if not workflow:
            continue
        tokens=[p for p in turn['output_media'] if p in tasks]
        leader=workflow['images'][0]
        rework_groups[leader]=(turn,tokens)
        for token in tokens:
            rework_for[token]=leader
            tasks[token]['workflow_leader']=leader
            tasks[token]['reasoner_version']=5
            if token!=leader:
                tasks[token]['reasoner_shared_with']=leader
    delivery_groups={}
    delivery_for={}
    if writer is None or callable(getattr(type(writer),'delivery_reasoning',None)):
        for turn in result['turns']:
            tokens=[p for p in turn.get('output_media',[]) if p in tasks]
            members=[tasks[p] for p in tokens]
            if (len(tokens)<2 or any(p in rework_for for p in tokens)
                    or not all(t.get('backend') and not t.get('clip_edits') for t in members)
                    or any(set(t.get('dependencies',[])) & set(tokens) for t in members)):
                continue
            leader=tokens[0]
            delivery_groups[leader]=(turn,tokens)
            for token in tokens:
                delivery_for[token]=leader
                tasks[token]['delivery_leader']=leader
                tasks[token]['reasoner_version']=6
                if token!=leader:
                    tasks[token]['reasoner_shared_with']=leader
    candidate_groups={}
    for turn in result['turns']:
        tokens=[p for p in turn.get('output_media',[]) if p in tasks]
        members=[tasks[p] for p in tokens]
        if (len(members)>1 and len({(t.get('backend'),t.get('kind')) for t in members})==1
                and all(t.get('backend') in ('h3_image','flux','minimax_h3','vdn_h3')
                        and not t.get('clip_edits') for t in members)):
            for token,member in zip(tokens,members):
                member['placeholder']=token
                member['writer_group_version']=3
                candidate_groups[member['placeholder']]=(str(turn['id']),members)
    import hashlib
    from tools.placeholder_prompter import writer_input,writer_system
    for task in tasks.values():
        from tools.prompt_reasoner import SYSTEM,REWORK_SYSTEM,DELIVERY_SYSTEM,OPERATION_SYSTEM,NARRATIVE_STYLE
        policy=(REWORK_SYSTEM if task.get('workflow_leader') else DELIVERY_SYSTEM if task.get('delivery_leader')
                else OPERATION_SYSTEM if task.get('assembly_version') else SYSTEM)
        task['reasoner_policy_hash']=hashlib.sha256((policy+'\n'+NARRATIVE_STYLE).encode()).hexdigest()
        for request in [task]+[part['generation'] for part in task.get('clip_edits',[])]:
            if request.get('backend'):
                request['writer_input_hash']=hashlib.sha256(writer_input(request).encode()).hexdigest()
                request['writer_system_hash']=hashlib.sha256(writer_system(request).encode()).hexdigest()
        task['fingerprint']=hashlib.sha256(json.dumps({k:v for k,v in task.items() if k!='fingerprint'},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    force_group=set()
    for key,members in candidate_groups.values():
        if any(rows[t['placeholder']].get('generation',{}).get('fingerprint')!=t['fingerprint'] or
               rows[t['placeholder']].get('generation',{}).get('status')!='ready' for t in members):
            force_group.update(t['placeholder'] for t in members)
    grouped_prompts={}
    state = dict(status='running', version=VERSION, text_only=True, construction_order=order,
                 completed=0, model=getattr(writer, 'model', model_options.get('model')))
    if not isinstance(state['model'], (str, type(None))):
        state['model'] = 'injected'
    result['prompt_rewriting'] = state
    owned, model = False, writer
    started = time.perf_counter()
    def checkpoint():
        save(output/'dialogue.json', result)
        save(output/'placeholders.json', placeholder_catalog(result))
        save(output/'generation_plan.json', dict(version=VERSION, status=state['status'],
            construction_order=order, tasks={p: rows[p]['generation'] for p in order if 'generation' in rows[p]}))
    checkpoint()
    try:
        for index, token in enumerate(order):
            task, row = tasks[token], rows[token]
            saved = row.get('generation', {})
            if (not force and token not in force_group and saved.get('fingerprint') == task['fingerprint']
                    and saved.get('status') == 'ready' and (not task['backend'] or saved.get('prompt'))
                    and (writer is not None and not callable(getattr(type(writer),'generation_reasoning',None))
                         or all(r.get('reasoner') or r.get('reasoner_shared_with') for r in ([saved] if saved.get('backend') else [])+[p['generation'] for p in saved.get('clip_edits',[])]))
                    and (not task.get('assembly_version') or saved.get('operation_reasoner')
                         or saved.get('reasoner_shared_with')
                         or writer is not None and not callable(getattr(type(writer),'operation_reasoning',None)))):
                state['completed'] += 1
                continue
            row.pop('generation_prompt', None)
            row['generation'] = task
            task['status'] = 'pending'
            requests = ([task] if task['backend'] else []) + [part['generation'] for part in task.get('clip_edits', [])]
            for request_index, request_task in enumerate(requests):
                if model is None:
                    model = QwenRealizer(output_dir=output/'prompt_rewriter/runtime', **model_options)
                    owned = True
                state['model'] = getattr(model, 'model', 'injected')
                if not isinstance(state['model'], str):
                    state['model'] = 'injected'
                request = model.generation_request if callable(getattr(type(model), 'generation_request', None)) else model
                name = token.strip('[]') + (f'-clip-{request_index+1}' if task.get('clip_edits') else '')
                prompt_seed=(result['realization'].get('rewrite_seed', result['seed']) + 100000 + index*31+request_index*3) % (2**31-3)
                group=candidate_groups.get(token)
                if group and callable(getattr(type(model),'generation_candidates',None)):
                    key,members=group
                    if key not in grouped_prompts:
                        grouped_prompts[key]=model.generation_candidates(members,seed=prompt_seed,
                            output_dir=output/'prompt_rewriter'/('candidates-'+key))
                    request_task['prompt']=grouped_prompts[key][token]
                else:
                    request_task['prompt'] = rewrite_placeholder(request_task, request,
                        seed=prompt_seed,output_dir=output/'prompt_rewriter'/name)
                reasoning_writer=getattr(model,'generation_reasoning',None)
                if token not in rework_for and token not in delivery_for and callable(getattr(type(model),'generation_reasoning',None)):
                    reasoned=reasoning_writer(request_task['sequence_context'],request_task['prompt'],request_task['kind'],
                        references=request_task['references'],
                        seed=(result['seed']+300000+index*31+request_index*3)%2**31,
                        output_dir=output/'prompt_reasoner'/name)
                    request_task['reasoner']=reasoned
                    request_task['latest_rules']=reasoned['latest_rules']
                request_task['status'] = 'ready' 
            if token not in rework_for and task.get('assembly_version') and (writer is None or callable(getattr(type(model),'operation_reasoning',None))):
                if model is None:
                    model=QwenRealizer(output_dir=output/'prompt_rewriter/runtime',**model_options)
                    owned=True
                from tools.video_editing import edit_tool_call
                planned_call=edit_tool_call(task,{p:p for p in task['dependencies']},token)
                task['operation_reasoner']=model.operation_reasoning(task['sequence_context'],planned_call,
                    seed=(result['seed']+400000+index*31)%2**31,
                    output_dir=output/'operation_reasoner'/token.strip('[]'))
            task['status'] = 'ready'
            if task.get('backend'):
                row['generation_prompt'] = task['prompt']
            state['completed'] += 1
            checkpoint()
            if progress:
                print(f"[generation prompts {state['completed']}/{len(order)}] {token}: {task['operation']}", file=sys.stderr, flush=True)
        for leader,(turn,tokens) in {**rework_groups,**delivery_groups}.items():
            workflow='asset_clip_rework' if leader in rework_groups else 'independent_outputs'
            method='rework_reasoning' if leader in rework_groups else 'delivery_reasoning'
            steps=[dict(rows[token]['generation'],placeholder=token) for token in tokens]
            prompts={step['placeholder']:step['prompt'] for step in steps if step.get('backend')}
            saved=rows[leader]['generation'].get('reasoner',{})
            if not force and saved.get('workflow')==workflow and saved.get('generation_prompts')==prompts:
                continue
            if model is None:
                model=QwenRealizer(output_dir=output/'prompt_rewriter/runtime',**model_options)
                owned=True
            if callable(getattr(type(model),method,None)):
                context=generation_context(result,leader)
                rows[leader]['generation']['reasoner']=getattr(model,method)(context,steps,
                    seed=(result['seed']+300000+turn['id']*31)%2**31,
                    output_dir=output/('rework_reasoner' if leader in rework_groups else 'delivery_reasoner')/str(turn['id']))
                checkpoint()
        from scripts.dialogue_generation.formal_sequence import generation_edit_scope
        for index, turn in enumerate(result['turns']):
            assemblies=[rows[p] for p in turn.get('output_media',[]) if rows[p].get('generation',{}).get('assembly_version')]
            if assemblies and not turn.get('intent',{}).get('asset_update_workflow'):
                season=assemblies[0].get('render_state',{}).get('season')
                turn['generation_opening']=("This turn you only asked me to join existing clips, so I'm not generating new media"
                    +(f" and the rule requiring a {season} background for new media does not apply." if season else "; I'll preserve their existing content."))
                continue
            generative = [rows[p].get('generation',{}) for p in turn.get('output_media',[])
                          if rows[p].get('generation',{}).get('backend') or rows[p].get('generation',{}).get('clip_edits')]
            if turn['role']=='assistant' and generative and not turn.get('generation_opening'):
                opening_writer = getattr(model, 'production_opening', None)
                if callable(getattr(type(model), 'production_opening', None)):
                    context = generation_context(result,turn['output_media'][0])
                    planned_outputs = []
                    for token in turn['output_media']:
                        task = tasks[token]
                        names = task.get('asset_names', {})
                        subjects = [names[asset] for asset in task.get('asset_ids', []) if asset in names]
                        planned = dict(kind=task['kind'], operation=task['operation'])
                        if subjects:
                            planned['subjects'] = subjects
                        edits = generation_edit_scope(task)
                        if edits:
                            planned['requested_edits'] = [
                                {key: edit[key] for key in ('subject', 'property', 'requested_value')}
                                for edit in edits
                            ]
                        planned_outputs.append(planned)
                    context['planned_outputs'] = planned_outputs
                    if index >= 2:
                        earlier = result['turns'][index-2]
                        context['answers_clarification'] = (
                            earlier['role'] == 'assistant'
                            and earlier.get('intent', {}).get('action', '').startswith('clarify_')
                        )
                    if turn.get('intent',{}).get('asset_update_workflow'):
                        context=dict(context,planned_workflow='First update and review the asset image, then rework and review the requested existing clip using it.')
                    turn['generation_opening'] = opening_writer(context,seed=(result['seed']+turn['id']+200000)%2**31)
        state.update(status='complete', elapsed_seconds=round(time.perf_counter()-started, 3))
        checkpoint()
    except BaseException as exc:
        state.update(status='failed', error=str(exc), elapsed_seconds=round(time.perf_counter()-started,3))
        checkpoint()
        raise
    finally:
        if owned:
            model.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--backend', choices=BACKENDS, default=DEFAULT_BACKEND)
    parser.add_argument('--model')
    parser.add_argument('--think-mode', choices=SONNET_EFFORTS, help='Sonnet reasoning effort (default: low)')
    parser.add_argument('--base-url', help='Default: localhost:11434 (Ollama), localhost:8000 (vLLM)')
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--enable-thinking', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    result = json.loads((args.run_dir/'dialogue.json').read_text())
    observation = json.loads((args.run_dir/'observation.json').read_text())
    rewrite_run(result, observation, args.run_dir, backend=args.backend, model=args.model, think_mode=args.think_mode,
                base_url=args.base_url, timeout=args.timeout, model_mode="thinking" if args.enable_thinking else "instruct", force=args.force, progress=not args.quiet)


if __name__ == '__main__':
    main()
