"""Official spoken/media sequence; planner annotations remain in internal files."""
from copy import deepcopy
import html
import json
from math import gcd
from pathlib import Path
import re
from urllib.parse import quote

from scripts.dialogue_generation.generation_retry import attempt_placeholder, clip_placeholder

TOKEN = re.compile(r'\[\[(?:image|video)_\d{4}(?:_clip\d+)?(?:_\d+)?\]\]')


def spoken(turn):
    # content is the realized dialogue, not intent.description or media captions.
    text = TOKEN.sub('',turn.get('content') or '')
    return re.sub(r'(?im)^references?:.*$', '',text).strip()


def generation_context(result, token, *, operation=None):
    """Actual spoken history, shown media IDs, and current applicable rule requirements."""
    turns = result.get('turns',[])
    delivery = next((i for i,t in enumerate(turns) if token in t.get('output_media',[])),None)
    if delivery is None:
        raise ValueError('Generated media must belong to an actual assistant delivery: '+token)
    history = []
    for turn in turns[:delivery]:
        text = spoken(turn)
        if not text:
            raise ValueError('Generation context requires realized spoken history')
        history.append(dict(role=turn['role'],content=text,
                            media=list(turn.get('output_media',[]))))
    current = next((t['content'] for t in reversed(history) if t['role']=='user'),None)
    if current is None:
        raise ValueError('Generation requires an actual user request')
    from scripts.dialogue_generation.planner_rules import media_rules, rule_applicability
    context = dict(current_user_request=current,visible_sequence=history,current_exchange=history[-3:],latest_rules=media_rules(result,token,operation=operation),output_placeholder=token,output_position=turns[delivery]['output_media'].index(token)+1,output_count=len(turns[delivery]['output_media']),
                rule_applicability=rule_applicability(result,token,operation=operation),
                attributes_instruction='The planned_output description and current attributes specify this deliverable. Interpret the actual conversation together with that plan and the exact references list; prior media is not an input unless listed.')
    workflow = turns[delivery].get('intent', {}).get('asset_update_workflow')
    if workflow:
        stages = [dict(placeholder=p, kind='image', role='updated_asset_image') for p in workflow['images']]
        stages += [dict(placeholder=p, kind='video', role='reworked_clip') for p in workflow['clips']]
        if workflow.get('compilation'):
            stages.append(dict(placeholder=workflow['compilation'], kind='video', role='assembled_video'))
        context['production_workflow'] = dict(kind='asset_clip_rework', stages=stages,
            current_stage=next(stage['role'] for stage in stages if stage['placeholder']==token),
            instruction='These are dependent stages of one requested edit, not alternatives. Update the asset image first; after it passes review, use it to edit the existing clip. Assemble last if listed. Planned outputs are not earlier observed media.')
    return context


def generation_edit_scope(task):
    """Resolved subject/property edits, separate from output rendering rules."""
    edits=[]
    for change in task.get('changes', []):
        field=change['field']
        if field.startswith('render.'):
            continue
        subject,_,prop=field.rpartition('.')
        subject=subject or task.get('subject')
        if not subject:
            names=task.get('asset_names', {})
            selected=[names[aid] for aid in task.get('asset_ids', []) if aid in names]
            subject=selected[0] if len(selected)==1 else None
        edit=dict(subject=subject,property=prop or field,requested_value=change['after'])
        if 'before' in change:
            edit['previous_value']=change['before']
        edits.append(edit)
    return edits


def review_context(result, token, *, context=None):
    """Scope dependent asset/clip reviews while retaining the full spoken request."""
    context = dict(context if context is not None else generation_context(result, token))
    row=next((row for row in result.get('placeholders', []) if row['placeholder']==token),{})
    task=row.get('generation', {})
    context['requested_edits']=generation_edit_scope(task)
    context['reference_bindings']=[{k:ref[k] for k in ('label','kind','placeholder','role','subjects') if k in ref}
                                   for ref in task.get('references', [])]
    turn = next(t for t in result['turns'] if token in t.get('output_media', []))
    workflow = turn.get('intent', {}).get('asset_update_workflow')
    if not workflow:
        return context
    if token in workflow['images']:
        stage = 'updated_asset_image'
        scope = ('Review this still image for the requested asset changes and retained identity/design. '
                 'The requested clip is a later stage of the same workflow; camera motion and '
                 'temporal action are requirements for that video, not this asset image.')
    elif token in workflow['clips']:
        stage = 'reworked_clip'
        scope = ('Review this reworked clip against the user request, applicable rules and actual '
                 'references, including the updated asset image. Other workflow outputs are '
                 'delivered separately; this clip must contain the requested motion and edits.')
    elif token == workflow.get('compilation'):
        stage = 'assembled_video'
        scope = ('Review this assembled video against the requested clip order and edits. '
                 'The updated asset images and individual clips are separate preceding stages.')
    else:
        raise ValueError('Output is missing from its asset update workflow: '+token)
    context['workflow_stage'] = dict(kind=stage, placeholder=token, review_scope=scope,
        delivery_output_position=turn['output_media'].index(token)+1,
        delivery_output_count=len(turn['output_media']),
        delivery_outputs=list(turn['output_media']))
    context.update(output_position=1, output_count=1)
    return context


def attempts_for(record, token):
    return record.get('attempts') or [dict(attempt, attempt_scope=clip_placeholder(token, index))
        for index, child in enumerate(record.get('clip_edits', []), 1)
        for attempt in child.get('attempts', [])]


def media_segment(attempt, token):
    number = attempt.get('attempt')
    placeholder = attempt_placeholder(attempt.get('attempt_scope', token), number or 1)
    segment = dict(type='media', kind=attempt['kind'], placeholder=placeholder, path=attempt['path'])
    if number:
        segment['attempt'] = number
    if placeholder != token:
        segment['logical_placeholder'] = token
    return segment

def text_segment(content):
    return dict(type='text',text=content)


def thinking_segment(content, source):
    return dict(type='thinking',source=source,text='<think>\n'+content+'\n</think>')


def generation_segment(attempt, token, record):
    """The exact executed call is a separate segment, outside all thinking."""
    backend=attempt.get('backend') or record.get('backend')
    kind=attempt.get('kind') or record.get('kind') or ('image' if token.startswith('[[image') else 'video')
    name={'h3_image':'minimax-h3-image','minimax_h3':'minimax-h3',
          'flux':'flux','vdn_h3':'vdn-h3'}.get(backend, 'minimax-h3-image' if kind=='image' else 'minimax-h3')
    parameters=deepcopy(attempt.get('effective_parameters') or attempt.get('options') or {})
    # Earlier receipts contain API options and the graph recipe separately.
    if attempt.get('inference_recipe') and not attempt.get('effective_parameters'):
        parameters.update(attempt['inference_recipe'])
    references = deepcopy(attempt.get('references', []))
    for ref in references:
        if ref.get('role') == 'edit_source' and ref.get('attempt'):
            ref['placeholder'] = attempt_placeholder(attempt.get('attempt_scope', token), ref['attempt'])
    call=dict(name=name,
        arguments=dict(prompt=attempt.get('prompt') or attempt['actual_prompt'],
                       references=references,parameters=parameters))
    return tool_call_segment(call)


def tool_call_segment(call):
    call=deepcopy(call)
    call.pop('id',None)
    arguments = call.get('arguments', {})
    if 'parameters' in arguments:
        options = arguments['parameters']
        parameters = {key: options[key] for key in ('seconds', 'ratio', 'res') if key in options}
        width, height = options.get('width'), options.get('height')
        if width and height:
            ratio = width / height
            nominal, value = min((('1:1', 1), ('16:9', 16/9), ('9:16', 9/16)),
                                 key=lambda item: abs(ratio/item[1] - 1))
            divisor = gcd(width, height)
            parameters.update(ratio=nominal if abs(ratio/value - 1) <= .025 else f'{width//divisor}:{height//divisor}',
                              res=f'{min(width, height)}p')
        arguments['parameters'] = parameters
    if 'references' in arguments:
        references = arguments.pop('references')
        for reference in references:
            reference.pop('path', None)
        call['arguments'] = dict(references=references, **arguments)
    return dict(type='tool_call',tool_calls=[call],
                text='<tool_call>\n'+json.dumps(call,ensure_ascii=False,indent=2)+'\n</tool_call>')


def ordinal(position):
    return {1:'first',2:'second',3:'third'}.get(position,str(position)+'th')


def generation_preface(kind, *, position=1, count=1):
    product=(ordinal(position)+' ' if count>1 else '')+kind
    return ("I have analyzed the user request, cross-turn rules, and applicable references. "
            "Now, I'll make a tool call to generate the "+product+'.')


def generation_segments(attempt, token, record, *, plan=None, preface=None):
    call = generation_segment(attempt, token, record)
    if attempt.get('attempt', 1) > 1:
        return [call]
    if plan is None:
        plan=attempt.get('reasoner',{}).get('think','')
    kind=attempt.get('kind') or record.get('kind') or ('image' if token.startswith('[[image') else 'video')
    preface=preface or generation_preface(kind)
    return [thinking_segment((plan+'\n\n' if plan else '')+preface,
                             'prompt_reasoner' if plan else 'generation_transition'),
            call]


def operation_segments(record, *, plan=None, preface=None):
    """Export the actual deterministic Python call without a generation prompt."""
    call=deepcopy(record['tool_call'])
    if plan is None:
        plan=record.get('operation_reasoner',{}).get('think','')
    preface=preface or ("I have analyzed the user request, cross-turn rules, and applicable references. "
                        "Now, I'll make a tool call to assemble the video.")
    return [thinking_segment((plan+'\n\n' if plan else '')+preface,'operation_reasoner'),
            tool_call_segment(call)]


def turn_review_segments(tokens,records):
    rounds=list(next((records[token]['turn_reviews'] for token in tokens if records.get(token,{}).get('turn_reviews')),[]))
    # A checkpoint can contain a freshly generated file before its review.
    # Publish those actual results without inventing a review or repeating a plan.
    represented={(member['placeholder'],member['attempt']['path'])
                 for round in rounds for member in round['members']}
    pending=[]
    owner=records.get(tokens[0],{})
    pending_tokens=tokens
    if 'turn_review_active' in owner:
        active=owner['turn_review_active']
        pending_tokens=[active] if active is not None else []
    for token in pending_tokens:
        record=records.get(token,{})
        attempts=attempts_for(record, token)
        candidate=attempts[-1] if attempts else record.get('turn_candidate')
        if candidate and (token,candidate['path']) not in represented:
            pending.append(dict(placeholder=token,attempt=candidate,
                operation_record={key:record[key] for key in ('tool_call','operation_reasoner','kind') if key in record}))
    if pending:
        rounds.append(dict(members=pending,changed=[m['placeholder'] for m in pending],
                           active_placeholder=pending[0]['placeholder'] if len(pending)==1 else None))
    segments=[]
    planned=False
    for index,round in enumerate(rounds):
        previous=None
        active=round.get('active_placeholder')
        for member in round['members']:
            token=member['placeholder']
            if active is not None:
                if token!=active:continue
            elif index and token not in round['changed']:
                continue
            attempt=member['attempt'];record=records.get(token,{})
            position=tokens.index(token)+1
            if attempt.get('attempt'):
                if not planned:
                    # Legacy candidate reports already describe the whole request.
                    # Keep the first shared analysis; raw per-output reports stay in the audit.
                    plan=attempt.get('reasoner',{}).get('think','')
                    preface=generation_preface(attempt['kind'],position=position,count=len(tokens))
                elif active is not None and attempt['attempt']==1:
                    plan=''
                    preface="I'll now generate the "+ordinal(position)+' '+attempt['kind']+'.'
                elif attempt['attempt'] > 1:
                    plan = preface = None
                else:
                    plan=''
                    preface=("I've generated the "+ordinal(previous[0])+' '+previous[1]+
                             "; I'll now proceed with the "+ordinal(position)+' generation.')
                segments.extend(generation_segments(attempt,token,record,plan=plan,preface=preface))
                planned=True
            elif member.get('operation_record',{}).get('tool_call'):
                segments.extend(operation_segments(member['operation_record'],plan='' if planned else None))
                planned=True
            segments.append(media_segment(attempt, token))
            previous=(position,attempt['kind'])
        report=round.get('report')
        if not report:
            continue
        segments.append(thinking_segment(report['thinking'],'media_critique'))
        if report['decision']=='rejected':segments.append(text_segment(report['reply']))
    accepted=bool(rounds and rounds[-1].get('report',{}).get('decision')=='accepted'
                  and all(records.get(token,{}).get('status')=='complete' for token in tokens))
    return segments,accepted


def rework_segments(turn,records):
    """One plan, then reviewed asset images, a spoken handoff, and reviewed clips."""
    workflow=turn['intent']['asset_update_workflow']
    images=workflow['images']
    segments=[]
    handed_off=False
    planned=False
    previous=None
    for token in turn['output_media']:
        record=records.get(token,{})
        if token not in images and not handed_off:
            if any(records.get(image,{}).get('status')!='complete' for image in images):
                return segments,False
            segments.append(text_segment(turn['rework_transition']))
            handed_off=True
        attempts=attempts_for(record, token)
        if attempts:
            for index,attempt in enumerate(attempts):
                if not planned:
                    plan=attempt.get('reasoner',{}).get('think','')
                    preface=generation_preface(attempt['kind'],count=len(images) if token in images else 1,
                        position=images.index(token)+1 if token in images else 1)
                elif index:
                    plan = preface = None
                elif token in images:
                    plan=''
                    preface="I've generated the previous asset image; I'll now generate the next asset image."
                elif previous in images:
                    plan=''
                    preface=("Now that I've updated the asset image, I'll proceed to edit the video clip "
                             "as the user requested.")
                else:
                    plan=''
                    preface="I've reworked the previous clip; I'll now proceed to edit the next requested clip."
                segments.extend(generation_segments(attempt,token,record,plan=plan,preface=preface))
                planned=True
                segments.append(media_segment(attempt, token))
                report=attempt.get('critique')
                if not report:
                    return segments,False
                segments.append(thinking_segment(report['thinking'],'media_critique'))
                if report['decision']=='rejected':
                    segments.append(text_segment(report['reply']))
            if attempts[-1]['critique']['decision']!='accepted' or record.get('status')!='complete':
                return segments,False
            if record.get('clip_edits'):
                segments.append(dict(type='media',kind='video',placeholder=token,path=record['path']))
        elif record.get('status')=='complete':
            if record.get('tool_call'):
                segments.extend(operation_segments(record,plan='' if planned else None))
                planned=True
            segments.append(dict(type='media',kind=record.get('kind','video'),placeholder=token,path=record['path']))
        else:
            return segments,False
        previous=token
    return segments,True


def export_sequence(directory, result, records, *, status='complete'):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    official, aliases = [], {}
    for turn in result.get('turns',[]):
        text = spoken(turn)
        if not text:
            continue
        segments = []
        if turn['role'] != 'assistant' or not turn.get('output_media'):
            segments.append(text_segment(text))
        else:
            tokens = turn['output_media']
            if all(token not in records for token in tokens):
                # On failure/cancellation, never publish future planned speech
                # that pretends ungenerated media has already been delivered.
                break
            has_generation = any(attempts_for(records.get(t,{}), t) for t in tokens)
            if has_generation or any(records.get(t,{}).get('tool_call') for t in tokens):
                opening = turn.get('generation_opening')
                if opening:
                    segments.append(text_segment(opening))
            all_accepted = True
            workflow=turn.get('intent',{}).get('asset_update_workflow')
            bundled=any(records.get(token,{}).get('turn_reviews')
                or records.get(token,{}).get('turn_review_leader') for token in tokens)
            if workflow and not bundled:
                workflow_segments,all_accepted=rework_segments(turn,records)
                segments.extend(workflow_segments)
            if bundled:
                turn_segments,all_accepted=turn_review_segments(tokens,records)
                segments.extend(turn_segments)
            for token in ([] if bundled or workflow else tokens):
                record = records.get(token,{})
                attempts = attempts_for(record, token)
                if attempts:
                    for attempt in attempts:
                        segments.extend(generation_segments(attempt,token,record))
                        segments.append(media_segment(attempt, token))
                        report = attempt.get('critique')
                        if report:
                            segments.append(thinking_segment(report['thinking'],'media_critique'))
                            if report['decision'] == 'rejected':
                                segments.append(text_segment(report['reply']))
                        else:
                            all_accepted = False
                    all_accepted &= bool(attempts[-1].get('critique',{}).get('decision')=='accepted')
                    if record.get('clip_edits') and record.get('status')=='complete' and all_accepted:
                        segments.append(dict(type='media',kind='video',placeholder=token,path=record['path']))
                elif record.get('tool_call'):
                    segments.extend(operation_segments(record))
                    if record.get('status')=='complete':
                        segments.append(dict(type='media',kind=record.get('kind','video'),placeholder=token,path=record['path']))
                    else:
                        all_accepted=False
                elif record.get('status') == 'complete':
                    if record.get('actual_prompt'):
                        segments.extend(generation_segments(record,token,record))
                    segments.append(dict(type='media',kind=record.get('kind') or ('image' if token.startswith('[[image') else 'video'),placeholder=token,path=record['path']))
                else:
                    all_accepted = False
            if all_accepted:
                segments.append(text_segment(text))
                for token in tokens:
                    record = records.get(token, {})
                    attempts = record.get('attempts', [])
                    if (record.get('status') == 'complete' and len(attempts) > 1
                            and attempts[-1].get('sha256') == record.get('sha256')):
                        aliases[token] = dict(placeholder=attempt_placeholder(token, attempts[-1]['attempt']),
                                              path=record['path'], after_turn=turn['id'])
            else:
                status = 'incomplete' if status == 'complete' else status
        official.append(dict(id=turn['id'],role=turn['role'],segments=segments,
                             content='\n\n'.join(s.get('text') or f"[[media:{s['path']}]]" for s in segments)))
        if turn['role']=='assistant' and turn.get('output_media') and not all_accepted:
            break
    artifact = dict(version=4,status=status,turns=official,media_aliases=aliases)
    _write(directory/'sequence.json',json.dumps(artifact,ensure_ascii=False,indent=2)+'\n')
    from scripts.dialogue_generation.generate import markdown_header
    lines = markdown_header(result, records=records, turns=official)
    for turn in official:
        lines += [f"**{turn['id']}. {turn['role'].title()}**",'']
        for segment in turn['segments']:
            if segment['type']=='media':
                url = html.escape(quote(segment['path'],safe='/'),quote=True)
                lines += [f"`{segment['placeholder']}`", '']
                if segment['kind']=='image':
                    lines += [f'<img src="{url}" alt="generated image" style="max-width:100%;height:auto;" loading="lazy">','']
                else:
                    lines += [f'<video src="{url}" controls preload="metadata" style="max-width:100%;height:auto;"></video>','']
            elif segment['type'] in ('thinking','generation_prompt','tool_call'):
                # A code fence keeps the literal special tokens visible in MD.
                longest = max([len(m[0]) for m in re.finditer(r'`+',segment['text'])] or [0])
                fence = '`'*max(3,longest+1)
                lines += [fence+('generation_prompt' if segment['type']=='generation_prompt' else 'text'),segment['text'],fence,'']
            else:
                lines += [segment['text'],'']
    _write(directory/'dialogue.md','\n'.join(lines))
    return artifact


def _write(path,text):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(text,encoding='utf-8')
    temporary.replace(path)


def _media_pointers(value):
    if isinstance(value, str) and TOKEN.fullmatch(value):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _media_pointers(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _media_pointers(key)
            yield from _media_pointers(item)


def dialogue_without_failed_media(result, tasks, exhausted):
    """Keep whole exchanges whose delivered assets and references are still available."""
    exchanges, exchange = [], []
    for turn in result.get('turns', []):
        exchange.append(turn)
        if turn['role'] != 'assistant':
            continue
        action = turn.get('intent', {}).get('action', '')
        if turn.get('output_media') or not (
            action.startswith('clarify_') or action in ('confirm_reference', 'confirm_switch')
        ):
            exchanges.append(exchange)
            exchange = []
    dangling_turns = {turn['id'] for turn in exchange}

    from scripts.dialogue_generation.cross_turn_rule import matches

    introduced = {row['placeholder']: row for row in result.get('placeholders', [])}
    rules_with_ends = [rule for rule in result.get('rules', []) if rule.get('ended_at') is not None]
    groups = []
    for exchange in exchanges:
        turns = {turn['id'] for turn in exchange}
        outputs = {token for turn in exchange for token in turn.get('output_media', [])}
        pointers = {token for turn in exchange
                    for token in _media_pointers([turn.get('input_media', []), turn.get('intent', {})])}
        rule_refs, new_rules = set(), set()
        for turn in exchange:
            event = turn.get('intent', {}).get('rule_event') or {}
            rule_refs.update(event.get('ids', []))
            if event.get('type') == 'end':
                rule_refs.add(event['id'])
            new_rules.update(row['id'] for row in (event.get('rules') or [event.get('rule')]) if row)
        rule_refs.update(rule['id'] for token in outputs
                         for rule in introduced.get(token, {}).get('applied_rules', []))
        rule_ends = {rule['ended_at'] for token in outputs if token in introduced
                     for rule in rules_with_ends
                     if rule['ended_at'] <= introduced[token]['introduced_at'] and matches(rule, introduced[token])}
        groups.append((turns, outputs, pointers, rule_refs - new_rules, new_rules, rule_ends - turns))

    omitted, turns, missing_rules = set(exhausted), dangling_turns, set()
    while True:
        before = (len(omitted), len(turns), len(missing_rules))
        for token, task in tasks.items():
            if omitted.intersection(task['dependencies']):
                omitted.add(token)
        for ids, outputs, pointers, rules, created, rule_ends in groups:
            if (omitted.intersection(outputs | pointers) or missing_rules.intersection(rules)
                    or turns.intersection(rule_ends)):
                turns.update(ids)
                omitted.update(outputs)
                missing_rules.update(created)
        if before == (len(omitted), len(turns), len(missing_rules)):
            break
    retained = dict(result, turns=[turn for turn in result.get('turns', []) if turn['id'] not in turns])
    if 'placeholders' in result:
        retained['placeholders'] = [row for row in result['placeholders'] if row['placeholder'] not in omitted]
    if 'rules' in result:
        retained['rules'] = [
            dict(rule, ended_at=None) if rule.get('ended_at') in turns else rule
            for rule in result['rules'] if rule['id'] not in missing_rules
        ]
    return retained, omitted, turns
