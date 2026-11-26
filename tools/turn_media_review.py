"""Review a delivery as it accumulates, using actual prompts and media."""
import json
from pathlib import Path

from tools.media_review import (SYSTEM as BASE_SYSTEM, SCHEMA as SINGLE_SCHEMA, SCORES,
    media_geometry, normalize_review_output, parse_review, review_conversation, review_media_block, review_output_contract, review_reference_caption, save)

POLICY_VERSION='sequential-delivery-v1'
CRITERIA=tuple(key for key in SCORES if key!='cross_turn_rules')
INTRO=BASE_SYSTEM.split('output_position/output_count')[0]
FULL_SCOPE='''

Review ALL numbered GENERATED OUTPUT files together as one assistant delivery.
Read the complete user request. Match its numbered alternatives to the numbered
files. A request for two images or two clips is fulfilled by two separate files;
do not demand that each file contain both alternatives or show a comparison collage
unless the user explicitly asked for that composition. Check both the individual
outputs and the delivery as a whole. Each output has its own applicable rules and
verified measured canvas. Do not transfer one output's exclusive requirement to
another output. User-approved references are labeled with the output they belong to.
A delivery can include dependent outputs, such as an edited clip followed by an
assembly using that clip. Check the clip for the requested edit and the assembly
for using that edited clip in the correct order. Do not apply the clip's requested
appearance change to unrelated clips inside the assembly. If an assembly issue
comes from a generated clip in this delivery, identify that clip for repair.
'''
STAGED_SCOPE='''
Review this delivery one asset at a time. delivery_stage gives the current
output position, the step and the total planned output count. Read this stage
metadata and retry_available at the end of the supplied trace. Earlier outputs
are already accepted. Their actual prompts, media and accepted critiques are
provided in chronological order as context for judging the current output.
Only score the current output. Treat previous accepted decisions as settled.
When final_stage is false, later outputs will be generated after this one
passes. Their absence is expected and must never reduce the score. For example,
one image at step 1 of a two-image request is the first alternative, not a
deficient two-image delivery. The same applies to videos and mixed deliveries.
When final_stage is true, consider all results together for coverage, comparison,
relationships and order. Verify that the current output fulfills its own part
and completes the set. If it materially fails, identify only the change needed
in the current output. Never demand both alternatives in one file unless the
user requested that composition. For a reference image followed by an edited
clip, the image demonstrates the requested visual change and the clip supplies
motion. A dependent assembly must use the accepted clip in the requested order;
unrelated clips keep their own content.
In request_check say how the current result fits its assigned part and any
applicable relationship to accepted outputs. In quality_check inspect only the
current result against its actual inputs and requirements. In decision say
whether I accept this output and what comes next, if anything.
Return exactly one output_checks entry for the current output_position.
'''
RUBRIC='''
The latest production request governs this delivery. Earlier requests explain
context and do not add requirements to these outputs unless the user carries them forward.

Score the outputs assigned for review on these criteria, using 0, 1 or 2:
request_fidelity: its assigned subject, variant, action, scene and requested edit.
edit_preservation: requested edits happened and unrelated content remains; 2 if
no edit was requested. Explicit edits replace the old reference property.
reference_alignment: identity/design continuity with its actual adopted inputs;
2 without references. Judge drift, not whether the adopted design is ideal.
visual_integrity: usable, coherent imagery and motion.
0 = a clear material visible failure of an explicit requirement.
1 = a minor deviation, subjective difference or uncertainty; ACCEPTABLE.
2 = met. Only 0 requires repair. Do not invent defects to fill a paragraph.
Check every applicable rule for each output exactly once, using its 1-based
rule_index, score and 1-3 sentences of visible or measured evidence.
The reason for an output is empty unless a score is 0; then identify that output's
concrete material failure and the explicit requirement it breaks.
Python derives acceptance and which outputs need repair from these scores.

Return JSON containing think, output_checks and reply.
think has three first-person prose fields: request_check, quality_check, decision.
In request_check compare the result with the user request in the assigned scope.
In quality_check inspect identity, preservation and visible coherence. In decision
state my conclusion and next action in 1-2 complete sentences. Aim for about 150
words for the whole reflection, including numbered rule evidence. Finish each
field with a complete sentence. Python adds stage introductions and numbered rule
checks; do not repeat them. Do not include tags, scores, input-field names or
system instructions in prose. If all scores are at least 1, reply is empty.
If a score is 0 and retry_available is true for that output, reply identifies the
affected output and its material problem, then says I will retry that output.
If that output has no retry left, say I could not complete the request. Write the
rejection reply in first person, in two short sentences of at least twelve words.
'''

SYSTEM=INTRO+FULL_SCOPE+RUBRIC
STAGED_SYSTEM=INTRO+STAGED_SCOPE+RUBRIC

OUTPUT_SCHEMA=dict(type='object',properties={
    'output_position':dict(type='integer',minimum=1),
    **{key:SINGLE_SCHEMA['properties'][key] for key in CRITERIA},
    'rule_checks':SINGLE_SCHEMA['properties']['rule_checks'],
    'reason':SINGLE_SCHEMA['properties']['reason']},
    required=['output_position',*CRITERIA,'rule_checks','reason'],additionalProperties=False)
SCHEMA=dict(type='object',properties={
    'think':dict(type='object',properties={key:dict(type='string',minLength=1)
                                        for key in ('request_check','quality_check','decision')},
                required=['request_check','quality_check','decision'],additionalProperties=False),
    'output_checks':dict(type='array',minItems=1,items=OUTPUT_SCHEMA),
    'reply':SINGLE_SCHEMA['properties']['reply']},
    required=['think','output_checks','reply'],additionalProperties=False)


def parse(result,outputs, *, active_position=None):
    if result.get('finish_reason') in ('length','max_tokens'):
        raise ValueError('Truncated turn media review')
    data=json.loads(result.get('text') or '')
    if set(data)!=set(SCHEMA['required']) or not isinstance(data['output_checks'],list):
        raise ValueError('Unexpected turn review fields')
    positions=[row.get('output_position') for row in data['output_checks']]
    by_position = {output.get('output_position', index): output for index, output in enumerate(outputs, 1)}
    expected = [active_position] if active_position is not None else sorted(by_position)
    if any(type(position) is not int for position in positions) or sorted(positions)!=expected:
        if active_position is not None:
            raise ValueError('Check only the current output in output_checks')
        raise ValueError('Check every output exactly once in output_checks')
    rows=sorted(data['output_checks'],key=lambda row:row['output_position'])
    flat_rules=[];flat_checks=[]
    for row in rows:
        output = by_position[row['output_position']]
        if set(row)!=set(OUTPUT_SCHEMA['required']):
            raise ValueError('Unexpected per-output review fields')
        if any(type(row[key]) is not int or row[key] not in (0,1,2) for key in CRITERIA):
            raise ValueError('Invalid per-output criterion scores')
        rules=output.get('latest_rules',[]);checks=row['rule_checks']
        if not isinstance(checks,list) or any(not isinstance(check,dict) or set(check)!={'rule_index','score','evidence'} for check in checks):
            raise ValueError('Invalid per-output rule checks')
        if any(type(check['rule_index']) is not int for check in checks) or sorted(check['rule_index'] for check in checks)!=list(range(1,len(rules)+1)):
            raise ValueError('Check every applicable rule for every output exactly once')
        for check in sorted(checks,key=lambda item:item['rule_index']):
            flat_rules.append(dict(requirement=f"Output {row['output_position']} ({output['placeholder']}): "+rules[check['rule_index']-1]['requirement']))
            flat_checks.append(dict(check,rule_index=len(flat_checks)+1))
        if any(type(check['score']) is not int or check['score'] not in (0,1,2) for check in checks):
            raise ValueError('Invalid per-output rule scores')
        row['cross_turn_rules']=min((check['score'] for check in checks),default=2)
        row['decision']='accepted' if all(row[key]>=1 for key in SCORES) else 'rejected'
        if not isinstance(row['reason'],str) or row['decision']=='rejected' and not row['reason'].strip():
            raise ValueError('A materially failed output needs a concrete reason')
        if row['decision'] == 'accepted':
            row['reason'] = ''
    scores={key:min(row[key] for row in rows) for key in SCORES}
    rejected=[row for row in rows if row['decision']=='rejected']
    reason=' '.join(f"Output {row['output_position']}: {row['reason']}" for row in rejected)
    assembled=dict(think=data['think'],**scores,rule_checks=flat_checks,
                   decision='rejected' if rejected else 'accepted',reason=reason,reply=data['reply'])
    aspects = [by_position[position]['aspect_ratio_rule'] for position in expected
               if 'aspect_ratio_rule' in by_position[position]]
    report=parse_review(dict(text=json.dumps(assembled),finish_reason=result.get('finish_reason')),
                        rules=flat_rules,aspect_ratios=aspects)
    report.update(policy_version=POLICY_VERSION,
                  scope='delivery_stage' if active_position is not None else 'assistant_turn',output_checks=rows,
                  retry_positions=[row['output_position'] for row in rejected])
    return report


def verify(model,context,outputs,*,seed,output_dir):
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    if not outputs:raise ValueError('A turn review needs at least one output')
    stage = context.get('delivery_stage')
    positions = [output.get('output_position', index) for index, output in enumerate(outputs, 1)]
    active = stage['current_output_position'] if stage else None
    if stage and (positions[-1] != active or len(set(positions)) != len(positions)
                  or len(positions) != stage['step']
                  or stage['final_stage'] != (len(positions) == stage['output_count'])):
        raise ValueError('Delivery stage does not match its supplied outputs')
    descriptions=[]
    for position,output in zip(positions,outputs):
        geometry=media_geometry(output['media'],output['kind'])
        if not geometry['available']:raise FileNotFoundError(output['media'])
        save(directory/f'output-{position}-geometry.json',geometry)
        descriptions.append(dict(output_position=position,placeholder=output['placeholder'],kind=output['kind'],
            verified_canvas=geometry.get('nominal_aspect_ratio') or geometry['aspect_ratio'],
            latest_rules=output.get('latest_rules',[]),retry_available=output['retry_available'],
            **{key:output[key] for key in ('operation','assembly_clips','aspect_ratio_rule') if key in output}))
    conversation = review_conversation(context)
    evidence = [dict(output_position=position, visual_source_count=len(output.get('references', [])),
                     historical_conversation_is_text_only=True)
                for position, output in zip(positions, outputs)]
    payload = dict(conversation, visual_evidence=evidence, outputs=descriptions)
    if stage:
        payload['delivery_stage'] = stage
    content = [dict(type='text', text=json.dumps(conversation if stage else payload, ensure_ascii=False))]
    audit=[]
    attachments=[]
    for position,output,description,visual_evidence in zip(positions,outputs,descriptions,evidence):
        refs=output.get('references',[])
        audit.append(dict(description,media=str(output['media']),
                          references=[dict(kind=ref.kind,path=str(ref.path)) for ref in refs],
                          **{key: output[key] for key in ('generation_prompt', 'operation_call', 'accepted_critique')
                             if key in output}))
        if stage:
            # Keep an accepted output's trace reusable when the next stage begins.
            unchanged = {key: value for key, value in description.items() if key != 'retry_available'}
            content.append(dict(type='text', text=json.dumps(
                dict(output=unchanged, visual_evidence=visual_evidence), ensure_ascii=False)))
        slots = output.get('reference_slots', [])
        for index, ref in enumerate(refs):
            slot = slots[index] if index < len(slots) else {}
            label = slot.get('label', str(index + 1))
            if ref.kind in ('image', 'video'):
                content.extend([dict(type='text', text=f'OUTPUT {position} · ' + review_reference_caption(slot, label)),
                                review_media_block(ref.kind, ref.path)])
        if stage and 'generation_prompt' in output:
            content.append(dict(type='text', text=f"ACTUAL GENERATION PROMPT FOR OUTPUT {position}:\n{output['generation_prompt']}"))
        if stage and 'operation_call' in output:
            call = json.dumps(output['operation_call'], ensure_ascii=False)
            content.append(dict(type='text', text=f'ACTUAL TOOL CALL FOR OUTPUT {position}:\n{call}'))
        result_blocks = [
            dict(type='text',text=f"GENERATED OUTPUT {position} ({output['placeholder']}): this separate {output['kind']} belongs to the same delivery."),
            review_media_block(output['kind'], output['media']),
        ]
        if stage:
            content.extend(result_blocks)
            if 'accepted_critique' in output:
                summary = output['accepted_critique']
                content.append(dict(type='text', text=f'ACCEPTED CRITIQUE FOR OUTPUT {position}:\n{summary}'))
        else:
            attachments.extend(result_blocks)
    content.extend(attachments)
    if stage:
        content.append(dict(type='text', text=json.dumps(
            dict(delivery_stage=stage, retry_available=outputs[-1]['retry_available']), ensure_ascii=False)))
    save(directory/'input.json',dict(context=payload,outputs=audit))
    system, schema, public = review_output_contract(model, STAGED_SYSTEM if stage else SYSTEM, SCHEMA)
    messages=[dict(role='system',content=system),dict(role='user',content=content)]
    errors=[]
    for attempt in range(2):
        result=model(messages=messages,enable_thinking=False,max_new_tokens=6144,temperature=.2,
                     response_schema=schema,seed=(seed+attempt)%2**63,output_dir=directory/'runtime')
        save(directory/f'raw-{attempt+1}.json',result)
        try:
            report=parse(normalize_review_output(result, public), outputs, active_position=active)
            if stage:
                report['delivery_stage'] = stage
            save(directory/'report.json',report)
            return report
        except (ValueError,KeyError,TypeError) as exc:
            errors.append(str(exc));save(directory/f'error-{attempt+1}.json',dict(error=str(exc)))
            correction = ('Return one output_checks entry for the current output only.' if stage
                          else 'Return the complete JSON with every output checked once.')
            messages.extend([dict(role='assistant',content=result.get('text') or ''),
                dict(role='user',content='Correct the same review. '+str(exc)+'. '+correction)])
    raise RuntimeError('Turn media review validation failed after two technical attempts: '+'; '.join(errors))
