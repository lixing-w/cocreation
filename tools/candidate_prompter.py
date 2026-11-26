"""Jointly write separate prompts for alternatives in one actual user request."""
import json,re
from pathlib import Path
from tools.placeholder_prompter import rewrite_placeholder, shared_writer_system, writer_contract

def rewrite_candidates(tasks,request,*,seed=0,output_dir=None,max_attempts=3):
    first=tasks[0]['sequence_context'];count=len(tasks)
    assert count>1 and [t['sequence_context']['output_position'] for t in tasks]==list(range(1,count+1))
    assert all(t['sequence_context']['current_user_request']==first['current_user_request'] and
               t['sequence_context']['visible_sequence']==first['visible_sequence'] and
               t['sequence_context']['output_count']==count for t in tasks)
    reference_modes = tuple(sorted({bool(task.get('references')) for task in tasks}))
    system = shared_writer_system(tasks[0]['backend'], image_reference_modes=reference_modes)
    scope='one continuous [Shot 1]' if tasks[0]['kind']=='video' else 'one complete image'
    system+= ('\nWrite all requested alternatives together, with one independent prompt per output, '
        'in the actual user\'s stated order. Keep the requested alternatives distinct. '
        f'Each prompt describes only its corresponding option as {scope}. '
        'Comparing the outputs together concerns their presentation after separate generation. '
        'Do not put the whole comparison or other options into each output. '
        'The guide applies to the text inside each prompt string. '
        'Return the enclosing response as JSON only: {"prompts":[{"output_position":1,"prompt":"..."}, ...]}. '
        'Within each prompt string, put every required section heading on a separate line; encode newlines as \\n in JSON strings.')
    outputs=[]
    for task in tasks:
        data=writer_contract(task)
        output={k:data[k] for k in ('output_position','operation','output_format','references')}
        output['latest_rules']=data.get('latest_rules',[])
        output.update({key: data[key] for key in ('production_workflow', 'requested_edits', 'output_subjects')
                       if data.get(key)})
        outputs.append(output)
    payload=dict(current_user_request=first['current_user_request'],visible_sequence=first['visible_sequence'],
                 current_exchange=first.get('current_exchange',first['visible_sequence'][-3:]),
                 output_count=count,outputs=outputs)
    user='Write the separate expert prompts for these requested outputs.\n'+json.dumps(payload,ensure_ascii=False)
    directory=Path(output_dir) if output_dir else None
    if directory:
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'input.txt').write_text(user);(directory/'system.txt').write_text(system)
    feedback='';last=None
    for attempt in range(max_attempts):
        text,finish=request(user+feedback,system=system,seed=seed+attempt,max_tokens=8192)
        try:
            if finish=='length':raise ValueError('truncated response')
            parsed=json.loads(text)
            if set(parsed)!={'prompts'} or len(parsed['prompts'])!=count:raise ValueError('wrong output count')
            prompts={};normalized=[]
            for position,(task,item) in enumerate(zip(tasks,parsed['prompts']),1):
                if set(item)!={'output_position','prompt'} or item['output_position']!=position:raise ValueError('wrong output order')
                prompt=rewrite_placeholder(task,lambda *a,**kw:(item['prompt'],'stop'),max_attempts=1,
                    output_dir=directory/f'attempt-{attempt+1}'/f'output-{position}' if directory else None)
                prompts[task['placeholder']]=prompt
                normalized.append(re.sub(r'\s+',' ',prompt).strip().casefold())
            if len(set(normalized))!=count:raise ValueError('alternatives repeated the same prompt')
            if directory:(directory/f'response-{attempt+1}.json').write_text(json.dumps(dict(text=text,finish_reason=finish,prompts=prompts),ensure_ascii=False,indent=2))
            return prompts
        except (ValueError,TypeError,KeyError,RuntimeError) as exc:
            last=exc
            if directory:(directory/f'response-{attempt+1}.json').write_text(json.dumps(dict(text=text,finish_reason=finish,error=str(exc)),ensure_ascii=False,indent=2))
            feedback=('\nValidation error: '+str(exc)+'\nReturn the required JSON with every output in order. '
                f'Each prompt must describe only its own distinct requested alternative as {scope}, '
                'using its supplied reference labels. Put every required section heading at the start of its own line, '
                'encoding newlines as \\n inside each JSON prompt string.')
    raise RuntimeError('Candidate prompts rejected after bounded retries') from last
