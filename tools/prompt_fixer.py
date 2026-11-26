"""Write a targeted visual edit for the latest rejected generation."""
import json
from pathlib import Path
import re

from tools._prompter import parse_prompt, REF_SECTIONS

SYSTEM = '''Write an edit instruction for the attached, latest rejected result.
Inspect the actual media and find what still differs from the user's request,
the current rules and the critique. The last labeled input is the media to edit.
The earlier reference labels and their roles are unchanged. Use the supplied
labels in the new edit instruction. Mention every supplied label and
make clear which one is the edit base. Do not reproduce the earlier prompt.
Change only the visible defects needed to satisfy this output. Tell the generator
what to add, remove, recolor, replace or change, where, and through which parts of
the clip when applicable. Preserve everything that already works, including
identity, unaffected materials, scene, composition, framing, timing, action,
lighting and sound unless a correction explicitly requires one of them to change.
For a material repair specify its visible construction or texture. For a motion
repair specify the required progression. The original references constrain
identity and design; their incidental backgrounds and actions should not replace
the edited scene. For multi-output and workflow requests, repair only this output.
Acknowledge the existing result only as an edit source. Omit status claims, canvas
dimensions and inference settings. Return a JSON object with one edit_prompt string.
'''
SCHEMA = dict(type='object', properties={'edit_prompt': dict(type='string', minLength=1)},
              required=['edit_prompt'], additionalProperties=False)


class PromptFixer:
    def __init__(self, model):
        self.model = model

    def fix(self, task, context, prompt, reason, *, media, seed, output_dir):
        from tools.media_review import save, review_media_block

        directory = Path(output_dir)
        refs = [{key: ref[key] for key in ('label', 'kind', 'role', 'placeholder', 'attempt', 'subjects')
                 if key in ref} for ref in task['references']]
        *originals, source = refs
        if source['role'] != 'edit_source':
            raise ValueError('Retry fixer requires the latest candidate as the last reference')
        target = task['backend'] if task['backend'] in ('flux', 'h3_image') else 'h3'
        output_format = (
            'English edit prose. Start by explicitly editing ' + source['label'] + '.'
            if target != 'h3' else
            'Use these headings in order, each at the start of its own line: ' + ', '.join(REF_SECTIONS)
            + '. Write exactly one continuous [Shot 1]. Begin summary with [video editing] and explicitly edit '
            + source['label'] + '.'
        )
        payload = dict(current_user_request=context['current_user_request'],
                       current_exchange=context.get('current_exchange', context.get('visible_sequence', [])[-3:]),
                       latest_rules=context.get('latest_rules', []),
                       output_position=context.get('output_position', 1), output_count=context.get('output_count', 1),
                       operation=task['operation'], edit_source=source, original_references=originals,
                       previous_prompt=prompt, critique_reason=reason)
        if context.get('workflow_stage'):
            payload['workflow_stage'] = context['workflow_stage']
        system = SYSTEM + '\nOutput format: ' + output_format
        save(directory/'input.json', dict(payload, media=str(media), output_format=output_format))
        (directory/'system.txt').write_text(system, encoding='utf-8')
        messages = [dict(role='system', content=system), dict(role='user', content=[
            dict(type='text', text=json.dumps(payload, ensure_ascii=False)),
            dict(type='text', text=f"LATEST RESULT TO EDIT, {source['label']}"),
            review_media_block(source['kind'], media),
        ])]
        catalog = [dict(kind=ref['kind'], observation_label=ref['label'], generator_label=ref['label'])
                   for ref in refs]
        last_error = None
        for attempt in range(3):
            result = self.model(messages=messages, enable_thinking=False, response_schema=SCHEMA,
                max_new_tokens=6144 if target == 'h3' else 4096, temperature=.4,
                seed=(seed+attempt)%2**63, output_dir=directory/'runtime')
            save(directory/f'raw-{attempt+1}.json', result)
            try:
                if result.get('finish_reason') in ('length', 'max_tokens'):
                    raise ValueError('Truncated edit instruction')
                response = json.loads(result.get('text') or '')
                if set(response) != {'edit_prompt'} or not isinstance(response['edit_prompt'], str):
                    raise ValueError('Return only the edit_prompt string field')
                revised = response['edit_prompt']
                if re.search(r'<think>|</think>|\[\[(?:image|video)_\d+(?:_clip\d+)?(?:_\d+)?\]\]', revised):
                    raise ValueError('Use only generator labels and edit instructions')
                errors = []
                parsed = parse_prompt({'text': revised}, catalog, target, task.get('seconds'),
                                      source_attachment=False, error_callback=errors.append)
                if not parsed:
                    raise ValueError(errors[-1]['message'] if errors else 'Invalid edit prompt')
                if target == 'h3':
                    if not re.search(r'(?m)^summary:\s*\[video editing\]', parsed):
                        raise ValueError('The summary must begin with [video editing]')
                    if re.findall(r'\[Shot \d+\]', parsed) != ['[Shot 1]']:
                        raise ValueError('Keep exactly one continuous [Shot 1]')
                if not re.search(r'\b(preserv\w*|retain\w*|unchanged|keep|maintain\w*|remain\w*)\b', parsed, re.I):
                    raise ValueError('Explicitly preserve content outside the correction')
                save(directory/'report.json', dict(previous_prompt=prompt, edit_prompt=parsed))
                return parsed
            except (ValueError, TypeError, KeyError) as exc:
                last_error = exc
                save(directory/f'error-{attempt+1}.json', dict(error=str(exc)))
                messages += [dict(role='assistant', content=result.get('text') or ''),
                             dict(role='user', content=f'Correct the edit prompt for the same attached result: {exc}')]
        raise RuntimeError('Prompt fixer failed after three validated edit instructions: '+str(last_error)) from last_error
