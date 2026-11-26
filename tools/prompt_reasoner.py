"""First-person reconstruction of generation decisions, aligned with the final prompt."""
import json
from pathlib import Path
import re
from tools.media_review import save
from scripts.dialogue_generation.output_geometry import SYSTEM_ASPECT_NOTE
from scripts.dialogue_generation.planner_rules import checked_rules

MEDIA_PLACEHOLDER = re.compile(r'\[\[(?:image|video)_\d{4}\]\]')

SYSTEM = '''Role-play the creative assistant preparing my NEXT media generation from the
actual user conversation. The final generation prompt is supplied so my plan matches
its concrete choices. Write from my first-person viewpoint BEFORE I compose that
prompt: derive each choice from the user's request, current rules and earlier media.
Do not describe reading, checking, explaining or reverse-engineering a finished
prompt. I have not seen the generated result. Use the supplied prompt only to keep
my planned content, references and choices consistent with the actual generation.
Keep the complete think to about 150 words total across all three fields combined,
including their opening sentences. Give each decision once, avoid repetition and
filler, and finish every sentence naturally. Allow room for the short closing
transition Python adds. Stop once the request, applicable rules and reference
choices are covered. The tool call is added separately, outside thinking.
Follow this standard procedure in three concise ordered paragraphs:
1. intent: Start "Let's analyze what the user is asking for." Identify the user's
requested image or video, subjects, properties, action, edit scope and preservation.
Explain step by step how I will turn this request into the next generation. If there
are alternatives, identify which ONE output I am preparing and keep the others separate.
2. rules: Start "Now let's look at which cross-turn rules are effective."
Explain why the supplied effective user rules apply and how I will satisfy them.
Explain why supplied inactive earlier rules do not apply to this output. If there
are no applicable user rules, say so. An earlier confirmed format still applies
when the latest request does not name one.
3. references: Start "Now let's gather visual references for this generation."
Reason from the conversation about whether this is an edit, a continuation of an
established identity/design, or a new identity. Explain the causal need for each
supplied reference binding and what I will retain or change. If no media is needed,
explain WHY from the request and earlier conversation: for example it requests a new
identity, is not an edit, and no earlier generated media needs to be reused, ONLY if
those facts are true. Earlier unrelated media alone does not require a reference.
Never say "the prompt specifies no prior image bindings", "the prompt uses no
references", or otherwise explain a finished prompt. Never justify a choice by
claiming that a finished prompt told me what to do. The user request motivates my decisions.
The supplied bindings are the complete set of actual media inputs. Earlier text
history can guide content without a visual attachment; do not claim exact visual
reuse or verified identity continuity for an unbound earlier image.
Return JSON {"think":{"intent":"...","rules":"...","references":"..."}}.
Write about 150 words total across intent, rules and references, with I/my for
my decisions. Allocate this shared budget flexibly to the actual task.
Use ordinary assistant language. Explain decisions from the user request itself.
Planning to write a prompt is a future action; explaining what an already written
prompt says is the wrong viewpoint. Explain reference needs from the task, without
mentioning input fields, media lists, arrays or system instructions.
Do not invent references, requested properties, invisible media details or successful
results. Do not write the generation prompt, tool call, headings or thinking tags;
the actual generation call follows the closed thinking block.'''
STEPS=('intent','rules','references')
OPENINGS=("Let's analyze what the user is asking for.",
          "Now let's look at which cross-turn rules are effective.",
          "Now let's gather visual references for this generation.")
OPERATION_OPENINGS=(OPENINGS[0],OPENINGS[1],"Now let's gather the source clips for this assembly.")
OPERATION_SYSTEM='''Write my concise first-person plan for joining existing video clips.
I am the assistant about to act on the current user request. I have not executed
the tool or seen an assembled result. The conversation supplies the clip content;
the planned Python call supplies the existing input clips and their exact order.

Return JSON with think.intent, think.rules and think.references. Keep the complete
think to about 150 words total across all three fields combined, allowing room
for the stage introductions Python adds. Each field is one concise paragraph.
Give each decision once, avoid repetition and filler, and finish every sentence
naturally. Stop once the clip order, applicable canvas rules and source roles
are covered. The exact tool call is added separately.
intent: Identify the requested clip order and the action/camera movement in each
clip. Explain why I can reuse those existing clips and will use Python concatenation.
rules: Explain every current_rules requirement. These are applicable OUTPUT CANVAS
rules: I will normalize the final canvas to that format while preserving the full
source pictures. Existing clips keep their content, identity, action and audio.
There is no new media generation, so later appearance/season rules do not apply.
source_formats states each existing clip's format. Use these facts; an old clip
need not match the current output canvas. Explain scaling/padding when formats
differ. Use the effective aspect ratio for the assembly canvas even if an input
clip has a different format.
references: Name each supplied source clip in order and explain its role using the
actual conversation. Explain that I will concatenate their complete timelines and
retain their existing audio. Describe each clip using the actual conversation.

Write about 150 words total, with I/my for my decisions and concrete steps. Write
the actual analysis in every field, not just a heading or an opening sentence.
Do not write stage introductions, headings, a generation prompt or a tool call.
Python adds the three stage introductions and the exact executed Python call.'''
SCHEMA=dict(type='object',properties={'think':dict(type='object',properties={
    key:dict(type='string',minLength=1) for key in STEPS},required=list(STEPS),additionalProperties=False)},
    required=['think'],additionalProperties=False)


PUBLIC_BRIEF_SYSTEM = """Write a short production brief suitable for showing to the user.
Summarize the requested deliverables and the concrete decisions for carrying them out.
Use three concise paragraphs totaling about 150 words. Return them as the fields
intent, rules and references inside production_brief. Base them on the supplied
conversation, applicable rules and actual reference bindings.
In intent, state the requested output count, kind, subjects, actions and changes.
Mention preserved content and any dependent steps if relevant. For an assembly,
describe what will be joined and in which order, with the existing audio retained.
In rules, explain the active constraints and the reason that earlier scoped,
canceled or replaced rules do not govern these outputs. Use the applicable output
format and distinguish continuity of source content from final canvas sizing.
In references, identify every actual bound source by its supplied placeholder.
Explain which output uses it and why. A reference-free output uses the text
request as its source. Discuss a planned dependency as future work.
Keep the description prospective, factual and ready to share. Use complete
sentences in the assistant's ordinary first-person voice. Include only the brief
and no private analysis, transcript, or tool execution claims."""
PUBLIC_BRIEF_SCHEMA = dict(SCHEMA, required=['production_brief'],
                           properties={'production_brief': SCHEMA['properties']['think']})


REWORK_SYSTEM='''Role-play the creative assistant preparing ONE coherent asset-and-clip rework
from the actual user conversation. I have not generated the replacement image or
edited the clip yet. The supplied ordered generation steps keep my plan aligned
with the actual calls. Explain my decisions from the user request and adopted
references, without describing or reverse-engineering finished prompts.

Return JSON {"think":{"intent":"...","rules":"...","references":"..."}}.
Write three concise first-person paragraphs, about 150 words TOTAL across all
three fields, including their opening sentences. State each decision once, avoid
repetition and filler, and finish every sentence naturally. The exact tool calls
are added separately. This is a single plan for the whole workflow.
intent: Start "Let's analyze what the user is asking for." Identify the genuine
requested before/after change and confirmed existing-clip scope. Explain that I
will first update the visual asset image, inspect it, present it to the user, and
then use that accepted asset to rework the requested clip while preserving its
unrelated identity, action, composition and camera movement. Any assembly is last.
rules: Start "Now let's look at which cross-turn rules are effective." Apply the
supplied current rules to the appropriate step. Distinguish image-only and
video-only requirements and use the applicable aspect ratio for each step.
references: Start "Now let's gather visual references for this generation."
Explain which existing asset image I will edit, which existing clip I will rework,
and which other adopted references I will retain. The updated image is a planned
output of the first step and becomes a reference for the later video step only
after it passes review; do not claim it already exists or has been seen.
Use each reference's supplied subject and role to explain this binding. The
asset-image stage updates that asset in its own base image; the existing clip's
full scene and camera motion belong to the video stage. An image-only season
does not transfer to the video through its asset reference. These stages are
dependent steps of the same edit, not independent alternatives.
Do not invent choices, successful results, references or changes. Return prose
without headings, tool calls or thinking tags. Do not repeat this plan before
the second tool call.'''


DELIVERY_SYSTEM='''Role-play the creative assistant preparing ALL independent outputs in one
assistant delivery from the actual user conversation. I have not generated any
of these outputs yet. The supplied ordered generation steps align my decisions
with the actual calls; do not describe reading or reverse-engineering prompts.

Return JSON {"think":{"intent":"...","rules":"...","references":"..."}}.
Write three concise first-person paragraphs, about 150 words TOTAL across all
three fields, including the opening sentences and allowing room for Python's
short closing transition. This is ONE shared analysis of the complete delivery.
intent: Start "Let's analyze what the user is asking for." State the exact number
and types of outputs, what they share and how each differs. Cover every supplied
output in its actual order. I will generate the first output and receive its
result, then generate and receive the next, and only then jointly critique the
complete set. I will not repeat the full analysis between those tool calls.
rules: Start "Now let's look at which cross-turn rules are effective." Apply
each output's current rules to that output, using its applicable aspect ratio.
Never revive superseded rules.
references: Start "Now let's gather visual references for this generation."
Explain each output's actual adopted reference bindings and why the task needs
them. Explain why reference-free outputs need no earlier media when applicable.
Do not invent a dependency between independent outputs, or treat a planned first
output as an available reference for the second. Do not claim to have seen results.
State each decision once, without filler, headings, tool calls or thinking tags.
Python adds brief stage transitions and the exact calls outside thinking.'''


NARRATIVE_STYLE='''Write only my concrete first-person decisions for this actual task.
Refer to the person making the request as "the user" or "the user's" throughout the brief.
Use "I" for the creative assistant. Keep any quoted dialogue faithful to the speaker.
Do not quote or echo directions addressed to a writer, phrases telling someone
what to state, or instructions about what not to claim. Express future dependencies
in my own words. Use the current rules belonging to each generation step; a rule
for its image is not a rule for its video. Use rule_applicability to distinguish
effective rules from actual earlier rules that are inactive for this output.
Each output has a governing aspect ratio. Use rule_applicability.aspect_ratio as
its effective ratio and source. A user_override reflects the user's format choice.
When it is system_default, Python adds the system default sentence to the trace.
Explain any difference in which outputs use the default and which use an override.
When the structured source is absent, use the supplied rules and confirmed dialogue.
In the rules paragraph explain WHY each applies or does not: media kind, subject
scope, cancellation/replacement, a one-request exception, or assembly without new
generation. For example a fall-background rule only for videos remains in force
for videos but does not govern this pair of images. Do not enforce an inactive
rule or omit it merely because it is absent from current_rules. In multi-output
workflows explain shared rules once and any scope differences by output.
For an edit, use the actual requested scope without inventing component-level
requirements. If the source already satisfies it, explain that no change is needed.
The visual_input_contract is the authority for what each actual call receives.
Read it before the conversation. A request to CREATE reference images is a request
for future outputs, not evidence that the user supplied images. For an initial
reference-free call, explain that I will create the requested assets from the
user's text. Do not describe nonexistent input images or bind output candidates
as their own sources. Existing assistant-generated media is not user-uploaded media.
Textual history describes earlier content but does not expose its pixels. Planned
earlier workflow outputs become inputs only at their stated later stage.
In the references paragraph, copy each placeholder from that call's actual_inputs
exactly, including both pairs of brackets, next to the subject or clip it
identifies. If actual_inputs is empty, say why I am generating from text and
mention no placeholders, even if the conversation discusses related images.
A planned earlier output counts only when listed in actual_inputs for a later
step; name its placeholder and explain when it will become available.
Treat each call's supplied bindings as fixed. Explain how the user's request and
the planned generation lead to those bindings, rather than choosing other inputs
from history. With zero actual inputs, explain why this standalone request needs
no old visual source; never say I will bind an earlier scene or prop image.
An earlier image can exist yet be irrelevant to this call; not selected does not
mean unavailable. Creative completion of the current brief is allowed, but an
invented visual binding is not a creative detail.'''


class PromptReasoner:
    def __init__(self, model, *, public_brief=False):
        self.model = model
        self.public_brief = public_brief

    def write(self, dialogue, first_prompt, latest_rules, *, seed, output_dir,
              kind=None, references=(), output_position=1, output_count=1, rule_applicability=None):
        visible=[{key:turn[key] for key in ('role','content','media') if key in turn} for turn in dialogue]
        bindings=[{key:ref[key] for key in ('label','kind','placeholder','role','subjects') if key in ref} for ref in references]
        payload=dict(dialogue=visible,generation_prompt=dict(text=first_prompt,kind=kind,
            output_position=output_position,output_count=output_count,references=bindings),
            current_rules=checked_rules(latest_rules))
        if rule_applicability is not None:
            payload['rule_applicability']=rule_applicability
        return self._write(payload, SYSTEM, OPENINGS, dict(generation_prompt=first_prompt),
                           seed=seed, output_dir=output_dir)

    @staticmethod
    def reference_history(dialogue, placeholders):
        visible = []
        for turn in dialogue:
            row = dict(role=turn['role'], content=MEDIA_PLACEHOLDER.sub(
                lambda match: match[0] if match[0] in placeholders else '', turn['content']))
            if 'media' in turn:
                row['media'] = [token for token in turn['media'] if token in placeholders]
            visible.append(row)
        return visible

    def write_operation(self, dialogue, tool_call, latest_rules, *, seed, output_dir, source_formats=(), rule_applicability=None):
        # Canvas pixels are resolved during execution. A null planning size is
        # not a decision to ignore the applicable format, so do not supply it.
        tool_call=json.loads(json.dumps(tool_call))
        if tool_call.get('arguments',{}).get('size','absent') is None:
            tool_call['arguments'].pop('size')
        arguments = tool_call.get('arguments', {})
        placeholders = {row['placeholder'] for row in source_formats}
        placeholders.update(value for value in [*arguments.get('clips', []), arguments.get('src')]
                            if isinstance(value, str) and MEDIA_PLACEHOLDER.fullmatch(value))
        planned_call = dict(tool_call, arguments={
            key: value for key, value in arguments.items() if key not in ('destination', 'dst')})
        payload=dict(dialogue=self.reference_history(dialogue, placeholders),
                     tool_call=planned_call,current_rules=checked_rules(latest_rules))
        if rule_applicability is not None:
            payload['rule_applicability']=rule_applicability
        if source_formats:
            payload['source_formats']=list(source_formats)
        return self._write(payload, OPERATION_SYSTEM, OPERATION_OPENINGS, dict(tool_call=tool_call),
                           seed=seed, output_dir=output_dir)

    def write_rework(self, dialogue, generation_steps, *, seed, output_dir):
        return self._write_group(dialogue,generation_steps,REWORK_SYSTEM,'asset_clip_rework',
                                 seed=seed,output_dir=output_dir)

    def write_delivery(self, dialogue, generation_steps, *, seed, output_dir):
        return self._write_group(dialogue,generation_steps,DELIVERY_SYSTEM,'independent_outputs',
                                 seed=seed,output_dir=output_dir)

    def _write_group(self, dialogue, generation_steps, system, workflow, *, seed, output_dir):
        from scripts.dialogue_generation.formal_sequence import generation_edit_scope
        visible=[{key:turn[key] for key in ('role','content','media') if key in turn} for turn in dialogue]
        steps=[]
        for position, row in enumerate(generation_steps, 1):
            steps.append(dict(placeholder=row['placeholder'],output_position=position,kind=row['kind'],
                generation_prompt=row.get('prompt'),operation=row['operation'],
                references=[{key:ref[key] for key in ('label','kind','placeholder','role','subjects') if key in ref}
                            for ref in row.get('references',[])],
                current_rules=checked_rules(row.get('latest_rules',[])),
                requested_edits=generation_edit_scope(row),
                rule_applicability=row.get('sequence_context',{}).get('rule_applicability',{}),
                workflow_stage=row.get('sequence_context',{}).get('production_workflow',{}).get('current_stage')))
        scoped=[dict(requirement=f"For {row['kind']} output {row['output_position']} only: "+rule['requirement'])
                for row in steps for rule in row['current_rules']]
        payload=dict(dialogue=visible,generation_steps=steps,current_rules=scoped)
        prompts={row['placeholder']:row['generation_prompt'] for row in steps if row['generation_prompt']}
        return self._write(payload, system, OPENINGS,
            dict(generation_prompt=steps[0]['generation_prompt'],generation_prompts=prompts,
                 workflow=workflow),seed=seed,output_dir=output_dir)

    def _write(self, payload, system, openings, report_fields, *, seed, output_dir):
        rule_contexts = [row.get('rule_applicability', {}) for row in payload.get('generation_steps', [payload])]
        use_system_aspect = any(context.get('aspect_ratio', {}).get('source') == 'system_default'
                                for context in rule_contexts)
        explain_user_rules = payload['current_rules'] or any(context.get('inactive') for context in rule_contexts)
        if 'generation_steps' in payload:
            calls=payload['generation_steps']
        elif 'generation_prompt' in payload:
            calls=[payload['generation_prompt']]
        else:
            calls=[]
        expected_references = {ref['placeholder'] for row in calls for ref in row.get('references', [])
                               if ref.get('placeholder')}
        if calls:
            planned={row.get('placeholder') for row in calls if row.get('placeholder')}
            origins={}
            for turn in payload.get('dialogue',[]):
                for token in turn.get('media',[]):
                    origins.setdefault(token,'assistant_generated' if turn['role']=='assistant' else 'user_attached')
            contract=[]
            for row in calls:
                refs=[]
                for ref in row.get('references',[]):
                    token=ref.get('placeholder')
                    refs.append(dict(ref,origin='planned_earlier_workflow_output' if token in planned
                                     else origins.get(token,'existing_bound_media')))
                entry = dict(output_position=row.get('output_position'), kind=row.get('kind'),
                    visual_input_count=len(refs),actual_inputs=refs,
                    mode='generation_from_text_only' if not refs else 'generation_with_bound_visual_inputs')
                if row.get('placeholder') in expected_references:
                    entry['output'] = row['placeholder']
                else:
                    row.pop('placeholder', None)
                contract.append(entry)
            payload['dialogue'] = self.reference_history(payload['dialogue'], expected_references)
            payload=dict(visual_input_contract=contract,**payload)
        if self.public_brief:
            system = PUBLIC_BRIEF_SYSTEM
            schema, output_key = PUBLIC_BRIEF_SCHEMA, 'production_brief'
        else:
            schema, output_key = SCHEMA, 'think'
        system += '\n' + NARRATIVE_STYLE
        if calls:
            if expected_references:
                system += (
                    '\nIn the references paragraph, explain how I will relabel every actual input for this '
                    'generation prompt. State the explicit mapping from each media placeholder to its exact '
                    'supplied label in actual_inputs, in that call\'s reference order. For example, say '
                    'I\'ll relabel the listed media as their supplied picture or video labels for this prompt. '
                    'For multiple outputs, state the mappings separately per generation call and use its labels. '
                    'Include these actual placeholders: ' + ', '.join(sorted(expected_references)) + '.'
                )
            else:
                system += '\nThis request has zero bound visual references. Explain why I will generate from text. Write no media placeholders.'

        directory=Path(output_dir)
        save(directory/'input.json',payload)
        (directory/'system.txt').write_text(system,encoding='utf-8')
        error=None;feedback=''
        for attempt in range(2):
            result=self.model(json.dumps(payload,ensure_ascii=False)+feedback,[],system_prompt=system,
                enable_thinking=False,response_schema=schema,max_new_tokens=3072,
                temperature=.2,seed=(seed+attempt)%2**63,output_dir=directory/'runtime')
            save(directory/f'raw-{attempt+1}.json',result)
            try:
                if result.get('finish_reason') in ('length','max_tokens'):
                    raise ValueError('Truncated prompt reasoner')
                parsed=json.loads(result.get('text') or '')
                if set(parsed)!={output_key} or not isinstance(parsed[output_key],dict) or set(parsed[output_key])!=set(STEPS):
                    raise ValueError('Invalid prompt reasoner JSON')
                steps=parsed[output_key]
                if any(not isinstance(p,str) or not p.strip() or '<think>' in p or '</think>' in p for p in steps.values()):
                    raise ValueError('Every step needs plain prose without thinking tags')
                # Enforce the sequence's stage framing deterministically. Prose
                # semantics belong in SYSTEM, not keyword-based rejection gates:
                # "I will construct a video prompt" is a valid prospective plan.
                steps={key:paragraph.strip() for key,paragraph in steps.items()}
                if calls:
                    cited = set(MEDIA_PLACEHOLDER.findall(steps['references']))
                    missing, unbound = expected_references - cited, cited - expected_references
                    if missing or unbound:
                        raise ValueError(f'Reference paragraph must cite {sorted(expected_references)} exactly; missing {sorted(missing)}, unbound {sorted(unbound)}')
                for key,opening in zip(STEPS,openings):
                    if not steps[key].startswith(opening):
                        steps[key]=opening+' '+steps[key]
                if use_system_aspect:
                    if not explain_user_rules:
                        steps['rules'] = openings[1] + ' No user-set cross-turn rules apply here.'
                    if SYSTEM_ASPECT_NOTE not in steps['rules']:
                        steps['rules'] += ' ' + SYSTEM_ASPECT_NOTE
                think='\n\n'.join(steps[key] for key in STEPS)
                report=dict(think=think,steps=steps,latest_rules=payload['current_rules'],**report_fields)
                save(directory/'report.json',report)
                return report
            except (ValueError,TypeError,KeyError) as exc:
                error=exc;save(directory/f'error-{attempt+1}.json',dict(error=str(exc)))
                feedback='\nCorrect the response format: '+str(exc)+f'. Return the complete JSON {output_key} again.'
        raise RuntimeError('Invalid prompt reasoner after two technical attempts') from error
