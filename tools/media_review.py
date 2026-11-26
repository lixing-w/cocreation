"""First-person JSON visual review and delegation to the independent prompt fixer."""
from contextlib import contextmanager
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request

from scripts.dialogue_generation.output_geometry import SYSTEM_ASPECT_NOTE

SCORES = ('request_fidelity', 'edit_preservation', 'reference_alignment', 'visual_integrity', 'cross_turn_rules')
SYSTEM = '''You are the creative assistant reviewing the media I just generated.
Write a concise first-person, step-by-step review grounded in visible evidence.
Aim for about 150 words total for the complete think, including the three prose
fields and numbered rule evidence. Allow room for the stage introductions Python
adds. State each finding once, avoid repetition and filler, and finish every
sentence naturally. Stop once the request, applicable rules, reference continuity
and decision are covered.
Inspect the actual GENERATED RESULT and the actual USER-APPROVED REFERENCES.
The Python planner has already selected the user's adopted reference versions.
These references are the authority for existing character identity and design.
Compare the result directly with them for clear identity/design drift. Do not
rejudge the adopted character's age or appearance. Words such as woman, girl or
young person do not request a different age. Only an explicit new appearance edit
changes the reference baseline, and only for the properties requested.
Use each reference for its selected subject/property. Incidental objects in one
reference cannot override the actual reference selected for another subject.
Check only obvious, material failures of the actual request. A general object
color request is satisfied by its main visible body; it does not require every
small component, handle, trim or reflection to share that color. A blue sword
blade with a gray hilt can satisfy "make the sword blue". Do not invent a
component-level requirement unless the user explicitly named that component.
requested_edits is planning context; the actual request and visible result decide
whether it is met. If the source already satisfies it, no visible change is
required. Explain that it was already satisfied instead of claiming a new edit.
With no attached references, do not claim visual matching to an earlier image
mentioned only in text.
Use visual_evidence to distinguish the attached visual sources from history.
Historical media markers and spoken descriptions are text, not attached pixels. With zero visual sources, compare the
generated result to the stated request only: say the scene fits its description,
not that it visually matches a selected earlier image. "Reference image" in a
request often names an output to create, not a source the user uploaded.

For a video, inspect its beginning, middle and end before deciding. Compare
framing across time to identify a push-in, pull-out, pan or camera shake. Do not
call the whole clip static or wide because its opening frame is static or wide.
Content appearing later in the clip is present. Ground a temporal rejection in
the observed progression and approximate time range, within the same think budget.
An explicit hand/forearm close-up does not require a face or full-body close-up.
Distinguish a clearly missing requested action from motion that is too subtle,
brief or outside the framing to verify. Uncertain breathing, jolting or shake
is score 1, not proof of a material failure. Do not infer breathing from audio;
this reviewer receives visual evidence only.

Read the actual conversation and current exchange as well as the latest user
message. The latest message may only resolve a rule conflict or say to proceed;
the preceding production request still specifies the requested content and motion.
Use latest_rules as the current user-set rule snapshot; do not revive superseded rules.
When aspect_ratio_rule is supplied, it identifies the output's governing ratio
and whether it comes from the system default or a user override. Use the verified
canvas as measurement. Python states the system default in the trace where it
applies. Keep numbered rule_checks matched to latest_rules only.
Interpret a season in the actual setting and framing. A season label alone does
not require trees, foliage, orange colors, snow or other unrequested scenery.
A bare coast, interior or tight close-up may have no diagnostic seasonal cue.
An unobservable or ambiguous season is score 1, not evidence of the wrong season.
Reject only a visibly supported conflict with the user's actual requirement;
do not replace their adopted setting to manufacture a generic seasonal backdrop.
For time-of-day rules, use visible lighting, sky or windows in the current scene.
Check dawn and dusk using scene evidence and context, not color temperature alone.
A neutral studio or windowless interior can be inconclusive; score 1 when time
is not observable. Preserve the setting without requiring new props or windows.

Read the current user request plainly. Check the requested content, action, edit
and active rules without adding idealized requirements or demanding perfection.
Static/fixed framing describes the camera, not frozen video or a motionless person.
Static handheld framing allows small natural handheld movement. Ordinary subject
motion does not itself show camera movement. A stopped/standing subject can meet
"stopping" without a staged walk-then-stop transition unless explicitly requested.
Natural variation in apparent age, smile intensity, eye narrowing, warm sunlight,
exposure or subtle motion is not a reason to regenerate an otherwise suitable clip.
Warm golden light need not be unusually dramatic. Reject a wrong core action only
when it is clearly visible, rather than inferred from small pose changes.

output_position/output_count identify the outputs in this review. Independent
alternatives are reviewed together. When workflow_stage is supplied, this is one
dependent stage of an asset-image-then-clip workflow: follow its review_scope.
Check an updated asset image for the requested visual asset change; the later
video supplies camera motion and temporal action. Check a reworked clip for the
requested motion and edits using the actual adopted references. The original
user request remains authoritative; other workflow outputs are delivered at their
own stages and need not appear inside this output.
Verified canvas is the measured format after backend pixel-grid rounding. Use it
as fact, without estimating another ratio from the preview.

Score every criterion 0, 1 or 2:
0 = a clear, material, visibly supported failure of an explicit requirement.
1 = a minor deviation, subjective difference or uncertainty; still acceptable.
2 = the requirement is met.
request_fidelity: requested subjects, core action, scene and edit are present.
edit_preservation: the requested edit occurred and unrelated content remains;
use 2 when no edit is requested. Requested changes override old reference texture.
reference_alignment: retained identity/design matches the adopted references;
use 2 without references. Judge drift, not whether the adopted design is ideal.
visual_integrity: usable coherent imagery and motion; minor imperfections are 1.
cross_turn_rules: the minimum rule score, or 2 without active rules.
Check EVERY latest_rules entry once in rule_checks with its 1-based rule_index,
score and specific visible/measured evidence. Keep each rule's evidence concise.
Accept when every criterion and rule is at least 1. Reject only if something is 0;
name the concrete visible material failure and the explicit requirement it breaks.
Do not invent a flaw just to justify a rejection or populate a review paragraph.

Return the required JSON. think contains three concise first-person prose fields:
request_check: what I actually see, compared with the plain request and references.
quality_check: identity/design continuity, requested edits and visual coherence.
decision: my conclusion and next action, in 1-2 complete sentences, at most 60 words.
Keep the whole reflection around 150 words. Python inserts the stage introductions
and numbered rule checks. Do not duplicate them. Use ordinary assistant language.
For acceptance, reason and reply are empty. For rejection with retry_available=true,
reply identifies the material failure and says I will retry, in two short first-person
sentences of at least twelve words total. Without a retry, say I could not complete
the request. Do not include thinking tags, scores, input-field names or instructions
in prose.'''
REVIEW_POLICY_VERSION = 'explicit-visual-provenance-v8'
THINK_STEPS=('request_check','quality_check','decision')

SCHEMA = dict(type='object', properties={
    'think':dict(type='object',properties={key:dict(type='string',minLength=1) for key in THINK_STEPS},required=list(THINK_STEPS),additionalProperties=False),
    **{k:dict(type='integer',minimum=0,maximum=2) for k in SCORES},
    'rule_checks':dict(type='array',items=dict(type='object',properties={
        'rule_index':dict(type='integer',minimum=1),
        'score':dict(type='integer',minimum=0,maximum=2),
        'evidence':dict(type='string',minLength=1)},
        required=['rule_index','score','evidence'],additionalProperties=False)),
    'decision':dict(type='string',enum=['accepted','rejected']),
    'reason':dict(type='string',maxLength=1500), 'reply':dict(type='string',maxLength=600)},
    required=['think',*SCORES,'rule_checks','decision','reason','reply'],additionalProperties=False)



def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def parse_review(result, rules=None, *, aspect_ratios=()):
    if result.get('finish_reason') in ('length','max_tokens'):
        raise ValueError('Truncated media review')
    report = json.loads(result.get('text') or '')
    if set(report) != set(SCHEMA['required']):
        raise ValueError('Unexpected media review fields')
    if any(type(report[k]) is not int or report[k] not in (0,1,2) for k in SCORES):
        raise ValueError('Invalid review scores')
    checks=report['rule_checks']
    if not isinstance(checks,list):
        raise ValueError('Invalid active rule checks')
    for item in checks:
        if not isinstance(item,dict) or set(item)!={'rule_index','score','evidence'}:
            raise ValueError('Invalid active rule check fields')
        if type(item['rule_index']) is not int or item['rule_index']<1 or type(item['score']) is not int or item['score'] not in (0,1,2) or not isinstance(item['evidence'],str) or not item['evidence'].strip():
            raise ValueError('Invalid active rule evidence or score')
    if rules is not None and sorted(item['rule_index'] for item in checks)!=list(range(1,len(rules)+1)):
        raise ValueError('Check every latest rule exactly once')
    aggregate=min((item['score'] for item in checks),default=2)
    if report['cross_turn_rules'] != aggregate:
        # Preserve the model's redundant summary; policy uses the strict
        # minimum of the individual model-written rule scores.
        report['reported_cross_turn_rules']=report['cross_turn_rules']
        report['cross_turn_rules']=aggregate
    accepted = all(report[k] >= 1 for k in SCORES)
    if report['decision'] not in ('accepted', 'rejected'):
        raise ValueError('Invalid review decision')
    overridden_rejection = accepted and report['decision'] == 'rejected'
    if accepted:
        report.update(decision='accepted', reason='', reply='')
    elif report['decision'] != 'rejected':
        raise ValueError('Review decision contradicts scores')
    thinking = report['think']
    if isinstance(thinking,dict):
        if set(thinking)!=set(THINK_STEPS) or any(not isinstance(v,str) or not v.strip() for v in thinking.values()):
            raise ValueError('Review think needs request_check, quality_check and decision prose')
        steps=dict(thinking)
        if any(not value.rstrip().endswith(('.', '!', '?', '"', "'")) for value in steps.values()):
            raise ValueError('Finish each think field with a complete sentence; decision needs only 1-2 sentences')
        if overridden_rejection:
            steps['decision'] = 'I accept the result based on the review scores.'
        rule_text="Let's think."
        if not report['rule_checks']:
            rule_text += ' No user-set cross-turn rules apply here.'
        has_system_default = any(rule['source'] == 'system_default' for rule in aspect_ratios)
        if has_system_default or (not aspect_ratios and not report['rule_checks']):
            rule_text += ' ' + SYSTEM_ASPECT_NOTE
        if report['rule_checks']:
            rule_text += ' The applicable user-set cross-turn rules are:'
        for item in sorted(report['rule_checks'],key=lambda row:row['rule_index']):
            index=item['rule_index']
            requirement=(rules[index-1]['requirement']+' ' if rules is not None and 1<=index<=len(rules) else '')
            rule_text+=f"\n{index}. "+requirement+item['evidence']
        thinking=("Let's carefully look at the generated result and check if it meets the user request and applicable cross-turn rules. "
                  +steps['request_check']+'\n\n'+rule_text+'\n\n'
                  +"Now I'll check reference continuity and visual integrity. "+steps['quality_check']
                  +'\n\n'+steps['decision'])
        report['think_steps']=steps
        report['think']=thinking
    if not isinstance(thinking, str) or not thinking.strip():
        raise ValueError('Media reviewer returned no JSON think analysis')
    if '<think>' in thinking or '</think>' in thinking:
        raise ValueError('JSON think must contain prose without thinking tags')
    if (not thinking.startswith("Let's carefully look at the generated result")
            or "Let's think." not in thinking
            or not re.search(r"\bI\b|\bmy\b|\bI'll\b",thinking)):
        raise ValueError('Write a first-person review with both prescribed stage openings')
    if rules and any(not re.search(r'\b'+str(i)+r'\.',thinking) for i in range(1,len(rules)+1)):
        raise ValueError('Enumerate and inspect every active rule in think as 1., 2., 3., etc.')
    if any(not isinstance(report[k],str) for k in ('reason','reply')) or (not accepted and not report['reply'].strip()):
        raise ValueError('Invalid review prose')
    if not accepted and not report['reason'].strip():
        raise ValueError('Rejected review needs a reason')
    if not accepted and (len(report['reply'].split()) < 12 or not re.search(r'\bI\b',report['reply'])):
        raise ValueError('Rejection reply must identify the specific visible failure in first person and explain the next action in at least twelve words')
    return dict(**report, thinking=thinking)


def cached_review(attempt):
    """Revisit old-policy rejections using existing media, preserving the audit."""
    review=attempt.get('critique')
    if review and review['decision']=='rejected' and review.get('policy_version')!=REVIEW_POLICY_VERSION:
        attempt.setdefault('critique_history',[]).append(review)
        attempt.pop('critique')
        # This repair came from the superseded critique. Review the existing
        # output before deciding whether any further repair is needed.
        attempt.pop('next_prompt',None)
        return None
    return review


def media_geometry(path,kind):
    path=Path(path)
    if not path.is_file():
        return dict(available=False)
    if kind=='image':
        from PIL import Image
        with Image.open(path) as source:
            width,height=source.size
    else:
        from tools.video_editing import probe
        stream=next(s for s in probe(path)['streams'] if s['codec_type']=='video')
        width,height=stream['width'],stream['height']
    from math import gcd
    divisor=gcd(width,height)
    nominal=min(('1:1','16:9','9:16'),key=lambda ratio:abs(width/height/(int(ratio.split(':')[0])/int(ratio.split(':')[1]))-1))
    nominal_value=int(nominal.split(':')[0])/int(nominal.split(':')[1])
    return dict(available=True,width=width,height=height,
                aspect_ratio=f'{width//divisor}:{height//divisor}',width_over_height=width/height,
                nominal_aspect_ratio=nominal if abs(width/height/nominal_value-1)<=.025 else None,
                note='Native dimensions may differ slightly from a nominal ratio due to the backend pixel grid.')


def review_output_contract(model, system, schema):
    """Request a shareable evidence summary from transports that prefer one."""
    system += (
        '\nIn the assessment prose, refer to the person requesting the media as "the user" or "the user\'s". '
        'Use "I" for the assistant. The separate spoken reply can address the user directly.'
    )
    if getattr(model, 'prefers_public_summaries', False) is not True:
        return system, schema, False
    system = system.replace('Write a concise first-person, step-by-step review grounded in visible evidence.',
                            'Write a concise assessment grounded in visible evidence.')
    system = re.sub(r'\bthink\b', 'review_summary', system).replace('whole reflection', 'whole summary')
    system = ('Return a public assessment with concrete observations, rule checks and a decision. '
              'Write a summary suitable to share with the user and omit private analysis or transcripts.\n'
              + system)
    fields = {('review_summary' if key == 'think' else key): value
              for key, value in schema['properties'].items()}
    required = ['review_summary' if key == 'think' else key for key in schema['required']]
    return system, dict(schema, properties=fields, required=required), True


def normalize_review_output(result, public_summary):
    if not public_summary or result.get('finish_reason') in ('length', 'max_tokens'):
        return result
    fields = json.loads(result.get('text') or '')
    fields['think'] = fields.pop('review_summary')
    return dict(result, text=json.dumps(fields, ensure_ascii=False))


def review_conversation(context):
    visible = context.get('visible_sequence', [])
    return dict(current_user_request=context['current_user_request'], visible_sequence=visible,
                current_exchange=context.get('current_exchange', visible[-3:]))


def review_media_block(kind, path):
    from cocreation.qwen_inputs import DEFAULT_VIDEO_FPS
    block = dict(type=kind, **{kind: str(Path(path).resolve())})
    if kind == 'video':
        block['fps'] = DEFAULT_VIDEO_FPS
    return block


def review_reference_caption(slot, label):
    if slot.get('role') == 'edit_source':
        version = f" ({slot['placeholder']})" if slot.get('placeholder') else ''
        return f'PREVIOUS REJECTED RESULT {label}{version}: edit base; compare what was preserved and corrected.'
    return f'USER-APPROVED REFERENCE {label}: original input and adopted identity/design baseline.'


class MediaReviewer:
    def __init__(self, model):
        self.model = model

    def verify_turn(self,context,outputs,*,seed,output_dir):
        from tools.turn_media_review import verify
        return verify(self.model,context,outputs,seed=seed,output_dir=output_dir)

    def latest_rules(self, context, kind, *, seed, output_dir):
        from scripts.dialogue_generation.planner_rules import checked_rules
        rules=checked_rules(context['latest_rules'])
        save(Path(output_dir)/'resolved.json',dict(rules=rules,source='planner_media_snapshot'))
        return rules

    def verify(self, context, prompt, media, kind, references, *, retry_available, seed, output_dir):
        directory = Path(output_dir)
        rules=self.latest_rules(context,kind,seed=seed,output_dir=directory/'rule_resolution')
        geometry=media_geometry(media,kind)
        save(directory/'measured_geometry.json',geometry)
        # The VLM only needs the verified canvas category. Raw 7:4 pixel-grid
        # fractions distract it into rejecting the supported nominal 16:9 mode.
        ratio=geometry.get('nominal_aspect_ratio') or geometry.get('aspect_ratio')
        canvas=dict(available=geometry['available'],aspect_ratio=ratio)
        current_request=context['current_user_request']
        save(directory/'request_scope.json',dict(original=context['current_user_request'],
            current_output_request=current_request,output_position=context['output_position'],output_count=context['output_count'],
            workflow_stage=context.get('workflow_stage')))
        conversation = review_conversation(context)
        data = dict(latest_rules=rules,
            output_position=context['output_position'],output_count=context['output_count'],
            retry_available=retry_available)
        if context.get('reviewed_placeholder'):
            data['reviewed_placeholder'] = context['reviewed_placeholder']
        aspect = context.get('rule_applicability', {}).get('aspect_ratio')
        if aspect is not None:
            data['aspect_ratio_rule'] = aspect
        data['requested_edits']=context.get('requested_edits',[])
        data['reference_bindings']=context.get('reference_bindings',[]) if references else []
        data['attached_reference_count']=len(references)
        data=dict(visual_evidence=dict(generated_results=1,visual_source_count=len(references),
            source_mode='attached_visual_sources' if references else 'no_visual_sources',
            historical_conversation_is_text_only=True),**data)
        if context.get('workflow_stage'):
            data['workflow_stage']=context['workflow_stage']
        if references:
            data['reference_status'] = ('Original inputs define adopted identity/design. A labeled rejected edit base '
                                        'is evidence of the preceding result, not an approved design.')
        content = [dict(type='text', text=json.dumps(conversation, ensure_ascii=False))]
        slots = context.get('reference_slots', [])
        counts = {}
        for index, ref in enumerate(references):
            counts[ref.kind] = counts.get(ref.kind, 0) + 1
            label_kind = ('Picture' if '<Picture ' in prompt else 'Image') if ref.kind == 'image' else ref.kind.title()
            slot = slots[index] if index < len(slots) else {}
            label = slot.get('label', f'<{label_kind} {counts[ref.kind]}>')
            content.extend([dict(type='text', text=review_reference_caption(slot, label)),
                            review_media_block(ref.kind, ref.path)])
        data['Verified canvas'] = canvas['aspect_ratio'] if canvas['available'] else 'unavailable'
        content.extend([dict(type='text', text=json.dumps(data, ensure_ascii=False)),
                        dict(type='text', text='GENERATED RESULT: inspect this output against the request and the labeled sources above.'),
                        review_media_block(kind, media)])
        system, schema, public = review_output_contract(self.model, SYSTEM, SCHEMA)
        messages = [dict(role='system',content=system),dict(role='user',content=content)]
        save(directory/'input.json',dict(context=dict(conversation, **data),media=str(media),kind=kind,
             references=[dict(kind=r.kind,path=str(r.path)) for r in references]))
        last_error = None
        errors=[]
        # Malformed transport/schema output is a technical retry of the SAME
        # image, not an extra critique/regeneration. Preserve all raw responses.
        for technical_attempt in range(2):
            result = self.model(messages=messages,enable_thinking=False,
                max_new_tokens=4096,temperature=.2,
                response_schema=schema,seed=(seed+technical_attempt) % 2**63,
                output_dir=directory/'runtime')
            save(directory/f'raw-{technical_attempt+1}.json',result)
            try:
                report = parse_review(normalize_review_output(result, public), rules=rules,
                                      aspect_ratios=(aspect,) if aspect is not None else ())
                if not retry_available and report['decision']=='rejected' and re.search(r"(?i)\bI(?: am retrying| will retry|'ll retry)",report['reply']):
                    raise ValueError('No retry remains: acknowledge the failure and say I could not complete the request')
                report['policy_version']=REVIEW_POLICY_VERSION
                save(directory/'report.json',report)
                return report
            except (ValueError,KeyError,TypeError) as exc:
                last_error = exc
                errors.append(str(exc))
                correction = ('Return the complete public evidence summary, rule checks and decision for the same media.'
                              if public else 'Return the complete JSON review for this same media, with three concise first-person think fields and rule-by-rule concrete observations, about 150 words total for the reflection, and a specific first-person reply.')
                messages.append(dict(role='user',content='Your review format needs correction: '+str(exc)+'. '+correction))
                save(directory/f'error-{technical_attempt+1}.json',dict(error=str(exc)))
        raise RuntimeError('Media review validation failed after two technical attempts: '+'; '.join(errors)) from last_error

    def fix(self, task, context, prompt, reason, *, media, seed, output_dir):
        from tools.prompt_fixer import PromptFixer
        return PromptFixer(self.model).fix(task, context, prompt, reason, media=media, seed=seed, output_dir=output_dir)


@contextmanager
def reviewer_session(output_dir, *, base_url='http://127.0.0.1:8019', timeout=600, require_running=False,
                     shared_model=None):
    """Attach to a shared service, or own an on-demand service when permitted."""
    from cocreation import VLLMQwen
    from urllib.parse import urlparse
    directory = Path(output_dir)
    directory.mkdir(parents=True,exist_ok=True)
    if shared_model is not None:
        yield MediaReviewer(shared_model)
        return
    base_url = base_url.rstrip('/')
    process = None
    model = None
    def healthy():
        try:
            with urllib.request.urlopen(base_url+'/health',timeout=2) as response:
                return response.status == 200
        except OSError:
            return False
    try:
        if not healthy():
            if require_running:
                raise RuntimeError('Shared review service is unavailable: '+base_url)
            url = urlparse(base_url)
            if url.hostname not in ('127.0.0.1','localhost'):
                raise RuntimeError('External review service is unavailable: '+base_url)
            repo = Path(__file__).resolve().parents[1]
            with (directory/'vllm.log').open('w') as log:
                process = subprocess.Popen(['bash','-l',str(repo/'scripts/vllm/start.sh'),
                    '--port',str(url.port or 8000),'--no-enable-thinking','--kv-cache-memory-gib','16'],cwd=repo,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline = time.monotonic()+timeout
            while not healthy():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Media review vLLM failed to start: '+str(directory/'vllm.log'))
                time.sleep(1)
        model = VLLMQwen(base_url=base_url,timeout=timeout,enable_thinking=False,
                        output_dir=directory/'runtime')
        yield MediaReviewer(model)
    finally:
        if model is not None:
            model.close()
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait(timeout=30)
