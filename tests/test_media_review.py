from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from tools.media_review import MediaReviewer,parse_review,SCORES,cached_review,REVIEW_POLICY_VERSION
from scripts.dialogue_generation.formal_sequence import generation_context,export_sequence
from scripts.dialogue_generation.materialize import materialize
from tools.placeholder_prompter import writer_input,writer_system


def report(accepted=True):
    return dict(**{k:2 if accepted or k!='request_fidelity' else 0 for k in SCORES},
                rule_checks=[],decision='accepted' if accepted else 'rejected',
                reason='' if accepted else 'The requested fabric still looks like wood.',
                reply='unused accepted reply' if accepted else "I notice the fabric still looks like wood. I am retrying now.",
                thinking=("Let's carefully look at the generated result and check if it meets the user request and applicable cross-turn rules. "
                    "I inspect this output's requested material and compare the visible surface with the requested change. "
                    "I check its shape, composition and identity against the source, preserving unrelated details. "
                    "Let's think. All currently applicable cross-turn rules are: 1. Use the requested output format when applicable; "
                    "I check the measured geometry for this rule. If there are no active rules I have no further rule checks to make. "
                    "I then inspect the compatible prompt instructions and the coherence of the visible details. "
                    "I base my decision on the visible result and will retry only when a specific requested requirement is still missing."))


def dialogue():
    return dict(turns=[
        dict(id=1,role='user',content='Make the city out of fabric.',output_media=[],intent={'hidden':'PLANNER_SECRET'}),
        dict(id=2,role='assistant',content='Here is your reference image.\n[[image_0001]]',
             generation_opening="I'll make that fabric city image.",output_media=['[[image_0001]]'],
             intent={'description':'PLANNER_SECRET'}),
        dict(id=3,role='user',content='FUTURE_SECRET',output_media=[])])


class ReviewContractTests(unittest.TestCase):
    def test_old_rejection_is_rechecked_and_its_misguided_repair_is_not_reused(self):
        previous=dict(decision='rejected',reason='A child is not a woman.')
        attempt=dict(critique=previous,next_prompt='Replace the adopted character with an adult.')
        self.assertIsNone(cached_review(attempt))
        self.assertEqual(attempt['critique_history'],[previous])
        self.assertNotIn('next_prompt',attempt)
        current=dict(decision='rejected',policy_version=REVIEW_POLICY_VERSION)
        self.assertIs(cached_review(dict(critique=current)),current)
        accepted=dict(decision='accepted')
        self.assertIs(cached_review(dict(critique=accepted)),accepted)

    def test_individual_review_can_receive_context_for_a_multi_output_turn(self):
        calls=[]
        def model(**options):
            calls.append(options)
            data=report()
            data['think']=data.pop('thinking')
            return dict(text=json.dumps(data),finish_reason='stop')
        context=dict(current_user_request='Two comparison images: Option 1 beige, Option 2 brown.',
                     output_count=2,output_position=2,latest_rules=[])
        with tempfile.TemporaryDirectory() as temp:
            result=MediaReviewer(model).verify(context,'brown',Path(temp)/'image.png','image',[],
                retry_available=True,seed=1,output_dir=temp)
        self.assertEqual(result['decision'],'accepted')
        self.assertEqual(len(calls),1)

    def test_minor_uncertainty_is_accepted_but_material_failure_still_rejects(self):
        good=report(); thinking=good.pop('thinking');good['think']=thinking
        parsed=parse_review(dict(text=json.dumps(good),finish_reason='stop'))
        self.assertEqual(parsed['thinking'],thinking)
        self.assertEqual(parsed['think'],thinking)
        native_only=dict(good);native_only.pop('think')
        with self.assertRaises(ValueError):parse_review(dict(text=json.dumps(native_only),reasoning=thinking))
        good['request_fidelity']=1
        self.assertEqual(parse_review(dict(text=json.dumps(good)))['decision'],'accepted')
        good['request_fidelity']=0
        with self.assertRaises(ValueError):parse_review(dict(text=json.dumps(good)))
        good['decision']='rejected';good['reason']='Missing fabric.';good['reply']=report(False)['reply']
        self.assertEqual(parse_review(dict(text=json.dumps(good)))['decision'],'rejected')
        with self.assertRaises(ValueError):parse_review(dict(text=json.dumps(good),finish_reason='length'))

    def test_json_think_must_be_nonempty_plain_prose(self):
        data=report();data['think']=data.pop('thinking')
        for bad in ('', '<think>I inspect the image.</think>'):
            data['think']=bad
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                parse_review(dict(text=json.dumps(data)))

    def test_review_length_is_prompt_guidance_without_runtime_rejection(self):
        from tools.media_review import SCHEMA
        data=report();data.pop('thinking')
        for prose in ('I see the requested change.', 'I see the requested change. '*180):
            data['think']=dict(request_check=prose,quality_check=prose,decision='I accept this result.')
            with self.subTest(prose=prose[:30]):
                parsed=parse_review(dict(text=json.dumps(data),finish_reason='stop'))
                self.assertEqual(parsed['think_steps'],data['think'])
        for field in SCHEMA['properties']['think']['properties'].values():
            self.assertNotIn('maxLength',field)
        self.assertNotIn('maxLength',SCHEMA['properties']['rule_checks']['items']['properties']['evidence'])

    def test_structured_review_renders_stage_openers_and_all_rule_evidence(self):
        data=report();data.pop('thinking')
        data['think']=dict(request_check='I inspect the generated city and verify its requested fabric material. '*5,
            quality_check='I compare the preserved geometry and identify the visible woven fibers. '*4,
            decision='I accept this result because the requested change is visible and the composition remains coherent.')
        data['rule_checks']=[dict(rule_index=1,score=2,evidence='I see the measured portrait canvas.')]
        parsed=parse_review(dict(text=json.dumps(data)),[dict(requirement='Use portrait 9:16.')])
        self.assertTrue(parsed['think'].startswith("Let's carefully look at the generated result"))
        self.assertIn('1. Use portrait 9:16. I see the measured portrait canvas.',parsed['think'])
        self.assertLess(parsed['think'].index('All currently applicable'),parsed['think'].index("Now I'll check"))
        self.assertEqual(parsed['think_steps'],data['think'])

    def test_rejected_reply_names_problem_in_first_person(self):
        data=report(False);data['think']=data.pop('thinking');data['reply']='I am retrying.'
        with self.assertRaisesRegex(ValueError,'specific visible failure'):
            parse_review(dict(text=json.dumps(data)))

    def test_failed_cross_turn_rule_rejects_otherwise_good_result(self):
        data=report(False);data['think']=data.pop('thinking');data['request_fidelity']=2
        data['cross_turn_rules']=0
        data['rule_checks']=[dict(rule_index=1,score=0,evidence='The result is landscape.')]
        rules=[dict(requirement='Use portrait 9:16.')]
        self.assertEqual(parse_review(dict(text=json.dumps(data)),rules)['decision'],'rejected')
        data['rule_checks']=[]
        with self.assertRaises(ValueError):parse_review(dict(text=json.dumps(data)),rules)

    def test_cross_turn_summary_uses_strict_individual_minimum(self):
        data=report(False);data['think']=data.pop('thinking');data['request_fidelity']=2
        data['rule_checks']=[dict(rule_index=1,score=0,evidence='The result violates the current format.')]
        parsed=parse_review(dict(text=json.dumps(data)),[{}])
        self.assertEqual(parsed['cross_turn_rules'],0)
        self.assertEqual(parsed['reported_cross_turn_rules'],2)
        self.assertEqual(parsed['decision'],'rejected')

    def test_critic_only_receives_latest_resolved_rules(self):
        context=generation_context(dialogue(),'[[image_0001]]')
        context['visible_sequence'][:0]=[
            dict(role='user',content='From now on, use portrait 9:16.',media=[]),
            dict(role='user',content='From now on, use landscape 16:9.',media=[])]
        calls=[]
        rule=dict(requirement='Use landscape 16:9.')
        context['latest_rules']=[rule]
        class Model:
            def __call__(self,*args,**kw):
                calls.append((args,kw))
                data=report();data['think']=data.pop('thinking')
                data['rule_checks']=[dict(rule_index=1,score=2,evidence='The result is landscape 16:9.')]
                return dict(text=json.dumps(data),finish_reason='stop')
        with tempfile.TemporaryDirectory() as temp:
            MediaReviewer(Model()).verify(context,'fabric city',Path(temp)/'image.png','image',[],retry_available=True,seed=1,output_dir=temp)
        self.assertEqual(len(calls),1)
        critic=calls[0]
        conversation, current = critic[1]['messages'][1]['content'][:2]
        payload=json.loads(current['text'])
        self.assertEqual(payload['latest_rules'],[rule])
        self.assertNotIn('portrait',json.dumps(payload))
        self.assertIn('visible_sequence',json.loads(conversation['text']))
        self.assertNotIn('PLANNER_SECRET',json.dumps(calls,default=str))

    def test_video_critic_receives_only_minimal_grounded_input(self):
        calls=[]
        class Model:
            def __call__(self,**kw):
                calls.append(kw)
                data=report();data['think']=data.pop('thinking')
                return dict(text=json.dumps(data),finish_reason='stop')
        context=dict(current_user_request='Make a video of a red toy car on a table.',
            visible_sequence=[dict(role='user',content='Make a video of a red toy car on a table.',media=[])],
            latest_rules=[],output_position=1,output_count=1)
        with tempfile.TemporaryDirectory() as temp:
            MediaReviewer(Model()).verify(context,'A red toy car driving on a wooden table.',
                Path(temp)/'native.mp4','video',[],retry_available=True,seed=1,output_dir=temp)
        conversation, current = calls[0]['messages'][1]['content'][:2]
        self.assertEqual(set(json.loads(conversation['text'])),
                         {'current_user_request','visible_sequence','current_exchange'})
        payload=json.loads(current['text'])
        self.assertEqual(payload['output_position'],1)
        self.assertEqual(payload['output_count'],1)
        system=calls[0]['messages'][0]['content'].lower()
        self.assertNotIn('sampled',system)
        self.assertNotIn('A red toy car driving on a wooden table.',json.dumps(payload))
        self.assertEqual(len([b for b in calls[0]['messages'][1]['content'] if b['type']=='video']),1)

    def test_actual_reference_images_are_labeled_as_adopted_identity_baselines(self):
        from cocreation import Reference
        from PIL import Image
        calls=[]
        def model(**options):
            calls.append(options)
            data=report();data['think']=data.pop('thinking')
            return dict(text=json.dumps(data),finish_reason='stop')
        context=dict(current_user_request='Show the woman smiling in a static handheld shot.',
            latest_rules=[],output_position=1,output_count=1)
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'actual-adopted.png';Image.new('RGB',(64,64),'blue').save(source)
            MediaReviewer(model).verify(context,'UNNEEDED_AGE_DESCRIPTION',Path(temp)/'result.mp4','video',
                [Reference('person',source,'image')],retry_available=True,seed=1,output_dir=temp)
        content=calls[0]['messages'][1]['content']
        self.assertEqual([block['image'] for block in content if block['type']=='image'],[str(source)])
        self.assertTrue(any('USER-APPROVED REFERENCE' in block.get('text','') for block in content))
        self.assertNotIn('UNNEEDED_AGE_DESCRIPTION',json.dumps(content))
        self.assertIn('Do not\nrejudge the adopted character',calls[0]['messages'][0]['content'])

    def test_uncertain_rule_is_nonblocking_and_clear_rule_failure_is_blocking(self):
        data=report();data['think']=data.pop('thinking')
        data['cross_turn_rules']=1
        data['rule_checks']=[dict(rule_index=1,score=1,evidence='The visual treatment is close with minor uncertainty.')]
        self.assertEqual(parse_review(dict(text=json.dumps(data)),[dict(requirement='Use warm light.')])['decision'],'accepted')
        data['rule_checks'][0]['score']=0
        with self.assertRaises(ValueError):parse_review(dict(text=json.dumps(data)),[dict(requirement='Use warm light.')])

    def test_latest_rules_use_planner_snapshot_without_any_vlm_call(self):
        context=generation_context(dialogue(),'[[image_0001]]')
        context['latest_rules']=[dict(requirement='Use landscape 16:9.')]
        def forbidden(*args,**options):raise AssertionError('No resolver VLM may be called')
        with tempfile.TemporaryDirectory() as temp:
            rules=MediaReviewer(forbidden).latest_rules(context,'image',seed=1,output_dir=temp)
            self.assertEqual(rules,context['latest_rules'])
            self.assertEqual(json.loads((Path(temp)/'resolved.json').read_text())['source'],'planner_media_snapshot')

    def test_context_and_fixer_never_serialize_planner_or_future(self):
        context=generation_context(dialogue(),'[[image_0001]]')
        task=dict(backend='flux',operation='local_edit',kind='image',references=[],
            desired_attributes={'shape':'PLANNER_SECRET'},subject='PLANNER_SECRET',
            sequence_context=context,render_state={},presentation={})
        payload=writer_input(task)+writer_system(task)
        self.assertNotIn('PLANNER_SECRET',payload)
        self.assertNotIn('FUTURE_SECRET',payload)
        self.assertIn('Make the city out of fabric.',payload)
        self.assertNotIn('\"changes\": []',writer_input(task))

    def test_export_order_and_no_accepted_reviewer_reply(self):
        with tempfile.TemporaryDirectory() as temp:
            records={'[[image_0001]]':dict(status='complete',attempts=[
                dict(attempt=1,prompt='Original fabric prompt',kind='image',path='one.png',critique=report(False)),
                dict(attempt=2,prompt='Repaired fabric prompt',kind='image',path='two.png',critique=report())])}
            out=export_sequence(temp,dialogue(),records)
            parts=out['turns'][1]['segments']
            self.assertEqual([p['type'] for p in parts],['text','thinking','media','thinking','text','thinking','media','thinking','text'])
            self.assertEqual(parts[-1]['text'],'Here is your reference image.')
            self.assertIn('<think>',parts[3]['text'])
            text=(Path(temp)/'dialogue.md').read_text()
            self.assertNotIn('PLANNER_SECRET',text)
            self.assertNotIn('unused accepted reply',text)
            self.assertLess(text.index('Original fabric prompt'),text.index('one.png'))
            self.assertLess(text.index('one.png'),text.index('I inspect'))

    def test_fixer_rejects_an_unchanged_failed_prompt_and_keeps_context_clean(self):
        context=generation_context(dialogue(),'[[image_0001]]')
        task=dict(backend='flux',kind='image',operation='local_edit',references=[dict(label='<Image 1>',kind='image',placeholder='[[image_0000]]')],desired_attributes=dict(material='PLANNER_SECRET'),render_state={})
        previous='Edit <Image 1>. Change the city to fabric. Preserve all other details.'
        replacement='Edit <Image 1>. Replace the wood grain with visible fabric weave and textile surfaces. Preserve the city silhouette and all other details.'
        calls=[]
        class Model:
            def __call__(self,**options):
                calls.append((options['messages'],options))
                return dict(text=json.dumps(dict(edits=[dict(before=previous,after=previous if len(calls)==1 else replacement,addresses='The city still looks wooden.',visible_success='Visible textile weave replaces the wood grain.')])),finish_reason='stop')
        with tempfile.TemporaryDirectory() as temp:
            fixed=MediaReviewer(Model()).fix(task,context,previous,'The city still looks like wood.',seed=1,output_dir=temp)
        self.assertEqual(fixed,replacement)
        self.assertEqual(len(calls),2)
        self.assertNotIn('PLANNER_SECRET',str(calls))
        first_call = calls[0][0]
        self.assertIn('Output format:',first_call[0]['content'])
        payload = json.loads(first_call[1]['content'])
        self.assertEqual(list(payload)[-2:],['failed_prompt','critique_reason'])

    def test_partial_candidate_delivery_keeps_failed_critique(self):
        data=dialogue()
        data['turns'][1]['output_media']=['[[image_0001]]','[[image_0002]]']
        context=generation_context(data,'[[image_0002]]')
        self.assertEqual((context['output_position'],context['output_count']),(2,2))
        with tempfile.TemporaryDirectory() as temp:
            records={'[[image_0001]]':dict(status='failed',attempts=[dict(attempt=1,prompt='fabric',kind='image',path='one.png',critique=report(False))])}
            result=export_sequence(temp,data,records,status='failed')
            self.assertIn('one.png',json.dumps(result))
            self.assertNotIn('Here is your reference image.',json.dumps(result))

    def test_references_are_labeled_and_result_is_last_with_measured_geometry(self):
        from PIL import Image
        from cocreation import Reference
        calls=[]
        def model(**kw):
            calls.append(kw)
            data=report();data['think']=data.pop('thinking')
            return dict(text=json.dumps(data),finish_reason='stop')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source.png';result=root/'result.png'
            Image.new('RGB',(96,64),'brown').save(source)
            Image.new('RGB',(114,64),'white').save(result)
            MediaReviewer(model).verify(generation_context(dialogue(),'[[image_0001]]'),
                'Use only the person from <Picture 1>.',result,'image',
                [Reference('source',source,'image')],retry_available=True,seed=1,output_dir=root/'review')
            content=calls[0]['messages'][1]['content']
            self.assertIn('REFERENCE <Picture 1>',content[1]['text'])
            self.assertEqual(content[2]['image'],str(source.resolve()))
            self.assertIn('GENERATED RESULT',content[-2]['text'])
            self.assertEqual(content[-1]['image'],str(result.resolve()))
            measured=json.loads(content[-3]['text'])
            self.assertEqual(measured['Verified canvas'],'16:9')
            self.assertNotIn('media_geometry',measured)
            audit=json.loads((root/'review/measured_geometry.json').read_text())
            self.assertEqual((audit['width'],audit['height']),(114,64))

    def test_real_wire_call_disables_native_thinking_and_keeps_only_full_result(self):
        calls=[]
        class Model:
            def __call__(self,*args,**kw):
                calls.append(kw)
                data=report(); data['think']=data.pop('thinking')
                return dict(text=json.dumps(data),finish_reason='stop')
        with tempfile.TemporaryDirectory() as temp:
            from PIL import Image
            media=Path(temp)/'result.png';Image.new('RGB',(64,96),'blue').save(media)
            result=MediaReviewer(Model()).verify(generation_context(dialogue(),'[[image_0001]]'),'fabric city',
                media,'image',[],retry_available=True,seed=1,output_dir=temp)
            images=[b['image'] for b in calls[0]['messages'][1]['content'] if b['type']=='image']
            self.assertEqual(images,[str(media.resolve())])
            self.assertNotIn('Detail crop',json.dumps(calls[0]['messages']))
            self.assertFalse(list(Path(temp).glob('result-detail-*')))
            self.assertFalse(calls[0]['enable_thinking'])
            self.assertNotIn('thinking_token_budget',calls[0])
            self.assertIn('think',calls[0]['response_schema']['required'])
            self.assertEqual(result['think'],result['thinking'])
            self.assertIn('step-by-step',calls[0]['messages'][0]['content'])
            self.assertTrue(result['thinking'])
            self.assertTrue((Path(temp)/'report.json').exists())


class ReviewLoopTests(unittest.TestCase):
    def run_loop(self,decisions,shared=False):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name)
        (root/'dialogue.md').write_text('Assistant: [[image_0001]]')
        (root/'dialogue.json').write_text(json.dumps(dialogue()))
        task=dict(placeholder='[[image_0001]]',kind='image',operation='text_generation',backend='flux',
             prompt='fabric city',dependencies=[],references=[],render_state={})
        task['reasoner']=dict(think='I specify fabric surfaces for the city.',generation_prompt='fabric city',latest_rules=[])
        task['latest_rules']=[]
        (root/'placeholders.json').write_text(json.dumps([task]))
        events=[];calls=[]
        class Generator:
            def __call__(self,prompt,refs,**options):
                from PIL import Image
                calls.append(prompt);events.append('generate')
                path=root/f'fake-{len(calls)}.png';Image.new('RGB',(64,64),'red').save(path)
                return dict(status='ok',path=str(path))
            def close(self):events.append('generator_closed')
        class Reviewer:
            def verify(self,context,prompt,media,kind,refs,**options):
                self_number=len(calls)
                assert events[-1]==('generate' if shared else 'generator_closed')
                events.append('critique')
                self_result=report(decisions[self_number-1])
                self_result['policy_version']=REVIEW_POLICY_VERSION
                if self_number==3 and not decisions[-1]:self_result['reply']='I could not complete this request.'
                return self_result
            def fix(self,task,context,prompt,reason,**options):
                events.append('fix')
                assert 'PLANNER_SECRET' not in json.dumps(context)
                return prompt+' repaired'
        @contextmanager
        def reviewer_factory(directory):
            yield Reviewer()
        options=dict(model_factory=lambda *args:Generator(),reviewer_factory=reviewer_factory,
                     verify_media=True,shared_vlm=shared,progress=False)
        if decisions[-1]:
            result=materialize(root,**options)
            self.assertEqual(result['status'],'complete')
        else:
            manifest = materialize(root, **options)
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(manifest['pruning']['removed_turn_ids'], [1, 2, 3])
            self.assertEqual(set(manifest['pruning']['exhausted']), {'[[image_0001]]'})
            self.assertFalse((root/'media/image_0001.png').exists())
            text=(root/'dialogue.md').read_text()
            self.assertNotIn('Here is your reference image.',text)
            self.assertNotIn('FUTURE_SECRET',text)
        if not decisions[-1]:
            before=(root/'generation_attempts/image_0001/attempt-03/output.png').read_bytes()
            resumed = materialize(root, resume=True, **options)
            self.assertEqual(resumed['pruning'], manifest['pruning'])
            self.assertEqual((root/'generation_attempts/image_0001/attempt-03/output.png').read_bytes(),before)
        else:
            materialize(root,resume=True,**options)
        self.assertEqual(len(calls),len(decisions))
        self.assertEqual(events.count('generator_closed'),1 if shared else len(decisions))
        for i in range(1,len(decisions)+1):
            self.assertTrue((root/f'generation_attempts/image_0001/attempt-{i:02d}/output.png').exists())
        official=json.loads((root/'sequence.json').read_text())
        if decisions[-1]:
            segments = official['turns'][1]['segments']
            self.assertEqual(segments[1]['source'], 'prompt_reasoner')
            call = next(segment for segment in segments if segment['type'] == 'tool_call')
            self.assertEqual(call['tool_calls'][0]['arguments']['prompt'], 'fabric city')
            self.assertEqual(sum(segment.get('source') == 'media_critique' for segment in segments), len(decisions))
        else:
            self.assertEqual(official['turns'], [])
            attempts = resumed['media']['[[image_0001]]']['attempts']
            self.assertEqual([attempt['critique']['decision'] for attempt in attempts], ['rejected'] * 3)
        return root,options

    def test_shared_vlm_keeps_generator_across_two_repairs(self):self.run_loop([False,False,True],shared=True)
    def test_shared_vlm_third_rejection_preserves_trace_and_closes_generator(self):self.run_loop([False,False,False],shared=True)

    def test_one_generation_always_reviewed(self):self.run_loop([True])
    def test_two_repairs_and_third_review(self):self.run_loop([False,False,True])
    def test_third_rejection_prunes_delivery_preserving_every_attempt(self):self.run_loop([False,False,False])

    def test_old_rejections_resume_by_reviewing_existing_media_without_generation(self):
        root,options=self.run_loop([False,False,False])
        manifest=json.loads((root/'media_manifest.json').read_text())
        record=manifest['media']['[[image_0001]]']
        originals=[(root/attempt['path']).read_bytes() for attempt in record['attempts']]
        for attempt in record['attempts']:
            attempt['critique'].pop('policy_version')
        (root/'media_manifest.json').write_text(json.dumps(manifest))
        reviewed=[]
        class Reviewer:
            def verify(self,context,prompt,media,kind,refs,**options):
                reviewed.append(Path(media))
                return dict(report(),policy_version=REVIEW_POLICY_VERSION)
            def fix(self,*args,**options):
                raise AssertionError('An accepted existing output needs no repair.')
        @contextmanager
        def reviewer_factory(directory):
            yield Reviewer()
        def forbid_generation(*args,**options):
            raise AssertionError('Existing media must be reviewed before any regeneration.')
        options.update(model_factory=forbid_generation,reviewer_factory=reviewer_factory)
        self.assertEqual(materialize(root,resume=True,**options)['status'],'complete')
        record=json.loads((root/'media_manifest.json').read_text())['media']['[[image_0001]]']
        self.assertEqual(len(reviewed),1)
        self.assertEqual(len(record['attempts']),1)
        self.assertEqual(len(record['superseded_attempts']),2)
        self.assertEqual(record['attempts'][0]['critique_history'][0]['decision'],'rejected')
        self.assertNotIn('next_prompt',record['attempts'][0])
        self.assertEqual((root/'media/image_0001.png').read_bytes(),originals[0])
        for attempt,original in zip(record['attempts']+record['superseded_attempts'],originals):
            self.assertEqual((root/attempt['path']).read_bytes(),original)
        official=json.loads((root/'sequence.json').read_text())
        segments=official['turns'][1]['segments']
        self.assertEqual(sum(s.get('source')=='media_critique' for s in segments),1)
        self.assertIn('Here is your reference image.',(root/'dialogue.md').read_text())

if __name__=='__main__':unittest.main()
