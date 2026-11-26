from contextlib import closing, contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cocreation import Reference
from tools.media_review import MediaReviewer
from tools.turn_media_review import CRITERIA,parse
from scripts.dialogue_generation.materialize import materialize,materialization_order,materialization_session
from scripts.dialogue_generation.media_queue import ReviewRequest

REQUEST='Two reference images of the polished base to compare: Option 1 is dark beige, and Option 2 is dark brown; both polished, but Option 2 must be 16:9 landscape.'

def raw_report(outputs,failed=()):
    positioned = [(row.get('output_position', index), row) for index, row in enumerate(outputs, 1)]
    return dict(think=dict(
        request_check=("I inspect the entire delivery together and match the numbered files to the user's requested alternatives. "
                       "The first output supplies the first variant and the second output supplies the second variant. "
                       "I compare their respective colors, shapes and finishes with the complete request. "
                       "I treat them as separate media files delivered in one turn, so their comparison does not require a collage. "
                       "I also check the measured format for each output independently."),
        quality_check=("I compare each output with its own adopted reference inputs where these are present. "
                       "I check that the retained subject identity and design remain consistent while the requested variant is visible. "
                       "I inspect the surfaces and framing for coherent, usable content. "
                       "Minor natural variation does not require regeneration. "
                       "I assess the complete set against the request rather than expecting one file to contain every alternative."),
        decision=("I will retry only the materially incorrect output while retaining the suitable alternative." if failed else
                  "I accept this complete delivery because the separate alternatives satisfy the request together.")),
        output_checks=[dict(output_position=position,**{key:0 if position in failed and key=='request_fidelity' else 2 for key in CRITERIA},
            rule_checks=[dict(rule_index=index,score=2,evidence='I verify this output against its measured canvas and visible content.')
                         for index,_ in enumerate(output.get('latest_rules',[]),1)],
            reason='The second output has the wrong requested color.' if position in failed else '')
            for position,output in positioned],
        reply="I see the second output has the wrong requested color. I will retry that output and retain the suitable first alternative." if failed else '')

def report(outputs,failed=(), *, active_position=None):
    scored = outputs[-1:] if active_position is not None else outputs
    return parse(dict(text=json.dumps(raw_report(scored, failed))), outputs, active_position=active_position)


class TurnReviewContract(unittest.TestCase):
    def test_full_request_all_images_and_real_reference_inputs_share_one_call(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);calls=[]
            for name,color in [('first','beige'),('second','brown'),('reference','white')]:
                Image.new('RGB',(64,64),color).save(root/(name+'.png'))
            outputs=[dict(placeholder='[[image_0001]]',kind='image',media=root/'first.png',latest_rules=[],retry_available=True,references=[]),
                     dict(placeholder='[[image_0002]]',kind='image',media=root/'second.png',latest_rules=[],retry_available=True,
                          references=[Reference('adopted',root/'reference.png','image')])]
            def model(**options):
                calls.append(options);return dict(text=json.dumps(raw_report(outputs)),finish_reason='stop')
            result=MediaReviewer(model).verify_turn(dict(current_user_request=REQUEST),outputs,seed=1,output_dir=root/'review')
            self.assertEqual(result['decision'],'accepted');self.assertEqual(len(calls),1)
            content=calls[0]['messages'][1]['content']
            self.assertEqual(json.loads(content[0]['text'])['current_user_request'],REQUEST)
            self.assertEqual([block['image'] for block in content if block['type']=='image'],
                             [str(root/'reference.png'),str(root/'first.png'),str(root/'second.png')])
            self.assertIn('USER-APPROVED REFERENCE',content[1]['text'])
            self.assertNotIn('generation_prompt',json.dumps(content))

    def test_multiple_videos_and_actual_source_video_are_native_video_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);calls=[]
            for name in ['first.mp4','second.mp4','source.mp4']:(root/name).touch()
            outputs=[dict(placeholder='[[video_0001]]',kind='video',media=root/'first.mp4',latest_rules=[],retry_available=True,
                          references=[Reference('adopted',root/'source.mp4','video')]),
                     dict(placeholder='[[video_0002]]',kind='video',media=root/'second.mp4',latest_rules=[],retry_available=True,references=[])]
            def model(**options):calls.append(options);return dict(text=json.dumps(raw_report(outputs)))
            with patch('tools.turn_media_review.media_geometry',return_value=dict(available=True,nominal_aspect_ratio='16:9')):
                MediaReviewer(model).verify_turn(dict(current_user_request='Two independent video clips.'),outputs,seed=1,output_dir=root/'review')
            content=calls[0]['messages'][1]['content']
            self.assertEqual([block['video'] for block in content if block['type']=='video'],
                             [str(root/'source.mp4'),str(root/'first.mp4'),str(root/'second.mp4')])
            self.assertFalse(any(block['type']=='image' for block in content))

    def test_staged_review_sends_actual_prompt_result_and_prior_critique_in_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls = []
            first = dict(placeholder='[[video_0001]]', output_position=1, kind='video',
                         media=root/'first.mp4', latest_rules=[], retry_available=True,
                         generation_prompt='Actual first video prompt')
            second = dict(placeholder='[[video_0002]]', output_position=2, kind='video',
                          media=root/'second.mp4', latest_rules=[], retry_available=True,
                          generation_prompt='Actual second video prompt')
            def model(**options):
                calls.append(options)
                return dict(text=json.dumps(raw_report([first if len(calls) == 1 else second])))

            with patch('tools.turn_media_review.media_geometry', return_value=dict(available=True, aspect_ratio='16:9')):
                reviewer = MediaReviewer(model)
                context = dict(current_user_request='Compare two videos.')
                report1 = reviewer.verify_turn(dict(context, delivery_stage=dict(current_output_position=1,
                    step=1, output_count=2, final_stage=False)), [first], seed=1, output_dir=root/'review1')
                earlier = dict(first, accepted_critique=report1['thinking'], retry_available=False)
                report2 = reviewer.verify_turn(dict(context, delivery_stage=dict(current_output_position=2,
                    step=2, output_count=2, final_stage=True)), [earlier, second], seed=2, output_dir=root/'review2')

            stage1, stage2 = (call['messages'][1]['content'] for call in calls)
            self.assertEqual(len([block for block in stage1 if block['type']=='video']), 1)
            self.assertFalse(json.loads(stage1[-1]['text'])['delivery_stage']['final_stage'])
            self.assertEqual(stage1[:-1], stage2[:len(stage1)-1])
            self.assertIn('Their absence is expected', calls[0]['messages'][0]['content'])
            self.assertTrue(json.loads(stage2[-1]['text'])['delivery_stage']['final_stage'])
            trace = [block.get('video', block.get('text', '')) for block in stage2]
            expected = ('Actual first video prompt', str(first['media']), 'ACCEPTED CRITIQUE FOR OUTPUT 1:',
                        'Actual second video prompt', str(second['media']))
            indices = [next(i for i, text in enumerate(trace) if piece in text) for piece in expected]
            self.assertEqual(indices, sorted(set(indices)))
            self.assertEqual([row['output_position'] for row in report2['output_checks']], [2])
            with self.assertRaisesRegex(ValueError, 'only the current output'):
                parse(dict(text=json.dumps(raw_report([first, second]))), [earlier, second], active_position=2)

    def test_scores_target_only_failed_candidate_and_keep_rules_per_output(self):
        outputs=[dict(placeholder='[[image_0001]]',latest_rules=[dict(requirement='Use portrait 9:16.')]),
                 dict(placeholder='[[image_0002]]',latest_rules=[dict(requirement='Use landscape 16:9.')])]
        result=report(outputs,[2])
        self.assertEqual(result['retry_positions'],[2])
        self.assertEqual([row['decision'] for row in result['output_checks']],['accepted','rejected'])
        self.assertIn('Output 1 ([[image_0001]]): Use portrait 9:16.',result['thinking'])
        self.assertIn('Output 2 ([[image_0002]]): Use landscape 16:9.',result['thinking'])
        data=raw_report(outputs);data['output_checks'][1]['request_fidelity']=1
        self.assertEqual(parse(dict(text=json.dumps(data)),outputs)['decision'],'accepted')
        data['output_checks'].pop()
        with self.assertRaisesRegex(ValueError,'every output'):parse(dict(text=json.dumps(data)),outputs)

    def test_delivery_is_scheduled_before_any_consumer_of_either_candidate(self):
        tasks=dict(first=dict(operation='text_generation',backend='flux',dependencies=[]),
                   consumer=dict(operation='reference_generation',backend='minimax_h3',dependencies=['first']),
                   second=dict(operation='reference_generation',backend='minimax_h3',dependencies=[]))
        order=materialization_order(tasks,groups=[['first','second']])
        self.assertLess(order.index('second'),order.index('consumer'))
        self.assertEqual(order[:2],['first','second'])


class TurnMaterialization(unittest.TestCase):
    def setup_run(self,decisions):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);root=Path(temporary.name)
        tokens=['[[image_0001]]','[[image_0002]]']
        dialogue=dict(turns=[dict(id=1,role='user',content=REQUEST,output_media=[]),
            dict(id=2,role='assistant',content='Here are both options. '+ ' '.join(tokens),output_media=tokens,
                 generation_opening="I'll generate both requested alternatives."),
            dict(id=3,role='user',content='FUTURE_SECRET',output_media=[])])
        (root/'dialogue.json').write_text(json.dumps(dialogue));(root/'dialogue.md').write_text('Assistant: '+' '.join(tokens))
        tasks=[dict(placeholder=token,kind='image',operation='text_generation',backend='flux',
                    prompt=prompt,dependencies=[],references=[],render_state={},latest_rules=[],
                    reasoner=dict(think='I plan the requested variant from the user request.',generation_prompt=prompt,latest_rules=[]))
               for token,prompt in zip(tokens,['beige base','brown base'])]
        (root/'placeholders.json').write_text(json.dumps(tasks));events=[];calls=[];views=[];fixes=[]
        class Generator:
            def __call__(self,prompt,references,**options):
                from PIL import Image
                calls.append(prompt);events.append('generate')
                path=root/f'fake-{len(calls)}.png';Image.new('RGB',(64,64),'brown' if 'brown' in prompt else 'beige').save(path)
                return dict(status='ok',path=str(path))
            def close(self):events.append('close')
        class Reviewer:
            def verify(self,*args,**options):raise AssertionError('Expected staged turn context.')
            def verify_turn(self,context,outputs,**options):
                self_decision=decisions[len(views)];views.append(deepcopy(outputs));events.append('review')
                assert context['current_user_request']==REQUEST
                stage=context['delivery_stage']
                assert stage['step']==len(outputs) and all(Path(row['media']).is_file() for row in outputs)
                assert stage['final_stage']==(len(outputs)==2)
                assert all('accepted_critique' in row for row in outputs[:-1])
                active=stage['current_output_position']
                if self_decision=='technical':raise RuntimeError('Fixture technical review failure')
                return report(outputs,[] if self_decision else [active], active_position=active)
            def fix(self,task,context,prompt,reason,**options):
                fixes.append(context['output_position']);return prompt+' repaired'
        @contextmanager
        def session(directory):yield Reviewer()
        options=dict(verify_media=True,shared_vlm=True,progress=False,model_factory=lambda *args:Generator(),reviewer_factory=session)
        return root,options,calls,views,fixes,events

    def test_each_asset_is_accepted_before_the_next_generation_and_export(self):
        root,options,calls,views,fixes,events=self.setup_run([True,True])
        materialize(root,**options)
        self.assertEqual([event for event in events if event!='close'],['generate','review','generate','review'])
        self.assertEqual(len(calls),2);self.assertEqual([len(outputs) for outputs in views],[1,2])
        segments=json.loads((root/'sequence.json').read_text())['turns'][1]['segments']
        media=[index for index,row in enumerate(segments) if row['type']=='media']
        reflection=[index for index,row in enumerate(segments) if row.get('source')=='media_critique']
        self.assertEqual(len(reflection),2)
        self.assertEqual(sorted(media+reflection),[media[0],reflection[0],media[1],reflection[1]])
        materialize(root,resume=True,**options)
        self.assertEqual(len(calls),2);self.assertEqual(len(views),2)

    def test_repair_only_the_second_output_with_first_acceptance_in_context(self):
        root,options,calls,views,fixes,events=self.setup_run([True,False,True])
        materialize(root,**options)
        self.assertEqual(calls,['beige base','brown base','brown base repaired'])
        self.assertEqual(fixes,[2]);self.assertEqual([len(outputs) for outputs in views],[1,2,2])
        self.assertEqual(views[0][0]['media'],views[2][0]['media'])
        self.assertEqual(views[1][0]['accepted_critique'],views[2][0]['accepted_critique'])
        self.assertNotEqual(views[1][1]['media'],views[2][1]['media'])
        segments=json.loads((root/'sequence.json').read_text())['turns'][1]['segments']
        self.assertEqual(sum(row.get('source')=='media_critique' for row in segments),3)
        self.assertEqual(sum(row['type']=='tool_call' for row in segments),3)

    def test_resume_after_second_review_technical_failure_reuses_both_files_and_first_review(self):
        root,options,calls,views,fixes,events=self.setup_run([True,'technical',True])
        with self.assertRaisesRegex(RuntimeError,'technical'):materialize(root,**options)
        materialize(root,resume=True,**options)
        self.assertEqual(len(calls),2)
        self.assertEqual([len(outputs) for outputs in views],[1,2,2])
        self.assertEqual(views[1][1]['media'],views[2][1]['media'])
        self.assertTrue((root/'media/image_0002.png').is_file())

    def test_three_rejections_of_first_asset_never_generate_second_or_repeat_on_resume(self):
        root,options,calls,views,fixes,events=self.setup_run([False,False,False])
        manifest = materialize(root, **options)
        self.assertEqual(manifest['status'], 'failed')
        self.assertEqual(manifest['pruning']['omitted_media'], ['[[image_0002]]'])
        self.assertEqual(manifest['pruning']['removed_turn_ids'], [1, 2, 3])
        self.assertEqual(calls,['beige base','beige base repaired','beige base repaired repaired'])
        self.assertEqual(fixes,[1,1]);self.assertEqual([len(outputs) for outputs in views],[1,1,1])
        self.assertFalse((root/'media/image_0001.png').exists())
        resumed = materialize(root, resume=True, **options)
        self.assertEqual(resumed['pruning'], manifest['pruning'])
        self.assertEqual(len(calls),3);self.assertEqual(len(views),3)

    def test_exhausted_bundle_prunes_dependents_but_later_independent_delivery_finishes(self):
        root, options, calls, _, _, _ = self.setup_run([True, False, False, False])
        first, failed, dependent, independent = [f'[[image_{i:04d}]]' for i in range(1, 5)]
        dialogue_file = root / 'dialogue.json'
        dialogue = json.loads(dialogue_file.read_text())
        dialogue['turns'][2:] = [
            dict(id=3, role='user', content='Edit the earlier first variant.', output_media=[],
                 intent=dict(context_media=[first])),
            dict(id=4, role='assistant', content='Here is the revised variant.', output_media=[dependent]),
            dict(id=5, role='user', content='Make one independent beige image.', output_media=[]),
            dict(id=6, role='assistant', content='Here is the beige image.', output_media=[independent]),
        ]
        dialogue_file.write_text(json.dumps(dialogue))
        original = root / 'dialogue.md'
        original.write_text(original.read_text() + f'\n{dependent}\n{independent}')
        plan_file = root / 'placeholders.json'
        tasks = json.loads(plan_file.read_text())
        tasks.append(dict(tasks[0], placeholder=dependent, backend=None, operation='copy', dependencies=[first],
                          prompt=None, reasoner=None))
        tasks.append(dict(tasks[0], placeholder=independent, prompt='independent beige', reasoner=None))
        plan_file.write_text(json.dumps(tasks))

        factory = options['reviewer_factory']

        @contextmanager
        def review_session(directory):
            with factory(directory) as reviewer:
                def verify(context, *args, **kwargs):
                    self.assertEqual(context['current_user_request'], 'Make one independent beige image.')
                    return report([dict(placeholder=independent)])
                reviewer.verify = verify
                yield reviewer

        options['reviewer_factory'] = review_session
        with closing(materialization_session(root, defer_retries=True, async_reviews=True, **options)) as runtime:
            next(runtime)
            self.assertEqual(runtime.send(runtime.send(first).run()).status, 'accepted')
            request = runtime.send(failed)
            for _ in range(2):
                self.assertEqual(runtime.send(request.run()).status, 'rejected')
                request = runtime.send(runtime.send(failed).run())
            event = runtime.send(request.run())
            self.assertEqual(event.status, 'exhausted')
            self.assertCountEqual(event.discarded, [first, failed, dependent])
            with self.assertRaises(StopIteration) as end:
                runtime.send(runtime.send(independent).run())
        manifest, _ = end.exception.value
        self.assertEqual(manifest['status'], 'partial')
        self.assertEqual(manifest['completed'], 1)
        self.assertEqual(manifest['pruning']['removed_turn_ids'], [1, 2, 3, 4])
        self.assertEqual(calls[-1], 'independent beige')
        self.assertEqual(manifest['media'][dependent]['status'], 'omitted')
        self.assertFalse((root / 'media/image_0003.png').exists())
        sequence = json.loads((root / 'sequence.json').read_text())
        self.assertEqual(sequence['status'], 'partial')
        self.assertEqual([turn['id'] for turn in sequence['turns']], [5, 6])
        self.assertEqual([row['placeholder'] for turn in sequence['turns'] for row in turn['segments']
                          if row['type'] == 'media'], [independent])

    def test_resume_interrupted_fixer_repairs_before_repeating_the_group_review(self):
        root,options,calls,views,fixes,events=self.setup_run([False,True,True])
        original=options['reviewer_factory'];interrupted=[]
        @contextmanager
        def session(directory):
            with original(directory) as reviewer:
                fix=reviewer.fix
                def interrupt_once(*args,**kwargs):
                    if not interrupted:
                        interrupted.append(True);raise RuntimeError('Interrupted fixer fixture')
                    return fix(*args,**kwargs)
                reviewer.fix=interrupt_once
                yield reviewer
        options['reviewer_factory']=session
        with self.assertRaisesRegex(RuntimeError,'Interrupted fixer'):materialize(root,**options)
        materialize(root,resume=True,**options)
        self.assertEqual(calls,['beige base','beige base repaired','brown base'])
        self.assertEqual([len(outputs) for outputs in views],[1,1,2]);self.assertEqual(fixes,[1])

    def test_native_dispatch_can_pause_a_delivery_and_run_an_unrelated_asset(self):
        root, options, calls, views, fixes, events = self.setup_run([False, True, True])
        first, second, independent = ('[[image_0001]]', '[[image_0002]]', '[[image_0003]]')
        # The single generation has its own dialogue turn and review.
        dialogue_file = root/'dialogue.json'
        dialogue = json.loads(dialogue_file.read_text())
        dialogue['turns'].append(dict(id=4, role='assistant', content='Here is the other image. '+independent,
                                      output_media=[independent]))
        dialogue_file.write_text(json.dumps(dialogue))
        plan_file = root/'placeholders.json'
        tasks = json.loads(plan_file.read_text())
        tasks.append(dict(tasks[0], placeholder=independent, prompt='independent beige', reasoner=None))
        plan_file.write_text(json.dumps(tasks))
        dialogue_markdown = root/'dialogue.md'
        dialogue_markdown.write_text(dialogue_markdown.read_text()+'\nAssistant: '+independent)
        original = options['reviewer_factory']
        @contextmanager
        def session(directory):
            with original(directory) as reviewer:
                reviewer.verify = lambda *args, **kwargs: report([dict(placeholder=independent)])
                yield reviewer
        options['reviewer_factory'] = session
        with closing(materialization_session(root, defer_retries=True, **options)) as runtime:
            plan = next(runtime)
            self.assertEqual(plan[second]['dependencies'], [first])
            self.assertEqual(runtime.send(first).status, 'rejected')
            self.assertEqual(calls, ['beige base'])
            self.assertEqual(runtime.send(independent).token, independent)
            self.assertEqual(runtime.send(first).status, 'accepted')
            with self.assertRaises(StopIteration) as end:
                runtime.send(second)
            manifest, event = end.exception.value
        self.assertEqual(event.token, second)
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(calls, ['beige base', 'independent beige', 'beige base repaired', 'brown base'])

    def test_native_dispatch_defers_retry_for_a_single_output_turn(self):
        root, options, calls, _, fixes, _ = self.setup_run([])
        first, second = ('[[image_0001]]', '[[image_0002]]')
        dialogue_file = root/'dialogue.json'
        dialogue = json.loads(dialogue_file.read_text())
        dialogue['turns'][1].update(output_media=[first], content='Here is the first image. '+first)
        dialogue['turns'].append(dict(id=4, role='assistant', content='Here is the second image. '+second,
                                      output_media=[second]))
        dialogue_file.write_text(json.dumps(dialogue))

        original = options['reviewer_factory']
        @contextmanager
        def session(directory):
            with original(directory) as reviewer:
                reviewer.verify = lambda context, prompt, *args, **kwargs: report(
                    [dict(placeholder=context['output_placeholder'])], [1] if prompt=='beige base' else [])
                yield reviewer
        options['reviewer_factory'] = session
        with closing(materialization_session(root, defer_retries=True, **options)) as runtime:
            next(runtime)
            self.assertEqual(runtime.send(first).status, 'rejected')
            self.assertFalse(fixes)
            self.assertEqual(runtime.send(second).token, second)
            with self.assertRaises(StopIteration) as end:
                runtime.send(first)
            manifest, event = end.exception.value
        self.assertEqual((event.token, event.status), (first, 'accepted'))
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(calls, ['beige base', 'brown base', 'beige base repaired'])

    def test_native_async_review_resumes_the_same_delivery_and_repair(self):
        root, options, calls, views, fixes, _ = self.setup_run([False, True, True])
        first, second = ('[[image_0001]]', '[[image_0002]]')
        with closing(materialization_session(root, defer_retries=True, async_reviews=True, **options)) as runtime:
            next(runtime)
            review = runtime.send(first)
            self.assertIsInstance(review, ReviewRequest)
            self.assertEqual(views, [])
            self.assertEqual(runtime.send(review.run()).status, 'rejected')

            fix = runtime.send(first)
            self.assertIsInstance(fix, ReviewRequest)
            self.assertEqual(calls, ['beige base'])
            review = runtime.send(fix.run())
            self.assertIsInstance(review, ReviewRequest)
            self.assertEqual(runtime.send(review.run()).status, 'accepted')

            review = runtime.send(second)
            self.assertIsInstance(review, ReviewRequest)
            with self.assertRaises(StopIteration) as finished:
                runtime.send(review.run())
            manifest, event = finished.exception.value

        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(event.token, second)
        self.assertEqual(calls, ['beige base', 'beige base repaired', 'brown base'])
        self.assertEqual(fixes, [1])
        self.assertEqual([len(outputs) for outputs in views], [1, 1, 2])

    def test_retry_versions_bind_exact_previous_bytes_and_later_call_uses_stable_alias(self):
        from PIL import Image
        root, options, *_ = self.setup_run([])
        base, later = '[[image_0001]]', '[[image_0002]]'
        version_1, version_2 = '[[image_0001_01]]', '[[image_0001_02]]'
        dialogue_file = root/'dialogue.json'
        dialogue = json.loads(dialogue_file.read_text())
        dialogue['turns'][1].update(output_media=[base], content='Here is the first image. '+base)
        dialogue['turns'].append(dict(id=4, role='assistant', content='Here is the next image. '+later,
                                      output_media=[later]))
        dialogue_file.write_text(json.dumps(dialogue))
        plan_file = root/'placeholders.json'
        tasks = json.loads(plan_file.read_text())
        tasks[1].update(operation='local_edit', dependencies=[base],
                        references=[dict(kind='image', label='<Image 1>', placeholder=base)])
        plan_file.write_text(json.dumps(tasks))
        generated, inputs, fixer_versions = [], [], []

        class Generator:
            def __call__(self, prompt, refs, **kwargs):
                inputs.append([Path(ref.path).read_bytes() for ref in refs])
                index = len(generated)
                path = root/f'generated-{index}.png'
                Image.new('RGB', (64, 64), ('red', 'blue', 'green', 'yellow')[index]).save(path)
                generated.append(path.read_bytes())
                return dict(status='ok', path=str(path))

            def close(self):
                pass

        class Reviewer:
            def verify(self, context, prompt, *args, **kwargs):
                return report([dict(placeholder=context['output_placeholder'])],
                              [1] if len(generated) < 3 else [])

            def fix(self, task, context, prompt, reason, *, media, **kwargs):
                source = task['references'][-1]
                fixer_versions.append(source['placeholder'])
                assert media.read_bytes() == generated[-1]
                return prompt + ' repaired'

        @contextmanager
        def session(directory):
            yield Reviewer()

        options.update(model_factory=lambda *args: Generator(), reviewer_factory=session)
        manifest = materialize(root, **options)
        self.assertEqual(fixer_versions, [base, version_1])
        self.assertEqual(inputs, [[], [generated[0]], [generated[1]], [generated[2]]])
        self.assertEqual(manifest['media'][base]['accepted_placeholder'], version_2)
        sequence = json.loads((root/'sequence.json').read_text())
        self.assertEqual(sequence['media_aliases'][base],
                         dict(placeholder=version_2, path='media/image_0001.png', after_turn=2))
        segments = sequence['turns'][1]['segments']
        self.assertEqual([s['placeholder'] for s in segments if s['type']=='media'], [base, version_1, version_2])
        self.assertEqual([s['tool_calls'][0]['arguments']['references'][-1]['placeholder']
                          for s in segments if s['type']=='tool_call' and s['tool_calls'][0]['arguments']['references']],
                         [base, version_1])
        later_call = next(s for s in sequence['turns'][3]['segments'] if s['type']=='tool_call')
        self.assertEqual(later_call['tool_calls'][0]['arguments']['references'][0]['placeholder'], base)

    def test_mixed_image_video_can_use_an_image_generated_in_the_same_delivery(self):
        import subprocess
        from PIL import Image
        root,options,_,_,_,_=self.setup_run([])
        tokens=['[[image_0001]]','[[video_0001]]']
        data=json.loads((root/'dialogue.json').read_text())
        data['turns'][0]['content']='Make one beige base reference image and one video of its brown variant using that reference.'
        data['turns'][1].update(output_media=tokens,content='Here are the reference image and video. '+' '.join(tokens))
        (root/'dialogue.json').write_text(json.dumps(data));(root/'dialogue.md').write_text('Assistant: '+' '.join(tokens))
        tasks=json.loads((root/'placeholders.json').read_text());video=tasks[1]
        video.update(placeholder=tokens[1],kind='video',operation='reference_generation',backend='vdn_h3',
                     prompt='Animate the brown base from <Picture 1>.',dependencies=[tokens[0]],
                     references=[dict(kind='image',placeholder=tokens[0],label='<Picture 1>')])
        video['reasoner']['generation_prompt']=video['prompt']
        (root/'placeholders.json').write_text(json.dumps(tasks))
        source=root/'generator.mp4'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','color=c=brown:s=256x256:r=8',
                        '-t','1.625','-c:v','libx264','-pix_fmt','yuv420p',str(source)],check=True)
        events=[]
        class Generator:
            def __init__(self,backend):self.backend=backend
            def __call__(self,prompt,refs,**kwargs):
                events.append(self.backend)
                if self.backend=='flux':
                    path=root/'generator.png';Image.new('RGB',(64,64),'red' if 'repaired' in prompt else 'beige').save(path)
                else:
                    assert refs and Path(refs[0].path).is_file()
                    with Image.open(refs[0].path) as accepted:
                        assert accepted.getpixel((0, 0)) == (255, 0, 0)
                    path=source
                return dict(status='ok',path=str(path))
            def close(self):pass
        class Reviewer:
            def verify_turn(self,context,outputs,**kwargs):
                events.append('review')
                active=context['delivery_stage']['current_output_position']
                assert [row['kind'] for row in outputs]==['image','video'][:active]
                if active==2:
                    assert outputs[1]['references'] and 'accepted_critique' in outputs[0]
                assert all(Path(row['media']).is_file() for row in outputs)
                return report(outputs, [1] if events.count('flux') == 1 else [], active_position=active)

            def fix(self, task, context, prompt, reason, **kwargs):
                return prompt + ' repaired'
        @contextmanager
        def session(directory):yield Reviewer()
        options.update(model_factory=lambda backend,*args:Generator(backend),reviewer_factory=session,
                       native_video_output=True,video_width=256,video_height=256)
        result = materialize(root, **options)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(events, ['flux', 'review', 'flux', 'review', 'vdn_h3', 'review'])
        self.assertEqual(result['media'][tokens[1]]['references'][0]['placeholder'], '[[image_0001_01]]')

if __name__=='__main__':unittest.main()
