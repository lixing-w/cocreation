"""Ground edits by subject and keep image/video rules separate."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.dialogue_generation.formal_sequence import generation_edit_scope,review_context
from tools.placeholder_prompter import writer_contract
from tools.prompt_reasoner import PromptReasoner
from tools.media_review import MediaReviewer,parse_review


class SubjectEditScope(unittest.TestCase):
    def test_subject_color_does_not_become_an_existing_subpart_color(self):
        task=dict(backend='h3_image',kind='image',operation='local_edit',subject='sword',
            sequence_context=dict(current_user_request='Make the sword blue.'),references=[],
            changes=[dict(field='color',after='blue'),dict(field='render.season',after='fall')])
        self.assertEqual(writer_contract(task)['requested_edits'],[
            dict(subject='sword',property='color',requested_value='blue')])
        video=dict(task,subject=None,changes=[dict(field='sword.color',after='blue')])
        self.assertEqual(generation_edit_scope(video),generation_edit_scope(task))
        task['changes'][0]['before']='gray'
        self.assertEqual(generation_edit_scope(task)[0]['previous_value'],'gray')

    def test_review_receives_the_same_resolved_edit_and_bound_subject(self):
        task=dict(subject='sword',changes=[dict(field='color',after='blue')],
            references=[dict(label='<Picture 1>',kind='image',subjects=['sword'],role='base to edit')])
        result=dict(placeholders=[dict(placeholder='[[image_0005]]',generation=task)],
            turns=[dict(output_media=['[[image_0005]]'],intent={})])
        context=review_context(result,'[[image_0005]]',context=dict(current_user_request='Make the sword blue.'))
        self.assertEqual(context['requested_edits'],generation_edit_scope(task))
        self.assertEqual(context['reference_bindings'][0]['subjects'],['sword'])

    def test_group_rules_are_scoped_for_every_output_not_only_first_image(self):
        captured=[]
        def model(user,refs,**options):
            captured.append(json.loads(user))
            return dict(text=json.dumps(dict(think=dict(intent='I will update the sword.',
                rules='I will apply each output format and season.',references='I will use the updated image after review.'))),finish_reason='stop')
        steps=[dict(placeholder='[[image_0005]]',kind='image',operation='local_edit',subject='sword',
                    changes=[dict(field='color',after='blue')],prompt='Edit the sword.',latest_rules=[dict(requirement='Use spring as the background season.')]),
               dict(placeholder='[[video_0002]]',kind='video',operation='video_edit',
                    prompt='Edit the clip.',latest_rules=[dict(requirement='Use fall as the background season.')])]
        with tempfile.TemporaryDirectory() as tmp:
            PromptReasoner(model).write_rework([],steps,seed=0,output_dir=tmp)
        rules=captured[0]['current_rules']
        self.assertIn('image [[image_0005]] only: Use spring',rules[0]['requirement'])
        self.assertIn('video [[video_0002]] only: Use fall',rules[1]['requirement'])
        self.assertEqual(captured[0]['generation_steps'][0]['requested_edits'][0]['subject'],'sword')

    def test_reviewer_transport_carries_edit_scope_without_claimed_visual_refs(self):
        def model(*,messages,**options):
            payload=json.loads(messages[1]['content'][0]['text'])
            self.assertEqual(payload['requested_edits'],[dict(subject='sword',property='color',requested_value='blue')])
            self.assertEqual(payload['attached_reference_count'],0)
            self.assertEqual(payload['reference_bindings'],[])
            report=dict(think=dict(request_check='I see the sword.',quality_check='Its target color is blue.',decision='I accept it.'),
                request_fidelity=2,edit_preservation=2,reference_alignment=2,visual_integrity=2,cross_turn_rules=2,
                rule_checks=[],decision='accepted',reason='',reply='')
            self.assertNotIn('edit_checks',options['response_schema']['properties'])
            return dict(text=json.dumps(report),finish_reason='stop')
        context=dict(current_user_request='Make the sword blue.',latest_rules=[],output_position=1,output_count=1,
            requested_edits=[dict(subject='sword',property='color',requested_value='blue')])
        with tempfile.TemporaryDirectory() as tmp,patch('tools.media_review.media_geometry',return_value=dict(available=True,aspect_ratio='16:9')):
            report=MediaReviewer(model).verify(context,'A blue sword.',Path(tmp)/'result.png','image',[],retry_available=True,seed=0,output_dir=Path(tmp)/'review')
        self.assertEqual(report['decision'],'accepted')



if __name__=='__main__':unittest.main()
