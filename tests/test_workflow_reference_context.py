"""Regression: a later clip edit must retain its updated asset reference."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from scripts.dialogue_generation.formal_sequence import generation_context
from tools.placeholder_prompter import writer_contract, writer_system, rewrite_placeholder


class WorkflowReferenceContext(unittest.TestCase):
    def setUp(self):
        self.result = dict(turns=[
            dict(role='user', content='Make the buildings in the existing clip out of thatch.'),
            dict(role='assistant', content='I will update them.',
                 output_media=['[[image_0010]]','[[video_0003]]'],
                 intent=dict(asset_update_workflow=dict(images=['[[image_0010]]'],
                     clips=['[[video_0003]]'],compilation=None)))])
        with patch('scripts.dialogue_generation.planner_rules.media_rules',return_value=[]):
            self.image_context=generation_context(self.result,'[[image_0010]]')
            self.video_context=generation_context(self.result,'[[video_0003]]')
        self.image=dict(backend='h3_image',kind='image',operation='local_edit',subject='buildings',
            render_state=dict(aspect_ratio='portrait'),sequence_context=self.image_context,
            references=[dict(label='<Picture 1>',kind='image',placeholder='[[image_0005]]',
                role='base to edit',subjects=['buildings'],appearance=[dict(material='leather')])])
        self.video=dict(backend='minimax_h3',kind='video',operation='video_edit',seconds=4,
            render_state=dict(aspect_ratio='portrait'),sequence_context=self.video_context,
            asset_ids=['A1'],asset_names=dict(A1='buildings'),references=[
                dict(label='<Video 1>',kind='video',placeholder='[[video_0002]]',role='source video to edit',subjects=['buildings']),
                dict(label='<Picture 1>',kind='image',placeholder='[[image_0010]]',role='identity/design reference',subjects=['buildings'])])

    def test_dependent_stages_remain_distinct_from_alternatives(self):
        self.assertEqual(self.image_context['production_workflow']['current_stage'],'updated_asset_image')
        self.assertEqual(self.video_context['production_workflow']['current_stage'],'reworked_clip')
        self.assertNotIn('Write only output 1 of 2',writer_system(self.image))
        self.assertNotIn('Write only output 2 of 2',writer_system(self.video))
        self.assertIn('not a comparison of alternatives',writer_system(self.image))
        independent=deepcopy(self.image)
        independent['sequence_context'].pop('production_workflow')
        self.assertEqual(writer_system(self.image),writer_system(independent))
        self.assertNotIn('production_workflow',writer_contract(independent))
        self.assertEqual(writer_contract(independent)['output_position'],1)
        self.assertEqual(writer_contract(independent)['output_count'],2)

    def test_bindings_carry_subject_role_and_planned_availability(self):
        asset=writer_contract(self.image)
        self.assertEqual(asset['output_subjects'],['buildings'])
        self.assertEqual(asset['references'][0]['role'],'base to edit')
        self.assertNotIn('appearance',asset['references'][0])
        donor=writer_contract(self.video)['references'][1]
        self.assertEqual(donor['subjects'],['buildings'])
        self.assertEqual(donor['workflow_role'],'updated_asset_image')
        self.assertIn('after its generation and review succeed',donor['availability'])

    def test_writer_scopes_asset_scene_and_names_exact_video_donor(self):
        image_scope=writer_contract(self.image)['production_workflow']['writer_scope']
        self.assertIn('Do not turn the asset image into the entire video scene',image_scope)
        system=writer_system(self.video)
        self.assertIn('explicitly cite its exact label',system)
        self.assertNotIn('buildings',system)
        video_scope=writer_contract(self.video)['production_workflow']['writer_scope']
        self.assertIn('image-only season does not transfer',video_scope)

    def test_video_guide_is_identical_across_sources_and_delivery_modes(self):
        independent = deepcopy(self.video)
        independent.update(operation='text_generation', references=[], render_state=dict(aspect_ratio='1:1'))
        independent['sequence_context'].pop('production_workflow')
        independent['sequence_context']['output_position'] = 1
        self.assertEqual(writer_system(self.video), writer_system(independent))
        self.assertEqual(writer_contract(independent)['references'], [])
        self.assertEqual(writer_contract(independent)['output_format']['orientation'], 'square')

    def test_omitted_reference_still_requires_model_rewrite(self):
        responses=[
            'subject_definitions: The buildings in <Video 1>.\nsummary: [video editing] Change buildings to thatch.\nretention_analysis: Preserve all else.\ndetailed_description: [Shot 1] Retain source action.\noverall_soundscape: N/A\nnon_diegetic_music: N/A',
            'subject_definitions: The buildings in <Video 1>, updated from <Picture 1>.\nsummary: [video editing] Change buildings to thatch.\nretention_analysis: Preserve all else.\ndetailed_description: [Shot 1] Retain source action.\noverall_soundscape: N/A\nnon_diegetic_music: N/A']
        calls=[]
        def request(user,**options):
            calls.append((user,options))
            return responses[len(calls)-1],'stop'
        prompt=rewrite_placeholder(self.video,request)
        self.assertEqual(len(calls),2)
        self.assertIn('<Picture 1>',prompt)
        self.assertIn('Include every exact reference label literally',calls[1][0])
        self.assertIn('"workflow_role": "updated_asset_image"',calls[0][0])


if __name__=='__main__':
    unittest.main()
