"""A requested reference output is not a supplied input image."""
import json,tempfile,unittest
from tools.prompt_reasoner import PromptReasoner


class ProvenanceTests(unittest.TestCase):
    def run_reasoner(self,dialogue,steps):
        captured=[]
        def model(prompt,images,**options):
            captured.append(json.loads(prompt))
            return dict(text=json.dumps(dict(think=dict(intent='I will generate the requested images.',
                rules='I will use the agreed format.',references='I will use only the actual sources.'))),finish_reason='stop')
        with tempfile.TemporaryDirectory() as tmp:
            PromptReasoner(model).write_delivery(dialogue,steps,seed=0,output_dir=tmp)
        return captured[0]['visual_input_contract']

    def test_two_requested_reference_images_have_zero_source_inputs(self):
        turns=[dict(role='user',content='Create two reference images.',media=[])]
        steps=[dict(placeholder=f'[[image_{n:04d}]]',kind='image',prompt='Waterfall.',operation='text_generation',references=[])
               for n in [1,2]]
        contract=self.run_reasoner(turns,steps)
        self.assertEqual([row['visual_input_count'] for row in contract],[0,0])
        self.assertTrue(all(row['mode']=='generation_from_text_only' for row in contract))

    def test_assistant_asset_and_future_workflow_output_keep_distinct_provenance(self):
        turns=[dict(role='assistant',content='Here is the asset.',media=['[[image_0003]]'])]
        steps=[dict(placeholder='[[image_0005]]',kind='image',prompt='Edit.',operation='local_edit',
            references=[dict(placeholder='[[image_0003]]',label='<Picture 1>',kind='image')]),
            dict(placeholder='[[video_0002]]',kind='video',prompt='Rework.',operation='video_edit',
            references=[dict(placeholder='[[image_0005]]',label='<Picture 1>',kind='image')])]
        contract=self.run_reasoner(turns,steps)
        self.assertEqual(contract[0]['actual_inputs'][0]['origin'],'assistant_generated')
        self.assertEqual(contract[1]['actual_inputs'][0]['origin'],'planned_earlier_workflow_output')


if __name__=='__main__':unittest.main()
