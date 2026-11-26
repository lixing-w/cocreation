import json,unittest
from scripts.dialogue_generation.formal_sequence import generation_context
from tools.placeholder_prompter import writer_contract,writer_system,rewrite_placeholder

class RequestFocusTests(unittest.TestCase):
    def context(self):
        rows=[('user','Make a candy-colored studio table.',[]),('assistant','Here it is.',['[[image_0001]]']),
              ('user','This one works.',[]),('assistant','Saved.',[]),
              ('user','Make small seating tables in landscape 16:9.',[]),
              ('assistant','Keep portrait, allow an exception, or replace it?',[]),
              ('user','Keep portrait 9:16 and proceed with the small tables.',[]),
              ('assistant','FUTURE_DELIVERY',['[[image_0002]]'])]
        result=dict(placeholders=[dict(placeholder='[[image_0002]]',kind='image',asset_ids=[],render_state=dict(aspect_ratio='9:16'))],turns=[dict(id=i,role=role,content=text,output_media=media,
            intent=dict(description='PLANNER_SECRET')) for i,(role,text,media) in enumerate(rows,1)])
        return generation_context(result,'[[image_0002]]')

    def test_focus_is_exact_actual_request_clarification_and_resolution(self):
        context=self.context()
        self.assertEqual(context['current_exchange'],context['visible_sequence'][-3:])
        self.assertEqual([r['role'] for r in context['current_exchange']],['user','assistant','user'])
        self.assertIn('small seating tables',context['current_exchange'][0]['content'])
        self.assertNotIn('candy-colored',str(context['current_exchange']))
        self.assertNotIn('PLANNER_SECRET',str(context));self.assertNotIn('FUTURE_DELIVERY',str(context))
        self.assertIn('candy-colored',str(context['visible_sequence']))

    def test_prompter_uses_same_actual_focus_without_hidden_attributes(self):
        task=dict(backend='h3_image',kind='image',operation='text_generation',references=[],
            render_state=dict(aspect_ratio='9:16'),sequence_context=self.context(),
            desired_attributes=dict(color='PLANNER_SECRET'))
        calls=[]
        def request(user,**options):
            calls.append((user,options));return 'A small seating table in a portrait composition.','stop'
        self.assertEqual(rewrite_placeholder(task,request,max_attempts=1),'A small seating table in a portrait composition.')
        self.assertNotIn('PLANNER_SECRET',str(calls))
        self.assertIn('Generate only that request',writer_system(task))
        self.assertEqual(writer_contract(task)['current_exchange'],self.context()['current_exchange'])

if __name__=='__main__':unittest.main()
