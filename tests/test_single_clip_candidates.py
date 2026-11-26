import json,re,subprocess,tempfile,unittest
from pathlib import Path
from tools.placeholder_prompter import rewrite_placeholder
from scripts.dialogue_generation.materialize import materialize
from tools.video_editing import probe

REQUEST='Make two comparative short clips of the miniature wood city model: one with a dark finish and one with a beige finish, both on a dark polished wood base with smooth rounded edges.'

def task(position):
    return dict(placeholder=f'[[video_{position+1:04d}]]',backend='minimax_h3',kind='video',
        operation='text_generation',references=[],dependencies=[],seconds=4.,
        render_state=dict(aspect_ratio='9:16'),desired_attributes={'hidden':'PLANNER_SECRET'},
        sequence_context=dict(current_user_request=REQUEST,
            visible_sequence=[dict(role='user',content=REQUEST,media=[])],
            output_position=position,output_count=2))

def prompt(finish):
    return ('integrated_multimodal_description: [Shot 1] The camera slowly pans right across a miniature wood city model with a '+finish+
        ' finish on a dark polished wood base with smooth rounded edges.\noverall_soundscape: N/A\nnon_diegetic_music: N/A')

class SingleClipCandidatesTests(unittest.TestCase):
    def test_multi_shot_alternatives_are_rejected_even_with_valid_timestamps(self):
        for position in (1,2):
            for timestamp in ('','At 00:02.000, '):
                calls=[];finish='dark' if position==1 else 'beige'
                bad=('integrated_multimodal_description: [Shot 1] A dark city. [Shot 2] '+timestamp+
                    'A beige city.\noverall_soundscape: N/A\nnon_diegetic_music: N/A')
                def writer(user,**options):
                    calls.append((user,options))
                    return (bad if len(calls)==1 else prompt(finish)),'stop'
                with self.subTest(position=position,timestamp=timestamp):
                    result=rewrite_placeholder(task(position),writer)
                    self.assertEqual(result,prompt(finish))
                    self.assertEqual(len(calls),2)
                    self.assertIn('match the output_position to the user',calls[0][1]['system'])
                    contract = json.loads(calls[0][0].split('instructions.\n',1)[1])
                    self.assertEqual((contract['output_position'],contract['output_count']),(position,2))
                    self.assertNotIn('[Shot 2]',calls[0][1]['system'])
                    self.assertIn('one continuous [Shot 1]',calls[1][0])
                    self.assertNotIn('PLANNER_SECRET',str(calls))

    def test_executor_refuses_a_combined_multishot_catalog(self):
        row=task(1)
        row['prompt']=('integrated_multimodal_description: [Shot 1] A dark city. '
            '[Shot 2] At 00:02.000, A beige city.\noverall_soundscape: N/A\nnon_diegetic_music: N/A')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'placeholders.json').write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError,'one single-shot clip'):
                materialize(root,dry_run=True)

    def test_inline_existing_sections_get_newlines_without_another_model_call(self):
        calls=[]
        def writer(user,**options):
            calls.append(user)
            return prompt('dark').replace('\n',' '),'stop'
        result=rewrite_placeholder(task(1),writer,max_attempts=1)
        self.assertEqual(result,prompt('dark'))
        self.assertEqual(len(calls),1)

    def test_section_formatting_does_not_supply_missing_or_reordered_content(self):
        for bad in (prompt('dark').replace('overall_soundscape: N/A\n',''),
                    'integrated_multimodal_description: [Shot 1] A city. non_diegetic_music: N/A overall_soundscape: N/A'):
            with self.subTest(bad=bad),self.assertRaises(RuntimeError):
                rewrite_placeholder(task(1),lambda *a,**kw:(bad,'stop'),max_attempts=1)

    def test_two_prompts_call_the_video_generator_twice_and_keep_native_frames(self):
        rows=[];rewrites=[]
        for position in (1,2):
            row=task(position)
            def writer(user,**options):
                data=json.loads(user.split('instructions.\n',1)[1]);rewrites.append(data)
                return prompt('dark' if data['output_position']==1 else 'beige'),'stop'
            row['prompt']=rewrite_placeholder(row,writer);rows.append(row)
        self.assertEqual([r['output_position'] for r in rewrites],[1,2])
        calls=[]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'placeholders.json').write_text(json.dumps(rows))
            (root/'dialogue.md').write_text('[[video_0002]] [[video_0003]]')
            def factory(backend,model_id,output_dir):
                class Model:
                    def __init__(self):self.output_dir=Path(output_dir)
                    def __call__(self,text,refs,**options):
                        calls.append(text);self.output_dir.mkdir(parents=True,exist_ok=True)
                        path=self.output_dir/'native.mp4'
                        subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i',
                            f"color=blue:s={options['width']}x{options['height']}:r=24:d={107/24}",
                            '-c:v','libx264','-preset','ultrafast','-pix_fmt','yuv420p',str(path)],check=True)
                        return dict(status='ok',path=str(path))
                    def close(self):pass
                return Model()
            result=materialize(root,model_factory=factory,native_video_output=True,
                video_width=512,video_height=512,video_steps=40,progress=False)
            self.assertEqual(calls,[prompt('dark'),prompt('beige')])
            self.assertEqual(len(result['media']),2)
            for row in result['media'].values():
                info=probe(root/row['path'])
                stream=next(s for s in info['streams'] if s['codec_type']=='video')
                self.assertEqual(stream['nb_frames'],'107')
                self.assertEqual(row['video_output_mode'],'native')

if __name__=='__main__':unittest.main()
