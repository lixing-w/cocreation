"""Full design history, explicit reference rollback, color defaults and immediate matching."""
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts.dialogue_generation.asset_variations import (
    AGES, AssetSpec, alternatives, canonical_value, design_key, empty_attributes,
)
from scripts.dialogue_generation.planner import Planner, build_plan, validate_plan
from scripts.dialogue_generation.generation import build_tasks
from tools.placeholder_prompter import rewrite_placeholder, writer_system


def source():
    person=dict(id='p',kind='person',name='person',description='A person wearing a sweater.',frames=[1.],
                attributes={**empty_attributes('person'),'age_group':'child','clothing_color':'blue'})
    return dict(assets=[person],shots=[dict(id='S1',description='The person waves.',assets=['p'])])


class VersionHistoryTests(unittest.TestCase):
    def test_four_distinct_ages_and_legacy_inputs_share_one_design(self):
        from tools.prompt_observer import attributes_schema
        self.assertEqual(AGES,('child','teenager','adult','elder'))
        age_schema=attributes_schema('person')['properties']['age_group']
        self.assertEqual(age_schema['enum'],[*AGES,'unknown','not applicable'])
        for alias,age in [('young adult','adult'),('middle-aged adult','adult'),('older adult','elder'),
                          ('teenager','teenager'),('school-age child','child')]:
            self.assertEqual(canonical_value('age_group',alias),age)
        row=source()['assets'][0]
        a={**row['attributes'],'age_group':'young adult','clothing_color':'white and yellow'}
        b={**row['attributes'],'age_group':'adult','clothing_color':'yellow and white'}
        self.assertEqual(design_key(row,a),design_key(row,b))
        row['attributes']['age_group']='young adult'
        self.assertEqual(AssetSpec(row).attributes['age_group'],'adult')
        self.assertNotIn('adult',alternatives(row)['age_group'])
        self.assertIn('teenager',alternatives(row)['age_group'])

    def test_sampler_excludes_tried_combinations_without_banning_age_in_other_designs(self):
        row=source()['assets'][0]
        tried={design_key(row,{**row['attributes'],'age_group':'adult'})}
        self.assertNotIn('adult',alternatives(row,tried=tried)['age_group'])
        blue_to_red=AssetSpec(row,{**row['attributes'],'clothing_color':'red'})
        self.assertIn('adult',alternatives(row,blue_to_red,tried=tried)['age_group'])
        p=Planner(source(),seed=5,reworks=0,extra_behaviors=False)
        for age in AGES:
            spec=AssetSpec(p.assets['p'],{**p.assets['p']['attributes'],'age_group':age})
            token=p.asset_media('p',description=spec)
            p.emit('assistant','deliver_asset','Here is the reference.',assets=['p'],outputs=[token])
        self.assertNotIn('age_group',alternatives(p.assets['p'],tried=p.tried_designs['p']))
        with self.assertRaisesRegex(ValueError,'already tried'):
            p.asset_media('p',description=AssetSpec(p.assets['p']))

    def test_return_to_child_reuses_earlier_reference_and_requests_restore(self):
        p=Planner(source(),seed=5,reworks=0,extra_behaviors=False)
        p.single('p','direct'); child=p.ready['p']
        adult=AssetSpec(p.assets['p'],{**p.assets['p']['attributes'],'age_group':'adult'})
        p.emit('user','request_variant','Try an adult.',assets=['p'],inputs=[child])
        token=p.asset_media('p',description=adult,parent=child,desired=False)
        p.emit('assistant','deliver_variant','Here is the adult.',assets=['p'],outputs=[token])
        p.ready['p']=token
        p.render_due()
        p.pending_reworks.add('p')
        count=p.counts['image']
        p.rework('p',AssetSpec(p.assets['p']))
        self.assertEqual(p.counts['image'],count)
        request=next(t for t in p.turns if t['intent']['action']=='request_version_restore')
        self.assertEqual(request['intent']['restore_media'],child)
        self.assertIn('prefer the earlier',request['intent']['description'])
        self.assertEqual(p.ready['p'],child)
        result=p.run()
        self.assertTrue(validate_plan(result))
        broken=deepcopy(result)
        delivered=next(t for t in broken['turns'] if t['intent']['action']=='deliver_reworked_clips')
        delivered['intent']['resolved_edits'][0]['restore_media']=token
        with self.assertRaisesRegex(ValueError,'Restored version'):
            validate_plan(broken)

    def test_many_seeded_plans_never_regenerate_a_tried_design(self):
        for seed in range(40):
            plan=build_plan(source(),seed=seed,reworks=seed%2)
            seen=set()
            for m in plan['placeholders']:
                if m.get('attributes'):
                    key=design_key(plan['target_assets']['p'],m['attributes'])
                    self.assertNotIn(key,seen,(seed,m['placeholder']))
                    seen.add(key)

    def test_final_still_prompt_explicitly_requests_natural_color(self):
        s=source();plan=build_plan(s,seed=4,reworks=0,modules=['direct'],extra_behaviors=False)
        task=next(t for t in build_tasks(plan,s).values() if t.get('layout')=='portrait')
        self.assertEqual(task['presentation']['color_mode'],'natural color photograph')
        self.assertIn('natural color photography',writer_system(task))
        raw='A child wearing blue. Natural portrait. Neutral studio background.'
        text=rewrite_placeholder(task,lambda *a,**k:(raw,'stop'))
        self.assertIn('Natural color photograph',text)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'ffmpeg required')
class ImmediateBrightnessTests(unittest.TestCase):
    def test_match_and_publish_finish_before_the_next_edit_model_is_called(self):
        import test_video_tone as fixtures
        import scripts.dialogue_generation.materialize as module
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[fixtures.task(1),fixtures.task(2,'video_edit',[1]),fixtures.task(3,'video_edit',[2])]
            (root/'placeholders.json').write_text(json.dumps(rows))
            (root/'dialogue.md').write_text(' '.join(fixtures.token(i) for i in (1,2,3)))
            models=fixtures.GradientModels();events=[]
            def factory(*args):
                underlying=models(*args)
                class Model:
                    def close(self): underlying.close()
                    def __call__(self,prompt,refs,**opts):
                        events.append(prompt)
                        if prompt=='clip 3':
                            manifest=json.loads((root/'media_manifest.json').read_text())
                            parent=manifest['media'][fixtures.token(2)]
                            self_test.assertEqual(parent['status'],'complete')
                            self_test.assertEqual(parent['brightness_matching']['status'],'matched')
                            self_test.assertEqual(parent['brightness_matching']['stage'],'post_generation_before_publish')
                            self_test.assertEqual(module.file_hash(refs[0].path),parent['sha256'])
                        underlying.output_dir=Path(tmp)/'generated'/prompt.replace(' ','-')
                        return underlying(prompt,refs,**opts)
                return Model()
            self_test=self
            original=module.match_video
            def match(*args,**kwargs):
                result=original(*args,**kwargs);events.append('matched');return result
            with patch.object(module,'match_video',side_effect=match):
                report=module.materialize(root,model_factory=factory,output_width=96,output_height=64,progress=False)
            self.assertEqual(events,['clip 1','clip 2','matched','clip 3','matched'])
            self.assertEqual(report['media'][fixtures.token(3)]['brightness_matching']['anchor']['placeholder'],fixtures.token(1))
