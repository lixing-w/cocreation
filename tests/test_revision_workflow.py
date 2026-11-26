"""User-visible revision transactions and conflict-free generation inputs."""
from copy import deepcopy
import json
from pathlib import Path
import random
import unittest
from unittest.mock import Mock
from scripts.dialogue_generation.asset_variations import AssetSpec, alternatives, alternate
from scripts.dialogue_generation.planner import Planner, build_plan
from scripts.dialogue_generation.generation import build_tasks
from tools.placeholder_prompter import writer_contract, rewrite_placeholder
from tests.test_materialize import MaterializeTests, FakeModels

ROOT=Path(__file__).resolve().parents[1]

class RevisionTests(unittest.TestCase):
    def source(self):
        return json.loads((ROOT/'scripts/dialogue_generation/examples/tokyo_vlog.json').read_text())



    def test_subtle_metadata_never_drives_a_variation(self):
        row=next(a for a in self.source()['assets'] if a['id']=='bag')
        choices=alternatives(row)
        for key in ('size','color_saturation','color_lightness','condition'):self.assertNotIn(key,choices)
        self.assertEqual({'color','material'}, set(choices))
        fields=[alternate(row,AssetSpec(row),random.Random(seed)).change['field'] for seed in range(300)]
        self.assertGreater(sum(k in ('color','material') for k in fields),150)

    def staged(self,scope):
        source=self.source(); bag=next(a for a in source['assets'] if a['id']=='bag')
        source.update(assets=[bag],constraints=[],shots=[dict(id=f'S{i}',description=f'Bag moves in shot {i}',assets=['bag']) for i in (1,2,3)],target_order=['S1','S2','S3'])
        p=Planner(source,seed=3,reworks=1,cross_turn_rules=False,extra_behaviors=False)
        p.single('bag','direct');old=p.ready['bag']
        for sid in ('S1','S2'):
            token=p.add_media('video',p.shots[sid]['description'],assets=['bag'],shots=[sid],inputs=[old],operation='generate_clip',bindings=dict(bag=old))
            p.emit('assistant','deliver_shot','Here is the clip.',outputs=[token])
            p.rendered[sid]=token;p.bindings[sid]=dict(bag=old);p.render_order.append(sid)
        p.compilation(['S1','S2'],'request_compilation')
        p.emit('user','request_rework','Change the bag in the first shot.',assets=['bag'],shots=['S1'])
        new=p.revision_preview('bag',old)
        p.emit('assistant','clarify_update_scope','Here is the revised reference. This shot or all existing shots?',assets=['bag'],outputs=[new])
        p.emit('user','choose_update_scope','This shot only.' if scope=='shot' else 'All existing shots.',scope=scope)
        edit=dict(index=1,asset_id='bag',target_media=old,target_shot='S1',rendered_uses=['S1','S2'],uses=['S1','S2','S3'],scope=scope,new_media=new)
        p.accept_revision(edit);p.deliver_revision([edit]);delivery=p.turns[-1]
        p.render_due()
        plan=p.run()
        return plan,old,new,delivery

    def test_local_scope_keeps_old_default_and_only_replaces_one_cut_clip(self):
        plan,old,new,delivery=self.staged('shot')
        media={m['placeholder']:m for m in plan['placeholders']}
        self.assertEqual(len(delivery['intent']['updated_clips']),1)
        self.assertEqual(plan['final']['shots']['S1']['bindings']['bag'],new)
        self.assertEqual(plan['final']['shots']['S2']['bindings']['bag'],old)
        self.assertEqual(plan['final']['shots']['S3']['bindings']['bag'],old)
        task=build_tasks(plan,self.source())[delivery['intent']['updated_clips'][0]]
        self.assertEqual([r['kind'] for r in task['references']],['video','image'])
        self.assertEqual(task['references'][1]['placeholder'],new)
        self.assertEqual(task['seconds'],4)
        self.assertEqual(task['output_seconds'],4)
        cut=media[delivery['intent']['updated_compilation']]
        self.assertEqual(cut['shot_versions']['S2'],next(m['placeholder'] for m in media.values() if m['operation']=='generate_clip' and m['shot_ids']==['S2']))

    def test_all_existing_scope_updates_together_and_changes_future_default(self):
        plan,old,new,delivery=self.staged('all_uses')
        self.assertEqual(len(delivery['intent']['updated_clips']),2)
        self.assertTrue(all(s['bindings']['bag']==new for s in plan['final']['shots'].values()))
        after=plan['turns'][delivery['id']:]
        self.assertFalse(any(t['intent']['action']=='request_shot_update' for t in after))

    def test_known_identity_preview_is_delivered_in_the_scope_question(self):
        for seed in range(12):
            plan=build_plan(self.source(),seed=seed,reworks=2)
            questions=[t for t in plan['turns'] if t['intent']['action']=='clarify_update_scope']
            if questions:break
        self.assertTrue(questions)
        media={m['placeholder']:m for m in plan['placeholders']}
        for question in questions:
            self.assertEqual(len(question['output_media']),1)
            self.assertEqual(media[question['output_media'][0]]['kind'],'image')
            self.assertNotIn('future',question['intent']['description'])
            choice=plan['turns'][question['id']]
            delivery=plan['turns'][question['id']+1]
            self.assertEqual(choice['intent']['action'],'choose_update_scope')
            self.assertEqual(delivery['intent']['action'],'deliver_reworked_clips')
            self.assertTrue(delivery['intent']['updated_clips'])

class MaterializePolicyTests(unittest.TestCase):
    setUp=MaterializeTests.setUp
    execute=MaterializeTests.execute
    persist=MaterializeTests.persist
    def test_timings_receipts_and_work_cleanup(self):
        result=self.execute()
        self.assertFalse((self.run/'_work').exists())
        for row in result['media'].values():
            self.assertGreaterEqual(row['elapsed_seconds'],0)
            if row.get('backend'):
                self.assertGreater(row['generation_seconds'],0)
                self.assertTrue((self.run/row['generation_record']).is_file())
                self.assertTrue(row['runtime_cleaned'])


    def test_assembly_appearance_edits_are_separate_short_calls(self):
        parent=next(t for t in self.catalog if t['placeholder']=='[[video_0003]]')
        parent['clip_edits']=[dict(placeholder='[[video_0001]]',generation=dict(operation='video_edit',backend='vdn_h3',
            status='ready',prompt='only this source clip <Video 1>',seconds=4,output_seconds=1,
            dependencies=['[[video_0001]]'],references=[dict(placeholder='[[video_0001]]',kind='video',label='<Video 1>')]))]
        self.persist();models=FakeModels();result=self.execute(models)
        calls=[c for c in models.calls if c['backend']=='vdn_h3']
        self.assertEqual(len(calls),2)
        self.assertTrue(all(c['options']['seconds']<=4 for c in calls))
        single=next(c for c in calls if c['prompt'].startswith('only this'))
        self.assertEqual(len(single['references']),1)
        self.assertAlmostEqual(single['reference_seconds'][0],1,places=1)
        self.assertEqual(len(result['media']['[[video_0003]]']['clip_edits']),1)
        self.assertFalse((self.run/'_work').exists())

if __name__=='__main__':unittest.main()
