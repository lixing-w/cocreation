"""Visible variation policy, current expert state, and native single-pass sampling."""
from copy import deepcopy
import json
from pathlib import Path
import random
import unittest
from cocreation import VDNH3
from cocreation.vdn import generation_command
from scripts.dialogue_generation.asset_variations import AssetSpec, alternatives, alternate
from scripts.dialogue_generation.planner import Planner, build_plan
from scripts.dialogue_generation.generation import build_tasks
from tools.placeholder_prompter import writer_contract, rewrite_placeholder

ROOT = Path(__file__).resolve().parents[1]

class VisualPolicyTests(unittest.TestCase):
    def source(self):
        return json.loads((ROOT/'scripts/dialogue_generation/examples/tokyo_vlog.json').read_text())

    def test_object_variations_only_change_color_and_requested_material_families(self):
        row = next(a for a in self.source()['assets'] if a['id']=='bag')
        row['attributes'].update(material='paper', shape='square', surface_finish='matte')
        choices = alternatives(row)
        self.assertNotIn('size', choices)
        self.assertEqual(set(choices), {'color', 'material'})
        self.assertNotIn('texture', choices)
        self.assertTrue({'wood', 'glass'} <= set(choices['material']))
        self.assertIn('leather', choices['material'])
        counts = {}
        for seed in range(200):
            changed = alternate(row, AssetSpec(row), random.Random(seed)).change['field']
            counts[changed] = counts.get(changed,0)+1
        self.assertGreater(counts['color']+counts['material'], 100)

    def test_every_studio_image_receives_seasonal_presentation_including_local_edits(self):
        for season in ('spring','summer','fall','winter'):
            source=self.source(); source['constraints']=[dict(scope='global', selector={}, effect=dict(season=season), text='')]
            plan=build_plan(source,seed=4,reworks=2,cross_turn_rules=False)
            tasks=build_tasks(plan,source)
            images=[t for t in tasks.values() if t.get('layout') in ('product','portrait')]
            self.assertTrue(any(t['operation']=='local_edit' for t in images))
            for task in images:
                contract=writer_contract(task)
                self.assertEqual(contract['presentation']['season'],season)
                self.assertIn('neutral',contract['presentation']['backdrop'])
                self.assertNotEqual(contract['presentation']['seasonal_accents'],'none')
                if season=='winter':
                    self.assertIn('subject surface',contract['presentation']['seasonal_accents'])

    def test_h3_current_paper_reference_does_not_get_guide_leather_caption(self):
        source=self.source(); source['constraints']=[]
        bag=next(a for a in source['assets'] if a['id']=='bag')
        bag['description']='A brown leather satchel with a bird emblem.'
        bag['attributes'].update(material='leather',surface_finish='matte',transparency='opaque')
        p=Planner(source,seed=1,reworks=0,extra_behaviors=False,cross_turn_rules=False)
        bag=p.assets['bag']
        paper=AssetSpec(bag,{**bag['attributes'],'material':'paper'})
        token=p.asset_media('bag',description=paper,desired=False)
        p.emit('assistant','deliver_asset','Here is the paper bag.',outputs=[token])
        trial=p.add_media('video','Short trial: brown leather satchel; material: paper',assets=['bag'],inputs=[token],operation='generate')
        p.emit('assistant','deliver_exploration','Here is the trial.',outputs=[trial])
        plan=p.run(); task=build_tasks(plan,source)[trial]
        contract=writer_contract(task)
        self.assertEqual(contract['references'][0]['appearance'][0]['attributes']['material'],'paper')
        self.assertNotIn('leather',json.dumps(contract).lower())

    def test_single_pass_steps_reach_actual_freevideo_command(self):
        install=dict(python='python',cache='cache',base='base',checkpoint='checkpoint',comfy_root='comfy',model_paths='paths')
        for steps in (16,20):
            request=VDNH3()('A sphere',steps=steps,dry_run=True)
            self.assertFalse(request['options']['two_pass'])
            self.assertEqual(request['options']['refine_steps'],0)
            command=generation_command(request,Path('/tmp/job'),install)
            self.assertIn('--no-two-pass',command)
            self.assertEqual(command[command.index('--base-steps')+1],str(steps))
        legacy=VDNH3()('A sphere',two_pass=True,dry_run=True)
        self.assertEqual(legacy['options']['base_steps'],8)
        self.assertEqual(legacy['options']['refine_steps'],3)
        with self.assertRaises(ValueError): VDNH3()('x',steps=16,two_pass=True,dry_run=True)

if __name__=='__main__': unittest.main()
