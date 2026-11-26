"""Source regression: blue blade must not survive a planned gray first version."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
import tempfile

from scripts.dialogue_generation.asset_variations import AssetSpec, empty_attributes
from scripts.dialogue_generation.planner import Planner, build_plan
from tools.observation_descriptions import separate_descriptions, validate_separation


def observation():
    return dict(assets=[dict(id='sword',kind='object',name='sword',
        description='Blue blade with a gray hilt, held in fingers and twisted toward the sky.',frames=[1.0],
        attributes={**empty_attributes('object'), 'color':'blue'})],
        shots=[dict(id='S1',start=0.0,end=3.0,assets=['sword'],
            description='Fingers twist the blue blade and swing it skyward as the camera pulls out from below.')],
        target_order=['S1'])


class DescriptionSeparationTests(unittest.TestCase):
    def test_separation_keeps_attributes_and_camera_then_planner_has_distinct_versions(self):
        value=observation();before=deepcopy(value)
        cleaned=dict(assets=[dict(id='sword',description='Blade with a hilt, held in fingers and twisted toward the sky.')],
            shots=[dict(id='S1',description='Fingers twist the sword and swing it skyward as the camera pulls out from below.')])
        class Observer:
            def _ask(self,prompt,images,schema,validator,directory):
                self.prompt=prompt
                assert images==[]
                return validator(dict(attribute_separation=2,action_preservation=2,detail_preservation=2,correction='')) if directory.name=='review' else validator(cleaned)
        observer=Observer()
        with tempfile.TemporaryDirectory() as tmp:
            separate_descriptions(observer,value['assets'],value['shots'],Path(tmp)/'separation')
        self.assertEqual(value['assets'][0]['attributes'],before['assets'][0]['attributes'])
        self.assertIn('pulls out from below',value['shots'][0]['description'])
        self.assertNotIn('blue',value['assets'][0]['description'].lower())
        self.assertIn('color',observer.prompt.data)
        planner=Planner(value,seed=8,reworks=1,extra_behaviors=False,cross_turn_rules=False)
        initial=planner.initial_specs['sword']
        self.assertNotEqual(initial.attributes['color'],'blue')
        self.assertNotIn('blue',str(initial).lower())
        plan=planner.run()
        request=next(t for t in plan['turns'] if t['intent']['action']=='request_rework')
        self.assertEqual(request['intent']['attribute_changes']['color']['after'],'blue')
        self.assertNotEqual(request['intent']['attribute_changes']['color']['before'],'blue')

    def test_already_satisfied_planner_emits_no_duplicate_user_edit_or_media(self):
        planner=Planner(observation(),seed=8,reworks=0,extra_behaviors=False,cross_turn_rules=False)
        planner.single('sword','direct')
        planner.pending_reworks.add('sword')
        count=len(planner.media);turns=len(planner.turns)
        planner.rework('sword')
        self.assertEqual(len(planner.media),count)
        self.assertEqual(len(planner.turns),turns+1)
        self.assertEqual(planner.turns[-1]['intent']['action'],'attribute_already_satisfied')
        self.assertEqual(planner.turns[-1]['role'],'assistant')
        self.assertFalse(planner.turns[-1].get('output_media'))

    def test_text_projection_cannot_mutate_identity_or_attributes(self):
        value=observation()
        bad=dict(assets=[dict(id='another',description='Sword')],
                 shots=[dict(id='S1',description='Pull-out')])
        with self.assertRaises(ValueError):validate_separation(bad,value['assets'],value['shots'])
        bad['assets']=[dict(id='sword',description='Sword',attributes={'color':'red'})]
        with self.assertRaises(ValueError):validate_separation(bad,value['assets'],value['shots'])


    def test_independent_check_rewrites_attribute_leakage_before_committing(self):
        value=observation();original=deepcopy(value);calls=[]
        neutral=dict(assets=[dict(id='sword',description='Sword held in fingers.')],
            shots=[dict(id='S1',description='Fingers twist and swing the sword skyward as the camera pulls out from below.')])
        class Observer:
            def _ask(self,prompt,images,schema,validator,directory):
                calls.append(directory.name)
                if directory.name=='writer':
                    answer=neutral if len(calls)>2 else dict(assets=[dict(id='sword',description='Blue blade.')],shots=neutral['shots'])
                else:
                    answer=dict(attribute_separation=2 if len(calls)>2 else 0,action_preservation=2,
                        detail_preservation=2,correction='' if len(calls)>2 else 'Remove blue from the asset description; color is stored in attributes.')
                return validator(answer)
        with tempfile.TemporaryDirectory() as tmp:
            separate_descriptions(Observer(),value['assets'],value['shots'],Path(tmp)/'separation')
        self.assertEqual(calls,['writer','review','writer','review'])
        self.assertEqual(value['assets'][0]['attributes'],original['assets'][0]['attributes'])
        self.assertEqual(value['assets'][0]['description'],'Sword held in fingers.')


if __name__=='__main__':unittest.main()
