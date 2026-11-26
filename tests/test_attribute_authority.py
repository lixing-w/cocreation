"""Regressions from the observed brown-bag / young-girl contradictions."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from scripts.dialogue_generation.asset_variations import AssetSpec, empty_attributes
from scripts.dialogue_generation.planner import Planner, build_plan
from scripts.dialogue_generation.generation import build_tasks
from tools.attribute_authority import neutral_name, ATTRIBUTE_AUTHORITY
from tools.placeholder_prompter import writer_contract, writer_system, rewrite_placeholder, attribute_conflicts
from tools.dialogue_realizer import QwenRealizer, SYSTEM as WRITER_SYSTEM
from tools.dialogue_verifier import CRITERIA, build_prompt

ROOT=Path(__file__).resolve().parents[1]


def source():
    value=json.loads((ROOT/'scripts/dialogue_generation/examples/tokyo_vlog.json').read_text())
    value['constraints']=[]
    return value


class AttributeAuthorityTests(unittest.TestCase):
    def test_identity_names_remove_mutable_color_material_and_age(self):
        for name,kind,wanted in [('Brown Bag','object','bag'),('Pink Sheep','object','sheep'),('Cardboard Box','object','box'),('Young Girl','person','person')]:
            self.assertEqual(neutral_name(name,kind),wanted)
        value=source();a=deepcopy(value['assets'][0]);a.update(id='another',name='Older Woman');value['assets'].append(a);value['shots'][0]['assets'].append('another')
        p=Planner(value,reworks=0)
        self.assertEqual(p.assets['woman']['name'],'person');self.assertEqual(p.assets['another']['name'],'person 2')

    def adult_task(self,editing=False):
        attrs=dict(age_group='adult',hair_color='white',hair_length='short hair',hair_style='sleek straight hair',clothing_color='white and yellow')
        return dict(backend='flux',operation='local_edit' if editing else 'text_generation',subject='person',layout='portrait',
            desired_description='A young girl with short white hair and a white and yellow jacket.',desired_attributes=attrs,
            attribute_overrides=[dict(subject='person',attributes=attrs)],
            changes=[dict(field='age_group',after='adult')] if editing else [],
            references=[dict(label='<Image 1>',kind='image',role='base to edit')] if editing else [],render_state={})

    def test_rewriter_receives_source_evidence_and_authoritative_override(self):
        task=self.adult_task();contract=writer_contract(task)
        self.assertIn('young girl',contract['desired_description'])
        self.assertEqual(contract['attribute_overrides'][0]['attributes']['age_group'],'adult')
        self.assertIn('Resolve each conflict',writer_system(task))
        good='A full-body portrait of an adult person with short white hair and a white and yellow jacket on a neutral studio background.'
        request=Mock(side_effect=[('A young adult girl with short white hair on a neutral studio background.','stop'),(good,'stop')])
        self.assertEqual(rewrite_placeholder(task,request),good);self.assertEqual(request.call_count,2)

    def test_related_bag_does_not_disable_primary_person_age_checks(self):
        task=self.adult_task()
        task['attribute_overrides'].append(dict(subject='bag',attributes=dict(color='green')))
        self.assertTrue(attribute_conflicts('A young adult girl carrying a green bag.',task))

    def test_age_edit_explicitly_preserves_hair_and_clothing(self):
        task=self.adult_task(editing=True)
        good='Edit <Image 1>. Make the person an adult. Preserve short, sleek straight white hair, white and yellow clothing, identity, pose and neutral studio background.'
        request=Mock(side_effect=[('Edit <Image 1>. Make the person an adult; preserve everything else.','stop'),(good,'stop')])
        self.assertEqual(rewrite_placeholder(task,request),good);self.assertEqual(request.call_count,2)

    def test_dialogue_writer_and_judge_get_current_attributes(self):
        model=QwenRealizer(asset_names={'person':'person'});model.verifier=Mock(return_value=dict(dict.fromkeys(CRITERIA,2),critique=''))
        overrides=self.adult_task()['attribute_overrides']
        turn=dict(id=1,role='user',intent=dict(action='request_asset',asset_ids=['person'],description='A young girl; apparent age group: adult.',attribute_overrides=overrides))
        with patch.object(model,'_request',return_value='Make a reference image of an adult person with short white hair.') as writer:
            model(turn,{})
        self.assertIn('Current reference attributes',writer.call_args.args[0])
        self.assertIn('Resolve each conflict',WRITER_SYSTEM)
        self.assertNotIn(ATTRIBUTE_AUTHORITY,writer.call_args.args[0])
        self.assertEqual(model.verifier.call_args.args[1]['attribute_overrides'],overrides)
        brief=model.verifier.call_args.args[1]
        self.assertIn('A young girl',brief['source_design'])
        self.assertNotIn('young girl',' '.join(brief['draft_content_requirements']))
        for prompt in (writer.call_args.args[0],build_prompt('candidate',brief)):
            self.assertIn('Keep every compatible distinctive detail',prompt)
            self.assertIn('State these concrete details in the spoken request',prompt)
            self.assertIn('Replace conflicting source attributes',prompt)

    def test_source_shot_appearance_does_not_become_a_required_design_change(self):
        model=QwenRealizer(asset_names={'person':'person','sheep':'sheep','street':'canal street'})
        model.verifier=Mock(return_value=dict(dict.fromkeys(CRITERIA,2),critique=''))
        staging='A wide shot of a canal street at sunset. A girl riding a pink sheep moves away from the camera down the center of the canal.'
        overrides=[dict(subject='sheep',attributes=dict(color='blue',material='unknown'))]
        turn=dict(id=35,role='user',intent=dict(action='request_shot',asset_ids=['person','sheep','street'],
            description='OBSOLETE RAW PLAN: use the pink sheep with voluminous pink wool.',
            shot_description=staging,attribute_overrides=overrides,generation_mode='new_video'))
        original=deepcopy(turn)
        with patch.object(model,'_request',return_value='Create a wide-shot video of the girl riding the blue sheep away down the canal at sunset.') as writer:
            model(turn,{})
        brief=model.verifier.call_args.args[1]
        self.assertEqual(brief['shot_staging'],staging)
        self.assertNotIn('pink',' '.join(brief['draft_content_requirements']))
        self.assertEqual(brief['attribute_overrides'],overrides)
        judge=build_prompt('candidate',brief)
        for prompt in (writer.call_args.args[0],judge):
            self.assertIn('sheep: color: blue',prompt)
            self.assertIn('Appearance in this source is background, not a requirement',prompt)
            self.assertIn("A generic request to preserve 'the staging' is incomplete",prompt)
            self.assertNotIn('OBSOLETE RAW PLAN',prompt)
            self.assertNotIn('material: unknown',prompt)
            self.assertEqual(prompt.count('pink sheep'),1)
        self.assertEqual(turn,original)

    def test_output_rule_overrides_source_reference_attributes_consistently(self):
        value=source()
        value['constraints']=[dict(scope='partial',selector=dict(kinds=['video']),effect=dict(asset_attributes={'bag':{'material':'wood'}}))]
        p=build_plan(value,seed=1,reworks=0,extra_behaviors=False,cross_turn_rules=False)
        tasks=build_tasks(p,value)
        for row in p['placeholders']:
            if row['kind']=='video' and 'bag' in row['asset_ids'] and row['operation']=='generate_clip':
                overrides={a['subject']:a['attributes'] for a in tasks[row['placeholder']]['attribute_overrides']}
                self.assertEqual(overrides['tote bag']['material'],'wood')

    def test_bag_current_color_overrides_stale_bag_inside_person_caption(self):
        value=source();next(a for a in value['assets'] if a['id']=='bag')['name']='Brown Bag'
        p=Planner(value,reworks=0,extra_behaviors=False,cross_turn_rules=False)
        bag=p.assets['bag'];p.initial_specs['bag']=AssetSpec(bag,{**bag['attributes'],'color':'green'})
        p.pending_reworks.add('bag')
        p.single('bag','direct');p.assets['woman']['description']='A young girl carrying a brown bag with a bird emblem.'
        p.single('woman','direct');token=p.ready['woman'];result=p.run();tasks=build_tasks(result,p.source)
        overrides={row['subject']:row['attributes'] for row in tasks[token]['attribute_overrides']}
        self.assertEqual(overrides['bag']['color'],'green')
        request=next(t for t in result['turns'] if t['intent']['action']=='request_asset' and t['intent']['asset_ids']==['woman'])
        self.assertTrue(any(row['subject']=='bag' and row['attributes']['color']=='green' for row in request['intent']['attribute_overrides']))


if __name__=='__main__':unittest.main()
