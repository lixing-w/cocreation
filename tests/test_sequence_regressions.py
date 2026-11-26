"""Ordinary requests carry rule changes, current identities and subject relations."""
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from scripts.dialogue_generation.asset_variations import AssetSpec
from scripts.dialogue_generation.planner import Planner, validate_plan
from scripts.dialogue_generation.generation import relationship_context
from tools.placeholder_prompter import rewrite_placeholder, writer_contract

ROOT=Path(__file__).resolve().parents[1]


class SequenceRegressions(unittest.TestCase):
    def planner(self):
        source=json.loads((ROOT/'scripts/dialogue_generation/examples/tokyo_vlog.json').read_text())
        source['constraints']=[]
        return Planner(source,seed=1,reworks=0,extra_behaviors=False,cross_turn_rules=False)

    def test_rule_decisions_modify_the_next_content_request_without_extra_media(self):
        for outcome in ('keep','once','override'):
            with self.subTest(outcome=outcome):
                p=self.planner()
                rule=p.rules.start(dict(scope='global',selector={},effect=dict(aspect_ratio='portrait'),text=''))
                p.single('bag','direct');count=len(p.media)
                p.rules.challenge(rule,outcome=outcome,kind='image')
                self.assertEqual(len(p.media),count)
                p.single('cup','direct')
                self.assertEqual(len(p.media),count+1)
                self.assertFalse(any(t['output_media'] and t['intent']['module']=='cross_turn_rule' for t in p.turns))
                request=next(t for t in p.turns if t['intent']['action']=='request_asset' and t['intent']['asset_ids']==['cup'])
                self.assertTrue(request['intent']['rule_overlay'])
                self.assertIn('cup',request['intent']['description'])
                self.assertTrue(validate_plan(p.run()))

    def test_clarification_shows_the_adopted_young_person_instead_of_old_draft(self):
        p=self.planner();p.single('bag','direct')
        target=p.ready['bag'];asset=p.assets['woman']
        adult=AssetSpec(asset,{**asset['attributes'],'age_group':'adult'})
        young=AssetSpec(asset,{**asset['attributes'],'age_group':'young adult'})
        p.emit('user','request_asset','Make an adult portrait.',assets=['woman'])
        old=p.asset_media('woman',description=adult,desired=False)
        p.emit('assistant','deliver_asset','Here is the portrait.',assets=['woman'],outputs=[old])
        p.emit('user','relative_request','Make her a young adult.',assets=['woman'])
        new=p.asset_media('woman',description=young,parent=old)
        p.emit('assistant','preview_for_confirmation','Here is the younger version.',assets=['woman'],outputs=[new])
        p.adopt('woman',new,temporary=True)
        self.assertEqual(p.current_references(exclude=[target]),[new])
        p.ambiguity('bag',target)
        question=next(t for t in reversed(p.turns) if t['intent']['action']=='clarify_reference')
        self.assertEqual(question['input_media'],[new])
        self.assertNotIn(old,question['input_media'])
        self.assertEqual(p.media[new]['attributes']['age_group'],'young adult')

    def test_relationship_context_only_links_subjects_in_this_shot(self):
        targets={'p':dict(name='person',description='A person carrying a brown bag.'),
                 'b':dict(name='bag',description='A brown bag.'),
                 's':dict(name='sheep',description='A sheep.')}
        context=relationship_context(['p','b'],targets)
        self.assertEqual(context[0]['related_subjects'],['bag'])
        self.assertFalse(relationship_context(['p'],targets))
        self.assertNotIn('sheep',json.dumps(context))

    def test_prompt_with_bag_only_in_definitions_is_rewritten(self):
        task=dict(backend='minimax_h3',operation='reference_generation',kind='video',seconds=4.,
            references=[dict(label='<Picture 1>',kind='image',role='identity/design reference',subjects=['person']),
                        dict(label='<Picture 2>',kind='image',role='identity/design reference',subjects=['bag'])],
            subject_relationship_context=[dict(subject='person',related_subjects=['bag'],relationship_source='A person carrying a brown bag.')],
            attribute_overrides=[dict(subject='person',attributes=dict(age_group='young adult')),
                                 dict(subject='bag',attributes=dict(color='green'))])
        prefix=('subject_definitions:\n<Subject 1>: A young adult from <Picture 1>.\n'
                '<Subject 2>: The green bag from <Picture 2>.\nsummary: [reference generation] A person walks.\n'
                'retention_analysis: Preserve the identities and supplied relationships.\n'
                'detailed_description: [Shot 1] ')
        suffix='\noverall_soundscape: N/A\nnon_diegetic_music: N/A'
        bad=prefix+'<Subject 1> walks along the path.'+suffix
        good=prefix+'<Subject 1> walks along the path while carrying <Subject 2> in her hand.'+suffix
        request=Mock(side_effect=[(bad,'stop'),(good,'stop')])
        prompt=rewrite_placeholder(task,request)
        self.assertEqual(request.call_count,2)
        self.assertIn('carrying <Subject 2>',prompt)
        self.assertIn('detailed_description',request.call_args.args[0])
        contract=writer_contract(task)
        self.assertEqual(contract['subject_relationship_context'],task['subject_relationship_context'])
        self.assertIn('relations',contract['relationship_instruction'].lower())
        self.assertEqual(contract['attribute_overrides'][1]['attributes']['color'],'green')


if __name__=='__main__':unittest.main()
