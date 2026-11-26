"""Sequence policy regressions: meaning, saved designs, persistent state and real files."""
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.dialogue_generation.asset_variations import (
    AssetSpec, VARIATION_VALUES, alternatives, alternate, canonical_value, empty_attributes,
)
from scripts.dialogue_generation.planner import Planner, build_plan, validate_plan
from scripts.dialogue_generation.generation import build_tasks
from scripts.dialogue_generation.cross_turn_rule import EFFECTS, output_aspect
from scripts.dialogue_generation.materialize import materialize
from tools.video_editing import probe
from tools.dialogue_realizer import QwenRealizer
from tools.dialogue_verifier import CRITERIA

ROOT = Path(__file__).resolve().parents[1]


def source():
    value = json.loads((ROOT/'scripts/dialogue_generation/examples/tokyo_vlog.json').read_text())
    value['constraints'] = []
    return value


class SequencePolicyTests(unittest.TestCase):
    def test_only_requested_fields_and_values_are_sampled(self):
        for kind, candidates in VARIATION_VALUES.items():
            row = dict(id='a', name='subject', kind=kind, description='subject', attributes=empty_attributes(kind))
            for key, values in candidates.items():
                row['attributes'][key] = values[0]
            for seed in range(100):
                result = alternate(row, AssetSpec(row), random.Random(seed))
                changed = [key for key in row['attributes'] if row['attributes'][key] != result.attributes[key]]
                self.assertEqual(changed, [result.change['field']])
                key = changed[0]
                self.assertIn(result.attributes[key], candidates[key])
                self.assertNotEqual(canonical_value(key, result.attributes[key]), canonical_value(key, row['attributes'][key]))

    def test_aliases_and_color_order_cannot_create_duplicate_variations(self):
        cases = [('material', 'cotton fabric', 'cotton'), ('season', 'autumn', 'fall'),
                 ('age_group', 'school-age child', 'child'), ('time_of_day', 'afternoon', 'daytime'),
                 ('clothing_color', 'yellow and white', 'white and yellow')]
        for key, before, after in cases:
            self.assertEqual(canonical_value(key, before), canonical_value(key, after))
            kind = next(kind for kind, candidates in VARIATION_VALUES.items() if key in candidates)
            row = dict(id='a', name='subject', kind=kind, description='subject', attributes=empty_attributes(kind),
                       attribute_extensions={key: dict(values=[before])})
            row['attributes'][key] = before
            self.assertNotIn(after, alternatives(row).get(key, []))

    def test_legacy_multicolored_clothing_does_not_repeat_visible_combination(self):
        row = dict(id='girl', kind='person', name='girl',
                   description='A girl with white hair wearing a white and yellow jacket and carrying a brown bag.',
                   attributes={**empty_attributes('person'), 'clothing_color': 'multicolored'})
        self.assertNotIn('white and yellow', alternatives(row)['clothing_color'])
        self.assertIn('blue and white', alternatives(row)['clothing_color'])


    def test_format_presets_replay_and_cancel_without_changing_past_outputs(self):
        for preset in EFFECTS['aspect_ratio']:
            p = Planner(source(), seed=1, reworks=0, extra_behaviors=False, cross_turn_rules=False)
            rule = p.rules.start(dict(scope='global', selector={}, effect=dict(aspect_ratio=preset), text=''))
            p.single('bag', 'direct')
            old = p.ready['bag']
            def ordinary_request():
                p.emit('user','request_trial','Make a still of the bag.',assets=['bag'])
                token=p.add_media('image','A still of the existing bag.',assets=['bag'],inputs=[old],operation='generate')
                p.emit('assistant','deliver_rule_preview','Here is the still.',assets=['bag'],outputs=[token])
                return p.media[token]
            before=len(p.media)
            p.rules.challenge(rule, outcome='once')
            self.assertEqual(len(p.media),before)  # The conflict waits for a real request.
            changed=ordinary_request()
            self.assertNotEqual(output_aspect(changed['render_state']['aspect_ratio'],'image'), output_aspect(preset,'image'))
            regular=ordinary_request()
            self.assertEqual(regular['render_state']['aspect_ratio'], preset)
            p.rules.challenge(rule, outcome='override')
            ordinary_request()
            replacement = next(iter(p.rules.active.values()))
            self.assertNotEqual(replacement['effect']['aspect_ratio'], preset)
            result = p.run()
            tasks = build_tasks(result, p.source)
            self.assertEqual(tasks[old]['render_state']['aspect_ratio'], output_aspect(preset, 'image'))
            self.assertTrue(validate_plan(result))
            self.assertIn('aspect_ratio', result['placeholders'][-1]['render_state'])

    def test_routine_dialogue_writer_does_not_receive_persistent_rule_context(self):
        model = QwenRealizer(asset_names={'bag': 'bag'})
        model.verifier = Mock(return_value=dict(dict.fromkeys(CRITERIA, 2), critique=''))
        turn = dict(id=10, role='user', intent=dict(action='request_rule_followup', description='Make another trial clip of the bag.',
                    asset_ids=['bag'], requested_media_kind='video', media_rule_context=['trial clips: winter as the background season']))
        with patch.object(model, '_request', return_value='Make another trial clip of the bag.') as writer:
            model(turn, {})
        self.assertNotIn('winter', writer.call_args.args[0])
        self.assertNotIn('media_rule_context', writer.call_args.args[0])
        self.assertIn('winter', ' '.join(model.verifier.call_args.args[1]['media_rule_context']))

    def test_validator_rejects_a_semantically_repeated_variation(self):
        plan = build_plan(source(), seed=3, reworks=1, cross_turn_rules=False)
        row = next(m for m in plan['placeholders'] if m.get('variation'))
        row['variation']['before'] = row['variation']['after']
        with self.assertRaisesRegex(ValueError, 'repeats an existing'):
            validate_plan(plan)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required')
    def test_mixed_format_assembly_normalizes_canvas_without_generating_again(self):
        import test_materialize as fixtures
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [dict(placeholder=f'[[video_{i:04d}]]', kind='video', operation='text_generation', backend='vdn_h3',
                         prompt='a moving bag', seconds=2, output_seconds=2, reference_materials=[], references=[],
                         render_state=dict(aspect_ratio=ratio)) for i,ratio in [(1,'9:16'),(2,'16:9')]]
            rows.append(dict(placeholder='[[video_0003]]', kind='video', operation='concatenate',
                reference_materials=['[[video_0001]]','[[video_0002]]'],
                clips=[dict(placeholder='[[video_0001]]'),dict(placeholder='[[video_0002]]')],
                render_state=dict(aspect_ratio='9:16')))
            (root/'placeholders.json').write_text(json.dumps(rows))
            (root/'observation.json').write_text(json.dumps(dict(shots=[])))
            (root/'dialogue.md').write_text('[[video_0003]]')
            models = fixtures.FakeModels()
            materialize(root, model_factory=models, progress=False)
            self.assertEqual(len(models.calls), 2)
            info = probe(root/'media/video_0003.mp4')
            stream = next(s for s in info['streams'] if s['codec_type']=='video')
            self.assertEqual((stream['width'],stream['height']), (720,1280))
            self.assertAlmostEqual(float(info['format']['duration']), 4, delta=.2)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required')
    def test_actual_files_and_generator_dimensions_follow_every_preset(self):
        import test_materialize as fixtures
        from PIL import Image
        for preset in EFFECTS['aspect_ratio']:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                models = fixtures.FakeModels()
                rows = [dict(placeholder='[[image_0001]]', kind='image', operation='text_generation', backend='flux', prompt='a bag', reference_materials=[], references=[], render_state=dict(aspect_ratio=output_aspect(preset, 'image'))),
                        dict(placeholder='[[video_0001]]', kind='video', operation='text_generation', backend='vdn_h3', prompt='a moving bag', seconds=2, output_seconds=2, reference_materials=[], references=[], render_state=dict(aspect_ratio=output_aspect(preset, 'video')))]
                (root/'placeholders.json').write_text(json.dumps(rows))
                (root/'observation.json').write_text(json.dumps(dict(shots=[])))
                (root/'dialogue.md').write_text('[[image_0001]] [[video_0001]]')
                result = materialize(root, model_factory=models, progress=False)
                self.assertEqual(result['status'], 'complete')
                for call, row in zip(models.calls, rows):
                    ratio = row['render_state']['aspect_ratio']; w,h = map(int, ratio.split(':'))
                    self.assertEqual(call['options']['width']*h, call['options']['height']*w)
                    path = root/'media'/('image_0001.png' if row['kind']=='image' else 'video_0001.mp4')
                    if row['kind']=='image':
                        with Image.open(path) as im: width,height = im.size
                    else:
                        info = next(s for s in probe(path)['streams'] if s['codec_type']=='video')
                        width,height = info['width'],info['height']
                    self.assertEqual(width*h, height*w)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required')
    def test_two_pass_generator_receives_canvas_parameters_without_ratio_prompt(self):
        import test_materialize as fixtures
        for ratio, native, final in [('16:9',(1024,576),(1280,720)),
                                     ('9:16',(576,1024),(720,1280)),
                                     ('1:1',(1024,1024),(1024,1024))]:
            with self.subTest(ratio=ratio), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);models=fixtures.FakeModels()
                rows=[dict(placeholder='[[video_0001]]',kind='video',operation='text_generation',
                    backend='vdn_h3',prompt='An object moves gently.',seconds=2,output_seconds=2,
                    reference_materials=[],references=[],render_state=dict(aspect_ratio=ratio))]
                (root/'placeholders.json').write_text(json.dumps(rows))
                (root/'observation.json').write_text(json.dumps(dict(shots=[])))
                (root/'dialogue.md').write_text('[[video_0001]]')
                materialize(root,video_steps=8,video_two_pass=True,model_factory=models,progress=False)
                call=models.calls[0]
                self.assertEqual(call['prompt'],'An object moves gently.')
                self.assertEqual((call['options']['width'],call['options']['height']),native)
                self.assertEqual(call['options']['steps'],8)
                self.assertTrue(call['options']['two_pass'])
                stream=next(s for s in probe(root/'media/video_0001.mp4')['streams'] if s['codec_type']=='video')
                self.assertEqual((stream['width'],stream['height']),final)


if __name__ == '__main__':
    unittest.main()
