"""Observation boundaries, catalog enforcement, and model ownership."""
import io
import base64
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.dialogue_generation.asset_variations import empty_attributes
from scripts.dialogue_generation.planner import build_plan, validate_plan
from tools.video_observer import VideoObserver, inventory_schema, validate_inventory, validate_review
from tools.video_observer_prompts import attributes_prompt


def inventory(count=4):
    return dict(assets=[dict(id='A1', kind='object', name='ball', frame_ids=[1, count])],
                shots=[dict(first_frame=1, last_frame=count, description='A ball is visible.', assets=['A1'])])


class InventoryTests(unittest.TestCase):
    def test_rejects_invented_references_and_frame_numbers(self):
        for mutation in ('unknown_id', 'duplicate_id', 'unknown_kind', 'bad_frame', 'bad_type', 'unused'):
            value = inventory()
            if mutation == 'unknown_id': value['shots'][0]['assets'] = ['missing']
            elif mutation == 'duplicate_id': value['assets'] *= 2
            elif mutation == 'unknown_kind': value['assets'][0]['kind'] = 'animal'
            elif mutation == 'bad_frame': value['assets'][0]['frame_ids'] = [5]
            elif mutation == 'bad_type': value['assets'][0]['frame_ids'] = [True]
            else: value['assets'].append(dict(id='A2', kind='object', name='extra', frame_ids=[1]))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_inventory(value, 4, 8)

    def test_rejects_shot_gaps_overlaps_and_false_visibility(self):
        for start in (2, 4):
            value = inventory()
            value['shots'] = [dict(first_frame=1, last_frame=2, description='First.', assets=['A1']),
                              dict(first_frame=start, last_frame=4, description='Second.', assets=['A1'])]
            with self.subTest(start=start), self.assertRaises(ValueError):
                validate_inventory(value, 4, 8)
        value = inventory()
        value['assets'].append(dict(id='A2', kind='object', name='cup', frame_ids=[1]))
        value['shots'] = [dict(first_frame=1, last_frame=2, description='First.', assets=['A1']),
                          dict(first_frame=3, last_frame=4, description='Second.', assets=['A2'])]
        with self.assertRaisesRegex(ValueError, 'frame_ids'):
            validate_inventory(value, 4, 8)

    def test_attribute_prompt_only_includes_relevant_category(self):
        prompt = attributes_prompt(inventory()['assets'][0], [0.0, 1.0])
        self.assertIn('surface_finish', prompt)
        self.assertIn('unknown', prompt)
        self.assertNotIn('hair_length', prompt)
        self.assertNotIn('dialogue', prompt)
        self.assertNotIn('Use description', prompt)

    def test_repeated_subject_cannot_be_split_into_per_frame_assets(self):
        value = inventory()
        value['assets'].append({**value['assets'][0], 'id': 'A2', 'frame_ids': [2]})
        value['shots'][0]['assets'].append('A2')
        with self.assertRaisesRegex(ValueError, 'Repeated asset'):
            validate_inventory(value, 4, 8)

    def test_review_can_remove_an_unsupported_value_but_cannot_invent_one(self):
        asset = inventory()['assets'][0]
        proposed = {**empty_attributes('object'), 'color': 'red'}
        reviewed = {**proposed, 'color': 'unknown'}
        self.assertEqual(validate_review(asset, proposed, reviewed), reviewed)
        with self.assertRaisesRegex(ValueError, 'only keep'):
            validate_review(asset, proposed, {**proposed, 'color': 'blue'})

    def test_ollama_uses_images_schema_and_keep_alive(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            image = directory / 'frame.jpg'
            from PIL import Image
            Image.new('RGB', (128, 128), 'red').save(image)
            response = dict(done_reason='stop', message=dict(content='{}'))
            observer = VideoObserver()
            schema = inventory_schema(1, 8)
            with patch('tools.video_observer.urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as call:
                self.assertEqual(observer._request('Observe.', [image], schema, directory), '{}')
            request = call.call_args.args[0]
            body = json.loads(request.data)
            self.assertTrue(request.full_url.endswith('/api/chat'))
            self.assertEqual(body['format'], schema)
            self.assertEqual(body['model'], 'qwen3-vl:30b-a3b-instruct-q4_K_M')
            self.assertEqual(body['keep_alive'], '15m')
            self.assertEqual(body['messages'][1]['images'], [base64.b64encode(image.read_bytes()).decode('ascii')])
            self.assertNotIn(str(image), body['messages'][1]['content'])

    def test_context_budget_is_checked_before_sending_images(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            image = directory / 'frame.jpg'
            Image.new('RGB', (128, 128), 'red').save(image)
            observer = VideoObserver(context_tokens=2048)
            with patch('tools.video_observer.urllib.request.urlopen') as call:
                with self.assertRaisesRegex(ValueError, 'context budget'):
                    observer._request('Observe.', [image], inventory_schema(1, 8), directory)
                call.assert_not_called()

    def test_instruct_model_and_caller_ownership(self):
        with self.assertRaisesRegex(ValueError, 'Instruct'):
            VideoObserver(model='qwen3-vl:30b')
        qwen = Mock(model_id='test-model')
        with VideoObserver(qwen=qwen) as observer:
            self.assertEqual(observer.backend, 'local')
        qwen.close.assert_not_called()


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'Requires ffmpeg and ffprobe')
class VideoPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / 'test.mp4'
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                        'color=c=red:s=128x128:r=8:d=1', '-an', str(self.video)], check=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_decoded_frame_times_retries_and_downstream_plan(self):
        attrs = {**empty_attributes('object'), 'color': 'red', 'color_saturation': 'high saturation'}
        bad = {**attrs, 'color': 'invented red'}
        qwen = Mock(model_id='test-model', side_effect=[dict(text=json.dumps(v)) for v in (inventory(), bad, attrs, attrs)])
        out = self.root / 'result'
        with VideoObserver(qwen=qwen) as observer:
            result = observer.observe(self.video, out, frames=4, progress=False)
            self.assertEqual(observer.qwen, qwen)
        qwen.close.assert_not_called()
        self.assertEqual(qwen.call_count, 4)
        self.assertIn('previous response was invalid', qwen.call_args_list[2].args[0])
        self.assertEqual(result['constraints'], [])
        self.assertEqual(result['attribute_schema_version'], 2)
        self.assertEqual(result['assets'][0]['attributes'], attrs)
        self.assertEqual(result['assets'][0]['frames'], [0.0, 0.875])
        self.assertEqual(result['shots'][0]['start'], 0.0)
        self.assertEqual(result['shots'][0]['end'], 1.0)
        self.assertEqual(result, json.loads((out / 'observation.json').read_text()))
        self.assertTrue(validate_plan(build_plan(result, seed=1, reworks=1)))
        with self.assertRaises(FileExistsError):
            VideoObserver(qwen=qwen).observe(self.video, out, frames=4, progress=False)

    def test_invalid_catalog_and_compatibility_never_publish_observation(self):
        for mode in ('missing', 'extra', 'incompatible'):
            attrs = empty_attributes('object')
            if mode == 'missing': attrs.pop('color')
            elif mode == 'extra': attrs['invented'] = 'value'
            else: attrs.update(material='wood', transparency='transparent')
            qwen = Mock(model_id='test-model', side_effect=[dict(text=json.dumps(v)) for v in (inventory(), attrs)])
            out = self.root / mode
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                with VideoObserver(qwen=qwen, retries=0) as observer:
                    observer.observe(self.video, out, frames=4, progress=False)
            self.assertFalse((out / 'observation.json').exists())

    def test_shot_boundaries_are_bounded_and_adjacent(self):
        value = inventory()
        value['shots'] = [dict(first_frame=1, last_frame=2, description='First.', assets=['A1']),
                          dict(first_frame=3, last_frame=4, description='Second.', assets=['A1'])]
        qwen = Mock(model_id='test-model', side_effect=[dict(text=json.dumps(v)) for v in (value, empty_attributes('object'), empty_attributes('object'))])
        with VideoObserver(qwen=qwen) as observer:
            result = observer.observe(self.video, self.root / 'shots', frames=4, progress=False)
        first, second = result['shots']
        self.assertEqual(first['end'], second['start'])
        self.assertLess(0, first['end'])
        self.assertLess(first['end'], 1.0)
        self.assertEqual(result['target_order'], ['S1', 'S2'])


if __name__ == '__main__':
    unittest.main()
