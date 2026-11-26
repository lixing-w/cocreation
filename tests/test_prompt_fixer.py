"""Retry edits see the latest candidate without changing earlier reference labels."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.dialogue_generation.formal_sequence import generation_segments
from scripts.dialogue_generation.generation_retry import retry_edit_task
from tools.prompt_fixer import PromptFixer
from tools.video_frames import rate_indices


class FixerTests(unittest.TestCase):
    def test_image_fixer_sees_the_latest_image_and_preserves_original_label(self):
        calls = []
        replacement = ('Edit <Picture 2>. Give the city the woven textile material of <Picture 1>. '
                       'Preserve the framing and silhouette.')

        def model(**options):
            calls.append(deepcopy(options))
            return dict(text=json.dumps(dict(edit_prompt=replacement)), finish_reason='stop')

        task = dict(kind='image', backend='h3_image', operation='reference_generation', hidden='PLANNER_SECRET')
        first = dict(references=[dict(kind='image', label='<Picture 1>', path='approved.png')])
        prior = dict(attempt=2, path='latest.png', sha256='hash')
        edit = retry_edit_task(task, '[[image_0001]]', first, prior)
        with tempfile.TemporaryDirectory() as temp:
            media = Path(temp) / prior['path']
            media.touch()
            actual = PromptFixer(model).fix(edit,
                dict(current_user_request='Make a fabric city.', latest_rules=[], visible_sequence=[]),
                'A wooden city.', 'The city still looks wooden.', media=media, seed=1, output_dir=temp)

        self.assertEqual(edit['operation'], 'local_edit')
        self.assertEqual(edit['references'][:-1], first['references'])
        self.assertEqual(edit['references'][-1]['attempt'], 2)
        self.assertEqual(edit['references'][-1]['placeholder'], '[[image_0001_01]]')
        self.assertEqual(actual, replacement)
        self.assertNotIn('PLANNER_SECRET', str(calls))
        content = calls[0]['messages'][1]['content']
        self.assertEqual(content[-1], dict(type='image', image=str(media.resolve())))
        self.assertEqual(json.loads(content[0]['text'])['edit_source']['label'], '<Picture 2>')
        self.assertFalse(calls[0]['enable_thinking'])

    def test_video_fixer_requires_new_edit_source_label_and_samples_latest_video_at_one_fps(self):
        prompt = """subject_definitions:
The boat design is from <Picture 1>. <Video 1> is a motion reference. <Video 2> is the clip to edit.
summary: [video editing] Edit <Video 2> to recolor the upper railing black.
retention_analysis: Preserve all unaffected scene details, movement, and sound.
detailed_description: [Shot 1] The boat rail stays black as the camera moves.
overall_soundscape: Preserve the existing sound.
non_diegetic_music: Preserve the existing music."""
        attempts = []

        def model(**options):
            attempts.append(deepcopy(options))
            text = prompt.replace('<Video 2>', '<Video 1>') if len(attempts) == 1 else prompt
            return dict(text=json.dumps(dict(edit_prompt=text)), finish_reason='stop')

        first = dict(options=dict(seconds=4.0), references=[
            dict(kind='image', label='<Picture 1>', path='approved.png'),
            dict(kind='video', label='<Video 1>', path='reference.mp4')])
        previous = dict(attempt=2, path='latest.mp4', sha256='hash')
        task = retry_edit_task(dict(backend='minimax_h3', kind='video', operation='reference_generation', seconds=4.0),
                               '[[video_0001]]', first, previous)
        with tempfile.TemporaryDirectory() as temp:
            media = Path(temp) / previous['path']
            media.touch()
            revised = PromptFixer(model).fix(task, dict(current_user_request='Make the rail black.'),
                                             'Previous result had a white rail.', 'White rail visible.',
                                             media=media, seed=2, output_dir=temp)
        self.assertEqual(revised, prompt)
        self.assertEqual(task['references'][:-1], first['references'])
        self.assertEqual(task['references'][-1]['placeholder'], '[[video_0001_01]]')
        self.assertEqual(attempts[0]['messages'][1]['content'][-1],
                         dict(type='video', video=str(media.resolve()), fps=1.0))
        self.assertEqual(task['retry_seconds'], 4.0)
        self.assertEqual(len(attempts), 2)

    def test_public_retry_trace_reuses_critique_and_exports_only_visual_parameters(self):
        attempt = dict(attempt=2, backend='minimax_h3', kind='video', prompt='Edit <Video 2>.',
                       options=dict(seconds=4.0, width=768, height=1344, steps=8, turbo=True,
                                    h3_vae='mt', transformer='/checkpoints/int8', cfg=1.0),
                       references=[dict(kind='video', label='<Video 2>', path='previous.mp4', attempt=1,
                                        placeholder='[[video_0001]]', role='edit_source')])
        call, = generation_segments(attempt, '[[video_0001]]', {})
        self.assertEqual(call['type'], 'tool_call')
        args = call['tool_calls'][0]['arguments']
        self.assertEqual(args['parameters'], dict(seconds=4.0, ratio='9:16', res='768p'))
        self.assertEqual(args['references'][0]['attempt'], 1)
        self.assertNotIn('path', args['references'][0])
        third = dict(attempt, attempt=3, references=[dict(attempt['references'][0], attempt=2)])
        third_call, = generation_segments(third, '[[video_0001]]', {})
        ref, = third_call['tool_calls'][0]['arguments']['references']
        self.assertEqual(ref['placeholder'], '[[video_0001_01]]')
        initial = dict(attempt, attempt=1)
        self.assertEqual(generation_segments(initial, '[[video_0001]]', {})[0]['type'], 'thinking')

    def test_one_fps_samples_integer_seconds_instead_of_spreading_endpoints(self):
        ticks = [index/24 for index in range(107)]
        self.assertEqual(rate_indices(ticks, 1.0), [0, 24, 48, 72, 96])


if __name__ == '__main__':
    unittest.main()
