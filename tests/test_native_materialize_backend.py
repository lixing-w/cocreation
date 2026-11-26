"""Run real materialization against the SGLang request contract without a GPU."""
import asyncio
from contextlib import closing, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from scripts.dialogue_generation.materialize import materialize, model_order
from scripts.dialogue_generation.media_queue import AssetEvent, ReviewRequest, ReviewResult, generation_time
from scripts.dialogue_generation.native_batch import background_vlm_loop, run_media
from scripts.dialogue_generation.sglang_media import SGLangMedia


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg required')
class NativeBackendTests(unittest.TestCase):
    def test_native_video_requests_apply_quality_before_any_export_transcode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root/'reference.png'
            image.touch()
            with closing(SGLangMedia(root, root, root)) as server, \
                 patch.object(server, 'call', side_effect=RuntimeError('capture request')) as request:
                for operation, refs in [('text_generation', []),
                                        ('video_edit', [SimpleNamespace(kind='image', path=image)])]:
                    with self.assertRaisesRegex(RuntimeError, 'capture request'):
                        server.video('Prompt', refs, server.recipe(operation), root,
                                     dict(width=768, height=1344, seconds=4., seed=1))
                    payload = request.call_args.kwargs['json']
                    self.assertEqual(payload['output_compression'], 64)
                    self.assertEqual(payload['task'], 'ref2va' if refs else 't2va')

    def test_base_video_and_turbo_edit_use_sglang_parameters(self):
        base, edit = '[[video_0001]]', '[[video_0002]]'
        calls = []

        class Generator:
            recipe = staticmethod(SGLangMedia.recipe)

            def generate(self, backend, prompt, references, recipe, directory, options):
                self_args = dict(backend=backend, recipe=recipe, options=options, references=references)
                calls.append(self_args)
                directory.mkdir(parents=True)
                path = directory / 'generated.mp4'
                subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                    'color=blue:s=256x256:r=24:d=4', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)],
                    check=True)
                return dict(status='ok', path=str(path), model_id='MiniMaxAI/MiniMax-H3',
                            effective_parameters=dict(recipe, fps=24, frames=96))

        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            tasks = [
                dict(placeholder=base, kind='video', operation='text_generation', backend='minimax_h3',
                     prompt='A plain blue image', references=[], dependencies=[]),
                dict(placeholder=edit, kind='video', operation='video_edit', backend='minimax_h3',
                     prompt='Lighten <Video 1>', dependencies=[base],
                     references=[dict(kind='video', label='<Video 1>', placeholder=base)]),
            ]
            (run / 'placeholders.json').write_text(json.dumps(tasks))
            (run / 'dialogue.md').write_text(base + '\n' + edit)
            (run / 'observation.json').write_text('{}')
            result = materialize(run, shared_generator=Generator(), native_video_output=True,
                video_width=256, video_height=256, match_video_brightness=False, progress=False)

        self.assertEqual(result['status'], 'complete')
        self.assertEqual([call['options']['steps'] for call in calls], [50, 8])
        self.assertFalse(calls[0]['references'])
        self.assertEqual([ref.kind for ref in calls[1]['references']], ['video'])
        self.assertEqual([call['recipe']['turbo'] for call in calls], [False, True])
        for token in (base, edit):
            with self.subTest(token=token):
                parameters = result['media'][token]['effective_parameters']
                self.assertEqual(parameters['precision'], 'int8')
                self.assertNotIn('sampler', parameters)


class NativeQueueTests(unittest.TestCase):
    def test_buffer_withholds_blocked_work_across_waves_and_logs_remaining_counts(self):
        first, video, assembly, independent = (
            '[[image_0001]]', '[[video_0001]]', '[[video_0002]]', '[[image_0002]]')
        plans = {
            'a': {
                first: dict(model=('h3_image', 'H3-Image'), dependencies=[], plan_position=1),
                video: dict(model=('minimax_h3', 'H3', True), dependencies=[first], plan_position=2),
                assembly: dict(model=None, dependencies=[video], plan_position=3),
            },
            'b': {independent: dict(model=('h3_image', 'H3-Image'), dependencies=[], plan_position=1)},
        }
        observed = []
        generator = SimpleNamespace(model=None)
        def materializer(directory, **options):
            self.assertTrue(options['defer_retries'])
            case_id = directory.parent.name
            plan = plans[case_id]
            remaining = set(plan)
            attempts = {}
            token = yield plan
            while remaining:
                self.assertIn(token, remaining)
                self.assertFalse(remaining.intersection(plan[token]['dependencies']))
                observed.append((case_id, token))
                attempts[token] = attempts.get(token, 0) + 1
                operation = {first: 'local_edit', video: 'video_edit', assembly: 'concatenate',
                             independent: 'text_generation'}[token]
                if token != assembly:
                    generator.model = 'video-turbo' if token == video else 'image'
                reject = token in (first, video) and attempts[token] == 1
                path = 'generation_attempts/output.png' if reject else 'media/'+token.strip('[]')+'.mp4'
                event = AssetEvent(token, operation, path, 2.3, 'rejected' if reject else 'accepted')
                if not reject:
                    remaining.remove(token)
                if remaining:
                    token = yield event
            return dict(media=plan), event

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for case_id in plans:
                (root/case_id/'dialogue').mkdir(parents=True)
            cases = [SimpleNamespace(id=case_id, seed=1, width=256, height=256) for case_id in plans]
            settings = SimpleNamespace(run_dir=root, base_url=None, match_video_brightness=False)
            output, results = StringIO(), {}
            with patch('scripts.dialogue_generation.native_batch.materialization_session', materializer):
                with redirect_stdout(output):
                    run_media(cases, settings, generator, results)
            self.assertEqual([row['status'] for row in results.values()], ['complete', 'complete'])
            self.assertEqual(results['a']['completed_units'], 3)

        self.assertEqual(observed, [('a', first), ('b', independent), ('a', first),
                                    ('a', video), ('a', video), ('a', assembly)])
        lines = output.getvalue().splitlines()
        self.assertTrue(lines[0].startswith('[generation][T1][a:1/3]'))
        self.assertTrue(lines[0].endswith('(rejected) 2.3s [main remain:1][queued:3]'))
        self.assertTrue(lines[1].endswith('[main remain:0][queued:3]'))
        self.assertTrue(lines[3].startswith('[generation][T2][a:2/3]'))
        self.assertTrue(lines[3].endswith('[main remain:0][queued:2]'))
        self.assertTrue(lines[4].startswith('[generation][T3][a:2/3]'))
        self.assertEqual(lines[-1], '[generation][T3][a:3/3] [[video_0002]]: concatenate -> '
                         'media/video_0002.mp4 2.3s [main remain:0][queued:0]')

    def test_next_vlm_batch_overlaps_media_and_joins_deferred_work_at_drain(self):
        group_started, release_group = Event(), Event()
        events, dispatched = [], []
        first, child = 'parent', 'child'
        plans = {
            'a': {
                first: dict(model=('h3_image', 'H3-Image'), dependencies=[], plan_position=1),
                child: dict(model=('minimax_h3', 'H3', True), dependencies=[first], plan_position=2),
            },
            'b': {'independent': dict(model=('h3_image', 'H3-Image'), dependencies=[], plan_position=1)},
            'c': {'video': dict(model=('minimax_h3', 'H3', True), dependencies=[], plan_position=1)},
        }

        async def vlm(group, settings, *, isolate_failures, summary_path):
            ids = [case.id for case in group]
            self.assertTrue(isolate_failures)
            if ids == ['a', 'b']:
                self.assertEqual(summary_path.name, '001.json')
                return {}
            self.assertEqual(ids, ['c', 'd'])
            self.assertEqual(summary_path.name, '002.json')
            group_started.set()
            while not release_group.is_set():
                await asyncio.sleep(0.001)
            return {'d': 'dialogue failed'}

        def materializer(directory, **options):
            case_id = directory.parent.name
            plan = plans[case_id]
            remaining, rejected = set(plan), set()
            token = yield plan
            while remaining:
                self.assertFalse(remaining.intersection(plan[token]['dependencies']))
                self.assertTrue(group_started.wait(timeout=5), 'VLM prefetch must start before media')
                dispatched.append((case_id, token))
                reject = case_id == 'a' and token not in rejected
                if reject:
                    rejected.add(token)
                else:
                    remaining.remove(token)
                event = AssetEvent(token, 'generate', 'output', 1., 'rejected' if reject else 'accepted')
                if remaining:
                    token = yield event
            return dict(media=plan), event

        def state(phase, **data):
            events.append((phase, data))
            if phase == 'media_wave_drained' and data['wave'] == 1:
                self.assertTrue(group_started.is_set())
                self.assertEqual(data['buffered'], 2)
                self.assertEqual(data['completed'], ['b'])
                release_group.set()

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = [SimpleNamespace(id=cid, seed=1, width=256, height=256) for cid in 'abcd']
            for case in cases:
                (root / case.id / 'dialogue').mkdir(parents=True)
            settings = SimpleNamespace(run_dir=root, base_url=None, match_video_brightness=False, concurrency=2)
            results = {case.id: dict(status='queued_vlm') for case in cases}
            with patch('scripts.dialogue_generation.native_batch.run_cases', vlm), \
                 patch('scripts.dialogue_generation.native_batch.materialization_session', materializer), \
                 background_vlm_loop() as loop, redirect_stdout(StringIO()):
                run_media(cases, settings, SimpleNamespace(model='image'), results, vlm_loop=loop, state=state)

        self.assertEqual([row['status'] for row in results.values()], ['complete', 'complete', 'complete', 'failed'])
        self.assertEqual(dispatched, [('a', first), ('b', 'independent'), ('a', first),
                                      ('a', child), ('c', 'video'), ('a', child)])
        self.assertEqual([(phase, data['batch']) for phase, data in events], [
            ('vlm', 1), ('media_buffered', 1), ('media', 1), ('vlm', 2),
            ('media_wave_drained', 1), ('media_buffered', 2), ('media', 2),
            ('media_wave_drained', 2), ('media', None), ('media_wave_drained', None),
        ])
        self.assertEqual(events[5][1]['buffered'], 3)

    def test_async_review_releases_gpu_and_retries_before_unlocking_dependents(self):
        review_started, independent_started = Event(), Event()
        plan = {
            'parent': dict(model=('h3_image', 'H3-Image'), dependencies=[], plan_position=1),
            'child': dict(model=('h3_image', 'H3-Image'), dependencies=['parent'], plan_position=2),
            'unrelated': dict(model=('minimax_h3', 'H3', True), dependencies=[], plan_position=3),
        }
        dispatched, verdicts, waves = [], [], []

        def reject_after_independent_starts():
            review_started.set()
            if not independent_started.wait(timeout=5):
                raise AssertionError('Expected another dispatch while critique was pending')
            return 'rejected'

        def materializer(directory, **options):
            self.assertTrue(options['async_reviews'])
            request = yield plan
            while True:
                if isinstance(request, ReviewResult):
                    token, verdict = request.token, request.result()
                    verdicts.append((token, verdict))
                else:
                    token = request
                    dispatched.append(token)
                    verdict = 'accepted'
                    if token == 'parent':
                        check = (reject_after_independent_starts if dispatched.count(token) == 1
                                 else lambda: 'accepted')
                        request = yield ReviewRequest(token, check)
                        continue
                    if token == 'unrelated':
                        self.assertTrue(review_started.wait(timeout=5))
                        independent_started.set()
                    if token == 'child':
                        self.assertEqual(verdicts[-1], ('parent', 'accepted'))
                event = AssetEvent(token, 'generate', 'output', 1.0, verdict)
                if token == 'child':
                    return dict(media=plan), event
                request = yield event

        def state(phase, **data):
            if phase == 'media_wave_drained':
                waves.append(data)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'a'/'dialogue').mkdir(parents=True)
            settings = SimpleNamespace(run_dir=root, base_url=None, match_video_brightness=False)
            case = SimpleNamespace(id='a', seed=1, width=256, height=256)
            results = {}
            with patch('scripts.dialogue_generation.native_batch.materialization_session', materializer), \
                 redirect_stdout(StringIO()):
                run_media([case], settings, SimpleNamespace(model='image'), results, state=state)
            timing = json.loads((root/'a'/'media_timings.json').read_text())

        self.assertEqual(dispatched, ['parent', 'unrelated', 'parent', 'child'])
        self.assertEqual(verdicts, [('parent', 'rejected'), ('parent', 'accepted')])
        self.assertEqual([row['buffered'] for row in waves], [2, 0])
        self.assertEqual(results['a']['status'], 'complete')
        self.assertEqual(results['a']['completed_units'], 3)
        self.assertGreater(timing['review'], 0.0)

    def test_resident_services_reuse_backbone_and_toggle_cached_adapter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video, image = Mock(), Mock()
            video.process.poll.return_value = image.process.poll.return_value = None
            response = MagicMock()
            response.__enter__.return_value.json.return_value = {'status': 'ok'}
            with closing(SGLangMedia(root, root, root, resident=True)) as server, \
                 patch('scripts.dialogue_generation.sglang_media.ProcessServer', side_effect=[video, image]) as launch, \
                 patch.object(server, 'call', return_value=response) as control, redirect_stdout(StringIO()):
                server.select('video-base')
                server.select('image')
                server.select('video-turbo')
                server.select('image')
                server.select('video-turbo')
                server.select('video-base')
                server.select('video-turbo')
                self.assertIs(server.server, video)
                self.assertEqual(server.loads, 2)
                self.assertEqual(launch.call_count, 2)
                self.assertNotIn('--lora-path', launch.call_args_list[0].args[0])
                video.close.assert_not_called()
                image.close.assert_not_called()
            video.close.assert_called_once()
            image.close.assert_called_once()
            self.assertEqual([call.args[1] for call in control.call_args_list],
                             ['/v1/set_lora', '/v1/unmerge_lora_weights', '/v1/set_lora'])
            first, _, last = [call.kwargs['json'] for call in control.call_args_list]
            self.assertEqual(first['lora_alpha'], 8)
            self.assertEqual(first['merge_mode'], 'dynamic')
            self.assertIn('lora_path', first)
            self.assertNotIn('lora_path', last)

    def test_nonresident_ref2va_toggles_adapter_without_reloading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video, image, new_video = Mock(), Mock(), Mock()
            for process in (video, image, new_video):
                process.process.poll.return_value = None
            response = MagicMock()
            response.__enter__.return_value.json.return_value = {'status': 'ok'}
            with closing(SGLangMedia(root, root, root)) as server, \
                 patch('scripts.dialogue_generation.sglang_media.ProcessServer',
                       side_effect=[video, image, new_video]) as launch, \
                 patch.object(server, 'call', return_value=response) as control, redirect_stdout(StringIO()):
                server.select('video-base')
                self.assertEqual(server.select('video-turbo'), 0.0)
                self.assertEqual(server.select('video-base'), 0.0)
                self.assertIs(server.server, video)
                self.assertEqual(launch.call_count, 1)
                video.close.assert_not_called()

                server.select('image')
                video.close.assert_called_once()
                server.select('video-turbo')
                image.close.assert_called_once()
                self.assertIs(server.server, new_video)
                self.assertEqual(launch.call_count, 3)
                self.assertEqual([call.args[1] for call in control.call_args_list],
                                 ['/v1/set_lora', '/v1/unmerge_lora_weights', '/v1/set_lora'])
                self.assertIn('lora_path', control.call_args.kwargs['json'])

    def test_next_wave_order_accounts_for_the_resident_model(self):
        units = {'video': dict(model='video-turbo', dependencies=[]),
                 'image': dict(model='image', dependencies=[])}
        self.assertEqual(model_order(units, current_model='image'), ['image', 'video'])

    def test_generation_log_time_excludes_model_reload(self):
        self.assertEqual(generation_time(dict(generation_seconds=12.0, model_reload_seconds=4.5)), 7.5)
        self.assertEqual(generation_time(dict(elapsed_seconds=2.3)), 2.3)

    def test_model_reload_logs_only_after_health_check_with_duration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with closing(SGLangMedia(root, root, root)) as server:
                process = Mock()
                process.process.poll.return_value = None
                with patch('scripts.dialogue_generation.sglang_media.free_ports', return_value=(8000, 8001, 8002)), \
                     patch('scripts.dialogue_generation.sglang_media.ProcessServer', return_value=process), \
                     patch('scripts.dialogue_generation.sglang_media.time.monotonic', side_effect=[4.0, 8.5]), \
                     patch('builtins.print') as printed:
                    process.wait_ready.side_effect = lambda client: printed.assert_not_called()
                    self.assertEqual(server.select('image'), 4.5)
                    self.assertEqual(server.select('image'), 0.0)
                    printed.assert_called_once()
                    self.assertTrue(printed.call_args.args[0].endswith('server.log 4.5s'))


if __name__ == '__main__':
    unittest.main()
