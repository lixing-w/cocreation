import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cocreation import VDNH3, VDNRuntime, Reference
from cocreation.vdn import generation_command, media_request
from cocreation.vdn_assets import verify_fp8
from cocreation.vdn_setup import replace_once, patch_sources, environment
from cocreation.vdn_worker import validate_media, prepare_assets, stop


class VDNTests(unittest.TestCase):
    def test_multimodal_labels_and_two_pass(self):
        refs = [Reference('v', 'v.mp4', 'video'), Reference('a', 'a.wav', 'audio'),
                Reference('i', 'i.png', 'image'), Reference('i2', 'i2.png', 'image')]
        r = VDNH3()('{{artifact:i2}} follows {{artifact:v}} with {{artifact:a}}', refs, dry_run=True)
        self.assertEqual(r['prompt'], '<Picture 2> follows <Video 1> with <Audio 1>')
        self.assertEqual([r['options'][k] for k in ('precision', 'two_pass', 'base_steps', 'refine_steps')],
                         ['fp8', True, 8, 3])
        media = media_request(r)
        self.assertEqual(media['references'][0]['sound'], False)
        self.assertEqual([v['kind'] for v in media['references']], ['video', 'audio', 'image', 'image'])
        installation = dict(python='python', cache='cache', base='base', checkpoint='checkpoint',
                            comfy_root='comfy', model_paths='paths')
        command = generation_command(r, Path('/tmp/job with spaces'), installation)
        self.assertIn('--two-pass', command)
        self.assertEqual(command[command.index('--refine-steps') + 1], '3')
        self.assertEqual(command[command.index('--prompt-file') + 1], '/tmp/job with spaces/prompt.txt')

    def test_reject_non_fp8_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'manifest.json'
            for precision, fmt in [('int8', 'freevideo-int8-slim-v1'), ('fp8', 'wrong')]:
                path.write_text(json.dumps(dict(precision=precision, format=fmt)))
                with self.assertRaises(ValueError):
                    verify_fp8(folder)
            path.write_text(json.dumps(dict(precision='fp8', format='freevideo-fp8-slim-v1')))
            verify_fp8(folder)

    def test_no_silent_reference_trimming(self):
        request = VDNH3()('x', [Reference('v', 'v.mp4', 'video')], seconds=5, dry_run=True)
        data = json.dumps({'streams': [{'codec_type': 'video', 'duration': '10'}]})
        with patch('subprocess.check_output', return_value=data), self.assertRaisesRegex(ValueError, 'trim'):
            validate_media(request)

    def test_contract_validation_and_imports(self):
        for kwargs in ({'seconds': float('nan')}, {'width': 256}, {'two_pass': 'yes'}, {'seed': -1}):
            with self.assertRaises(ValueError):
                VDNH3()('x', dry_run=True, **kwargs)
        for kwargs in ({'attention': 'fa4'}, {'ram_gib': float('inf')}, {'ram_gib': 4}):
            with self.assertRaises(ValueError):
                VDNRuntime(**kwargs)
        script = "from cocreation import VDNH3; import sys; VDNH3()('x', dry_run=True); assert not {'torch','diffusers','transformers'} & set(sys.modules)"
        subprocess.run([sys.executable, '-c', script], check=True)

    def test_patch_is_idempotent_and_fails_on_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'code.py'
            path.write_text('old')
            replace_once(path, 'old', 'new')
            replace_once(path, 'old', 'new')
            self.assertEqual(path.read_text(), 'new')
            with self.assertRaises(RuntimeError):
                replace_once(path, 'missing', 'other')

    def test_environment_isolation(self):
        env = environment(Path('/tmp/vdn'), {'FREEVIDEO_RESIDENT_SESSION': 'stale', 'PYTHONPATH': 'bad'})
        self.assertNotIn('FREEVIDEO_RESIDENT_SESSION', env)
        self.assertEqual(env['FREEVIDEO_KEEP_MODELS'], 'off')
        self.assertEqual(env['PYTHONPATH'], '/tmp/vdn/vendor/FreeVideo')

    def test_missing_assets_do_not_download_without_setup(self):
        with tempfile.TemporaryDirectory() as tmp, patch('cocreation.vdn_worker.run') as run:
            with self.assertRaisesRegex(RuntimeError, 'assets not prepared'):
                prepare_assets(Path(tmp), 'python', {'auto_setup': False}, {})
            run.assert_not_called()

    def test_cleanup_targets_process_group(self):
        from unittest.mock import Mock
        process = Mock(pid=12345)
        with patch('os.killpg') as kill:
            stop(process)
        import signal
        self.assertEqual(kill.call_args_list[0].args, (12345, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[-1].args, (12345, signal.SIGKILL))

    def test_worker_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fake(command, **kw):
                self.assertEqual(command[1:3], ['-m', 'cocreation.vdn_worker'])
                job = Path(command[-1])
                (job / 'result.json').write_text(json.dumps(dict(status='ok', kind='video')))
            with patch.dict(os.environ, {'COCREATION_SCRATCH': tmp}), patch('sys.platform', 'linux'), patch('subprocess.run', side_effect=fake):
                self.assertEqual(VDNH3(output_dir=tmp)('x')['kind'], 'video')


if __name__ == '__main__':
    unittest.main()
