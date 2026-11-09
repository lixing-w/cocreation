import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from cocreation import MiniMaxH3, Reference
from cocreation.comfy_worker import download_models, prepare_runtime, server, stage_media, wait_result
from cocreation.h3 import COMFY_REVISION, H3Runtime, model_files


class ComfyWorkerTests(unittest.TestCase):
    def test_no_setup_in_offline_mode(self):
        with tempfile.TemporaryDirectory() as tmp, patch('cocreation.comfy_worker.run') as run:
            with self.assertRaisesRegex(RuntimeError, 'not prepared'):
                prepare_runtime(dict(root=tmp, auto_setup=True, local_files_only=True))
            run.assert_not_called()

    def test_download_only_selected_partition_with_offline_flag(self):
        with patch('cocreation.comfy_worker.run') as run:
            download_models('python', Path('/tmp/runtime'), 'ref2va', True)
        payload = json.loads(run.call_args.args[0][-1])
        self.assertTrue(payload[3])
        self.assertEqual(payload[4], model_files('ref2va'))
        self.assertFalse(any('fl2va' in x for x in payload[4]))

    def test_history_failure_and_timeout(self):
        process = Mock()
        process.poll.return_value = None
        with patch('cocreation.comfy_worker.api', return_value={'id': {'status': {'status_str': 'error'}}}):
            with self.assertRaisesRegex(RuntimeError, 'execution failed'):
                wait_result('http://localhost', 'id', process, 10)
        with self.assertRaises(TimeoutError):
            wait_result('http://localhost', 'id', process, 0)

    def test_server_terminated_on_api_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = Mock(pid=1234)
            process.poll.return_value = None
            with patch('cocreation.comfy_worker.subprocess.Popen', return_value=process), \
                 patch('cocreation.comfy_worker.api', return_value={}), \
                 patch('cocreation.comfy_worker.os.killpg') as kill, \
                 patch('cocreation.comfy_worker.socket.socket') as socket:
                socket.return_value.__enter__.return_value.getsockname.return_value = ('127.0.0.1', 12345)
                with self.assertRaisesRegex(ValueError, 'request failed'):
                    with server(root, root/'python', root, root, 1):
                        raise ValueError('request failed')
                kill.assert_called_once()
                process.wait.assert_called_once_with(timeout=15)

    def test_reference_video_resampling_and_no_silent_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root/'source.mp4'
            video.write_bytes(b'placeholder')
            request = MiniMaxH3()('x', [Reference('v', video, 'video')], seconds=5, dry_run=True)
            with patch('cocreation.comfy_worker.shutil.which', return_value='/bin/ffmpeg'), \
                 patch('cocreation.comfy_worker.subprocess.check_output', return_value='{"streams": [{"index": 1}]}'), \
                 patch('cocreation.comfy_worker.probe_media', return_value=3), \
                 patch('cocreation.comfy_worker.run') as run:
                names, metadata = stage_media(request, root/'input')
                command = run.call_args.args[0]
                self.assertNotIn('-an', command)
                self.assertIn('0:a:0?', command)
                self.assertIn('fps=24', command[command.index('-vf') + 1])
                self.assertTrue(metadata[0]['video_audio_used'])
                self.assertEqual(names, ['ref_0.mp4'])
            with patch('cocreation.comfy_worker.shutil.which', return_value='/bin/ffmpeg'), \
                 patch('cocreation.comfy_worker.subprocess.check_output', return_value='{"streams": [{"index": 1}]}'), \
                 patch('cocreation.comfy_worker.probe_media', return_value=10), \
                 patch('cocreation.comfy_worker.run') as run:
                with self.assertRaisesRegex(ValueError, 'exceeds output'):
                    stage_media(request, root/'input2')
                run.assert_not_called()
