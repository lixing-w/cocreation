import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cocreation import Flux, MiniMaxH3, Qwen, Reference, Runtime, H3Runtime
from cocreation.h3 import build_graph, model_files
from cocreation.worker import qwen_messages


class Contracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        env_patch = patch.dict(os.environ, {"COCREATION_SCRATCH": temporary.name})
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def test_numbering_preserves_order_per_kind(self):
        refs = [Reference("v4", "4.mp4", "video"), Reference("i7", "7.png", "image"),
                Reference("v5", "5.mp4", "video")]
        request = Qwen()("Video 99 stays literal; follow {{artifact:v5}} and {{artifact:v4}}", refs, dry_run=True)
        self.assertEqual(request["prompt"], "Video 99 stays literal; follow <Video 2> and <Video 1>")
        content = qwen_messages(request)[0]["content"]
        self.assertEqual(content[0]["text"], "<Image 1>")
        self.assertEqual(content[2]["text"], "Video 99 stays literal; follow <Video 2>")
        self.assertEqual(content[4]["text"], " and <Video 1>")
        self.assertEqual(content[3]["video"], request["bindings"]["v5"]["path"])
        self.assertEqual(content[5]["video"], request["bindings"]["v4"]["path"])
        self.assertEqual(content[5]["max_frames"], 16)

    def test_bad_references(self):
        ref = Reference("x", "x.png", "image")
        for prompt, refs in [("{{artifact:missing}}", [ref]), ("x", [ref, ref]), ("{{artifact:x}", [ref])]:
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                Qwen()(prompt, refs, dry_run=True)

    def test_flux_rejects_video(self):
        with self.assertRaises(ValueError):
            Flux()("x", [Reference("v", "v.mp4", "video")], dry_run=True)

    def test_h3_routing_labels_and_graph(self):
        refs = [Reference("v", "v.mp4", "video"), Reference("i", "i.png", "image"),
                Reference("a", "a.wav", "audio")]
        for selected, variant in [([], "fl2va"), (refs, "ref2va")]:
            prompt = "x" if not selected else "{{artifact:i}} moves like {{artifact:v}} with {{artifact:a}}"
            request = MiniMaxH3()(prompt, selected, seconds=5, dry_run=True)
            self.assertEqual(request["options"]["variant"], variant)
            self.assertEqual(request["options"]["frames"], 124)
            graph = build_graph(request, [Path(r.path).name for r in selected])
            self.assertIn(variant + "_pruned_fp8_scaled", graph["model"]["inputs"]["unet_name"])
            self.assertEqual(graph["video"]["inputs"]["audio"], ["decode_audio", 0])
            if selected:
                self.assertEqual(request["prompt"], "<Picture 1> moves like <Video 1> with <Audio 1>")
                self.assertEqual(graph["condition"]["inputs"]["ref_videos.ref_video_0"], ["media_0_components", 0])
                self.assertEqual(graph["condition"]["inputs"]["ref_video_audios.ref_video_audio_0"], ["media_0_components", 1])
                audio_graph = build_graph(request, [Path(r.path).name for r in selected], [{"video_audio_used": True}])
                self.assertIn("<Audio 2>", audio_graph["condition"]["inputs"]["prompt"])
            self.assertEqual(len(model_files(variant)), 4)

    def test_h3_input_limits(self):
        with self.assertRaises(ValueError):
            MiniMaxH3()("x", seconds=0, dry_run=True)
        with self.assertRaises(ValueError):
            MiniMaxH3()("x", [Reference("a", "a.wav", "audio")], dry_run=True)
        with self.assertRaises(ValueError):
            MiniMaxH3()("x", [Reference(str(i), "a.png", "image") for i in range(10)], dry_run=True)
        with self.assertRaises(ValueError):
            Qwen()("x", [Reference("a", "a.wav", "audio")], dry_run=True)

    def test_dry_run_has_no_side_effects_or_heavy_imports(self):
        script = '''
import sys
from unittest.mock import patch
from cocreation import Qwen, Flux, MiniMaxH3
with patch('subprocess.run', side_effect=AssertionError('spawned')):
    for cls in (Qwen, Flux, MiniMaxH3):
        assert cls()('hello', dry_run=True)['status'] == 'planned'
assert not {'torch','transformers','diffusers','qwen_vl_utils'} & set(sys.modules)
'''
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_runtime_validation(self):
        with self.assertRaises(ValueError):
            Runtime(device="mps")
        with self.assertRaises(ValueError):
            Runtime(device="cpu")
        self.assertEqual(Runtime(device="cpu", dtype="float32").device, "cpu")

    def test_missing_media_fails_before_inference(self):
        with patch("subprocess.run", side_effect=AssertionError("spawned")):
            with self.assertRaises(FileNotFoundError):
                Qwen()("x", [Reference("absent", "/no/such/media.png", "image")])

    def test_worker_job_roundtrip_and_failure_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fake_call(session, job, timeout):
                request = json.loads((job / 'request.json').read_text())
                self.assertEqual(request['prompt'], 'hello')
                (job / 'result.json').write_text(json.dumps(dict(status='ok', text='answer')))
            with patch('cocreation.session.ProcessSession.call', autospec=True, side_effect=fake_call):
                with Qwen(output_dir=tmp) as model:
                    result = model('hello')
                    session = model._session
                    model('hello')
                    self.assertIs(model._session, session)
            self.assertEqual(result['text'], 'answer')
            self.assertTrue(Path(result['request_path']).is_file())
            with patch('cocreation.session.ProcessSession.call', side_effect=RuntimeError('worker failed')):
                with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                    Qwen(output_dir=tmp)('hello')

    def test_h3_worker_command_and_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fake_call(session, job, timeout):
                self.assertEqual(session.backend, 'minimax_h3')
                (job / 'result.json').write_text(json.dumps(dict(status='ok', kind='video')))
            with patch('cocreation.models.sys.platform', 'linux'), \
                 patch('cocreation.session.ProcessSession.call', autospec=True, side_effect=fake_call):
                with MiniMaxH3(runtime=H3Runtime(accept_model_license=True), output_dir=tmp) as model:
                    self.assertEqual(model('hello')['kind'], 'video')



if __name__ == "__main__":
    unittest.main()
