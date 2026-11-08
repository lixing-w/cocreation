import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cocreation import Flux, FreeVideo, Qwen, Reference, Runtime, VideoRuntime
from cocreation.models import freevideo_media, FREEVIDEO_REVISION
from cocreation.worker import qwen_messages


class Contracts(unittest.TestCase):
    def test_numbering_preserves_order_per_kind(self):
        refs = [Reference("v4", "4.mp4", "video"), Reference("i7", "7.png", "image"),
                Reference("v5", "5.mp4", "video")]
        request = Qwen()("Video 99 stays literal; follow {{artifact:v5}} and {{artifact:v4}}", refs, dry_run=True)
        self.assertEqual(request["prompt"], "Video 99 stays literal; follow <Video 2> and <Video 1>")
        content = qwen_messages(request)[0]["content"]
        self.assertEqual([content[i]["text"] for i in (0, 2, 4)], ["<Video 1>", "<Image 1>", "<Video 2>"])
        self.assertEqual(content[1]["video"], request["bindings"]["v4"]["path"])
        self.assertEqual(content[1]["max_frames"], 16)

    def test_bad_references(self):
        ref = Reference("x", "x.png", "image")
        for prompt, refs in [("{{artifact:missing}}", [ref]), ("x", [ref, ref]), ("{{artifact:x}", [ref])]:
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                Qwen()(prompt, refs, dry_run=True)

    def test_flux_rejects_video(self):
        with self.assertRaises(ValueError):
            Flux()("x", [Reference("v", "v.mp4", "video")], dry_run=True)

    def test_video_tasks_and_media(self):
        image = Reference("i", "i.png", "image")
        last = Reference("j", "j.png", "image")
        video = Reference("v", "v.mp4", "video")
        for kwargs, task in [({}, "t2va"), ({"first": image}, "i2va"),
            ({"last": last}, "l2va"), ({"first": image, "last": last}, "fl2va"),
            ({"references": [video, image]}, "ref2va")]:
            request = FreeVideo()("x", dry_run=True, **kwargs)
            self.assertEqual(request["options"]["task"], task)
            media = freevideo_media(request)
            self.assertEqual(media["version"], 1)
            if task == "ref2va":
                self.assertEqual(media["references"][0]["sound"], False)
                self.assertNotIn("artifact_id", media["references"][0])
            if task == "fl2va":
                self.assertNotIn("references", media)
        with self.assertRaises(ValueError):
            FreeVideo()("x", [video], first=image, dry_run=True)

    def test_dry_run_has_no_side_effects_or_heavy_imports(self):
        script = '''
import sys
from unittest.mock import patch
from cocreation import Qwen, Flux, FreeVideo
with patch('subprocess.run', side_effect=AssertionError('spawned')):
    for cls in (Qwen, Flux, FreeVideo):
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
            def fake_run(command, **kwargs):
                job = Path(command[-1])
                request = json.loads((job / "request.json").read_text())
                self.assertEqual(request["prompt"], "hello")
                (job / "result.json").write_text(json.dumps(dict(status="ok", text="answer")))
            with patch("subprocess.run", side_effect=fake_run):
                result = Qwen(output_dir=tmp)("hello")
            self.assertEqual(result["text"], "answer")
            self.assertTrue(Path(result["request_path"]).is_file())
            with patch("subprocess.run", side_effect=subprocess.CalledProcessError(2, "worker")):
                with self.assertRaisesRegex(RuntimeError, "runtime.log"):
                    Qwen(output_dir=tmp)("hello")

    def test_freevideo_command_and_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source, root = base / "source", base / "runtime"
            source.mkdir()
            root.mkdir()
            (root / "machine.json").write_text('{"ready":true}')
            calls = []
            def fake_run(command, **kwargs):
                calls.append(command)
                output = Path(command[command.index("--out") + 1])
                output.write_bytes(b"fake video")
            rt = VideoRuntime(source_dir=str(source), root=str(root), auto_setup=False)
            with patch("cocreation.models.sys.platform", "linux"), \
                 patch("subprocess.check_output", return_value=FREEVIDEO_REVISION), \
                 patch("subprocess.run", side_effect=fake_run):
                result = FreeVideo(runtime=rt, output_dir=base / "jobs")("hello")
            self.assertEqual(result["kind"], "video")
            self.assertEqual(calls[0][calls[0].index("--task") + 1], "t2va")
            self.assertNotIn("--media", calls[0])


if __name__ == "__main__":
    unittest.main()
