import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sample_download", ROOT / "scripts/download_samples.py")
samples = importlib.util.module_from_spec(spec)
spec.loader.exec_module(samples)


class Samples(unittest.TestCase):
    def test_only_selected_frames_are_used(self):
        names = ["DAVIS/JPEGImages/480p/bear/00001.jpg", "DAVIS/Annotations/480p/bear/00000.png",
                 "DAVIS/JPEGImages/480p/bear/00000.jpg", "DAVIS/JPEGImages/480p/bear/../outside.jpg"]
        self.assertEqual(samples.selected_frames(names, "bear", 1), ["DAVIS/JPEGImages/480p/bear/00000.jpg"])
        with self.assertRaises(ValueError):
            samples.selected_frames(names, "absent", 10)

    def test_download_dry_run_does_not_touch_network_or_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "not-created"
            with patch.object(sys, "argv", ["download_samples.py", "--dry-run", "--output-dir", str(dest)]), \
                 patch("urllib.request.urlopen", side_effect=AssertionError("network")), \
                 patch("subprocess.run", side_effect=AssertionError("ffmpeg")), \
                 patch("sys.stdout", new_callable=io.StringIO) as output:
                samples.main()
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["images"], 4)
            self.assertEqual(plan["sequences"], ["bear", "blackswan"])
            self.assertFalse(dest.exists())

    def test_local_archive_respects_limit_and_records_playback_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "local.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for i in range(30):
                    archive.writestr(f"DAVIS/JPEGImages/480p/bear/{i:05d}.jpg", b"test-frame")
            def fake_save(path, data):
                path.write_bytes(data)
            with patch.object(samples, "save_jpeg", side_effect=fake_save), \
                 patch.object(samples, "encode_video") as encode:
                results = samples.download_videos(root, ["bear"], 24, 24, "ffmpeg", archive_path)
            self.assertEqual(results[0]["frame_count"], 24)
            self.assertEqual(results[0]["duration_seconds"], 1)
            self.assertIn("not asserted", results[0]["timing_note"])
            self.assertEqual(encode.call_args.args[2:4], (24, 24))
            self.assertEqual(len(list((root / "davis/bear/frames").glob("*.jpg"))), 24)

    def test_example_run_fails_before_model_loading_on_missing_input(self):
        result = subprocess.run([sys.executable, str(ROOT / "examples/plan.py"), "--run",
                                 "--image", "/no/such/image.jpg"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("download_samples.py", result.stderr)


if __name__ == "__main__":
    unittest.main()
