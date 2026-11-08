import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cocreation import Flux, Qwen, FreeVideo, Runtime, VideoRuntime
from cocreation.storage import scratch_root, runtime_environment, ensure_scratch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("setup_scratch", ROOT / "scripts/setup_scratch.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class Storage(unittest.TestCase):
    def test_all_default_paths_use_override_without_creating_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = (Path(tmp) / "scratch").resolve()
            with patch.dict(os.environ, {"COCREATION_SCRATCH": str(target)}):
                for cls in (Qwen, Flux, FreeVideo):
                    self.assertEqual(cls().output_dir, target / "runs")
                self.assertEqual(Runtime().cache_dir, str(target / "cache/huggingface/hub"))
                self.assertEqual(VideoRuntime().root, str(target / "runtime/freevideo-runtime"))
                Qwen()("hello", dry_run=True)
                self.assertFalse(target.exists())

    def test_link_idempotent_and_conflicting_files_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            target = (Path(tmp) / "outside").resolve()
            link = setup.create_link(repo, target)
            self.assertEqual(setup.create_link(repo, target), link)
            with patch.dict(os.environ, {}, clear=True), patch("cocreation.storage.REPO_ROOT", repo):
                self.assertEqual(scratch_root(), target)
            with self.assertRaises(ValueError):
                setup.create_link(repo, Path(tmp) / "other")
            link.unlink()
            link.write_text("keep me")
            with self.assertRaises(ValueError):
                setup.create_link(repo, target)
            self.assertEqual(link.read_text(), "keep me")

    def test_runtime_caches_and_temp_files_redirected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"COCREATION_SCRATCH": tmp, "HF_HOME": "/old/home/cache"}):
            env = runtime_environment()
            for key in ("HF_HOME", "HF_HUB_CACHE", "TMPDIR", "TORCH_HOME", "PIP_CACHE_DIR", "MAMBA_ROOT_PREFIX"):
                self.assertTrue(Path(env[key]).is_relative_to(Path(tmp).resolve()))
                self.assertTrue(Path(env[key]).is_dir())

    def test_linux_does_not_silently_write_into_repo(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True), \
             patch("cocreation.storage.REPO_ROOT", Path(tmp)), patch("cocreation.storage.sys.platform", "linux"):
            with self.assertRaisesRegex(RuntimeError, "setup_scratch"):
                ensure_scratch()
            self.assertFalse((Path(tmp) / "scratch").exists())
