from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
from cocreation import MiniMaxH3, H3Runtime, VDNH3, VDNRuntime
from cocreation.comfy_worker import ComfySession
from cocreation.vdn_worker import NativeVDNSession, VDNSession


class VideoSessionTests(unittest.TestCase):
    def test_comfy_server_reused_and_outputs_are_job_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); entries=[]; exits=[]; home=[]
            @contextmanager
            def server(source, python, modelroot, job, timeout):
                entries.append(True); home.append(job)
                yield 'http://local', Mock(pid=123)
                exits.append(True)
            def api(base, route, payload=None):
                if route=='/object_info':
                    return {n['class_type']:{} for n in self.graph.values()}
                graph=payload['prompt']
                output=home[0]/'comfy_output'/graph['save']['inputs']['filename_prefix']
                output.parent.mkdir(parents=True); output.with_suffix('.mp4').write_bytes(b'video')
                return {'prompt_id':'id'}
            request=MiniMaxH3(runtime=H3Runtime(root=str(root),accept_model_license=True))('hi',dry_run=True)
            from cocreation.h3 import build_graph
            self.graph=build_graph(request, [])
            with patch('cocreation.comfy_worker.prepare_runtime',return_value=(root,root/'python')) as setup, \
                 patch('cocreation.comfy_worker.download_models',return_value=['model']) as download, \
                 patch('cocreation.comfy_worker.server',side_effect=server), \
                 patch('cocreation.comfy_worker.api',side_effect=api), \
                 patch('cocreation.comfy_worker.wait_result',return_value={}):
                session=ComfySession(); results=[]
                for name in ('one','two'):
                    job=root/name; job.mkdir()
                    results.append(session.execute(request,job))
                self.assertEqual(len(entries),1); self.assertFalse(exits)
                setup.assert_called_once(); download.assert_called_once()
                self.assertNotEqual(results[0]['path'],results[1]['path'])
                self.assertTrue(all(Path(r['path']).is_file() for r in results))
                session.close(); self.assertEqual(len(exits),1)

    def test_vdn_native_passes_same_owned_resident_endpoint(self):
        resident=types.ModuleType('freevideo_engine.resident_process')
        owner=Mock(); owner.start.return_value='/private/session/endpoint.json'
        resident.SessionOwner=Mock(return_value=owner)
        resident.ENV='FREEVIDEO_RESIDENT_SESSION'; resident.status=Mock(return_value={'models':['engine']})
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules', {'freevideo_engine.resident_process':resident}):
            root=Path(tmp)
            installation=dict(python='python',cache='cache',base='base',checkpoint='checkpoint',
                              comfy_root='comfy',model_paths='paths',model_revision='revision')
            request=VDNH3(runtime=VDNRuntime(root=str(root)))('hi',dry_run=True)
            def launch(command, **kwargs):
                self.assertEqual(kwargs['env'][resident.ENV],owner.start.return_value)
                self.assertEqual(kwargs['env']['FREEVIDEO_KEEP_MODELS'],'on')
                Path(command[command.index('--out')+1]).write_bytes(b'video')
                return Mock(wait=Mock(return_value=0))
            with patch('cocreation.vdn_worker.subprocess.Popen',side_effect=launch), \
                 patch('cocreation.vdn_worker.stop'):
                session=NativeVDNSession()
                for name in ('one','two'):
                    job=root/name; job.mkdir(); (job/'installation.json').write_text(json.dumps(installation))
                    result=session.execute(request,job)
                    self.assertEqual(result['resident']['models'],['engine'])
                resident.SessionOwner.assert_called_once()
                owner.close.assert_not_called()
                owner.finished.assert_not_called()  # no idle timer
                session.close(); owner.close.assert_called_once()
