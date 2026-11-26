import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

from scripts.qwen import serve
from tools.media_review import reviewer_session
from scripts.dialogue_generation.materialize import crop_generated_reference
from cocreation.qwen_client import MODEL_ID

class PersistentVLMTests(unittest.TestCase):
    def test_explicit_kv_budget_reaches_engine_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);runtime=root/'runtime/qwen3.6-35b-a3b';runtime.mkdir(parents=True)
            checkpoint=root/'checkpoint';checkpoint.mkdir()
            (runtime/'model.json').write_text(json.dumps(dict(model_id=MODEL_ID,path=str(checkpoint))))
            cuda=root/'cuda';(cuda/'bin').mkdir(parents=True);(cuda/'bin/nvcc').touch()
            arguments=['serve.py','--cuda-home',str(cuda),'--kv-cache-memory-gib','16']
            with patch.object(serve,'scratch_root',return_value=root),patch.object(serve,'runtime_environment',return_value={'PATH':'/bin'}),patch('sys.argv',arguments),patch.object(serve.os,'execve') as execute:
                serve.main()
            command=execute.call_args.args[1]
            self.assertEqual(command[command.index('--kv-cache-memory-bytes')+1],str(16*1024**3))
            self.assertEqual(json.loads((runtime/'server.json').read_text())['kv_cache_memory_bytes'],16*1024**3)

    def test_invalid_kv_budget_rejected_before_runtime_start(self):
        for value in ['0','-1','nan','inf']:
            with self.subTest(value=value),patch('sys.argv',['serve.py','--kv-cache-memory-gib',value]),self.assertRaises(SystemExit):
                serve.main()

    def test_missing_shared_reviewer_never_spawns_another_model(self):
        with tempfile.TemporaryDirectory() as temp,patch('urllib.request.urlopen',side_effect=OSError('offline')),patch('tools.media_review.subprocess.Popen') as start:
            with self.assertRaisesRegex(RuntimeError,'Shared review service is unavailable'):
                with reviewer_session(temp,require_running=True):pass
            start.assert_not_called()

    def test_attached_shared_reviewer_is_not_stopped(self):
        response=Mock();response.__enter__=Mock(return_value=Mock(status=200));response.__exit__=Mock(return_value=False)
        client=Mock()
        with tempfile.TemporaryDirectory() as temp,patch('urllib.request.urlopen',return_value=response),patch('cocreation.VLLMQwen',return_value=client),patch('tools.media_review.subprocess.Popen') as start,patch('tools.media_review.os.killpg') as kill:
            with reviewer_session(temp,require_running=True):pass
            start.assert_not_called();kill.assert_not_called();client.close.assert_called_once()

    def test_missing_shared_localizer_never_spawns_another_model(self):
        with tempfile.TemporaryDirectory() as temp,patch('urllib.request.urlopen',side_effect=OSError('offline')),patch('scripts.dialogue_generation.materialize.subprocess.Popen') as start:
            with self.assertRaisesRegex(RuntimeError,'Shared localization service is unavailable'):
                crop_generated_reference('input.mp4','a building',Path(temp)/'crop.png',require_running=True)
            start.assert_not_called()

if __name__=='__main__':unittest.main()
