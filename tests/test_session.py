import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from cocreation import Qwen, Flux, MiniMaxH3, VDNH3, H3Runtime
from cocreation.session import ProcessSession
from cocreation.worker import run_qwen, run_flux


# Real IPC/process lifecycle, no weights and no GPU. Each job sees an incrementing
# state in the SAME OS process; stdout noise must not corrupt the reply channel.
FAKE = '''
import json, os, sys, time
from pathlib import Path
from multiprocessing.connection import Connection
connection=Connection(int(sys.argv[1]))
count=0
while True:
    try: message=json.loads(connection.recv_bytes())
    except EOFError: break
    job=Path(message['job'])
    if job.name == 'timeout': time.sleep(30)
    if job.name == 'crash': os._exit(7)
    count+=1
    print('model diagnostic output', flush=True)
    (job/'result.json').write_text(json.dumps(dict(pid=os.getpid(), count=count)))
    connection.send_bytes(b'{"ok": true}')
'''


class SessionTests(unittest.TestCase):
    def fake_popen(self, command, **kwargs):
        return self.real_popen([sys.executable, '-u', '-c', FAKE, command[-2]], **kwargs)

    def setUp(self):
        self.real_popen = subprocess.Popen

    def test_actual_process_reuse_and_explicit_close(self):
        with tempfile.TemporaryDirectory() as tmp, patch('cocreation.session.subprocess.Popen', side_effect=self.fake_popen):
            session = ProcessSession('qwen', sys.executable, os.environ.copy(), Path(tmp)/'session')
            self.assertFalse(session.state)
            results=[]
            for name in ('one', 'two'):
                job=Path(tmp)/name; job.mkdir()
                session.call(job, 5)
                results.append(json.loads((job/'result.json').read_text()))
            process=session.state['process']
            self.assertEqual(results[0]['pid'], results[1]['pid'])
            self.assertEqual([r['count'] for r in results], [1, 2])
            session.close(); session.close()
            self.assertIsNotNone(process.poll())

    def test_deletion_releases_worker(self):
        with tempfile.TemporaryDirectory() as tmp, patch('cocreation.session.subprocess.Popen', side_effect=self.fake_popen):
            session=ProcessSession('qwen', sys.executable, os.environ.copy(), Path(tmp)/'session')
            job=Path(tmp)/'job'; job.mkdir(); session.call(job, 5)
            process=session.state['process']
            del session
            gc.collect()
            self.assertIsNotNone(process.poll())

    def test_timeout_and_crash_cleanup(self):
        for name in ('timeout','crash'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp, \
                 patch('cocreation.session.subprocess.Popen', side_effect=self.fake_popen):
                session=ProcessSession('qwen', sys.executable, os.environ.copy(), Path(tmp)/'session')
                job=Path(tmp)/name; job.mkdir()
                with self.assertRaises((TimeoutError, EOFError, ConnectionError)):
                    session.call(job, .1)
                self.assertFalse(session.state)

    def test_all_four_objects_own_independent_sessions(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'COCREATION_SCRATCH': tmp}), \
             patch('sys.platform', 'linux'):
            def call(session, job, timeout):
                (job/'result.json').write_text('{"status":"ok"}')
            with patch('cocreation.session.ProcessSession.call', autospec=True, side_effect=call):
                models=[Qwen(), Flux(), MiniMaxH3(runtime=H3Runtime(accept_model_license=True)), VDNH3()]
                sessions=[]
                for model in models:
                    model('first'); original=model._session
                    model.generate('second'); self.assertIs(original, model._session)
                    sessions.append(original)
                self.assertEqual(len({id(s) for s in sessions}), 4)
                for model in models:
                    previous=model._session
                    model.close(); self.assertFalse(previous.finalizer.alive)
                    model('new'); self.assertIsNot(model._session, previous)
                    model.close()

    def test_qwen_load_once_and_fresh_messages(self):
        torch=MagicMock(); transformers=MagicMock(); utils=MagicMock()
        model=transformers.Qwen3VLForConditionalGeneration.from_pretrained.return_value.eval.return_value
        model.generate.return_value=[[1,2,3]]
        processor=transformers.AutoProcessor.from_pretrained.return_value
        inputs=MagicMock(); inputs.input_ids=MagicMock(); inputs.input_ids.shape=[1,2]
        inputs.input_ids.__iter__.side_effect=lambda: iter([[1,2]])
        processor.return_value.to.return_value=inputs
        processor.batch_decode.return_value=['ok']
        utils.process_vision_info.return_value=(None,None,{})
        request=Qwen()('first',dry_run=True)
        state={}
        with patch.dict(sys.modules, {'torch':torch,'transformers':transformers,'qwen_vl_utils':utils}):
            run_qwen(request, Path('/tmp'), state)
            request['prompt']='second'
            run_qwen(request, Path('/tmp'), state)
        transformers.Qwen3VLForConditionalGeneration.from_pretrained.assert_called_once()
        transformers.AutoProcessor.from_pretrained.assert_called_once()
        messages=[c.args[0] for c in processor.apply_chat_template.call_args_list]
        self.assertEqual([len(m) for m in messages],[1,1])
        self.assertEqual(messages[1][0]['content'][-1]['text'],'second')
        self.assertNotIn('first', str(messages[1]))

    def test_worker_routes_opt_in_prefix_cache(self):
        torch=MagicMock(); transformers=MagicMock(); utils=MagicMock()
        processor=transformers.AutoProcessor.from_pretrained.return_value
        inputs=MagicMock(); inputs.input_ids=MagicMock(); inputs.input_ids.shape=[1,2]
        inputs.input_ids.__iter__.side_effect=lambda: iter([[1,2]])
        processor.return_value.to.return_value=inputs
        processor.batch_decode.return_value=['ok']
        utils.process_vision_info.return_value=(None,None,{})
        request=Qwen()('user',system_prompt='rules',system_cache_path='/tmp/prefix',dry_run=True)
        info={'status':'memory','prefix_tokens':1,'path':'/tmp/prefix'}
        with patch.dict(sys.modules, {'torch':torch,'transformers':transformers,'qwen_vl_utils':utils}), \
             patch('cocreation.qwen_prefix_cache.generate_with_prefix',return_value=([[1,2,3]],info)) as generate:
            state={}
            result=run_qwen(request,Path('/tmp'),state)
            self.assertIs(generate.call_args.args[-1],state)
            self.assertEqual(result['system_cache'],info)
            self.assertEqual(result['output_tokens'],1)
            transformers.Qwen3VLForConditionalGeneration.from_pretrained.return_value.eval.return_value.generate.assert_not_called()

    def test_flux_load_once(self):
        torch=MagicMock(); diffusers=MagicMock(); pil=MagicMock()
        request=Flux()('first', dry_run=True); state={}
        with patch.dict(sys.modules, {'torch':torch,'diffusers':diffusers,'PIL':pil}):
            run_flux(request, Path('/tmp'), state)
            run_flux(request, Path('/tmp'), state)
        diffusers.Flux2KleinPipeline.from_pretrained.assert_called_once()

    def test_official_video_metadata_forwarded_without_resampling(self):
        from cocreation import Reference
        from cocreation.worker import qwen_messages
        torch=MagicMock(); transformers=MagicMock(); utils=MagicMock()
        model=transformers.Qwen3VLForConditionalGeneration.from_pretrained.return_value.eval.return_value
        model.generate.return_value=[[1,2,3]]
        processor=transformers.AutoProcessor.from_pretrained.return_value
        inputs=MagicMock(); inputs.input_ids=MagicMock(); inputs.input_ids.shape=[1,2]
        inputs.input_ids.__iter__.side_effect=lambda: iter([[1,2]])
        processor.return_value.to.return_value=inputs
        processor.batch_decode.return_value=['{"timestamp_seconds": 0.6}']
        video=object(); metadata={'fps':24., 'frames_indices':[0,12,24,36], 'total_num_frames':48}
        utils.process_vision_info.return_value=(None,[(video,metadata)],{'do_sample_frames':False})
        request=Qwen()('Locate the subject', [Reference('v','video.mp4','video')],
                       video_fps=2.,video_max_frames=24,dry_run=True)
        with patch.dict(sys.modules, {'torch':torch,'transformers':transformers,'qwen_vl_utils':utils}):
            run_qwen(request, Path('/tmp'), {})
        media=qwen_messages(request)[0]['content'][1]
        self.assertEqual((media['type'],media['fps'],media['max_frames']),('video',2.,24))
        self.assertTrue(utils.process_vision_info.call_args.kwargs['return_video_metadata'])
        kwargs=processor.call_args.kwargs
        self.assertEqual(kwargs['video_metadata'],[metadata])
        self.assertEqual(kwargs['videos'],[video])
        self.assertFalse(kwargs['do_sample_frames'])
        self.assertFalse(kwargs['do_resize'])
