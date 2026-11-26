"""Ordered conversation contracts and real HTTP concurrency, without a GPU model."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from cocreation import Qwen, VLLMQwen, Runtime
from cocreation.qwen_client import MODEL_ID
from tools.dialogue_models import resolve_model, OLLAMA_INSTRUCT, DEFAULT_LOCAL_MODEL


class StructuredMessages(unittest.TestCase):
    def test_multiturn_media_order_repetition_and_input_immutability(self):
        messages = [dict(role='system', content='rules'),
            dict(role='user', content=[dict(type='text', text='before'),
                dict(type='image', image='image.png'), dict(type='text', text='middle'),
                dict(type='video', video='video.mp4'), dict(type='image', image='image.png')]),
            dict(role='assistant', content='earlier answer'), dict(role='user', content='follow-up')]
        original = json.loads(json.dumps(messages))
        request = Qwen().chat(messages, dry_run=True)
        result = request['conversations'][0]
        self.assertEqual([m['role'] for m in result], ['system','user','assistant','user'])
        self.assertEqual([x['type'] for x in result[1]['content']], ['text','image','text','video','image'])
        self.assertEqual(result[1]['content'][1]['image'], result[1]['content'][4]['image'])
        self.assertEqual(messages, original)

    def test_batch_is_recorded_and_missing_media_fails_before_worker(self):
        conversations = [[dict(role='user', content='one')], [dict(role='user', content='two')]]
        result = Qwen().generate_batch(conversations, dry_run=True)
        self.assertTrue(result['batched'])
        self.assertEqual(result['conversations'], conversations)
        with self.assertRaises(FileNotFoundError):
            Qwen().generate_batch([[dict(role='user', content=[dict(type='image', image='/missing/file.png')])]])

    def test_empty_audio_and_ambiguous_inputs_are_rejected(self):
        for messages in ([], [dict(role='user', content=[dict(type='audio', audio='a.wav')])]):
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                Qwen().chat(messages, dry_run=True)
        with self.assertRaises(ValueError):
            Qwen()('prompt', messages=[dict(role='user', content='other')], dry_run=True)
        with self.assertRaises(ValueError):
            Qwen().generate_batch([], dry_run=True)

    def test_server_default_does_not_change_mac_native_or_ollama_default(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch('cocreation.models.sys.platform', 'darwin'):
                self.assertEqual(Qwen().model_id, 'Qwen/Qwen3-VL-8B-Instruct')
            with patch('cocreation.models.sys.platform', 'linux'):
                self.assertEqual(Qwen().model_id, DEFAULT_LOCAL_MODEL)
        self.assertEqual(resolve_model('ollama'), (OLLAMA_INSTRUCT, 'instruct'))
        self.assertEqual(resolve_model('vllm', 'qwen3.6-35b-a3b'), (MODEL_ID, 'instruct'))

    def test_scratch_native_runtime_is_linux_only_and_explicit_runtime_wins(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'COCREATION_SCRATCH':tmp}):
            executable = Path(tmp)/'runtime/qwen3.6-35b-a3b/vllm/bin/python'
            executable.parent.mkdir(parents=True)
            executable.touch()
            (executable.parents[2]/'model.json').write_text(json.dumps(
                dict(model_id=DEFAULT_LOCAL_MODEL, revision='pinned-revision')))
            with patch('cocreation.models.sys.platform', 'linux'):
                self.assertEqual(Qwen().runtime.python, str(executable.resolve()))
                self.assertEqual(Qwen().runtime.revision, 'pinned-revision')
                self.assertTrue(Qwen().runtime.local_files_only)
                self.assertIsNone(Qwen('other/model').runtime.revision)
                explicit = Runtime(python='/custom/python')
                self.assertIs(Qwen(runtime=explicit).runtime, explicit)
            with patch('cocreation.models.sys.platform', 'darwin'):
                self.assertNotEqual(Qwen().runtime.python, str(executable.resolve()))
                self.assertIsNone(Qwen().runtime.revision)

    def test_schema_is_recorded_and_cannot_mix_with_custom_prefix_cache(self):
        schema = dict(type='object', properties=dict(value=dict(type='string')))
        request = Qwen().generate_batch([[dict(role='user', content='JSON')]],
                                       response_schema=schema, dry_run=True)
        self.assertEqual(request['options']['response_schema'], schema)
        with self.assertRaisesRegex(ValueError, 'Schema decoding'):
            Qwen('Qwen/Qwen3-VL-8B-Instruct')('JSON', response_schema=schema, system_prompt='rules',
                   system_cache_path='/tmp/cache', dry_run=True)


class ServingHTTP(unittest.TestCase):
    def test_order_schema_per_request_seed_and_concurrent_requests(self):
        received = []
        barrier = threading.Barrier(2)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                received.append((self.path, body))
                barrier.wait(timeout=5)  # A serialized client would fail this test.
                text = body['messages'][-1]['content']
                payload = json.dumps(dict(choices=[dict(message=dict(content=text), finish_reason='stop')],
                    usage=dict(prompt_tokens=5, completion_tokens=2))).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                client = VLLMQwen(base_url=f'http://127.0.0.1:{server.server_port}/v1', output_dir=tmp, enable_thinking=True)
                schema = dict(type='object', properties=dict(value=dict(type='string')), required=['value'], additionalProperties=False)
                result = client.generate_batch([[dict(role='user', content='ALPINE')],
                    [dict(role='user', content='TOKYO')]], seed=10, response_schema=schema, concurrency=2)
                self.assertEqual([r['text'] for r in result['results']], ['ALPINE','TOKYO'])
                self.assertEqual({body['seed'] for _,body in received}, {10,11})
                self.assertEqual({path for path,_ in received}, {'/v1/chat/completions'})
                self.assertTrue(all(body['response_format']['json_schema']['schema'] == schema for _,body in received))
                self.assertTrue(all(body['thinking_token_budget'] == 512 for _,body in received))
                for row in result['results']:
                    self.assertEqual(row['thinking_token_budget'], 512)
                    self.assertTrue(Path(row['request_path']).is_file())
                    self.assertTrue((Path(row['job_dir'])/'response.json').is_file())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_thinking_defaults_off_and_is_explicitly_overridable(self):
        messages = [dict(role='user', content='Hello')]
        client = VLLMQwen()
        self.assertEqual(client.chat(messages, dry_run=True)['request']['chat_template_kwargs'],
                         dict(enable_thinking=False))
        self.assertEqual(client.chat(messages, dry_run=True, enable_thinking=True)['request']['chat_template_kwargs'],
                         dict(enable_thinking=True))
        self.assertFalse(client.enable_thinking)
        with self.assertRaises(TypeError):
            client.chat(messages, dry_run=True, enable_thinking='false')
        with self.assertRaises(ValueError):
            resolve_model('vllm', 'qwen3-vl-30b-instruct')

    def test_thinking_budget_reserves_final_output_and_is_overridable(self):
        client = VLLMQwen(enable_thinking=True)
        messages = [dict(role='user', content='Return JSON.')]
        for total, expected in ((512, 256), (4096, 2048), (8192, 2048)):
            body = client.chat(messages, dry_run=True, max_new_tokens=total)['request']
            self.assertTrue(body['chat_template_kwargs']['enable_thinking'])
            self.assertEqual(body['thinking_token_budget'], expected)
            self.assertEqual(body['max_tokens'], total)
        body = client.chat(messages, dry_run=True, max_new_tokens=8192, thinking_token_budget=1024)['request']
        self.assertEqual(body['thinking_token_budget'], 1024)
        self.assertNotIn('thinking_token_budget', client.chat(messages, dry_run=True, enable_thinking=False)['request'])
        for invalid in (-1, 8192, True, 1.5, '1024'):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'thinking_token_budget'):
                client.chat(messages, dry_run=True, max_new_tokens=8192, thinking_token_budget=invalid)
        with self.assertRaisesRegex(ValueError, 'enable_thinking'):
            client.chat(messages, dry_run=True, enable_thinking=False, max_new_tokens=4096, thinking_token_budget=1024)

    def test_vllm_realizer_thinking_uses_template_option_without_prefill(self):
        from tools.dialogue_realizer import QwenRealizer
        from unittest.mock import Mock
        client = Mock(return_value=dict(text='{"ok":true}', finish_reason='stop'))
        with patch('cocreation.VLLMQwen', return_value=client):
            model = QwenRealizer(backend='vllm', model_mode='thinking')
            text, finish = model._generate_once('review', 1, system='rules',
                max_tokens=512, stage='verifier', response_schema={'type':'object'})
        self.assertEqual(text, '{"ok":true}')
        self.assertEqual(finish, 'stop')
        self.assertIsNone(client.call_args.kwargs['assistant_prefill'])
        self.assertTrue(client.call_args.kwargs['enable_thinking'])

    def test_video_is_a_video_block_and_media_order_survives_transport(self):
        result = VLLMQwen().chat([dict(role='user', content=[
            dict(type='text', text='before'), dict(type='image', image='/tmp/a.png'),
            dict(type='text', text='between'), dict(type='video', video='/tmp/b.mp4'),
            dict(type='text', text='after')])], dry_run=True)
        body = result['request']
        self.assertEqual([x['type'] for x in body['messages'][0]['content']],
            ['text','image_url','text','video_url','text'])
        self.assertEqual(body['messages'][0]['content'][3]['video_url']['url'], Path('/tmp/b.mp4').resolve().as_uri())
        self.assertEqual(body['mm_processor_kwargs']['fps'], 1.0)
        self.assertFalse(body['mm_processor_kwargs']['do_sample_frames'])
        from cocreation.qwen_inputs import DEFAULT_VIDEO_MAX_FRAMES
        self.assertEqual(body['mm_processor_kwargs']['max_frames'], DEFAULT_VIDEO_MAX_FRAMES)
        with self.assertRaisesRegex(ValueError, 'fixed at server launch'):
            VLLMQwen().chat([dict(role='user',content=[dict(type='video',video='/tmp/b.mp4')])],
                           dry_run=True, video_fps=0.5, video_max_frames=8)


if __name__ == '__main__':
    unittest.main()
