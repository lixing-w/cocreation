"""Sampling reaches both backends; wording seeds never change the planned path."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.dialogue_generation.generate import run
from tools.dialogue_realizer import QwenRealizer
from tools.dialogue_verifier import CRITERIA

EXAMPLE = Path(__file__).resolve().parents[1] / 'scripts/dialogue_generation/examples/tokyo_vlog.json'


class Response:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self): return b'{"response":"ok"}'


class SamplingTests(unittest.TestCase):
    def setUp(self):
        report = dict(dict.fromkeys(CRITERIA, 2), critique="")
        mocked = patch("tools.dialogue_verifier.QwenVerifier.__call__", return_value=report)
        mocked.start()
        self.addCleanup(mocked.stop)

    def test_ollama_receives_explicit_default_sampling(self):
        model = QwenRealizer(backend="ollama")
        with patch('urllib.request.urlopen', return_value=Response()) as call:
            model._request('Only text', 123)
        options = json.loads(call.call_args.args[0].data)['options']
        self.assertEqual({key:options[key] for key in ['temperature','top_p','top_k','seed']},
                         dict(temperature=.9,top_p=.95,top_k=50,seed=123))

    def test_custom_sampling_reaches_ollama_and_local_adapter(self):
        model = QwenRealizer(backend="ollama", temperature=1.1,top_p=.9,top_k=80)
        with patch('urllib.request.urlopen', return_value=Response()) as call:
            model._request('Only text', 321)
        options = json.loads(call.call_args.args[0].data)['options']
        self.assertEqual({key:options[key] for key in ['temperature','top_p','top_k']},model.sampling)
        model.local = Mock(return_value={'text':'ok'})
        self.assertEqual(model._request('Only text',321),'ok')
        kwargs = model.local.call_args.kwargs
        self.assertEqual({key:kwargs[key] for key in model.sampling},model.sampling)
        self.assertEqual(kwargs['seed'],321)

    def test_local_uses_thinking_checkpoint_and_same_answer_prefill_as_ollama(self):
        prompt = "Speak as the User with this tone:\nCasual.\nDraft: Request an image of the bag."
        model = QwenRealizer(backend="ollama", model="qwen3-vl-30b-thinking")
        with patch('urllib.request.urlopen', return_value=Response()) as call:
            model._request(prompt, 11)
        body = json.loads(call.call_args.args[0].data)
        with patch('cocreation.Qwen') as factory:
            factory.return_value.return_value = {'text':'ok'}
            local = QwenRealizer(backend='local', model='qwen3.6-35b-a3b', model_mode='thinking')
            self.assertEqual(local._request(prompt, 11), 'ok')
            self.assertEqual(factory.call_args.kwargs['model_id'], 'Qwen/Qwen3.6-35B-A3B')
        prefix = local.local.call_args.kwargs['assistant_prefill']
        self.assertIn("dialogue act has been decided", prefix)
        self.assertTrue(prefix.endswith("</think>\n\n"))
        self.assertTrue(body['prompt'].endswith('<|im_start|>assistant\n' + prefix))
        self.assertEqual(local.local.call_args.kwargs['system_prompt'],
                         body['prompt'].split('<|im_start|>system\n',1)[1].split('<|im_end|>',1)[0])

    def test_invalid_sampling_fails_before_opening_a_model(self):
        for params in [dict(temperature=0),dict(temperature=float('nan')),dict(top_p=0),dict(top_p=1.1),dict(top_k=-1),dict(top_k=2.5)]:
            with self.subTest(params=params), self.assertRaises(ValueError):
                QwenRealizer(**params)

    def test_direct_rewrites_draw_fresh_seeds_and_log_retry_sampling(self):
        turn=dict(id=1,role='assistant',intent=dict(action='confirm_selection',description='Confirm the choice.'))
        with tempfile.TemporaryDirectory() as tmp:
            model=QwenRealizer(backend="ollama", output_dir=tmp)
            with patch('tools.dialogue_realizer.secrets.randbits',side_effect=[111,222]), patch.object(model,'_request',side_effect=['{}','Okay, we\'ll use that one.','Okay, we\'ll use that one.']) as call:
                model(turn,{})
                model(turn,{})
            rows=[json.loads(line) for line in (Path(tmp)/'calls.jsonl').read_text().splitlines()]
        self.assertEqual([c.args[1] for c in call.call_args_list],[111,112,222])
        self.assertEqual([r['seed'] for r in rows],[111,112,222])
        self.assertTrue(all(r['sampling']==model.sampling for r in rows))

    def test_default_wording_seeds_change_without_changing_the_plan(self):
        sequences=[]
        def writer(turn,media,previous,*,seed):
            sequences[-1].append(seed)
            return 'This is a test line.'
        with tempfile.TemporaryDirectory() as tmp, patch('scripts.dialogue_generation.generate.secrets.randbits',side_effect=[111,222]):
            plans=[]
            for i in range(2):
                sequences.append([])
                output=Path(tmp)/str(i)
                result=run(EXAMPLE,output,seed=2,realizer=writer,progress=False)
                plans.append(json.loads((output/'plan.json').read_text()))
                self.assertEqual(result['realization']['rewrite_seed'],[111,222][i])
        self.assertEqual(plans[0],plans[1])
        self.assertEqual(sequences[0][0],116)
        self.assertEqual(sequences[1][0],227)
        self.assertTrue(all(b-a==5 for a,b in zip(sequences[0],sequences[0][1:])))

    def test_explicit_wording_seed_reuses_the_same_sampling_sequence(self):
        sequences=[]
        def writer(turn,media,previous,*,seed):
            sequences[-1].append(seed)
            return 'This is a test line.'
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(2):
                sequences.append([])
                run(EXAMPLE,Path(tmp)/str(i),seed=2,rewrite_seed=77,realizer=writer,progress=False)
        self.assertEqual(sequences[0],sequences[1])

    def test_resume_restores_wording_seed_and_checks_sampling(self):
        interrupted_seeds,resumed_seeds=[],[]
        def interrupted(turn,media,previous,*,seed):
            interrupted_seeds.append(seed)
            if len(previous)==1:
                raise RuntimeError('test interruption')
            return 'First test line.'
        def continued(turn,media,previous,*,seed):
            resumed_seeds.append(seed)
            return 'A test line after resuming.'
        with tempfile.TemporaryDirectory() as tmp, patch('scripts.dialogue_generation.generate.secrets.randbits',return_value=111) as random_seed:
            output=Path(tmp)/'run'
            with self.assertRaisesRegex(RuntimeError,'interruption'):
                run(EXAMPLE,output,seed=2,realizer=interrupted,progress=False)
            with self.assertRaisesRegex(ValueError,'temperature'):
                run(EXAMPLE,output,seed=2,resume=True,temperature=1.1,realizer=continued,progress=False)
            with self.assertRaisesRegex(ValueError,'rewrite_seed'):
                run(EXAMPLE,output,seed=2,resume=True,rewrite_seed=222,realizer=continued,progress=False)
            result=run(EXAMPLE,output,seed=2,resume=True,realizer=continued,progress=False)
            self.assertEqual(random_seed.call_count,1)
        self.assertEqual(resumed_seeds[0],interrupted_seeds[1])
        self.assertEqual(result['realization']['rewrite_seed'],111)


if __name__=='__main__':
    unittest.main()
