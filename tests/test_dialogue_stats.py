"""Coverage counts realized data; runtime counters survive interrupted/resumed runs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.dialogue_generation.generate import run
from scripts.dialogue_generation.planner import BEHAVIORS
from tools.dialogue_models import resolve_model, OLLAMA_INSTRUCT, OLLAMA_THINKING, LOCAL_THINKING

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'scripts/dialogue_generation/examples/tokyo_vlog.json'


class Writer:
    backend, model, model_mode = 'test', 'counted-writer', 'instruct'

    def __init__(self, fail_at=None):
        self.fail_at = fail_at
        self.rewrite_attempts = 0
        self.metrics = dict(model_requests=0, completion_tokens=0)
        self.verifier = type('Verifier', (), dict(attempt_count=0))()

    def __call__(self, turn, media, previous, *, seed):
        self.rewrite_attempts += 1
        self.metrics['model_requests'] += 1
        self.metrics['completion_tokens'] += 5
        if turn['id'] == self.fail_at:
            raise RuntimeError('Interrupted')
        self.verifier.attempt_count += 1
        self.metrics['model_requests'] += 1
        return 'Test line.'


class StatsTests(unittest.TestCase):
    def test_complete_counts_and_exact_documented_behavior_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run(EXAMPLE, Path(tmp)/'run', seed=4, reworks=2, p_compile=.5,
                         realizer=Writer(), progress=False)
        s = result['stats']
        self.assertEqual(s['number_of_turns'], len(result['turns']))
        self.assertEqual(s['number_of_attempts'], len(result['turns']))
        self.assertEqual(s['generation']['verifier_attempts'], len(result['turns']))
        self.assertEqual(s['generation']['model_requests'], 2*len(result['turns']))
        self.assertEqual(s['number_of_assets'], len(result['placeholders']))
        self.assertEqual(s['number_of_logical_assets'], len(result['target_assets']))
        generated_videos = [m for m in result['placeholders'] if m['kind']=='video' and m['operation'] not in ('assemble', 'export')]
        self.assertEqual(s['number_of_generated_videos'], len(generated_videos))
        self.assertEqual(s['cross_turn_rules']['active'], sum(r['ended_at'] is None for r in result['rules']))
        self.assertTrue(s['cross_turn_rules']['introduced'])
        self.assertEqual(s['reference_depth_counts']['0'], sum(not row['reference_materials'] for row in result['placeholders']))
        self.assertEqual(sum(s['media_by_kind'].values()), s['number_of_assets'])
        expected = sorted({b for t in result['turns'] if t['role']=='user' for b in t['intent']['behavior_ids']})
        self.assertEqual(s['user_behaviors_covered'], [BEHAVIORS[b] for b in expected])
        doc = (ROOT/'scripts/dialogue_generation/README.md').read_text()
        for k, name in BEHAVIORS.items():
            self.assertIn(f'| {k} | {name} |', doc)
        self.assertGreater(s['time_elapsed_seconds'], 0)

    def test_failure_counts_attempt_and_only_completed_coverage_then_resume_accumulates(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'run'
            with self.assertRaisesRegex(RuntimeError, 'Interrupted'):
                run(EXAMPLE, output, seed=1, realizer=Writer(fail_at=2), progress=False)
            checkpoint = json.loads((output/'dialogue.json').read_text())
            self.assertEqual(checkpoint['stats']['number_of_turns'], 1)
            self.assertEqual(checkpoint['stats']['number_of_attempts'], 2)
            self.assertEqual(checkpoint['stats']['number_of_assets'], 0)
            self.assertEqual(checkpoint['stats']['user_behavior_ids_covered'], checkpoint['turns'][0]['intent']['behavior_ids'])
            resumed = run(EXAMPLE, output, seed=1, realizer=Writer(), progress=False, resume=True)
            s = resumed['stats']
            self.assertEqual(s['number_of_attempts'], len(resumed['turns'])+1)
            self.assertEqual(s['generation']['verifier_attempts'], len(resumed['turns']))
            self.assertEqual(s['generation']['model_requests'], 2*len(resumed['turns'])+1)
            self.assertGreaterEqual(s['time_elapsed_seconds'], checkpoint['stats']['time_elapsed_seconds'])
            writer = Writer()
            again = run(EXAMPLE, output, seed=1, realizer=writer, progress=False, resume=True)
            self.assertEqual(again['stats'], s)
            self.assertEqual(writer.rewrite_attempts, 0)
            from unittest.mock import patch
            with patch("scripts.dialogue_generation.generate.QwenRealizer") as factory:
                with self.assertRaisesRegex(ValueError, "same realization"):
                    run(EXAMPLE, output, seed=1, resume=True, progress=False)
                factory.assert_not_called()

    def test_real_qwen_stage_counters_survive_failure_and_resume(self):
        from tools.dialogue_realizer import QwenRealizer
        from tools.dialogue_verifier import CRITERIA

        for reuse_writer in (False, True):
            with self.subTest(reuse_writer=reuse_writer), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)/'run'
                interrupt = True
                def response(*args, stage='rewrite', **kwargs):
                    nonlocal interrupt
                    if stage == 'rewrite' and writer.rewrite_attempts == 2 and interrupt:
                        interrupt = False
                        raise RuntimeError('Transport interrupted')
                    text = json.dumps(dict(dict.fromkeys(CRITERIA, 2), critique='')) if stage == 'verifier' else 'Test line.'
                    return text, 'stop'

                options = dict(seed=1, reworks=0, modules=['direct'], extra_behaviors=False,
                               cross_turn_rules=False, rewrite_prompts=False, progress=False)
                writer = QwenRealizer(backend='vllm', model_mode='thinking')
                with patch.object(writer, '_generate_once', side_effect=response):
                    # The model may be resident and already have nested counters.
                    writer.generation_request('Earlier request.', system='Test.', seed=0, max_tokens=4096)
                    with self.assertRaisesRegex(RuntimeError, '^Transport interrupted$'):
                        run(EXAMPLE, output, realizer=writer, **options)
                saved = json.loads((output/'dialogue.json').read_text())
                self.assertEqual(saved['realization']['status'], 'failed')
                self.assertEqual(saved['realization']['error'], 'Transport interrupted')
                self.assertEqual(saved['realization']['completed_turns'], 1)
                self.assertEqual(saved['stats']['generation']['stages']['rewrite']['requests'], 2)
                self.assertEqual(saved['stats']['generation']['stages']['verifier']['requests'], 1)

                if not reuse_writer:
                    writer = QwenRealizer(backend='vllm', model_mode='thinking')
                with patch.object(writer, '_generate_once', side_effect=response):
                    writer.generation_request('Another earlier request.', system='Test.', seed=0, max_tokens=4096)
                    result = run(EXAMPLE, output, realizer=writer, resume=True, **options)
                generation = result['stats']['generation']
                turns = len(result['turns'])
                self.assertEqual(result['realization']['status'], 'complete')
                self.assertEqual(generation['model_requests'], 2*turns+1)
                self.assertEqual(generation['rewrite_attempts'], turns+1)
                self.assertEqual(generation['verifier_attempts'], turns)
                self.assertEqual(generation['stages']['rewrite']['requests'], turns+1)
                self.assertEqual(generation['stages']['verifier']['requests'], turns)
                self.assertEqual(generation['stages']['prompt_rewriter']['requests'], 0)
                self.assertEqual(generation['token_counts_source'], 'vllm')
                self.assertAlmostEqual(generation['model_seconds'], sum(s['seconds'] for s in generation['stages'].values()))

    def test_generation_snapshot_does_not_share_nested_counters(self):
        from scripts.dialogue_generation.stats import generation_snapshot
        writer = Writer()
        writer.metrics['stages'] = dict(rewrite=dict(requests=2, seconds=1.5))
        snapshot = generation_snapshot(writer)
        writer.metrics['stages']['rewrite']['requests'] += 1
        writer.metrics['stages']['verifier'] = dict(requests=1, seconds=.5)
        self.assertEqual(snapshot['stages'], dict(rewrite=dict(requests=2, seconds=1.5)))

    def test_model_aliases_keep_thinking_separate_from_instruct(self):
        self.assertEqual(resolve_model('ollama'), (OLLAMA_INSTRUCT, 'instruct'))
        self.assertEqual(resolve_model('ollama', 'qwen3-vl-30b-thinking'), (OLLAMA_THINKING, 'thinking'))
        self.assertEqual(resolve_model('local', 'qwen3.6-35b-a3b', 'thinking'), (LOCAL_THINKING, 'thinking'))
        self.assertEqual(resolve_model('ollama', 'qwen3-vl:30b'), ('qwen3-vl:30b', 'thinking'))
        self.assertEqual(resolve_model('ollama', 'custom', 'thinking'), ('custom', 'thinking'))

    def test_reference_confirmation_is_a_question_not_adoption(self):
        from tools.dialogue_realizer import QwenRealizer, ACKNOWLEDGMENTS
        from unittest.mock import Mock, patch
        from tools.dialogue_verifier import CRITERIA
        from scripts.dialogue_generation.planner import build_plan
        p = build_plan(json.loads(EXAMPLE.read_text()), seed=4, reworks=2, p_compile=.5)
        turn = next(t for t in p['turns'] if t['intent']['action']=='confirm_reference')
        self.assertEqual(turn['intent']['action'], 'confirm_reference')
        self.assertIn('ask what they want to change', turn['intent']['description'])
        self.assertNotIn('confirm_reference', ACKNOWLEDGMENTS)
        model = QwenRealizer(asset_names={'cup':'coffee cup'})
        model.verifier = Mock(return_value=dict(dict.fromkeys(CRITERIA, 2), critique=''))
        with patch.object(model, '_request', return_value='You meant the coffee cup. What would you like to change?') as writer:
            model(turn, {m['placeholder']:m for m in p['placeholders']})
        self.assertIn('ask what they want to change', model._writer_control)
        self.assertNotIn('ask no question', writer.call_args.args[0])
        self.assertNotIn('acknowledgment_requirement', model.verifier.call_args.args[1])


if __name__ == '__main__':
    unittest.main()
