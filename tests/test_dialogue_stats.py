"""Coverage counts realized data; runtime counters survive interrupted/resumed runs."""
import json
from pathlib import Path
import tempfile
import unittest

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
            result = run(EXAMPLE, Path(tmp)/'run', seed=1, reworks=2, p_compile=.5,
                         realizer=Writer(), progress=False)
        s = result['stats']
        self.assertEqual(s['number_of_turns'], 64)
        self.assertEqual(s['number_of_attempts'], 64)
        self.assertEqual(s['generation']['verifier_attempts'], 64)
        self.assertEqual(s['generation']['model_requests'], 128)
        self.assertEqual(s['number_of_assets'], len(result['placeholders']))
        self.assertEqual(s['number_of_logical_assets'], len(result['target_assets']))
        self.assertEqual(s['number_of_trial_clips'], 2)
        self.assertEqual(s['reference_depth_counts']['0'], 1)
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

    def test_model_aliases_keep_thinking_separate_from_instruct(self):
        self.assertEqual(resolve_model('ollama'), (OLLAMA_INSTRUCT, 'instruct'))
        self.assertEqual(resolve_model('ollama', 'qwen3-vl-30b-thinking'), (OLLAMA_THINKING, 'thinking'))
        self.assertEqual(resolve_model('local', 'qwen3-vl-30b-thinking'), (LOCAL_THINKING, 'thinking'))
        self.assertEqual(resolve_model('ollama', 'qwen3-vl:30b'), ('qwen3-vl:30b', 'thinking'))
        self.assertEqual(resolve_model('ollama', 'custom', 'thinking'), ('custom', 'thinking'))

    def test_reference_confirmation_is_a_question_not_adoption(self):
        from tools.dialogue_realizer import QwenRealizer, ACKNOWLEDGMENTS
        from unittest.mock import Mock, patch
        from tools.dialogue_verifier import CRITERIA
        from scripts.dialogue_generation.planner import build_plan
        p = build_plan(json.loads(EXAMPLE.read_text()), seed=1, reworks=2, p_compile=.5)
        turn = p['turns'][55]
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
