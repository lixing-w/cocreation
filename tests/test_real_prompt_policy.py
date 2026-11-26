import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cocreation.real_prompt_policy import real_prompt_cases


def cases(n=4):
    return [dict(id=f'case_{i}', seed=i, source_format='h3',
                 input_file=f'case_{i}.json', reworks=2,
                 aspect_ratio_rule='portrait', video_width=768, video_height=1344)
            for i in range(n)]


class RealPromptPolicyTests(unittest.TestCase):
    def test_sampling_is_stable_per_case_and_does_not_mutate_source(self):
        inputs = cases(100)
        before = copy.deepcopy(inputs)
        tsv, report = real_prompt_cases(inputs)
        self.assertEqual((tsv, report), real_prompt_cases(inputs))
        counts = [c['selected_reworks'] for c in report['cases']]
        self.assertEqual(set(counts), {0, 1})
        self.assertTrue(30 < sum(counts) < 70)
        self.assertEqual(inputs, before)
        self.assertEqual(report['cases'][0], real_prompt_cases([inputs[0]])[1]['cases'][0])

    def test_resume_preserves_legacy_two_reworks_even_under_new_default(self):
        previous, _ = real_prompt_cases(cases(), policy='manifest')
        legacy_rows = [line.split('\t') for line in previous.splitlines()]
        legacy = ''.join('\t'.join([*row[:5], 'portrait', *row[5:]])+'\n'
                         for row in legacy_rows)
        current_cases = cases()
        for case in current_cases:
            del case['aspect_ratio_rule']
        resumed, report = real_prompt_cases(current_cases, previous_tsv=legacy)
        self.assertEqual(previous, resumed)
        self.assertEqual({c['selected_reworks'] for c in report['cases']}, {2})
        self.assertEqual(report['source'], 'existing_cases_tsv')

    def test_explicit_override_and_resume_metadata_validation(self):
        previous, report = real_prompt_cases(cases(), policy='0')
        self.assertEqual({c['selected_reworks'] for c in report['cases']}, {0})
        changed = cases()
        changed[0]['seed'] = 100
        with self.assertRaisesRegex(ValueError, 'metadata changed'):
            real_prompt_cases(changed, previous_tsv=previous)
        with self.assertRaisesRegex(ValueError, 'case set'):
            real_prompt_cases(cases()[:2], previous_tsv=previous)
        with self.assertRaises(ValueError):
            real_prompt_cases(cases(), policy='random')


if __name__ == '__main__':
    unittest.main()
