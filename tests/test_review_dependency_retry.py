"""Regression coverage for score-driven acceptance and dependent turn outputs."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest

from scripts.dialogue_generation.reviewed_turn import materialize_turn
from tools.media_review import SCORES, parse_review


class ReviewPolicyTests(unittest.TestCase):
    def test_passing_scores_drop_rejected_disposition_and_repair_advice(self):
        raw = dict(think=dict(request_check='I see the requested scene.',
            quality_check='I see slight framing variation.', decision='I will retry the shot.'),
            **dict.fromkeys(SCORES, 1), rule_checks=[], decision='rejected',
            reason='Change the framing.', reply='I will change the framing and retry.')
        report = parse_review(dict(text=json.dumps(raw)), rules=[])
        self.assertEqual((report['decision'], report['reason'], report['reply']), ('accepted', '', ''))
        self.assertEqual(report['think_steps']['decision'], 'I accept the result based on the review scores.')
        raw['request_fidelity'] = 0
        raw['reply'] = 'I see a clear framing error, and I will retry this shot.'
        self.assertEqual(parse_review(dict(text=json.dumps(raw)), rules=[])['decision'], 'rejected')

    def test_trace_includes_system_aspect_with_or_without_other_user_rules(self):
        raw = dict(think=dict(request_check='I see the requested scene.',
            quality_check='I see the expected framing.', decision='I accept the result.'),
            **dict.fromkeys(SCORES, 2), rule_checks=[], decision='accepted', reason='', reply='')
        note = 'The system default for aspect ratio is 1:1 for images and 16:9 landscape for videos.'
        default = dict(source='system_default', ratio='16:9')
        report = parse_review(dict(text=json.dumps(raw)), rules=[], aspect_ratios=[default])
        self.assertIn("Let's think. No user-set cross-turn rules apply here. " + note, report['thinking'])
        self.assertEqual(report['rule_checks'], [])

        raw['rule_checks'] = [dict(rule_index=1, score=2, evidence='I see the expected treatment.')]
        for rule, aspect, has_default in [
            ('Use winter.', default, True),
            ('Use portrait 9:16.', dict(source='user_override', ratio='9:16'), False),
        ]:
            with self.subTest(rule=rule):
                report = parse_review(dict(text=json.dumps(raw)), rules=[dict(requirement=rule)],
                                      aspect_ratios=[aspect])
                self.assertEqual(note in report['thinking'], has_default)
                self.assertIn('1. ' + rule, report['thinking'])

    def test_clip_is_repaired_and_accepted_before_assembly_is_generated(self):
        edit, assembly = '[[video_0001]]', '[[video_0002]]'
        tasks = {
            edit: dict(kind='video', backend='minimax_h3', operation='video_edit', prompt='Edit clip', dependencies=[]),
            assembly: dict(kind='video', operation='concatenate', dependencies=[edit],
                           clips=[dict(placeholder=edit)]),
        }
        fixed = []
        reviews = []
        records, resolved = {}, {}

        def fingerprint(token):
            return ':'.join([token] + [records[dep]['sha256'] for dep in tasks[token]['dependencies']])

        def generate(token, task, job, out, seed, receipt):
            if task.get('retry_edit'):
                source = task['references'][-1]
                self.assertEqual(source['role'], 'edit_source')
                self.assertEqual(source['attempt'], 1)
                self.assertEqual((work / source['path']).read_text(), 'Edit clip')
            out.write_text(task['prompt'])
            receipt.update(actual_prompt=task['prompt'], generation_seconds=1)

        def execute(token, task, job, out, record):
            out.write_text('assembled ' + resolved[edit].read_text())

        class Reviewer:
            def verify_turn(self, context, outputs, **kwargs):
                reviews.append([item['media'].read_text() for item in outputs])
                active = context['delivery_stage']['current_output_position']
                if active == 2:
                    assert outputs[1]['assembly_clips'] == [edit]
                    assert 'accepted_critique' in outputs[0]
                rejected = len(reviews) == 1
                return dict(decision='rejected' if rejected else 'accepted',
                    thinking='I assessed the current asset.', reply='I will repair the unchanged clip.',
                    retry_positions=[active] if rejected else [],
                    output_checks=[dict(output_position=active, **dict.fromkeys(SCORES, 0 if rejected else 2),
                                        decision='rejected' if rejected else 'accepted',
                                        reason='The edited clip has the wrong material.')])

            def fix(self, task, context, prompt, reason, **kwargs):
                fixed.append(context['output_position'])
                self_source = kwargs['media']
                assert self_source.read_text() == 'Edit clip'
                assert task['operation'] == 'video_edit'
                return 'Repaired clip'

        @contextmanager
        def session(directory):
            yield Reviewer()

        with tempfile.TemporaryDirectory() as temp:
            root = work = Path(temp)
            (root/'media').mkdir()
            materialize_turn([edit, assembly], directory=root, tasks=tasks, records=records, resolved=resolved,
                context={}, fingerprint=fingerprint, generate=generate, execute=execute, session=session,
                checkpoint=lambda: None, export=lambda: None, seed=13, progress=False)
            self.assertEqual(resolved[assembly].read_text(), 'assembled Repaired clip')

        self.assertEqual(fixed, [1])
        self.assertEqual(reviews, [['Edit clip'], ['Repaired clip'], ['Repaired clip', 'assembled Repaired clip']])
        self.assertEqual(records[assembly]['fingerprint'], fingerprint(assembly))


if __name__ == '__main__':
    unittest.main()
