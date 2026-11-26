"""Rubric protocol and review/rewrite integration, with no semantic keyword oracle."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools.dialogue_realizer import QwenRealizer, MAX_REWRITE_ATTEMPTS, _format_problems, reference_requirements
from tools.dialogue_verifier import CRITERIA, QwenVerifier, SYSTEM, build_prompt, parse_report, passed


def review(score=2, criterion="content_fidelity", critique=""):
    report = dict.fromkeys(CRITERIA, 2)
    report[criterion] = score
    return dict(report, critique=critique)



class ReportTests(unittest.TestCase):
    def test_each_criterion_has_the_same_zero_to_two_scale(self):
        for key in CRITERIA:
            for score in range(3):
                with self.subTest(key=key, score=score):
                    result = parse_report(json.dumps(review(score, key, "Restore the required change." if score < 2 else "")))
                    self.assertEqual(passed(result), score == 2)

    def test_missing_items_noninteger_scores_and_missing_critique_are_rejected(self):
        invalid = [None, [], {}, review(1)]
        for value in [True, "2", 2.0, -1, 3]:
            report = review(); report[CRITERIA[0]] = value; invalid.append(report)
        report = review(); del report[CRITERIA[-1]]; invalid.append(report)
        report = review(); report["extra"] = []; invalid.append(report)
        for report in invalid:
            with self.subTest(report=report), self.assertRaises(ValueError):
                parse_report(json.dumps(report))

    def test_report_is_flat_and_needs_actionable_critique_only_on_failure(self):
        self.assertEqual(set(review()), {*CRITERIA, "critique"})
        self.assertFalse(passed(parse_report(json.dumps(review(1, critique="Restore the missing finish.")))))
        with self.assertRaises(ValueError):
            parse_report(json.dumps(review(critique="An unsupported complaint.")))

    def test_malformed_report_retries_the_judge_without_changing_candidate(self):
        request = Mock(side_effect=[("not JSON", "stop"), (json.dumps(review()), "stop")])
        judge = QwenVerifier(request)
        self.assertTrue(passed(judge("Got it, drop the cuts.", "Resolve the conflict.",
                                    turn_id=35, rewrite_attempt=0, seed=100)))
        self.assertEqual(request.call_args_list[0].args[0], request.call_args_list[1].args[0])
        self.assertEqual([call.args[1] for call in request.call_args_list], [100, 101])

    def test_truncated_or_unavailable_review_never_approves_candidate(self):
        for response in [('{"content_fidelity":', "length"), ("{}", "stop")]:
            with self.subTest(response=response):
                request = Mock(return_value=response)
                with self.assertRaisesRegex(RuntimeError, "verifier failed"):
                    QwenVerifier(request)("A line.", "A brief.", turn_id=1, rewrite_attempt=0, seed=1)
                self.assertEqual(request.call_count, 3)

    def test_complete_valid_report_is_accepted_even_at_token_limit(self):
        request = Mock(return_value=(json.dumps(review()), "length"))
        result = QwenVerifier(request)("A complete line.", "A brief.", turn_id=1, rewrite_attempt=0, seed=1)
        self.assertTrue(passed(result))
        request.assert_called_once()

    def test_rubric_explicitly_allows_paraphrases_and_contextual_acknowledgments(self):
        for phrase in ["wider shot", "short black hair", "no opening phrase is forbidden",
                       "Intentionally incompatible user requests", "all scores are 2"]:
            self.assertIn(phrase, SYSTEM)


class EvaluationContextTests(unittest.TestCase):
    def test_deliberate_same_turn_conflict_requires_both_halves_without_fixing_it(self):
        for action in ("incompatible_requirements", "incompatible_framing"):
            with self.subTest(action=action):
                prompt = build_prompt("Make the background sharp and completely blurred.", dict(
                    speaker="user", action=action,
                    draft_content_requirements=["Make every background detail visible and blur away all background detail in one view."]))
                self.assertIn("must ask for BOTH incompatible requirements", prompt)
                self.assertIn("do not lower any score because", prompt)
                self.assertIn("Omitting either requirement", prompt)
                self.assertIn("single-view/shot/file restrictions", prompt)

    def test_keep_decision_applies_to_current_request_without_an_exception(self):
        prompt = build_prompt("Keep the rule but do the rejected treatment anyway.", dict(
            speaker="user", action="resolve_rule_conflict", intent_facts=dict(rule_decision="keep")))
        self.assertIn("continue the pending media request under that rule", prompt)
        self.assertIn("Canceling or abandoning the media request fails content_fidelity", prompt)
        self.assertIn("No exception is authorized", prompt)

    def test_keep_check_dispatches_to_short_prompt_without_unrelated_planner_fields(self):
        general=Mock(return_value=(json.dumps(review()),'stop'))
        keep=Mock(return_value=(json.dumps(review()),'stop'))
        brief=dict(action='resolve_rule_conflict',intent_facts=dict(rule_decision='keep'),
            draft_content_requirements=['Keep portrait for all images and videos; continue the tables request.'],
            previous_messages=[dict(role='user',content='OLD_UNRELATED'),dict(role='user',content='Make landscape tables.'),dict(role='assistant',content='Keep portrait or change it?')],
            attribute_overrides=['PLANNER_SECRET'],source_design='UNRELATED_DESIGN')
        judge=QwenVerifier(general,keep_request=keep)
        self.assertTrue(passed(judge('Keep portrait and make the tables.',brief,turn_id=3,rewrite_attempt=0,seed=1)))
        general.assert_not_called();keep.assert_called_once()
        payload=json.loads(keep.call_args.args[0])
        self.assertEqual(set(payload), {'required_decision', 'requested_outputs', 'additional_requirements', 'candidate'})
        self.assertNotIn('Make landscape tables.',str(payload))
        self.assertEqual(payload['required_decision'],brief['draft_content_requirements'][0])
        self.assertNotIn('PLANNER_SECRET',str(payload));self.assertNotIn('UNRELATED',str(payload))
        judge('Here it is.',dict(action='deliver_asset'),turn_id=4,rewrite_attempt=0,seed=2)
        general.assert_called_once();self.assertEqual(keep.call_count,1)

    def test_keep_check_has_its_own_system_on_the_same_model(self):
        from tools.dialogue_verifier import KEEP_SYSTEM,KEEP_REPORT_SCHEMA,VERIFIER_MAX_TOKENS
        writer=QwenRealizer(backend='ollama',model_mode='instruct')
        with patch.object(writer,'_generate',return_value=(json.dumps(review()),'stop')) as generate:
            writer._verify_keep_request('short decision input',7)
        self.assertEqual(generate.call_args.kwargs['system'],KEEP_SYSTEM)
        self.assertEqual(generate.call_args.kwargs['response_schema'],KEEP_REPORT_SCHEMA)
        self.assertEqual(KEEP_REPORT_SCHEMA['properties']['critique']['maxLength'],240)
        self.assertEqual(generate.call_args.kwargs['max_tokens'],VERIFIER_MAX_TOKENS)

    def test_sonnet_verifier_has_separate_effort_and_budget(self):
        writer = QwenRealizer(backend='sonnet', think_mode='medium', model_mode='instruct')
        self.addCleanup(writer.close)
        self.assertEqual(writer.local.effort, 'medium')
        with patch.object(type(writer.local), '__call__',
                          return_value=dict(text=json.dumps(review()), finish_reason='stop')) as call:
            writer._verify_request('verifier input', 1)
            writer._verify_keep_request('keep input', 2)
            writer._generate('rewrite input', 3, system='writer', max_tokens=4096)
            writer._generate('expert prompt', 4, system='writer', max_tokens=2048, stage='prompt_rewriter')
        for invocation in call.call_args_list[:2]:
            self.assertEqual(invocation.kwargs['effort'], 'low')
            self.assertEqual(invocation.kwargs['max_new_tokens'], 1024)
        self.assertEqual(call.call_args_list[2].kwargs['effort'], 'low')
        self.assertEqual(call.call_args_list[2].kwargs['max_new_tokens'], 4096)
        self.assertNotIn('effort', call.call_args.kwargs)

    def test_conflict_allowance_does_not_apply_to_assistant_clarification(self):
        prompt = build_prompt("Which do you prefer?", dict(speaker="assistant", action="clarify_conflict"))
        self.assertIn("must explain the incompatibility", prompt)
        self.assertIn("must not claim it can satisfy both", prompt)
        self.assertNotIn("must ask for BOTH incompatible requirements", prompt)

    def test_candidate_current_attributes_retain_option_bindings(self):
        prompt = build_prompt('Two separate images, cyan first and black second.', dict(
            action='request_candidates', attribute_overrides=[
                dict(subject='handbag', output_position=1, attributes=dict(color='cyan')),
                dict(subject='handbag', output_position=2, attributes=dict(color='black'))]))
        self.assertIn('Option 1 / handbag: color: cyan', prompt)
        self.assertIn('Option 2 / handbag: color: black', prompt)
        self.assertIn('two separate images', prompt)

    def test_older_version_binding_survives_flat_prompt_without_history(self):
        brief = dict(speaker="user", action="rollback", subjects=["tote bag"],
                     semantic_purpose="Return to the earlier version.",
                     draft_content_requirements=["Return to the earlier canvas tote bag."],
                     intent_facts=dict(required_attribute=dict(label="main material", value="canvas")),
                     bound_references=[dict(kind="image", operation='create_reference', subjects=["tote bag"])])
        prompt = build_prompt("Let's go back to the canvas tote bag.", brief)
        for fact in ["earlier reference version of tote bag", "version of tote bag being chosen: canvas",
                     "still image of tote bag", "outside the recent messages"]:
            self.assertIn(fact, prompt)
        self.assertNotIn('"subjects":', prompt)
        self.assertNotIn('"intent_facts":', prompt)

    def test_legacy_decision_metadata_does_not_require_a_duration_declaration(self):
        for decision in ("temporary", "accepted", "rejected"):
            with self.subTest(decision=decision):
                prompt = build_prompt("I'll take that version.", dict(speaker="user", action="adopt",
                    offered_choice_count=1, selected_choice_number=1,
                    draft_content_requirements=["Accept the reference just shown."],
                    intent_facts=dict(decision=decision)))
                self.assertNotIn("Decision status:", prompt)
                self.assertNotIn("temporar", prompt.lower())
                self.assertIn("accepting that single reference", prompt)

    def test_required_property_in_two_subject_test_belongs_to_one_subject(self):
        prompt = build_prompt("Try the wood bag with the traveler.", dict(
            speaker="user", action="temporary_adoption_trial", subjects=["tote bag", "traveler"],
            intent_facts=dict(required_attribute=dict(label="main material", value="wood", subject="tote bag"))))
        self.assertIn("Required main material of tote bag: wood.", prompt)
        self.assertNotIn("Required main material of tote bag, traveler", prompt)

    def test_original_plan_and_reference_policy_are_visible_without_oracle_targets(self):
        intent = dict(action="request_variant", reference_policy="clear",
                      description="Try the previous tote bag with wood.", target_media="PRIVATE_ORACLE")
        prompt = build_prompt("Try that with wood.", dict(
            python_description=intent["description"], reference_requirements=reference_requirements(intent)))
        self.assertIn("Original Python plan description", prompt)
        self.assertIn(intent["description"], prompt)
        self.assertIn("does not call for an ambiguous", prompt)
        self.assertIn("Planner-only asset bindings cannot supply words missing", prompt)
        self.assertNotIn("PRIVATE_ORACLE", prompt)

    def test_mixed_batch_uses_ambiguity_policy_for_each_edit(self):
        notes = reference_requirements(dict(action="request_reworks", reference_policy="per_edit", edits=[
            dict(index=1, ambiguous=True, target_media="PRIVATE_ONE"),
            dict(index=2, ambiguous=False, asset_id="PRIVATE_TWO")]))
        self.assertIn("Edit 1: the Python plan allows an ambiguous", notes[0])
        self.assertIn("Edit 2: the Python plan does not allow an ambiguous", notes[1])
        self.assertNotIn("PRIVATE", " ".join(notes))
        numbering = reference_requirements(dict(action="ambiguous_shot_number"))
        self.assertIn("ordering convention ambiguous", numbering[0])

    def test_catalog_camera_distance_explains_long_shot_alias(self):
        prompt = build_prompt("A wider shot of the street.", dict(
            python_description="Try the previous street in a long shot.",
            intent_facts=dict(required_attribute=dict(field="camera_distance", label="shot size", value="long shot"))))
        self.assertIn("long shot means a wide shot", prompt)
        self.assertIn("does not need the literal words long shot", prompt)

    def test_shot_operation_and_reference_correction_are_explicit_prose(self):
        prompt = build_prompt("Swap shots two and three in the current cut.", dict(
            intent_facts=dict(numbering_operation=dict(type="swap", positions=[2, 3]), numbering_basis="current"),
            reference_correction=dict(shown="traveler", intended="coffee cup")))
        for fact in ["Swap shot 2 and shot 3.", "positions in the current cut", "just showed: traveler",
                     "earlier reference of: coffee cup"]:
            self.assertIn(fact, prompt)
        self.assertNotIn("{'type':", prompt)


class ReviewLoopTests(unittest.TestCase):
    def turn(self):
        return dict(id=27, role="user", intent=dict(action="request_variant",
                    description="Try the same Tokyo street with a different shot size: long shot.",
                    required_attribute=dict(label="shot size", value="long shot")))

    def test_perfect_paraphrase_and_got_it_are_not_subject_to_python_semantic_checks(self):
        for line in ["Can we try the Tokyo street with a wider shot instead?",
                     "Got it, let's drop the hard cuts.",
                     "Use the medium shot; the long shot was too wide.",
                     "Can we get her with short black hair?"]:
            with self.subTest(line=line):
                model = QwenRealizer(backend="ollama")
                model.verifier = Mock(return_value=review())
                with patch.object(model, "_request", return_value=line) as writer:
                    self.assertEqual(model(self.turn(), {}), line)
                writer.assert_called_once()
                self.assertEqual(model.verifier.call_args.args[0], line)

    def test_any_nonmaximum_item_sends_critique_and_candidate_to_rewriter(self):
        for criterion in CRITERIA:
            model = QwenRealizer(backend="ollama")
            critique = "Restore the shot framing; keep the rest of this request."
            model.verifier = Mock(side_effect=[review(1, criterion, critique), review()])
            with patch.object(model, "_request", side_effect=["First candidate.", "Revised candidate."]) as writer:
                self.assertEqual(model(self.turn(), {}), "Revised candidate.")
            prompt = writer.call_args.args[0]
            self.assertIn("Important turn requirement: " + critique, prompt)
            self.assertNotIn(critique, model._writer_control)
            self.assertNotIn("First candidate.", prompt)
            self.assertNotIn('"' + criterion + '"', prompt)
            # Independent evaluation: the new judge sees the task, not prior grades.
            self.assertEqual(model.verifier.call_args_list[0].args[1], model.verifier.call_args_list[1].args[1])
            self.assertNotIn(critique, model.verifier.call_args.args[1])

    def test_retry_uses_latest_critique_instead_of_accumulating_stale_diagnoses(self):
        model = QwenRealizer(backend="ollama")
        model.verifier = Mock(side_effect=[review(1, critique="FIRST_ISSUE"),
                                          review(1, "grounding", "SECOND_ISSUE"), review()])
        with patch.object(model, "_request", side_effect=["One.", "Two.", "Three."]) as writer:
            model(self.turn(), {})
        prompt = writer.call_args.args[0]
        self.assertIn("SECOND_ISSUE", prompt)
        self.assertNotIn("FIRST_ISSUE", prompt)
        self.assertNotIn("SECOND_ISSUE", model._writer_control)

    def test_ambiguous_shot_retry_keeps_system_and_history_prefix_identical(self):
        model = QwenRealizer(backend="ollama")
        turn = dict(id=29, role="user", intent=dict(action="ambiguous_shot_number",
                    description="Swap shot 1 and shot 2."))
        previous = [dict(role="assistant", content="The clips are stitched.", intent={})]
        critique = "You should swap shot 1 and shot 2 without saying which clip is which."
        model.verifier = Mock(side_effect=[review(1, "conversation_coherence", critique), review()])
        replies = [("Use the last clip first.", "stop"), ("Can you swap shot 1 and shot 2?", "stop")]
        with patch.object(model, "_generate", side_effect=replies) as wire:
            self.assertEqual(model(turn, {}, previous), replies[-1][0])
        first, retry = wire.call_args_list
        self.assertEqual(first.kwargs["system"], retry.kwargs["system"])
        self.assertNotIn("Current speech act", first.kwargs["system"])
        first_prompt, retry_prompt = first.args[0], retry.args[0]
        prefix, suffix = first_prompt.split("\n\nReply intention:", 1)
        self.assertLess(prefix.index("Assistant: The clips are stitched."), prefix.index("Current speech act"))
        self.assertIn("Swap shot 1 and shot 2. Keep your reference deliberately ambiguous.", suffix)
        self.assertEqual(retry_prompt, prefix + "\n\nImportant turn requirement: " + critique +
                         "\n\nReply intention:" + suffix)

    def test_production_opening_only_uses_current_request_plan_and_needed_clarification(self):
        model = QwenRealizer(backend="ollama")
        current = dict(role='user', content='Use the landscape version.')
        plan = [dict(kind='video', operation='reference_generation', subjects=['boat'])]
        context = dict(current_user_request=current['content'], planned_outputs=plan,
                       visible_sequence=[dict(role='user', content='UNRELATED_EARLIER_TURN')],
                       current_exchange=[dict(role='user', content='Make one boat video.'),
                                         dict(role='assistant', content='Portrait or landscape?'), current])
        with patch.object(model, "_generate", return_value=("I'll make the landscape boat video.", "stop")) as wire:
            model.production_opening(context, seed=1)
            clarified = json.loads(wire.call_args.args[0])
            model.production_opening(dict(context, current_exchange=[current]), seed=1)
            ordinary = json.loads(wire.call_args.args[0])
        self.assertEqual(ordinary, dict(current_user_request=current['content'], planned_outputs=plan))
        self.assertEqual(clarified['preceding_clarification'], context['current_exchange'][:2])
        self.assertNotIn('UNRELATED_EARLIER_TURN', str(clarified))

    def test_failed_verification_exhausts_bounded_rewrites_without_template_fallback(self):
        model = QwenRealizer(backend="ollama")
        model.verifier = Mock(return_value=review(0, critique="Wrong production state."))
        with patch.object(model, "_request", return_value="Candidate.") as writer:
            with self.assertRaisesRegex(RuntimeError, "Wrong production state"):
                model(self.turn(), {})
        self.assertEqual(writer.call_count, MAX_REWRITE_ATTEMPTS)

    def test_format_failures_skip_judge_but_not_semantic_looking_text(self):
        model = QwenRealizer(backend="ollama")
        model.verifier = Mock(return_value=review())
        with patch.object(model, "_request", side_effect=["Use [[image_0001]].", "Got it, use this one."]):
            self.assertEqual(model(self.turn(), {}), "Got it, use this one.")
        model.verifier.assert_called_once()
        # These need semantic judgment, not token/substring tests.
        for line in ["The draft says short hair.", "Here's the video.", "Make it brighter.", "Can we stop at the"]:
            self.assertEqual(_format_problems(line), [])

    def test_followup_uses_current_rules_without_priming_expired_exception(self):
        model = QwenRealizer(asset_names=dict(street="Tokyo street"))
        model.verifier = Mock(return_value=review())
        turn = dict(id=49, role="user", intent=dict(action="request_rule_followup",
                    description="Make another trial clip under the continuing rules.",
                    asset_ids=["street"], requested_media_kind="video",
                    media_rule_context=["trial clips: fall as the background season"]))
        previous = [dict(role="user", content="Allow one exception in spring.", intent={}),
                    dict(role="assistant", content="Here is the spring clip.", intent={})]
        with patch.object(model, "_request", return_value="Make another Tokyo street trial clip.") as writer:
            model(turn, {}, previous)
        self.assertNotIn("spring", writer.call_args.args[0])
        self.assertNotIn("fall as the background season", writer.call_args.args[0])
        self.assertEqual(model.verifier.call_args.args[1]["previous_messages"], [])
        self.assertEqual(model.verifier.call_args.args[1]["media_rule_context"], turn["intent"]["media_rule_context"])

    def test_keep_rule_outcome_reaches_writer_and_verifier(self):
        model = QwenRealizer(backend="ollama")
        model.verifier = Mock(return_value=review())
        turn = dict(id=39, role="user", intent=dict(action="resolve_rule_conflict",
                    description="Keep the winter rule and make this video in winter.",
                    rule_event=dict(type="resolve", outcome="keep", effect=dict(season="winter"))))
        with patch.object(model, "_request", return_value="Keep the rule and make the video in winter."):
            model(turn, {})
        self.assertIn("Only the conflicting setting is withdrawn", model._writer_control)
        self.assertEqual(model.verifier.call_args.args[1]["intent_facts"]["rule_decision"], "keep")

    def test_mixed_language_line_is_rewritten_before_semantic_review(self):
        model = QwenRealizer(backend="ollama")
        model.verifier = Mock(return_value=review())
        english = "The rule covers all images and videos—including assets and trials."
        with patch.object(model, "_request", side_effect=[
                "The rule covers all images and videos—包括 assets and trials.", english]) as writer:
            self.assertEqual(model(self.turn(), {}), english)
        self.assertEqual(writer.call_count, 2)
        model.verifier.assert_called_once()
        self.assertEqual(model.verifier.call_args.args[0], english)
        self.assertEqual(model.metrics["format_rejections"], 1)
        self.assertIn("entire spoken line in English", writer.call_args.args[0])

    def test_language_check_allows_english_unicode_punctuation_and_latin_names(self):
        self.assertEqual(_format_problems("Let’s film Zoë at the café—including the outdoor tables."), [])
        for fragment in ("包括", "東京", "プレビュー", "미리보기", "𠀀"):
            with self.subTest(fragment=fragment):
                self.assertTrue(_format_problems("Here is the " + fragment + "."))

    def test_token_limit_is_evidence_for_verifier_not_automatic_semantic_rejection(self):
        model = QwenRealizer(backend="ollama"); model.verifier = Mock(return_value=review())
        def writer(*args):
            model._last_finish_reason = "length"
            return "We'll use this one."
        with patch.object(model, "_request", side_effect=writer) as request:
            self.assertEqual(model(self.turn(), {}), "We'll use this one.")
        self.assertEqual(request.call_count, 1)
        self.assertEqual(model.verifier.call_args.kwargs["generation_finish_reason"], "length")

    def test_incomplete_token_limited_line_is_rewritten_from_model_critique(self):
        model = QwenRealizer(backend="ollama")
        model.verifier = Mock(side_effect=[review(1, "naturalness", "Finish the interrupted request."), review()])
        responses = iter([("Can we get the", "length"), ("Can we get the wide shot?", "stop")])
        def writer(*args):
            text, model._last_finish_reason = next(responses)
            return text
        with patch.object(model, "_request", side_effect=writer) as request:
            self.assertEqual(model(self.turn(), {}), "Can we get the wide shot?")
        self.assertEqual(request.call_count, 2)
        self.assertIn("Finish the interrupted request.", request.call_args.args[0])

    def test_multiline_or_long_prose_has_no_python_style_gate(self):
        self.assertEqual(_format_problems("A natural part of the turn.\n" * 4), [])
        self.assertEqual(_format_problems("A longer spoken turn. " * 100), [])

    def test_candidate_and_review_are_persisted_together_and_judge_failure_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = QwenRealizer(backend="ollama", output_dir=tmp)
            report = review()
            model.verifier = Mock(return_value=report)
            with patch.object(model, "_request", return_value="A candidate."):
                model(self.turn(), {})
            row = json.loads((Path(tmp)/"calls.jsonl").read_text())
            self.assertEqual(row["verification"], report)
            self.assertTrue(row["accepted"])
            model.verifier.side_effect = RuntimeError("Verifier unavailable")
            with patch.object(model, "_request", return_value="Another candidate."):
                with self.assertRaisesRegex(RuntimeError, "Verifier unavailable"):
                    model(self.turn(), {})
            row = json.loads((Path(tmp)/"calls.jsonl").read_text().splitlines()[-1])
            self.assertFalse(row["accepted"])
            self.assertEqual(row["verification_error"], "Verifier unavailable")


if __name__ == "__main__":
    unittest.main()
