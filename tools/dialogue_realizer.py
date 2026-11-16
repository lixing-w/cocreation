"""Realize one pre-decided dialogue act as short prose; never plan with Qwen."""
from __future__ import annotations

import json
import re
import secrets
import time
import urllib.error
import urllib.request
from pathlib import Path
from tools.dialogue_personas import choose_personas
from cocreation.qwen_sampling import generation_options
from tools.dialogue_verifier import (QwenVerifier, passed, SYSTEM as VERIFIER_SYSTEM,
    REPORT_SCHEMA, CRITERIA, VERIFIER_MAX_TOKENS, THINKING_VERIFIER_MAX_TOKENS, VERIFIER_TEMPERATURE)

from tools.dialogue_models import DEFAULT_OLLAMA_MODEL, DEFAULT_LOCAL_MODEL, resolve_model
MAX_REWRITE_ATTEMPTS = 5


SYSTEM = """Write one natural English spoken turn for the specified user or creative assistant making a video together. The current intention decides meaning; history and media notes supply context. Express requirements in your own words, rather than copying the notes or tone examples.
Keep all requested details, option bindings, shot order and scope. For edits, state the change and enough to identify the subject; leave unchanged properties implicit. Distinguish images from videos, temporary from final choices, new from earlier versions, and pending from completed work. Correcting a referent is not adopting a version. Keep deliberate ambiguity and conflicting requirements until the planned clarification.
The user directs the assistant to create media. The assistant delivers briefly, acknowledges a decision, or asks the specified question. Do not invent designs, prior discussion, production, preferences or scope. Use ordinary words and the supplied persona; vary structure when stock openings repeat. Usually use one or two sentences, allowing more for multiple edits/questions. Return dialogue only, without headings, role labels, arrow notation, field lists, JSON, analysis or media markers.
"""

PLACEHOLDER = re.compile(r"\[\[(?:image|video)_\d{4}\]\]")
RULE_ACTIONS = {"set_persistent_rule", "confirm_rule", "override_rule", "confirm_override"}
NO_DESIGN_ACTIONS = {"switch_for_trial", "suspend_task", "vague_request", "clarify_goal", "confirm_switch",
                     "clarify_update_scope", "ambiguous_reference", "request_preview",
                     "request_compilation", "reorder_shots", "replace_compilation_shots"} | RULE_ACTIONS
REFERENCE_REQUESTS = {"request_asset", "specify_goal", "request_trial", "begin_task"}
EDIT_REQUESTS = {"reject_and_edit", "request_variant", "request_rework", "request_reworks",
                 "relative_request", "resume_task", "request_shot_update"}
ACKNOWLEDGMENTS = {"confirm_selection", "confirm_final", "confirm_synchronization",
                   "confirm_project_rule", "confirm_rule", "confirm_override"}
ACT_GUIDANCE = {
    "set_cross_turn_rule": "Introduce the supplied rule and its exact global or partial scope. Do not invent exemptions for a global rule.",
    "confirm_cross_turn_rule": "Confirm the supplied effect and scope. For a partial rule, explicitly say other output categories are outside this particular rule; other existing rules still apply.",
    "request_rule_conflict": "Request the specified media category explicitly, with the new treatment. A trial clip, standalone asset image and formal shot have different rule scopes. Leave the conflict for the assistant to point out; do not pre-authorize an exception.",
    "clarify_rule_conflict": "Name the existing rule and why the new request conflicts. Offer every stated decision, including keeping the existing rule, before producing media.",
    "resolve_rule_conflict": "State the chosen decision and its full stated scope: keep the earlier rule, allow this request only, or replace it going forward. If a rule covers all trial clips or all outputs, say so explicitly; do not narrow it to the current subject.",
    "cancel_cross_turn_rule": "Explicitly cancel only the specified rule. Keep other rules and already produced media unchanged.",
    "confirm_rule_cancel": "Confirm cancellation of this rule for future outputs. Other rules and existing media remain unchanged.",
    "confirm_rule_resolution": "Confirm the supplied rule decision and its exact duration and scope.",
    "request_rule_followup": "Request the specified media category under its current applicable rules. The earlier one-request exception has expired; do not repeat it or request another exception.",
    "restore_rule_asset": "Ask for a fresh reference without the canceled treatments. Preserve the agreed underlying design.",
    "confirm_reference": "Acknowledge which earlier object the user meant and ask what they want to change about it. This resolves a referent, not a version choice.",
    "set_project_rule": "State the project-wide footage requirement and its exact scope. Include any exemption for standalone references.",
    "confirm_project_rule": "Confirm the footage requirement, scope, and standalone-reference exemption stated in the content requirements.",
    "begin_task": "Request one initial still test image and state the subject's specified appearance.",
    "request_shot": "Request a video shot using the existing references, with the intention's action, framing, and camera movement.",
    "request_compilation": "Ask to assemble the existing clips in the specified order. Name them in spoken English, using first/then/finally rather than arrow notation.",
    "reorder_shots": "Ask to reorder the existing clips. Name each clip in exactly the intention's sequence; a generic 'this order' or 'this shot only' cannot convey that sequence.",
    "replace_compilation_shots": "Ask to replace the updated clips in the existing cut while keeping the other clips and order.",
    "request_rework": "Request the specified change to the existing subject in the named clip. Preserve the scope.",
    "suspend_task": "Pause the current task and switch to the other named subject. Leave the unfinished design unchanged.",
    "resume_task": "Return to the unfinished task and request only the specified change.",
    "choose_update_scope": "State whether the agreed change applies just here or also to other and future shots.",
    "synchronize_revision": "Ask to use the already-agreed revised reference in its other appearances.",
    "accept_project": "Accept the existing cut as final. Do not request more production.",
    "correct_reference": "Correct the assistant: distinguish what was just shown from the earlier reference actually intended.",
    "clarify_reference": "Ask whether the existing reference shown is the one the user means. Do not propose edits.",
    "conflicting_request": "Ask for the preview in the specified colors. Leave the conflict with the earlier rule for the assistant to clarify.",
    "request_asset": "Request one still reference of this subject and retain the visible details in the content requirements.",
    "request_variant": "Request the specified variant of the existing reference. When returning after discussion of another subject, reestablish which subject you want changed; a bare pronoun must have a clear conversational referent.",
    "specify_goal": "Request one still and state the visual details that clarify the preceding request.",
    "request_trial": "Request one still as an initial test of the specified design.",
    "request_candidates": "Ask the assistant to make two alternatives in the stated medium. Include every shared attribute once, even in a concise voice. Bind each differing detail to the first or second option.",
    "adopt": "Choose or accept the existing version. Leave further production for another turn. For numbered options, explicitly keep the chosen number.",
    "rollback": "Choose to return to the specified earlier version after trying the new one. This is a deliberate switch back, not a correction of what you meant earlier.",
    "vague_request": "Open a conversation about designing the subject without settling any visual details.",
    "clarify_goal": "Ask what the current subject should look like. Let the user supply the details.",
    "switch_for_trial": "Say which subjects to prepare now and combine in a test later. Leave their appearance open until the user defines it.",
    "confirm_switch": "Ask the user to describe the new subject's appearance.",
    "incompatible_framing": "Deliberately ask for both incompatible framings in the same shot at the same time. Make clear that the prop alone is to fill the close-up. Do not remove either requirement or resolve the contradiction; the assistant will ask about it next.",
    "incompatible_requirements": "Deliberately request both incompatible requirements at the same time. Keep restrictions such as a single shot, camera, or file. Do not remove either requirement or resolve the contradiction; the assistant will ask about it next.",
    "clarify_conflict": "Explain the physical or editing reason specified for this turn. Ask the user which approach to take.",
    "resolve_conflict": "Propose the new action and framing yourself, as the user. The assistant gave a choice, so this is your own alternative shot description.",
    "request_reworks": "Keep each requested change, shot number, and numbering basis. Leave deliberately vague subjects unnamed. Separate the two requests clearly.",
    "clarify_reworks": "Ask all reference and scope questions in this one turn. Use ‘edit one’ and ‘edit two’ where helpful. Keep the candidates and scope questions with the right edit.",
    "resolve_reworks": "Identify the intended versions and specify whether each change stays in that shot or carries over. Use clear labels if the two edits have different scopes.",
    "temporary_adoption_trial": "Temporarily choose the newly delivered variant for the test, identifying its distinguishing detail. Request a short test clip showing a plausible action or relationship between the two specified subjects.",
    "reject_and_edit": "Change only the specified field of the named subject or the whole setting. Preserve the distinction between changing the subject and changing a detail within it.",
    "request_preview": "Request another preview in a short sentence. Leave color processing and all visual design decisions unstated.",
    "clarify_rule_override": "Ask whether color is a one-time exception or the default from now on.",
    "clarify_update_scope": "Ask whether the change carries over to other and future shots. Keep the design details from the last task out of the question.",
    "ambiguous_reference": "Ask to change an unspecified thing in one of the earlier still images. Leave the object, appearance, image number, and shot number unspecified.",
    "set_persistent_rule": "Introduce black and white as the default for standalone previews and preserve the underlying designs.",
    "confirm_rule": "Confirm black and white as the preview default and preserve the final footage.",
    "one_time_exception": "Say that original colors apply just this once and that subsequent previews remain black and white.",
    "override_rule": "End the black-and-white preview rule and restore original colors for all subsequent previews.",
    "confirm_override": "Confirm that original colors are now the default for previews.",
}

# Content requirements, deliberately not fully worded character lines.
ASSISTANT_CONTENT = {
    "deliver_rule_preview": "Completed result: the preview under the agreed rule decision.",
    "deliver_restored_rule_asset": "Completed result: the regenerated reference without canceled treatments; clip updates are still pending.",
    "deliver_asset": "Completed result: the requested still reference image.",
    "deliver_trial": "Completed result: the initial still test image.",
    "deliver_edit": "Completed result: the revised reference image with the requested edit.",
    "deliver_candidates": "Completed results: both requested alternatives. The user has not chosen yet.",
    "confirm_selection": "Decision to acknowledge: use the version the user just selected. No new production.",
    "save_task": "Task state: the current intention is saved; work switches to the other requested task.",
    "deliver_resumed_task": "Completed result: the resumed draft with the specified change.",
    "deliver_variant": "Completed result: the newly requested variant, available to try now.",
    "deliver_exploration": "Completed result: the action/relationship test clip, for exploration.",
    "confirm_reference": "Clarified referent: the earlier object the user meant; next question: what change do they want? No version has been selected or revised.",
    "deliver_reworked_assets": "Completed results: both revised reference images. Clip updates are still pending, with the scope agreed for each edit.",
    "confirm_synchronization": "Decision to acknowledge: use the revised reference in the other shots too. Keep the earlier version available.",
    "deliver_exception": "Completed result: this preview in original colors. Later previews remain black and white.",
    "restore_persistent_rule": "Completed result: a preview following the black-and-white default.",
    "deliver_shot": "Completed result: the requested video shot.",
    "deliver_compilation": "Completed result: the requested assembly of existing clips.",
    "deliver_refreshed_compilation": "Completed result: updated shots substituted into the existing cut without changing the order. Earlier cut remains available.",
    "confirm_final": "Project state: the user accepted the cut as final; it is saved and reference versions remain available.",
}


def _format_problems(text):
    """Only objective output-contract checks; meaning belongs to the verifier."""
    if not isinstance(text, str) or not text.strip():
        return ["Return a nonempty dialogue line."]
    problems = []
    if (text.startswith(("{", "[", "```"))
            or any(marker in text for marker in ("<think>", "</think>", "<|"))):
        problems.append("Return plain dialogue, without JSON, code fences or model control markers.")
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af\u3040-\u30ff\uac00-\ud7af]", text):
        problems.append("Return the entire spoken line in English; translate non-English fragments into English.")
    if re.search(r"\b(?:image|video)_\d{4}\b", text) or "[[" in text or "{{artifact:" in text:
        problems.append("Leave media markers to the program.")
    return problems


def reference_requirements(intent):
    """Translate the plan's reference policy, without exposing hidden targets."""
    policy = intent.get("reference_policy") or {
        "ambiguous_reference": "ambiguous_subject",
        "ambiguous_shot_number": "ambiguous_shot_number",
        "request_reworks": "per_edit",
    }.get(intent["action"], "clear")
    if policy == "ambiguous_subject":
        return ["The Python plan deliberately leaves the object reference ambiguous. An unnamed thing or earlier image is allowed; do not require the hidden object's identity or a unique referent before the planned clarification."]
    if policy == "ambiguous_shot_number":
        return ["The Python plan deliberately leaves the shot-number ordering convention ambiguous. Keep the requested operation and numbers, but do not require the speaker to resolve generation order versus current-cut order yet."]
    if policy == "per_edit":
        return [f"Edit {edit['index']}: " + (
            "the Python plan allows an ambiguous object reference; do not require a unique object or name yet."
            if edit["ambiguous"] else
            "the Python plan does not allow an ambiguous object reference; the listener must be able to identify this edit's subject.")
            for edit in intent.get("edits", [])]
    return ["The Python plan does not call for an ambiguous object reference in this turn. Strictly judge whether the listener can identify the intended subject from the spoken line, the recent conversation and any media actually shown. A pronoun is fine when its referent is clear. After a topic switch, the line must reestablish the earlier subject. Planner-only asset bindings cannot supply words missing from the spoken line. A planned clarification may name alternatives without choosing one; a brief delivery may point to its accompanying result."]


def spoken_context(turn):
    text = PLACEHOLDER.sub("", turn["content"])
    return re.sub(r"(?im)^references?:.*$", "", text).strip()


class QwenRealizer:
    """Text-only Ollama, or the project's persistent CUDA Qwen adapter.

    The writer returns dialogue; an independent system prompt requests rubric
    reports from the same model. One adapter/keep-alive serves both stages.
    """

    def __init__(self, *, backend="ollama", model=None,
                 base_url="http://127.0.0.1:11434", output_dir=None,
                 max_tokens=256, timeout=180, temperature=0.9, top_p=0.95, top_k=50,
                 personas=None, context_tokens=32768, asset_names=None, model_mode=None):
        if backend not in ("ollama", "local"):
            raise ValueError("backend must be ollama or local")
        if max_tokens < 32 or timeout <= 0:
            raise ValueError("Invalid Qwen generation budget")
        sampling = generation_options(dict(temperature=temperature, top_p=top_p, top_k=top_k))
        self.backend = backend
        self.model, self.model_mode = resolve_model(backend, model, model_mode)
        self.rewrite_attempts = 0
        self.metrics = dict(model_requests=0, prompt_tokens=0, completion_tokens=0,
                            model_seconds=0.0, load_seconds=0.0, prompt_seconds=0.0, decode_seconds=0.0,
                            accepted_candidates=0, format_rejections=0, semantic_rejections=0, verifier_failures=0)
        self.base_url = base_url.rstrip("/")
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.max_tokens, self.timeout = max_tokens, timeout
        self.sampling = {key:value for key,value in sampling.items() if key != "do_sample"}
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k
        if context_tokens < 2048:
            raise ValueError("Context budget must be at least 2048 tokens")
        self.context_tokens = context_tokens
        self.personas = personas or choose_personas(0)
        self.asset_names = asset_names or {}
        self.verifier = QwenVerifier(self._verify_request,
                                    output_dir=self.output_dir / "verifier" if self.output_dir else None)
        self.local = None
        if backend == "local":
            from cocreation import Qwen
            self.local = Qwen(model_id=self.model,
                              output_dir=self.output_dir / "runtime" if self.output_dir else None)

    def close(self):
        if self.local is not None:
            self.local.close()

    def _request(self, prompt, seed):
        speaker = getattr(self, "_writer_speaker", "specified speaker")
        purpose = getattr(self, "_writer_purpose", "Express the supplied content requirements.")
        prefix = (f"<think>\nThe {speaker}'s dialogue act has been decided: {purpose} "
                  "The content facts are fixed. Only the natural spoken wording remains.\n</think>\n\n")
        control = getattr(self, "_writer_control", "")
        system = SYSTEM + ("\nCurrent speech act (instructions, not spoken content): " + control if control else "")
        text, self._last_finish_reason = self._generate(prompt, seed, system=system,
                                                       max_tokens=self.max_tokens,
                                                       assistant_prefill=prefix if self.model_mode == "thinking" else "")
        return text

    def generation_request(self, prompt, *, system, seed, max_tokens):
        """Reuse this model for text-only expert rewriting after the dialogue."""
        return self._generate(prompt, seed, system=system, max_tokens=max_tokens,
                              stage='prompt_rewriter', temperature=0.3)

    def _verify_request(self, prompt, seed):
        # One invocation produces the whole report. No second finalization call.
        budget = THINKING_VERIFIER_MAX_TOKENS if self.model_mode == "thinking" else VERIFIER_MAX_TOKENS
        return self._generate(prompt, seed, system=VERIFIER_SYSTEM, stage="verifier",
                              max_tokens=budget, temperature=VERIFIER_TEMPERATURE,
                              response_schema=REPORT_SCHEMA)

    def _generate(self, *args, **kwargs):
        started = time.perf_counter()
        self.metrics["model_requests"] += 1
        try:
            return self._generate_once(*args, **kwargs)
        finally:
            self.metrics["model_seconds"] += time.perf_counter() - started

    def _generate_once(self, prompt, seed, *, system, max_tokens, stage="rewrite",
                  temperature=None, response_schema=None, assistant_prefill=None):
        # Short spoken rewriting and semantic review have different budgets.
        # The judge must not inherit the writer's closed reasoning prefix.
        if assistant_prefill is None:
            assistant_prefill = ("<think>\n" if stage == "verifier" else "<think>\n\n</think>\n\n") if self.model_mode == "thinking" else ""
        sampling = dict(self.sampling)
        if temperature is not None:
            sampling["temperature"] = temperature
        if len((system + prompt + assistant_prefill).encode("utf-8")) + max_tokens > self.context_tokens:
            raise RuntimeError(f"{stage.title()} prompt exceeds context budget (recent messages are never truncated)")
        if self.local is not None:
            result = self.local(prompt, [], system_prompt=system,
                                assistant_prefill=assistant_prefill or None,
                                max_new_tokens=max_tokens, seed=seed, **sampling)
            text = result.get("text", "")
            if stage == "verifier" and self.model_mode == "thinking":
                if "</think>" in text:
                    self._verifier_thinking, text = text.rsplit("</think>", 1)
                    text = text.strip()
                else:
                    self._verifier_thinking = text
                    return "", "length"
            return text, result.get("finish_reason")
        chatml = ("<|im_start|>system\n" + system + "<|im_end|>\n"
                  "<|im_start|>user\n" + prompt + "<|im_end|>\n"
                  "<|im_start|>assistant\n" + assistant_prefill)
        body = dict(model=self.model, stream=False, keep_alive="15m",
                    options=dict(num_predict=max_tokens, **sampling, seed=seed, num_ctx=self.context_tokens))
        endpoint = "/api/generate"
        if stage == "verifier":
            body.update(messages=[dict(role="system", content=system), dict(role="user", content=prompt)])
            if self.model_mode == "thinking":
                body["think"] = True
            endpoint = "/api/chat"
        else:
            body.update(raw=True, prompt=chatml)
        if response_schema is not None:
            body["format"] = response_schema
        # Keep the exact wire payload, including SYSTEM and the closed thinking
        # prefix, separate from the shorter per-turn response log.
        if self.output_dir:
            directory = self.output_dir / "verifier" if stage.startswith("verifier") else self.output_dir
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "requests.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(body, ensure_ascii=False) + "\n")
        request = urllib.request.Request(self.base_url + endpoint,
                                         data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach Ollama at {self.base_url}: {exc.reason}") from exc
        self.metrics["prompt_tokens"] += result.get("prompt_eval_count", 0)
        self.metrics["completion_tokens"] += result.get("eval_count", 0)
        for key, remote in (("load_seconds", "load_duration"), ("prompt_seconds", "prompt_eval_duration"),
                            ("decode_seconds", "eval_duration")):
            self.metrics[key] += result.get(remote, 0) / 1e9
        if self.output_dir and stage == "rewrite":
            with (self.output_dir / "responses.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        if result.get("error"):
            raise RuntimeError("Ollama: " + str(result["error"]))
        if stage == "verifier":
            self._verifier_thinking = result.get("message", {}).get("thinking", "")
            if self.output_dir:
                with (self.output_dir / "verifier" / "responses.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            return result.get("message", {}).get("content", ""), result.get("done_reason")
        return result.get("response", ""), result.get("done_reason")

    def __call__(self, turn, media, previous=(), *, seed=None, review_feedback=None):
        if seed is None:
            seed = secrets.randbits(31)
        # Keep three consecutive messages in chronological order. They provide
        # the request/result/choice context without exposing future plan state.
        history = list(previous[-3:])
        last = history[-1] if history else None
        intent = turn["intent"]
        action = intent["action"]
        speaker = turn["role"].title()
        self._writer_speaker = speaker
        self._writer_purpose = ACT_GUIDANCE.get(action, "Express the supplied content requirements.")
        persona = self.personas[turn["role"]]
        prior_subject_discussion = []
        if intent.get("first_creation"):
            subjects = set(intent.get("asset_ids", []))
            prior_subject_discussion = [dict(role=old["role"], content=spoken_context(old)) for old in previous
                                        if subjects & set(old.get("intent", {}).get("asset_ids", []))]
        draft = intent["description"]
        if turn["role"] == "assistant":
            draft = ASSISTANT_CONTENT.get(action, draft)
        current_names = [self.asset_names[a] for a in intent.get("asset_ids", []) if a in self.asset_names]
        shown_references = [dict(kind=media[p].get("kind"), description=media[p]["description"],
                                 subjects=[self.asset_names.get(a, a) for a in media[p].get("asset_ids", [])])
                            for p in turn.get("reference_media", []) if p in media]
        if action in {"request_compilation", "reorder_shots"}:
            # These pointers already contain the exact chronological clip versions
            # in the requested order. Their generated caption prefix is the shot
            # description; reference attributes after it are background only.
            clips = [media[p]["description"].split(". References used:", 1)[0].rstrip(".")
                     for p in intent.get("context_media", []) if p in media and media[p].get("kind") == "video"]
            draft = ("Assemble the existing video clips in this order; " + "; ".join(
                f"{('First', 'Second', 'Third', 'Fourth')[i]}: {description}" for i, description in enumerate(clips))) if clips else draft.replace(" → ", "; Next clip: ")
        if action == "request_shot" and intent.get("shot_description"):
            draft = "Video content: " + intent["shot_description"] + ". Use the existing " + ", ".join(current_names) + " references."
        if action == "confirm_switch" and current_names:
            draft = f"Ask the user how the {current_names[0]} should look."
        if action == "request_preview":
            draft = "Requested result: another preview, with no new design or color-processing decision."
        if action == "vague_request" and current_names:
            draft = "Task to open: designing the " + current_names[0] + ". Appearance is not decided yet."
        if action in {"adopt", "rollback"}:
            subject = " of the " + ", ".join(current_names) if current_names else ""
            draft = (f"Version decision: return to the specified earlier reference{subject}." if action == "rollback"
                     else f"Version decision: select the referenced existing version{subject}.")
            choices = last.get("output_media", []) if last else []
            selected = intent.get("context_media", [])
            if action == "adopt" and len(choices) == 1 and selected == choices:
                draft = f"Version decision: accept the single reference{subject} just shown."
            if len(choices) > 1 and len(selected) == 1 and selected[0] in choices:
                draft += f" Selected option number: {choices.index(selected[0]) + 1}."
            draft += " Decision status: " + intent.get("decision", "accepted") + "."
        if action == "deliver_reworked_asset":
            draft = ("Completed result: the revised reference image. Clip updates are still pending. Agreed scope: "
                     + ("this shot only; other uses keep the earlier version."
                        if intent.get("scope") == "shot" else "this clip and other/future uses."))
        if "required_attribute" in intent and action not in REFERENCE_REQUESTS | {"request_candidates"}:
            attribute = intent["required_attribute"]
            draft += f"; Required {attribute['label']} of this version: {attribute['value']}."
        if action == "deliver_reworked_assets":
            for edit in intent.get("resolved_edits", []):
                name = self.asset_names.get(edit["asset_id"], edit["asset_id"])
                scope = "this shot only" if edit["scope"] == "shot" else "this shot and all other/future uses"
                draft += f"; {name}: revised reference ready, clip updates still pending for {scope}."
        # Project only facts this speech act needs to express. Detailed media
        # evidence belongs to the independent judge, not the spoken writer.
        minimal_delivery = (turn["role"] == "assistant" and bool(turn.get("output_media"))
                            and action in ASSISTANT_CONTENT and "required_attribute" not in intent
                            and action not in {"deliver_reworked_asset", "deliver_reworked_assets"})
        self_contained = (turn["role"] == "assistant" and bool(turn.get("output_media"))) or action in (
            ACKNOWLEDGMENTS | {"request_asset", "request_trial", "begin_task", "request_candidates",
                              "request_shot", "request_shot_update", "request_compilation", "reorder_shots",
                              "replace_compilation_shots", "clarify_goal", "confirm_switch",
                              "relative_request", "adopt", "rollback", "temporary_adoption_trial",
                              "ambiguous_reference", "request_reworks", "request_rule_followup"})
        writer_history = [] if self_contained else history
        draft = draft.replace("long shot", "wide shot")
        self._writer_control = self._writer_purpose
        if not previous and turn["role"] == "user":
            self._writer_control += " A fresh conversation starts. You are the user originating this request or these project requirements. Address the assistant directly with what you want them to do. The assistant is receiving your requirements for the first time; they have not given you instructions or an explanation to confirm."
        if turn["role"] == "assistant" and turn.get("output_media") and action in ASSISTANT_CONTENT:
            self._writer_control += " This is a delivery-only turn: use a single brief presentation, without a follow-up offer or question."
        rule_decision = intent.get("rule_event", {}).get("outcome") if action == "resolve_rule_conflict" else None
        if rule_decision == "keep":
            self._writer_control += " The user chooses to keep the earlier rule and withdraw the conflicting change. The current request must follow the earlier rule too. Do not ask for the rejected treatment as a test or exception."
        base_control = self._writer_control
        if review_feedback:
            self._writer_control += "\nReviewer correction to apply: " + str(review_feedback)
        lines = [f"Speak as the {speaker} with this tone:", persona["style"]]
        if persona.get("example"):
            lines += ["", "Tone examples:", "<tone_examples>", persona["example"], "</tone_examples>",
                      "Imitate tone only; use the current intention for content."]
        # Style evidence is judge-only: showing repeated openings to the writer
        # can prime precisely the language we are trying to diversify.
        openings = [" ".join(spoken_context(old).split()[:5])[:80] for old in previous
                    if old["role"] == turn["role"]][-8:]
        lines += ["", "Previous messages (oldest to newest):"]
        lines += [f"{old['role'].title()}: {spoken_context(old)}" for old in writer_history]
        if not writer_history:
            lines.append("(No previous message.)" if not previous else "(The current intention contains the necessary context.)")
        reply_instruction = (f"Now, write a natural reply to this message from the {speaker}'s point of view."
                             if writer_history else f"Now, write the current turn from the {speaker}'s point of view.")
        lines += ["", reply_instruction,
                  "", "Reply intention:", "Content requirements (meaning to express, not wording to copy):"]
        lines += ["- " + part.strip() for part in draft.split(";") if part.strip()]
        if not minimal_delivery and action not in {"ambiguous_reference", "ambiguous_shot_number", "request_reworks", "clarify_reworks", "resolve_reworks"} and current_names:
            lines.append("Subject(s): " + ", ".join(current_names))
        if turn["role"] == "assistant" and turn.get("output_media"):
            kinds = list(dict.fromkeys(media[p].get("kind", "media") for p in turn["output_media"] if p in media))
            lines.append("Ready media: " + ", ".join(kinds) + "; count: " + str(len(turn["output_media"])))
        correction = intent.get("reference_correction")
        if correction:
            shown = correction["shown_name"]
            target = correction["target_name"]
        bound_references = []
        if action not in {"ambiguous_reference", "ambiguous_shot_number", "request_reworks", "clarify_reworks", "resolve_reworks"}:
            bound_references = [dict(kind=media[p].get("kind"), purpose=media[p].get("purpose"),
                                     subjects=[self.asset_names.get(a, a) for a in media[p].get("asset_ids", [])])
                                for p in intent.get("context_media", []) if p in media]
        return_instruction = f"Return only the {speaker.lower()}'s spoken line."
        rule_context = intent.get('media_rule_context', [])
        if rule_context and turn['role'] == 'user':
            lines += ['', 'Exact rule scope for this generation request (other partial rules do not apply):'] + rule_context
            lines.append('These conditions are already agreed. Do not extend them to another output category. Color processing and background season are separate: original-color rendering is not a season. Underlying design colors are distinct from rendering treatment; current output conditions replace conflicting source-shot conditions.')
        if turn['role'] == 'assistant' and turn.get('output_media'):
            lines.append('This is only a delivery of the prepared result. Do not add appearance, scene, rule or processing claims, a question, or an offer of further work.')
        if intent.get('requested_media_role'):
            from scripts.dialogue_generation.cross_turn_rule import ROLE_NAMES
            lines += ['', 'Required media category: ' + ROLE_NAMES[intent['requested_media_role']] + '.',
                      'Name this category explicitly. A generic version or preview is insufficient when its category determines whether the rule applies.']
        if action == 'clarify_rule_conflict':
            lines += ['', 'Required question structure:',
                      '1. Remind the user of the existing rule and its scope.',
                      '2. Ask which of these three choices they want: keep the rule, an exception for this request only, or replacement going forward.',
                      'Do not choose, promise or announce any of these decisions yourself. The user has not answered yet.']
        if action == 'confirm_cross_turn_rule' and intent.get('scope') == 'partial':
            lines.append('Explicitly say that other output categories are not restricted by this particular rule. Do not leave this boundary implicit.')
        prompt = "\n".join(lines + ["", return_instruction])
        # Give the judge a factual evaluation brief, not a role-playing prompt.
        # Each semicolon-delimited planning note stays separate so a less salient
        # requirement is not hidden behind the main action. This only formats
        # supplied data; it does not judge meaning or match candidate wording.
        # Choices need the immediately preceding test/result as well as the
        # selected reference. Older selected versions remain explicit facts.
        verification_history = history if action in {
            "adopt", "rollback", "temporary_adoption_trial", "confirm_selection", "confirm_final", "relative_request"
        } else writer_history
        explicit_subjects = (current_names if not minimal_delivery and action not in {
            "ambiguous_reference", "ambiguous_shot_number", "request_reworks",
            "clarify_reworks", "resolve_reworks"
        } else [])
        brief = dict(
            speaker=turn["role"], speaking_style=persona["style"], action=action,
            subjects=explicit_subjects,
            conversation_start=not previous,
            python_description=intent["description"],
            reference_requirements=reference_requirements(intent),
            draft_content_requirements=[part.strip() for part in draft.split(";") if part.strip()],
            semantic_purpose=ACT_GUIDANCE.get(action, "Express the current content requirements."),
            intent_facts={key:intent[key] for key in ("decision", "scope", "first_creation", "required_attribute",
                         "numbering_operation", "numbering_basis", "requested_media_role") if key in intent},
            previous_messages=[dict(role=old["role"], content=spoken_context(old)) for old in verification_history],
            recent_same_speaker_openings=openings,
            bound_references=bound_references,
            shown_reference_media=[{key: value for key, value in row.items() if key != "description"} for row in shown_references],
            delivered_media=[dict(kind=media[p].get("kind"), purpose=media[p].get("purpose"),
                                 subjects=[] if minimal_delivery else [self.asset_names.get(a, a) for a in media[p].get("asset_ids", [])])
                            for p in turn.get("output_media", []) if p in media],
        )
        if rule_decision:
            brief["intent_facts"]["rule_decision"] = rule_decision
        if rule_context:
            brief['media_rule_context'] = rule_context
        if (action in EDIT_REQUESTS | {"request_variant"} and len(explicit_subjects) == 1 and previous
                and intent.get("reference_policy", "clear") == "clear"):
            last_subjects = previous[-1].get("intent", {}).get("asset_ids", [])
            if last_subjects and set(intent.get("asset_ids", [])) == set(last_subjects):
                target = explicit_subjects[0]
                attribute = intent.get("required_attribute", {})
                example = (f"try that with {attribute['value']} instead" if attribute.get("field") == "material"
                           else "try that with the requested change")
                brief["reference_transition"] = (
                    f"The preceding turn and this request concern the same {target}. "
                    "This is a continuation, not a topic switch. A clear pronoun may refer to the object just shown or discussed. "
                    f"For example, '{example}' can clearly modify {target} in this continuing context. "
                    "Do not require the speaker to repeat the object name merely because the Python plan requires clear references. "
                    "Judge the actual visible history; a continuing, unambiguous pronoun satisfies this check."
                )
            elif last_subjects and not set(intent.get("asset_ids", [])) <= set(last_subjects):
                before = ", ".join(self.asset_names.get(a, a) for a in last_subjects)
                target = explicit_subjects[0]
                brief["reference_transition"] = (
                    f"The preceding turn concerns {before}. The intended object now is the earlier {target}. "
                    f"PASS this reference-clarity check when the line names {target} or gives an identifying description. "
                    f"For example, 'the {target}', 'that {target}' and 'the previous {target}' all clearly name the object and satisfy this check. "
                    "No extra topic-switch announcement or particular wording is required. "
                    f"FAIL this check only when the line leaves the object unidentified, such as saying only 'that' after discussing {before}. "
                    f"'That {target}' includes the object's name; it is not the same as a bare 'that'. "
                    "For an unidentified object, lower conversation_coherence; for an identified object, do not lower a score on this check."

                )
        if action == "adopt" and len(selected) == 1 and selected[0] in choices:
            brief["offered_choice_count"] = len(choices)
            brief["selected_choice_number"] = choices.index(selected[0]) + 1
        # An alternative may be a future planned variant. It is not a fact
        # about the current conversation and is unnecessary for this judgment.
        if "required_attribute" in brief["intent_facts"]:
            brief["intent_facts"]["required_attribute"] = {
                key: value for key, value in intent["required_attribute"].items() if key != "alternative"
            }
            if current_names:
                # In a composition, the contrast belongs to the first asset,
                # not to every subject appearing alongside it in the test.
                brief["intent_facts"]["required_attribute"]["subject"] = current_names[0]
        if action == "rollback" and intent.get("required_attribute", {}).get("field") and current_names:
            attribute = brief["intent_facts"]["required_attribute"]
            selected = set(intent.get("context_media", []))
            asset = intent["asset_ids"][0]
            # Inspect only images already delivered in this conversation. The
            # full media table also contains future versions, which must stay hidden.
            earlier_outputs = [media[p] for old in reversed(previous) for p in old.get("output_media", [])
                               if p in media and p not in selected and media[p].get("kind") == "image"
                               and asset in media[p].get("asset_ids", [])]
            other = next((row for row in earlier_outputs
                          if row.get("attributes", {}).get(attribute["field"]) not in (None, attribute["value"])), None)
            if other:
                brief["previous_variant"] = dict(subject=current_names[0], label=attribute["label"],
                                                 value=other["attributes"][attribute["field"]])
        if correction:
            brief["reference_correction"] = dict(shown=shown, intended=target)
        if intent.get("first_creation"):
            brief["prior_subject_discussion"] = prior_subject_discussion
            brief["subject_previously_discussed"] = bool(prior_subject_discussion)
        if turn["role"] == "assistant" and turn.get("output_media"):
            brief["delivery_requirement"] = "Briefly present the ready media, without a follow-up offer, question, or extra claims about appearance, scene, rules or processing. Such added claims are outside this delivery intention."
        if action in ACKNOWLEDGMENTS:
            brief["acknowledgment_requirement"] = "Confirm the already-made decision; no question or next production step."
        problems = []
        for attempt in range(MAX_REWRITE_ATTEMPTS):
            self._last_finish_reason = None
            self.rewrite_attempts += 1
            text = self._request(prompt, seed + attempt)
            finish_reason = self._last_finish_reason
            if isinstance(text, str) and "</think>" in text:
                text = text.rsplit("</think>", 1)[1]
            text = text.strip() if isinstance(text, str) else text
            problems = _format_problems(text)
            if problems:
                self.metrics["format_rejections"] += 1
            report = None
            verifier_error = None
            if not problems:
                try:
                    report = self.verifier(text, brief, turn_id=turn["id"], rewrite_attempt=attempt,
                                           seed=((seed + attempt) ^ 0x5F3759DF) & 0x7FFFFFFF,
                                           generation_finish_reason=finish_reason)
                except RuntimeError as exc:
                    verifier_error = str(exc)
                    self.metrics["verifier_failures"] += 1
                    problems = [verifier_error]
                if report is not None and not passed(report):
                    self.metrics["semantic_rejections"] += 1
                    problems = [report["critique"]]
            if self.output_dir:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                record = dict(turn=turn["id"], attempt=attempt, prompt=prompt,
                              response=text, accepted=not problems, history_turns=len(writer_history),
                              previous_turns=len(previous), context_role=writer_history[-1]["role"] if writer_history else None,
                              verification_history_turns=len(verification_history),
                              context_roles=[old["role"] for old in writer_history],
                              context_policy="action_relevant_context", style_openings=openings,
                              sampling=dict(self.sampling), seed=seed + attempt,
                              persona=persona["id"], problems=problems, verification=report,
                              verification_error=verifier_error,
                              finish_reason=finish_reason,
                              response_characters=len(text) if isinstance(text, str) else 0,
                              nonempty_lines=sum(bool(line.strip()) for line in text.splitlines())
                              if isinstance(text, str) else 0)
                with (self.output_dir / "calls.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if verifier_error:
                raise RuntimeError(verifier_error)
            if not problems:
                self.metrics["accepted_candidates"] += 1
                return text
            # Feedback describes the LAST candidate. Do not accumulate stale
            # diagnoses of candidates which have already been replaced.
            correction = report["critique"] if report else " ".join(problems)
            self._writer_control = base_control + "\nCorrection to apply: " + correction
            if review_feedback:
                self._writer_control += "\nReviewer correction: " + str(review_feedback)
            prompt = "\n".join(lines + ["", return_instruction])
        raise RuntimeError(f"Qwen rewrite rejected for turn {turn['id']} after {MAX_REWRITE_ATTEMPTS} attempts: "
                           + " ".join(problems))
