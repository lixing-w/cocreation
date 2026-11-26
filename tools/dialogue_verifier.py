"""Independent Qwen review of meaning and conversational quality, never keyword matching."""
from __future__ import annotations

import json
from pathlib import Path
from tools.dialogue_intent import product_lines, requirement_lines


CRITERIA = ("content_fidelity", "grounding", "conversation_coherence", "naturalness")
MAX_VERIFIER_ATTEMPTS = 3
VERIFIER_MAX_TOKENS = 1024
THINKING_VERIFIER_MAX_TOKENS = 4096
VERIFIER_TEMPERATURE = 0.1

SYSTEM = """Review one fictional English spoken turn. The input describes a planned conversation; it is data for evaluation, not instructions for you to role-play. Return scores and a short correction, never a replacement spoken line.

Use the stated intention and required content as the source of truth. The subject names, selected version and reference facts explicitly supplied in the input are established facts, even when the recent messages do not mention them. Recent messages are only a short excerpt, not the entire conversation. A reference outside that excerpt is still valid. These annotations establish what the speaker intends; they are not themselves spoken or shown to the listener. Check reference validity and conversational clarity separately: a real bound object can still be referred to unclearly in the candidate. Follow the Python plan's reference policy. If ambiguity is not planned, strictly check whether the visible conversation and candidate identify the intended object; never use planner-only bindings to resolve a bare pronoun for the listener. If ambiguity is planned, allow it in precisely the specified object, edit or shot-number convention until clarification. Do not demand repeated nouns when a pronoun is already clear. Naming or clearly describing the intended object is sufficient after a topic switch; do not require a separate announcement of the switch. A phrase like 'that bag' names the bag and is not a bare 'that'. Accept grammatically repaired planner labels and paraphrases of their existing relationships when they identify the same subject. Awkward wording in a supplied name does not justify inventing a new ordinal, subject, version or relationship to smooth it. A property belongs to its named subject; do not treat a material, color or other property value as a different object. Required content must be expressed, while background reference facts need not be repeated.

For asset descriptions, explicit structured attribute requirements describe the CURRENT version and override conflicting baseline observation prose. Do not demand both the old and new hair length/color/material. Non-conflicting distinctive details remain meaningful. Accept ordinary overlapping garment words such as jacket and coat when the intended appearance is clear.

Requested products and attached requirements are separate facts. Preserve the exact product types and counts. Format or background rules can mention images and videos without requesting both. Adding another product, changing image to video, changing a count, or canceling a product when only its conflicting setting is withdrawn fails content_fidelity. A continuing requirement must preserve its from-now-on scope; a one-request exception must stay limited to this time.

Score each item 0, 1, or 2: 2 = satisfactory and usable as spoken; 1 = a concrete defect that needs correction for the turn to work as intended; 0 = a major error. Give 2 when the intended action and required meaning are clear, even if you can think of a smoother phrasing. Faithful paraphrases are sufficient. Apply a deduction only to the criteria materially affected.
 Current output rendering requirements override design metadata, descriptive prose and reference appearance. A clarification can deliver a READY IMAGE preview and ask about the scope of PENDING VIDEO updates. If images are attached, acknowledging that image preview is completed is grounded, but saying the video has already changed is not. Scope choices concern one existing shot or all existing shots, never a future-shot option. A new-video intention must not be rewritten as reformatting an earlier clip.
content_fidelity: Does the candidate perform this speaker's intended action and convey the required details? Check medium, appearance, option/version choice, framing, shot order and revision scope when required. Expressions such as 'wider shot' and 'short black hair' are valid. In shot-size terminology, 'long shot', 'wide shot' and 'wider shot' describe wide framing rather than clip duration; accept these equivalent framing expressions. For a requested wood variant, 'wood texture design' is also an acceptable visual paraphrase unless the line explicitly contradicts the requested change; do not impose a materials-engineering distinction on ordinary creative dialogue. Initial appearance requirements must be conveyed; an existing reference's unrelated properties need not be repeated. A delivery of ready media can be complete without describing its appearance. Phrases like "take a look whenever you like", "ready for you to explore", and "you can watch it now" present the result already being delivered, even in a separate short sentence. Give full credit for these ordinary delivery cues. A follow-up offer proposes additional work, such as offering to make another version. A follow-up question asks the user to answer or make a decision. A version choice is complete when the intended version is clear; no declaration about how long it will be used is required.
grounding: Is each claimed subject, version, prior event, result and scope supported by the supplied facts? Naming an explicitly supplied subject is grounded, even if it appears only in the subject or reference section. Choosing a named earlier version is grounded when the intention specifies that earlier version. Keep each property attached to the correct subject. The user requests the assistant to make media, rather than offering to upload it. For a first reference creation, do not invent an earlier generated image; earlier discussion alone is allowed. For a fresh conversation, the user originates the request or requirements. An acknowledgment implying that the user has received or understood the assistant's instructions invents a prior event and merits grounding below 2. Judge this implication even when the rest of the line repeats the correct requirements.
conversation_coherence: Does the line make sense for this speaker at this point? A version choice and a correction of the intended object are different actions. For referent clarification, acknowledge the object and ask what change the user wants, without adopting a version. In an open appearance question, mentioning optional dimensions such as color or texture can help the user answer. Accept these as prompts for the user to supply details when the candidate leaves the choices open. Preserve the difference between an image and a clip, requested and completed work, and one-shot and wider revision scope. Intentionally incompatible user requests and deliberately vague references are valid until their planned clarification. If the plan calls for an ambiguous reference, do not require the hidden object or edit to be named. "Keep your reference deliberately ambiguous" is an instruction to the writer. Judge whether the line leaves the reference unresolved without announcing the instruction aloud.
naturalness: Score 2 for clear, plausible spoken English in the immediate exchange. Allow conversational ellipsis and fragments when context supplies the missing words, as well as ordinary lead-ins, hedges, pauses and light repetition. A detailed description or instruction can sound natural in speech, including a dense shot brief with filmmaking terms. Both a casual "This one, please" and a full description of a camera move can earn 2. Allow acknowledgments grounded in the exchange, and remember that no opening phrase is forbidden. Give 2 for readily understandable speech even when phrasing is a little clunky, a supplied label sounds awkward, or some words repeat. When a clause is only optional commentary, give 2 if the rest of the line is clear and complete and the clause adds no conflicting meaning. For example, "The lighting test clip is ready, one video to see how those details come across" earns 2 even if "those details" is vague. The listener already knows what is ready. A vague reference warrants a deduction when resolving it is necessary to understand the intended action or result. Score 1 only when a concrete fluency problem makes the turn materially hard to understand or use, and 0 for largely incoherent or unusable speech. Persona adherence, assigned tone, length and sentence count are outside this rubric. Judge factual errors, role reversals and contradictions under the semantic criteria.

Before lowering a score, check the complete intention and established facts again. Identify a contradiction, omission or conversational defect that changes the meaning or gets in the way of the turn. A merely preferable rewrite is not a correction. Do not invent another object, version or requirement to explain a concern. Judge completeness by meaning, not length or token-limit status.
The critique is sent directly to the writer as a turn requirement. For any score below 2, begin with "You should" and give one or two short sentences of direct revision advice. State the intended subject, property or action clearly. Skip diagnosis and summaries of passing items. For example: "You should swap shot 1 and shot 2 without describing the clips or resolving the numbering convention." If all scores are 2, critique is empty. Return one flat JSON object only: content_fidelity, grounding, conversation_coherence, naturalness (integer scores), and critique (string). No nested reports, evidence arrays, markdown or analysis.
"""


KEEP_SYSTEM = 'Evaluate whether a user reply expresses the required_decision. This is a completed rule decision: its stated setting applies to the pending generation and its stated continuing scope. The pending media content is still to be generated using that setting.\nScore content_fidelity (required decision, scope, and continued media generation), grounding (supported setting and scope), conversation_coherence (consistent meaning), and naturalness (fluent spoken English; ordinary lead-ins, hesitations and light repetition can earn 2): 2 correct, 1 minor defect, 0 major defect. Persona adherence, assigned tone, sentence count and word count are outside this rubric. Judge matching meanings, not literal wording. For example, retaining winter and making the requested scene in winter fulfills a keep-winter decision; retaining winter only for later while making summer now does not.\nRequested_outputs gives the authoritative product types and counts. The user must continue those same products. Image/video categories in a rule describe settings, not additional requested products. Changing an image request to video, adding or dropping a product, or changing a count MUST score content_fidelity 0. An unambiguous reference to the same pending products is sufficient without repeating every count.\nReturn only JSON with those four integer scores and critique. All four 2 means critique is empty. Otherwise critique is one short direct instruction beginning with "You should", at most 240 characters. Do not put deliberation or analysis in critique.\nA cancellation, deferral, or abandonment of the pending media request MUST score content_fidelity 0. Making the conflicting setting now, even while retaining the rule for later, MUST score content_fidelity and grounding below 2. Narrowing the required continuing scope also fails those items. Check all these obligations before scoring.'


REPORT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {**{key: {"type": "integer", "enum": [0, 1, 2]} for key in CRITERIA},
                   "critique": {"type": "string"}},
    "required": [*CRITERIA, "critique"],
}

# Bound correction prose in the wire schema so a judge cannot fill its token
# budget with self-debate inside the JSON string. Do not truncate responses.
KEEP_REPORT_SCHEMA = {**REPORT_SCHEMA, "properties": {**REPORT_SCHEMA["properties"],
    "critique": {"type": "string", "maxLength": 240}}}



def parse_report(text):
    """Check only the small report protocol; the model judges meaning."""
    try:
        report = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ValueError("Verifier must return one valid JSON report") from exc
    if not isinstance(report, dict) or set(report) != {*CRITERIA, "critique"}:
        raise ValueError("Verifier report needs four scores and critique")
    if any(type(report[key]) is not int or report[key] not in (0, 1, 2) for key in CRITERIA):
        raise ValueError("Verifier scores must be integers from 0 to 2")
    if not isinstance(report["critique"], str):
        raise ValueError("Verifier critique must be a string")
    if not passed(report) and not report["critique"].strip():
        raise ValueError("A below-maximum score needs an actionable critique")
    if passed(report) and report["critique"].strip():
        raise ValueError("A passing report must have an empty critique")
    return report


def passed(report):
    return all(report[key] == 2 for key in CRITERIA)


def shot_staging_lines(description):
    return ["Shot staging (setting, framing, camera, action and layout):", description,
            "The spoken request must state the concrete scene, framing, actions and movement direction. A generic request to preserve 'the staging' is incomplete; the listener cannot see these notes.",
            "Appearance in this source is background, not a requirement. Current reference attributes override conflicting source details. Appearance may be omitted, but any stated property MUST match the current reference state. An obsolete source color, material or age is incorrect. No redesign or recoloring is requested."]


def source_design_lines(description):
    return ["Required visual design content (current attributes override conflicting values):", description,
            "Express these supplied visual details as the desired image content. Current attributes replace conflicting descriptions. Do not require details that are absent from these notes and attributes."]


def reference_state_lines(overrides):
    lines = ["Current visual attributes (use these values where they conflict with design notes):"]
    for row in overrides:
        attributes = ", ".join(key.replace('_', ' ') + ": " + str(value)
                               for key, value in row['attributes'].items()
                               if value not in (None, '', 'unknown', 'not applicable'))
        if attributes:
            option = ("Option " + str(row['output_position']) + " / ") if row.get('output_position') else ""
            lines.append("- " + option + row["subject"] + ": " + attributes)
    return lines if len(lines)>1 else []


def build_prompt(candidate, brief, *, generation_finish_reason=None):
    """Render supplied planning facts as explicit prose, without a JSON context tree."""
    if not isinstance(brief, dict):
        brief = {"draft_content_requirements": [str(brief)]}
    role = brief.get("speaker", "specified speaker")
    action = brief.get("action", "unspecified")
    subjects = brief.get("subjects", [])
    subject = ", ".join(subjects)
    facts = brief.get("intent_facts", {})
    lines = ["Evaluate the candidate against the following planned turn.",
             "", "Speaker: " + role.title() + "."]
    if brief.get("conversation_start"):
        lines += ["Conversation position: A fresh conversation starts.",
                  "The user is introducing their own request or requirements to the assistant for the first time.",
                  "The assistant has not provided any rules, instructions or explanation for the user to acknowledge or confirm.",
                  "Opening-turn grounding check: Does the candidate originate the request, or imply a response to earlier instructions? A response to earlier instructions is unsupported, even if the remaining content is correct. For example, 'I understand your requirements' would invent prior instructions in this opening turn."]
    else:
        lines.append("Conversation position: The conversation is already in progress.")
    spoken_action = "choose_variant_and_request_test" if action == "temporary_adoption_trial" else action
    lines += ["", "Planned action: " + spoken_action,
              "Meaning of this action: " + brief.get("semantic_purpose", "Express the required content."),
              "", "Content the candidate must express:"]
    lines += ["- " + part for part in brief.get("draft_content_requirements", [])]
    if brief.get('requested_outputs'):
        lines += ['', *product_lines(brief['requested_outputs'])]
    if brief.get('requested_outputs') or brief.get('additional_requirements'):
        lines += ['', *requirement_lines(brief.get('additional_requirements', []))]
    if brief.get('shot_staging'):
        lines += ['', *shot_staging_lines(brief['shot_staging'])]
    if brief.get('source_design'):
        lines += ['', *source_design_lines(brief['source_design'])]
    if brief.get("python_description"):
        lines += ["", "Original Python plan description (intent.description):", brief["python_description"],
                  "Use this to understand the planned action. Background design details in the plan need not all be repeated; the content requirements above state what this line must convey."]
    if brief.get('attribute_overrides'):
        lines += ['', *reference_state_lines(brief['attribute_overrides'])]
    if brief.get("reference_requirements"):
        lines += ["", "Python-planned reference clarity requirements:"]
        lines += ["- " + requirement for requirement in brief["reference_requirements"]]
    if brief.get('media_rule_context'):
        lines += ['', 'Exact applicable rules for this particular output category (authoritative; rules limited to other categories do not apply):'] + brief['media_rule_context']
        lines.append('These are already stored rules, not content the speaker must repeat. Routine requests and deliveries should omit them. Season and aspect ratio are independent settings. Current rules override conflicting source-shot conditions. Keep deliveries focused on the prepared result. Do not add unsupported appearance, processing or rule claims. Penalize a statement that extends a partial rule to another output category.')
    if action == 'request_candidates':
        lines += ['', 'Output unit: two separate images or two independent single-shot video clips, matching the stated medium. The user must make these two outputs clear. A comparison layout combining both options in one output is a different request and fails content_fidelity. Comparing two separate results is valid.']
    if action in {"incompatible_requirements", "incompatible_framing"}:
        lines += ["", "This user request is deliberately impossible. It must ask for BOTH incompatible requirements at the same time. The next assistant turn will explain the conflict and ask the user to choose.",
                  "Judge whether BOTH requirements and the stated single-view/shot/file restrictions are expressed. The contradiction itself is correct for this turn: do not lower any score because the requested combination is impossible, and do not tell the writer to remove either requirement or make them compatible.",
                  "Omitting either requirement, choosing one, or changing the requirements to make them compatible is a content_fidelity error. For example, if the plan asks for clearly visible background details AND background blur that hides those details, a candidate must request both; asking only for sharp details or only for blur is incomplete."]
    if action == "clarify_conflict":
        lines += ["", "The preceding user deliberately requested incompatible requirements. This assistant turn must explain the incompatibility and ask the specified choice. It must not claim it can satisfy both together."]
    if action in {'request_rule_conflict', 'propose_rule'}:
        lines += ['', 'This user turn deliberately conflicts with an earlier user rule. That is the intended training behavior, not a grounding or coherence error. The user may request a change. A later assistant turn will point out the conflict and ask for a decision. Do not require this user turn to resolve or acknowledge the conflict yet.']
    if action == 'clarify_rule_conflict':
        lines += ['', 'If the content requirements offer three decisions, the question must include all three: retain the earlier rule, allow this request only, or replace the rule going forward. Omitting the keep option is incomplete. The assistant must not announce an exception or replacement before the user has chosen it. Saying it will make an exception while asking about later outputs is a premature decision and must fail content_fidelity.']
    if action in {'set_cross_turn_rule','confirm_cross_turn_rule','resolve_rule_conflict','confirm_rule_resolution'}:
        lines += ['', 'Rule-scope fidelity is required. A rule for all videos must not become a rule only for the current kayak or person, and a global rule covers every generated image/video. Keep the full scope in the content requirements. A partial-rule confirmation must explicitly say other output types are outside this particular rule. Do not invent other rules when none are supplied.']
    if action == 'resolve_rule_conflict':
        lines += ['', 'The user is authorized to keep, make a one-request exception to, or replace their own earlier rule. Judge the specified decision, rather than treating a chosen override as disobedience.']
    if facts.get("rule_decision") == "keep":
        lines += ["", "Required user decision: keep the earlier rule with its full stated scope AND continue the pending media request under that rule. Only the conflicting setting is withdrawn, not the media request. Accept a clear request to create the same content using the retained setting; the literal word withdraw is not required. Referring to the earlier request while explicitly changing its setting to the retained one is valid. Canceling or abandoning the media request fails content_fidelity. Continuing with the conflicting treatment also fails. No exception is authorized."]
    if facts.get('requested_media_kind'):
        lines += ['', 'Requested media kind: ' + facts['requested_media_kind'] + '.',
                  'The request must distinguish image from video. Do not require any trial/formal/final category.']
    if action == 'request_rule_exception':
        lines += ['', 'The user explicitly authorizes a differing requirement for this single request only. Preserve the continuing rule for later outputs. No further clarification or approval is required.']
    if action == 'request_rule_followup':
        lines += ['', 'Any earlier one-request exception has expired. The exact applicable rules supplied for this request are now back in force. Repeating the exception is an incorrect new request, even if the recent history mentions it.']
    if subjects:
        lines += ["", "Established subject(s) for this turn: " + subject + ".",
                  "These subject names are supplied facts. Naming them is not inventing a new object."]
    if action in {"adopt", "rollback"}:
        target = "the earlier reference version" if action == "rollback" else "the referenced existing option"
        lines += ["", "Version being chosen: " + target + (" of " + subject if subject else "") + ".",
                  "This existing version is explicitly established by the plan, even if its original delivery is outside the recent messages."]
    if brief.get("offered_choice_count") == 1:
        lines += ["The assistant just delivered one reference, and the user is accepting that single reference.",
                  "No numbered alternatives were offered. Any ordinal or numbered choice, including first, first one, option 1 and version 1, invents a comparison and must fail grounding and conversation_coherence. A natural acceptance is This one works or Let us use this one. Do not require the object name again when the reference is already clear."]
    elif brief.get("offered_choice_count"):
        lines += [f"The assistant offered {brief['offered_choice_count']} alternatives in the preceding delivery.",
                  f"The user must choose option {brief['selected_choice_number']}; keep this option binding."]
    attribute = facts.get("required_attribute")
    if attribute:
        attribute_subject = attribute.get("subject") or subject or "the subject specified in the content requirements"
        owner = ("the version of " + attribute_subject + " being chosen" if action in {"adopt", "rollback"}
                 else attribute_subject)
        lines.append("Required " + attribute["label"] + " of " + owner + ": " + attribute["value"] + ".")
        if attribute.get("field") == "camera_distance" and attribute["value"] == "long shot":
            lines.append("Terminology for this requirement: the Python plan's long shot means a wide shot. Wide shot or wider shot is a faithful expression of this framing; it does not need the literal words long shot.")
    if action == "rollback":
        other = brief.get("previous_variant")
        if other:
            lines.append("Newer variant being left behind: " + other["subject"] + "; " + other["label"] + ": " + other["value"] + ".")
        lines.append("For this rollback, the required choice is the earlier version described above. The newer variant or test in recent messages is background, not the version to accept now.")
    if facts.get("scope"):
        lines.append("Scope of the current requirement or change: " + {
            "shot": "only the specified shot; other uses keep their existing versions.",
            "all_uses": "all currently existing shots that use the specified subject. Do not add a future-shot choice.",
            "shots": "project footage, subject to the stated reference exemptions."
        }.get(facts["scope"], str(facts["scope"])))
    if brief.get('reference_creation_requirement'):
        lines += ['', 'Required source availability check:',brief['reference_creation_requirement'],
                  'A request to preserve or match an unavailable current/source reference invents media and must score grounding below 2. Merely asking to create a new reference image is valid.']
    if facts.get("first_creation"):
        lines.append("Reference creation state: this is the first generated reference for these subjects. Earlier discussion may exist, but an earlier generated reference does not.")
    operation = facts.get("numbering_operation")
    if operation:
        a, b = operation["positions"]
        instruction = (f"Move shot {a} after shot {b}." if operation["type"] == "move_after"
                       else f"Swap shot {a} and shot {b}.")
        lines.append("Required operation involving shot numbers: " + instruction)
    if facts.get("numbering_basis"):
        lines.append("Shot-number meaning: " + {
            "current": "use the positions in the current cut, not generation order."
        }.get(facts["numbering_basis"], str(facts["numbering_basis"])))
    correction = brief.get("reference_correction")
    if correction:
        lines += ["", "The assistant just showed: " + correction["shown"] + ".",
                  "The user actually meant the earlier reference of: " + correction["intended"] + ".",
                  "This corrects the referent; it does not select a version or request a completed edit."]
    for key, heading in (("bound_references", "Planner-only target bindings; these are not newly spoken or shown to the listener"),
                         ("shown_media", "References visibly shown in this turn"),
                         ("delivered_media", "Prepared results being delivered in this turn")):
        rows = brief.get(key, [])
        if not rows:
            continue
        lines += ["", heading + " (background facts; a visual description need not be spoken):"]
        for i, row in enumerate(rows, 1):
            medium = {"image": "still image", "video": "video clip"}.get(row.get("kind"), "media")
            name = ", ".join(row.get("subjects", []))
            lines.append("- " + medium + (" of " + name if name else "") + ".")
    for key in ("delivery_requirement", "acknowledgment_requirement"):
        if brief.get(key):
            lines += ["", "Requirement for this response: " + brief[key]]
    discussion = brief.get("prior_subject_discussion", [])
    if discussion:
        lines += ["", "Earlier discussion specifically about the current subject (context, not additional requirements):"]
        lines += ["- " + row["role"].title() + ": " + row["content"] for row in discussion]
    recent = brief.get("previous_messages", [])
    lines += ["", "Recent spoken messages, oldest to newest (partial history; current requirements take priority):"]
    if recent:
        lines += [f"{i}. {row['role'].title()}: {row['content']}" for i, row in enumerate(recent, 1)]
    elif brief.get("conversation_start"):
        lines.append("There are no previous messages.")
    else:
        lines.append("Recent messages are omitted for this action. Use the explicit subject, version and reference facts above; an omitted older message does not make those facts ungrounded.")
    openings = brief.get("recent_same_speaker_openings", [])
    if openings:
        lines += ["", "Earlier openings from this speaker (fragments for judging repetition only):"]
        lines += [f"{i}. {opening}" for i, opening in enumerate(openings, 1)]
    if generation_finish_reason == "length":
        lines += ["", "Generation reached its token budget. Judge whether the actual sentence is complete; this status alone is not an error."]
    if brief.get("reference_transition"):
        lines += ["", "Specific reference-coherence check for this turn:", brief["reference_transition"]]
    lines += ["", "Candidate spoken line to evaluate:", json.dumps(candidate, ensure_ascii=False),
              "", "Return only the five-field score JSON: content_fidelity, grounding, conversation_coherence, naturalness, critique."]
    return "\n".join(lines)


def build_keep_prompt(candidate,brief):
    """The chosen decision already resolves the old request's conflicting setting."""
    return json.dumps(dict(required_decision=' '.join(brief.get('draft_content_requirements',[])),
        requested_outputs=brief.get('requested_outputs', []),
        additional_requirements=brief.get('additional_requirements', []),
        candidate=candidate),ensure_ascii=False)



class QwenVerifier:
    def __init__(self, request, *, output_dir=None, keep_request=None):
        self.request = request
        self.keep_request = keep_request
        self.attempt_count = 0
        self.output_dir = Path(output_dir) if output_dir is not None else None

    def __call__(self, candidate, brief, *, turn_id, rewrite_attempt, seed, generation_finish_reason=None):
        keeping=isinstance(brief,dict) and brief.get('intent_facts',{}).get('rule_decision')=='keep'
        prompt=build_keep_prompt(candidate,brief) if keeping else build_prompt(candidate,brief,generation_finish_reason=generation_finish_reason)
        request=self.keep_request if keeping and self.keep_request is not None else self.request
        for attempt in range(MAX_VERIFIER_ATTEMPTS):
            self.attempt_count += 1
            raw, finish_reason = request(prompt, seed + attempt)
            report, problem = None, None
            try:
                report = parse_report(raw)
            except ValueError as exc:
                problem = str(exc) + (" (generation hit its token limit)" if finish_reason == "length" else "")
            if self.output_dir:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                record = dict(turn=turn_id, rewrite_attempt=rewrite_attempt, attempt=attempt,
                              prompt=prompt, response=raw, finish_reason=finish_reason,
                              seed=seed + attempt, report=report, error=problem,
                              passed=passed(report) if report else False)
                with (self.output_dir / "calls.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if problem is None:
                return report
            # A malformed judge response is a judge failure, not evidence that
            # the spoken candidate needs changing. Retry the SAME candidate.
        raise RuntimeError(f"Qwen verifier failed for turn {turn_id}: {problem}")
