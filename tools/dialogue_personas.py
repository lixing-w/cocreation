"""One seeded speaking style per conversation, independent of action sampling."""
from copy import deepcopy
import random


USER_PERSONAS = {
    "casual": dict(name="Alex",
                   style="Chat like a collaborator you know. Use contractions and relaxed phrasing. Fragments are fine if the request stays clear.",
                   example="1. Wait, I meant the bag from earlier. 2. Can we try that one first? 3. Yeah, that works."),
    "cute": dict(name="Pip",
                 style="Sound warm, playful, and a little excited. Vary your openings and sentence rhythm. Use occasional exclamations without relying on catchphrases or emoji.",
                 example="1. Ooh, that's closer! 2. Yep, that one looks right. 3. Could we give it a tiny tweak?"),
    "precise": dict(name="Morgan",
                    style="Be direct and even-toned. Say exactly what the change applies to. Use complete sentences when they clarify scope.",
                    example="1. I mean the one from the first preview. 2. Apply the change to this shot only."),
    "verbose": dict(name="Jamie",
                    style="Think aloud in natural English, using up to three short sentences. You can restate the current request for clarity. Do not add a reason, story, or preference the draft does not give.",
                    example="1. Let me go back a second. The one I meant was the version from the first preview. I'd like to use that one here."),
    "concise": dict(name="Sam",
                     style="Get straight to the point. Use one compact sentence or a fragment, retaining the target, scope, order, and any exclusions.",
                     example="1. The earlier one. 2. Swap those two. 3. This shot only."),
}
ASSISTANT_PERSONA = dict(
    name="Creative assistant",
    style="Sound calm, clear, and collaborative. Deliver the result or ask the specific question needed. Use everyday English and vary brief acknowledgments.",
)


def choose_personas(seed, user_persona=None):
    if user_persona is not None and user_persona not in USER_PERSONAS:
        raise ValueError("Unknown user persona")
    key = user_persona or random.Random(f"persona:{seed}").choice(tuple(USER_PERSONAS))
    return dict(user=dict(id=key, **deepcopy(USER_PERSONAS[key])),
                assistant=dict(id="creative_assistant", **deepcopy(ASSISTANT_PERSONA)))
