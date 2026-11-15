"""Explicitly incompatible requests with their distinct reasons."""

CONFLICTS = {
    "framing": {
        "requires_any": ("object",),
        "request": "In the same shot, show the whole setting in a wide shot and have the object alone fill the frame in close-up at the same time. Keep it to one shot.",
        "question": "A wide shot needs the whole setting in view. A close-up filled by the object leaves the setting out. Which would you like to see first, the wide shot or the close-up?",
        "terms": (r"\b(?:wide(?:[- ]angle)? (?:shot|view)|wide|whole (?:setting|scene))\b", r"\bclose[- ]?up\b"),
        "reason": r"(?:leav\w*|cut\w*|crop\w*) .*out|los\w* (?:the )?(?:setting|surroundings|background)|(?:can(?:not|'t)|won't|wouldn't) see|(?:hide|obscur|exclud)\w*|fill\w* the frame|no room",
    },
    "camera_motion": {
        "requires_any": ("person", "object"),
        "request": "Keep the camera position and orientation completely fixed throughout the shot, and make that same camera orbit the subject a full 360 degrees without a cut or camera switch.",
        "question": "A locked camera stays in one place, but a full orbit requires moving the camera around the subject. Which should we shoot first, the locked-off view or the orbit?",
        "terms": (r"\b(?:fixed|stationary|locked(?:[- ]off)?|still)\b", r"\b(?:orbit\w*|circl\w*|360|(?:mov\w*|go\w*) around)\b"),
        "reason": r"mov\w*|chang\w* position|stay\w* (?:put|in (?:place|one place))|can't (?:do both|stay)",
    },
    "depth_of_field": {
        "requires_any": ("indoor_environment", "outdoor_environment"),
        "request": "In the same frame, make every background detail clearly visible and blur the entire background until none of its details can be made out. Use a single view, with no split screen.",
        "question": "Blurring the whole background would hide those details. Which should we go with, a sharp background or a completely blurred one?",
        "terms": (r"\b(?:sharp|clear(?:ly visible|ly defined)?|readable|legible|in focus|crisp)\b", r"\b(?:blur\w*|out of focus)\b"),
        "reason": r"hid\w*|obscur\w*|los\w*|wip\w* out|can(?:not|'t) (?:see|make out|keep)|wouldn't (?:be visible|see)",
    },
    "shot_structure": {
        "request": "Make this one continuous unbroken take, and put exactly two obvious hard cuts in the middle. Keep the cuts visible instead of disguising them with transitions.",
        "question": "A continuous take stays unbroken, and two hard cuts would break it into separate shots. Would you like the single uninterrupted take or the two hard cuts?",
        "terms": (r"\b(?:continuous|unbroken|uninterrupted|single take|one take|one long take|no (?:edits|cuts))\b", r"\bhard[- ]cuts?\b"),
        "reason": r"\b(?:break\w*|interrupt\w*|split\w*|separat\w*|end\w*)\b",
    },
    "aspect_ratio": {
        "request": "Deliver one video file with one full frame that is both 9:16 portrait and 16:9 landscape. No crop, borders, or separate versions.",
        "question": "A single uncropped frame has one aspect ratio, so it can't be both 9:16 portrait and 16:9 landscape. Which format should we use?",
        "terms": (r"\b(?:9\s*:\s*16|portrait|vertical)\b", r"\b(?:16\s*:\s*9|landscape|horizontal)\b"),
        "reason": r"(?:one|single|different|same) (?:fixed )?(?:aspect ratio|ratio|shape|frame|orientation)|can't be both",
    },
}
