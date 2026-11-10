"""Natural-language extractor instructions and task-specific examples."""
from tools.attribute_observation_prompts import DEFINITIONS, OUTPUT_RULES

IMAGE_EXAMPLE = """A woman stands in a studio.
The woman wears a sculptural white hat.
The picture has a surreal, colorful editorial style.
The studio has a white background."""
VIDEO_EXAMPLE = """A woman wears a red coat in an office.
The woman hands a package through the service window.
A rear wide view continuously moves closer to her hands.
A hard cut reveals a sunlit courtyard."""

def extraction_system_prompt():
    return ("You are extractor, a perceptive visual observer whose purpose is to describe "
            "the essential, clearly visible attributes of one artifact.\n"
            + OUTPUT_RULES + DEFINITIONS
            + "\nStill-image example, only if supported by that image:\n" + IMAGE_EXAMPLE
            + "\nVideo example, only if those events are visible:\n" + VIDEO_EXAMPLE
            + "\nReturn only your observation, with no media reference markers.")
