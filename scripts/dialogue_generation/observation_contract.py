"""Export the observation schema and upstream VLM prompt from one catalog.

No video loading or model calls happen here.
"""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dialogue_generation.asset_variations import CATALOG, CATEGORIES, empty_attributes


def input_schema():
    variants = []
    for kind, category in CATEGORIES.items():
        attrs = dict(type="object", required=list(category["attributes"]), additionalProperties=False,
                     properties={key: dict(type="string", enum=spec["values"])
                                 for key, spec in category["attributes"].items()})
        if category.get("constraints"):
            attrs["allOf"] = [
                {"if": {"properties": {key: {"enum": vals} for key, vals in rule["if"].items()}},
                 "then": {"properties": {key: {"enum": vals} for key, vals in rule["then"].items()}}}
                for rule in category["constraints"]]
        variants.append(dict(type="object", required=["id", "kind", "name", "description", "frames", "attributes"],
                             properties={"id": dict(type="string", minLength=1), "kind": dict(const=kind),
                                         "name": dict(type="string", minLength=1),
                                         "description": dict(type="string", minLength=1),
                                         "frames": dict(type="array", minItems=1, maxItems=2,
                                                        items=dict(type="number", minimum=0)),
                                         "attributes": attrs}))
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Video observation with finite asset attributes", "type": "object",
        "required": ["assets", "shots"],
        "properties": {
            "attribute_schema_version": {"const": CATALOG["schema_version"]},
            "video": {"type": "string"},
            "assets": {"type": "array", "minItems": 1, "items": {"oneOf": variants}},
            "shots": {"type": "array", "minItems": 1, "maxItems": 4,
                      "items": {"type": "object", "required": ["id", "description", "assets"],
                                "properties": {"id": {"type": "string", "minLength": 1},
                                               "description": {"type": "string", "minLength": 1},
                                               "assets": {"type": "array", "minItems": 1, "uniqueItems": True,
                                                          "items": {"type": "string"}},
                                               "start": {"type": "number", "minimum": 0},
                                               "end": {"type": "number", "minimum": 0}}}},
            "target_order": {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
            "constraints": {"type": "array", "items": {"type": "object", "required": ["text", "scope"],
                "properties": {"text": {"type": "string"}, "scope": {"enum": ["shots", "all"]},
                               "asset_attributes": {"type": "object", "additionalProperties": {
                                   "type": "object", "additionalProperties": {"type": "string"}}}}}},
        },
    }


def example_asset(kind, name, description, **known):
    return dict(id=kind, kind=kind, name=name, description=description, frames=[1.0],
                attributes={**empty_attributes(kind), **known})


def observation_prompt():
    lines = [
        "# Target video observation prompt", "",
        "Observe the supplied final video. Return compact JSON only. Record observations and choose enum values. Leave editing and dialogue planning to later stages.",
        f"Output video, attribute_schema_version ({CATALOG['schema_version']}), assets, shots, and target_order. Describe one to four shots.",
        "For each asset include id, kind, name, description, frames (one or two timestamps in seconds where it is visible), and attributes.",
        "Choose kind from person, object, indoor_environment, outdoor_environment. Track people and objects separately.",
        "Reuse an id for the same person, object, or environment across shots. If the final video has distinct visible versions, use a separate id for each version.",
        "Track the principal people, objects, and environments. List each shot's id, start, end, description (action and composition), and assets (the relevant asset ids).",
        "List ids in final playback order in target_order. Every asset must appear in at least one shot.",
        *CATALOG["notes"],
        "Write names, descriptions, and any constraint text in English. Use exact enum strings as printed below. Use short, version-neutral names, such as traveler or tote bag.",
        "Use unknown for details such as actual material, apparent height, wind strength, and season when the video cannot establish them. Use numbers only in numeric fields.",
        "Include constraints only when the input explicitly supplies lasting production requirements. A visible property alone does not establish a persistent user rule.",
        "For lasting rules, constraints[].asset_attributes may map asset ids to locked attribute values. These must agree with the target observation.",
        "Follow the compatibility rules below.", "",
        "## Attribute values", "",
    ]
    for kind, category in CATEGORIES.items():
        lines += [f"### {category['label']} ({kind})", "",
                  "| Field | Meaning | Allowed values |", "| --- | --- | --- |"]
        for key, spec in category["attributes"].items():
            lines.append(f"| {key} | {spec['label']} | " + ", ".join(spec["values"]) + " |")
        lines.append("")
        for rule in category.get("constraints", []):
            condition = " and ".join(key + " is " + " or ".join(vals) for key, vals in rule["if"].items())
            consequence = ", and ".join(key + " must be " + " or ".join(vals) for key, vals in rule["then"].items())
            lines.append(f"If {condition}, then {consequence}.")
        lines.append("")
    examples = [
        ("A traveler has short hair and a beige coat. Other details are unclear.", [example_asset(
            "person", "traveler", "Traveler", hair_length="short hair", clothing_type="coat",
            clothing_color="beige")]),
        ("A blue matte canvas tote bag has a wide strap. Other details are unclear.", [example_asset(
            "object", "tote bag", "Tote bag with a wide strap", color="blue", material="canvas",
            surface_finish="matte")]),
        ("A warmly lit coffee shop interior and an alpine lake under clear skies. The exact time and season are unclear.", [
            example_asset("indoor_environment", "coffee shop", "Wooden tables and glass entrance",
                          room_type="coffee shop", lighting_temperature="warm-toned"),
            example_asset("outdoor_environment", "alpine lake", "Reflections of distant peaks",
                          setting_type="lake", geography="alpine terrain", weather="clear skies")]),
    ]
    lines += ["## Complete examples", ""]
    for index, (text, assets) in enumerate(examples, 1):
        shots = []
        for offset, asset in enumerate(assets):
            asset["frames"] = [offset * 2 + 1.0]
            shots.append(dict(id=f"S{offset + 1}", start=offset * 2.0, end=(offset + 1) * 2.0,
                              description=f"Static shot of the {asset['name']}", assets=[asset["id"]]))
        observed = dict(video=f"fictional://example-{index}.mp4", attribute_schema_version=CATALOG["schema_version"],
                        assets=assets, shots=shots, target_order=[shot["id"] for shot in shots])
        lines += [f"Example {index}. Input: {text}", "", "Output:", "```json",
                  json.dumps(observed, ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "observation.schema.json").write_text(json.dumps(input_schema(), ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "observation_prompt.md").write_text(observation_prompt())


if __name__ == "__main__":
    main()
