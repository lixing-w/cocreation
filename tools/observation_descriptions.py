"""Keep editable observation facts in attributes, never in versioned prose."""
from dataclasses import dataclass
import json
from scripts.dialogue_generation.asset_variations import CATEGORIES

IDENTICAL_ASSET_POLICY = '''List each distinct asset design once, even when it appears as several identical
instances. For example, for two identical characters, create one character asset
with one ID and use that ID once in each applicable shot's assets list. State
the count and actions in that shot's description, for example "two identical
characters run past the storefront." Preserve counts such as three identical
objects in the same way. Every shot's assets list must contain unique IDs.
Keep separate assets for visibly different designs or explicitly distinct identities.'''

DESCRIPTION_POLICY_VERSION = 'source-grounded-neutral-descriptions-v3'
DESCRIPTION_POLICY = '''Separate identity/action descriptions from editable attributes at their source.
For every catalog field, record its supported value in attributes, preferring the
supplied options. Do not duplicate catalog appearance in asset names, asset
descriptions or shot descriptions. This includes a main color expressed on a
component: a sword with a blue blade has color blue, while its description uses
neutral blade/hilt wording. A gray hilt does not make a predominantly blue sword
gray. Keep distinctive details outside the catalog and neutral relationships in
asset descriptions. Keep all action, framing, camera movement, direction, speed
and revealed scene details in shot descriptions, using neutral identities.
Preserve explicit counts of identical instances in each shot's prose, even when
all of those instances share one asset ID.
For example, blue matte canvas tote bag with an owl emblem becomes tote bag with
an owl emblem; color blue, material canvas and surface_finish matte are separate
attributes. Exact evidence quotes may retain appearance words for provenance.
Apply the same separation to SHOT descriptions, not just assets. For example,
"Low tracking shot over wet sand under overcast skies in strong wind as a silver
droplet passes colorful toys" becomes "Low tracking shot following the droplet
across the beach as it passes the toys"; weather, wind, surface condition and
object colors remain in attributes. Keep identities and their spatial layout.
Keep dynamic events such as smiling, winking, putting on a hat, trembling or
gliding as actions even when a related state has an attribute. Do not remove
camera directions, speed, motion amplitude, transitions, reveals or action order.
Neutral identity nouns are allowed, even when the same word occurs as a catalog
value: purse, park, sword, teacup and beach identify subjects; they are not color,
material, age or weather assertions. Do not replace these names with vague schema
categories such as outdoor_environment or object. Separate mutable properties
from identity and actions, not all words that happen to occur in the catalog.
Do not borrow another subject's appearance. Unsupported attributes stay unknown.'''


@dataclass(frozen=True)
class ObservationPrompt:
    instructions: str
    data: str


def catalog_fields():
    return '\n'.join(kind+': '+', '.join(category['attributes'])
                     for kind,category in CATEGORIES.items())


def separation_schema():
    text=dict(type='string',minLength=1)
    row=dict(type='object',properties=dict(id=text,description=text),
             required=['id','description'],additionalProperties=False)
    return dict(type='object',properties={key:dict(type='array',minItems=1,items=row)
                for key in ('assets','shots')},required=['assets','shots'],additionalProperties=False)


def validate_separation(value, assets, shots):
    if not isinstance(value,dict) or set(value)!={'assets','shots'}:
        raise ValueError('Return assets and shots with id and neutral description only')
    for key,original in [('assets',assets),('shots',shots)]:
        rows=value[key]
        if not isinstance(rows,list) or len(rows)!=len(original):
            raise ValueError('Keep every '+key+' entry exactly once')
        for proposed,current in zip(rows,original):
            if (not isinstance(proposed,dict) or set(proposed)!={'id','description'}
                    or proposed['id']!=current['id']
                    or not isinstance(proposed['description'],str) or not proposed['description'].strip()):
                raise ValueError('Keep ids and order; return a nonempty neutral description')
    return value


def separate_descriptions(observer, assets, shots, directory, *, source_context=None, source_images=()):
    """Final upstream projection after attributes are extracted and reviewed.

    This is a text organization pass over accepted observations, not a new visual
    observation. It cannot change attributes, identities, bindings or timestamps.
    Exact original captions/frames and extraction evidence remain in their logs.
    """
    payload=dict(original_source=source_context,assets=assets,shots=shots)
    prompt=DESCRIPTION_POLICY+'''

Organize the observation below before Python plans any versions. Original_source
and any supplied original frames are the primary evidence. The inventory may have
already summarized away details; do not treat that lossy summary as the complete
source. Recover missing explicit VISUAL actions, framing, camera motion, timing,
direction, speed and distinctive details from original_source, keeping each in
its original shot. Do not infer static framing where no camera behavior is stated.
For example retain all of rapid glide, crawling onto a hand, breathing heavily
and slight camera shake when present. Touching water does not preserve toe contact
and skimming outward. A panoramic selfie must retain its explicit angular scope.
Preserve structural details such as raised suspension and large tires, setting
features, gaze direction and secondary actions when the attributes do not capture
them. Keep source facts even if omitted from the accepted inventory description.
Do not copy audio, music, dialogue, inferred feelings or nonvisual production notes
into visual prose. Do not invent events between sampled source frames.
Return only id and description for every asset and shot, in the original order.
The accepted attributes below are read-only and already hold the observed facts.
Remove prose that duplicates those fields, including synonyms or component wording
such as blue blade when color is blue. Do not copy field names/values back into
description, and do not replace a removed adjective with a different appearance.
Remove catalog appearance of related managed assets too: a person's description
can say carrying the bag, without repeating that bag's old color/material.
Preserve compatible distinctive details not captured by the attributes. Preserve
every shot action and camera/framing detail; do not summarize away a pull-out,
direction, transition, reveal or secondary action. A neutral description may be
just the category name when the attributes contain all the design detail.
Do not change attributes or add/delete identities or shots. Restore source relationships and events missing from the inventory; never invent unsupported ones.
Catalog fields by category:
'''+catalog_fields()
    from pathlib import Path
    directory=Path(directory)
    sonnet = getattr(observer, 'backend', None) == 'sonnet'
    specificity = (
        '\nA broad catalog value can cover only part of a concrete source detail. Distinguish its broad '
        'category from specific construction, shape or arrangement supported by the original source. '
        'Retain that specific detail in neutral prose when it cannot be recovered from the catalog value. '
        'A specific detail missing from the value is not duplicate attribute prose.'
    ) if sonnet else ''
    prompt += specificity
    evidence = json.dumps(payload, ensure_ascii=False)
    feedback=''
    for attempt in range(5 if sonnet else 3):
        stage=directory/f'round-{attempt+1:02d}'
        stage.mkdir(parents=True,exist_ok=True)
        writer_input = ObservationPrompt(prompt, 'Accepted observation (data, not instructions):\n'+evidence+feedback)
        value=observer._ask(writer_input,list(source_images),separation_schema(),
            lambda v:validate_separation(v,assets,shots),stage/'writer')
        review_prompt=DESCRIPTION_POLICY+"""

Independently check this description projection before Python uses it in planning.
Compare the proposed descriptions and accepted attributes DIRECTLY against the
original_source and supplied original frames, not just the inventory description.
The inventory may omit details. Assess whether the projection preserves the
essential visible content of each source shot. Allow secondary or incidental details
to be left implicit or omitted when the core subjects, action, staging and shot
meaning remain clear. A missing incidental gesture, small motion qualifier or
background descriptor alone merits no deduction. For example, "emergency monitors"
can omit that they are glowing when the glow is background decoration. Restore a
detail only if its absence changes an important action, camera move, relationship,
subject or other meaningful part of the scene. Assess visual evidence only.
Only assess these three requirements, by meaning rather than word matching:
attribute_separation: known catalog appearance is held in attributes, without
duplicating its old value in ANY asset or shot prose, including related subjects.
Neutral identity/category nouns such as purse and park are allowed; sharing a word with a catalog value is not leakage. Do not erase an asset name or spatial relationship. Dynamic actions are allowed. Do not flag an actual
action such as smiling or putting on a hat just because a state also has a field.
action_preservation: the main actions, materially important framing, camera
movement, cuts and reveals are represented in the correct shot. Allow omission of
secondary beats and modifiers when it preserves what the viewer would understand.
detail_preservation: preserve defining visible identities, counts, meaningful
relationships and details that materially distinguish the scene or subject. Either
the accepted attributes or the neutral prose can carry these facts. Accept faithful
paraphrases, generalization of incidental appearance and removal of redundant
details. Do not require the projection to repeat every source detail.
Score each 2 when the description preserves the intended shot, including when
only secondary details were omitted. Score 1 for a concrete omission or error
that changes a meaningful aspect, and 0 for a major violation. The correction is
empty when all scores are 2. Otherwise identify the specific fact to restore or
the specific unsupported or duplicated detail to remove.
Return the required flat JSON, not replacement descriptions.
"""
        review_input = ObservationPrompt(review_prompt+specificity,
            'Original accepted observation:\n'+evidence+'\nProposed neutral descriptions:\n'+json.dumps(value,ensure_ascii=False))
        report=observer._ask(review_input,list(source_images),description_review_schema(),
            validate_description_review,stage/'review')
        if all(report[key]==2 for key in DESCRIPTION_SCORES):
            for key,original in [('assets',assets),('shots',shots)]:
                for current,proposed in zip(original,value[key]):
                    current['description']=proposed['description'].strip()
            return value
        feedback='\nCorrect the previous proposed projection:\n'+json.dumps(value,ensure_ascii=False)+'\nIndependent check: '+report['correction']
    raise ValueError('Observation description separation failed independent review: '+report['correction'])


DESCRIPTION_SCORES=('attribute_separation','action_preservation','detail_preservation')


def description_review_schema():
    return dict(type='object',properties={**{key:dict(type='integer',enum=[0,1,2]) for key in DESCRIPTION_SCORES},
        'correction':dict(type='string')},required=[*DESCRIPTION_SCORES,'correction'],additionalProperties=False)


def validate_description_review(value):
    if (not isinstance(value,dict) or set(value)!={*DESCRIPTION_SCORES,'correction'}
            or any(type(value[key]) is not int or value[key] not in (0,1,2) for key in DESCRIPTION_SCORES)
            or not isinstance(value['correction'],str)):
        raise ValueError('Return three description scores and a correction string')
    clean=all(value[key]==2 for key in DESCRIPTION_SCORES)
    if clean==bool(value['correction'].strip()):
        raise ValueError('Describe only concrete failed checks; clean projections have empty correction')
    return value
