"""Finite attributes shared by observation, sampling and version validation."""
from copy import deepcopy
import json
from pathlib import Path

CATALOG = json.loads(Path(__file__).with_name('attribute_catalog.json').read_text())
CATEGORIES = CATALOG['categories']
UNKNOWN, NA = CATALOG['unknown_value'], CATALOG['not_applicable_value']
UNOBSERVED = {UNKNOWN, NA}
_MISSING = object()
POOLS = {kind: tuple(row['attributes']) for kind, row in CATEGORIES.items()}
RELATIVE_POOLS = {kind: tuple(key for key, spec in row['attributes'].items() if 'ordered_values' in spec)
                  for kind, row in CATEGORIES.items()}

def asset_type(row):
    kind = row.get('kind')
    if kind not in CATEGORIES:
        raise ValueError('Asset kind must be person/object/indoor_environment/outdoor_environment')
    return kind

def fields(row):
    specs = deepcopy(CATEGORIES[asset_type(row)]['attributes'])
    for key, extra in row.get('attribute_extensions', {}).items():
        current = specs.setdefault(key, dict(label=extra.get('label', key.replace('_', ' ')), values=[UNKNOWN]))
        current['values'] = list(dict.fromkeys(current['values'] + extra['values']))
    return specs


def register_attributes(row, attrs):
    """Persist model additions per observed identity, without changing the catalog."""
    if not isinstance(attrs, dict):
        return
    if row['kind'] == 'person' and 'age_group' in attrs:
        attrs['age_group'] = canonical_value('age_group', attrs['age_group'])
    specs = fields(row)
    for key, value in attrs.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip():
            raise ValueError('Attribute names and values must be nonempty strings')
        if key not in specs or value not in specs[key]['values']:
            extra = row.setdefault('attribute_extensions', {}).setdefault(key,
                dict(label=key.replace('_', ' '), values=[UNKNOWN]))
            extra['values'] = list(dict.fromkeys(extra['values'] + [value]))

def validate_attributes(row, attributes=_MISSING):
    kind = asset_type(row)
    attrs = row.get('attributes') if attributes is _MISSING else attributes
    specs = fields(row)
    if not isinstance(attrs, dict) or set(attrs) != set(specs):
        missing = sorted(set(specs) - set(attrs)) if isinstance(attrs, dict) else list(specs)
        extra = sorted(set(attrs) - set(specs)) if isinstance(attrs, dict) else []
        raise ValueError(f"Asset {row.get('id', '?')} requires every {kind} attribute; missing={missing}, extra={extra}")
    for key, value in attrs.items():
        if key == 'age_group' and isinstance(value, str):
            value = canonical_value(key, value)
        if not isinstance(value, str) or value not in specs[key]['values']:
            raise ValueError(f"Invalid enumerated attribute {row.get('id', '?')}.{key}: {value!r}")
    for rule in CATEGORIES[kind].get('constraints', []):
        if all(attrs[key] in values for key, values in rule['if'].items()):
            if any(attrs[key] not in values for key, values in rule['then'].items()):
                raise ValueError(f"Incompatible attributes for {row.get('id', '?')}: {rule}")
    return True

def empty_attributes(kind):
    """For explicit unknowns in examples; production input must supply all keys."""
    return dict.fromkeys(POOLS[kind], UNKNOWN)

class AssetSpec(str):
    """Source prose plus complete current state; writers resolve attribute conflicts."""
    def __new__(cls, row, attributes=None, *, change=None, focus=(), max_fields=7):
        attrs = deepcopy(row['attributes'] if attributes is None else attributes)
        if row['kind'] == 'person' and 'age_group' in attrs:
            attrs['age_group'] = canonical_value('age_group', attrs['age_group'])
        validate_attributes(row, attrs)
        kind = asset_type(row)
        deviations = [key for key, value in attrs.items() if value != row['attributes'][key]]
        keys = list(dict.fromkeys(list(focus) + ([change['field']] if change else []) + deviations
                                  + list(row.get('attribute_extensions', {})) + CATEGORIES[kind]['summary_fields']))
        visible = [key for key in keys if attrs[key] not in UNOBSERVED][:max_fields]
        text = row['description'] + ("; " if visible else "") + ", ".join(
            f"{fields(row)[key]['label'].lower()}: {attrs[key]}" for key in visible)
        obj = super().__new__(cls, text)
        obj.attributes, obj.change, obj.row = attrs, deepcopy(change), row
        return obj

def specification(row, value=None):
    return value if isinstance(value, AssetSpec) else AssetSpec(row)

# Observation retains detailed metadata. Only these fields/values are sampled.
COLORS = ('black', 'white', 'gray', 'beige', 'brown', 'red', 'orange', 'yellow',
          'green', 'cyan', 'blue', 'purple', 'pink', 'gold', 'silver')
MATERIALS = ('paper', 'glass', 'wood', 'fabric', 'metal', 'cotton', 'powder', 'leather')
AGES = ('child', 'teenager', 'adult', 'elder')
SEASONS = ('spring', 'summer', 'fall', 'winter')
TIMES = ('night', 'dawn', 'daytime', 'dusk')
WEATHER = ('sunny with few clouds', 'sunny with many clouds', 'light rain', 'thunderstorm',
           'light snow', 'heavy snow', 'overcast')
CLOTHING_COLORS = COLORS + ('red and white', 'blue and white', 'black and white',
                            'white and yellow', 'green and yellow', 'pink and purple')
VARIATION_VALUES = {
    'object': {'color': COLORS, 'material': MATERIALS},
    'person': {'clothing_color': CLOTHING_COLORS, 'age_group': AGES},
    'indoor_environment': {'season': SEASONS, 'time_of_day': TIMES, 'weather': WEATHER},
    'outdoor_environment': {'season': SEASONS, 'time_of_day': TIMES, 'weather': WEATHER},
}
VISIBLE_FIELDS = {kind: set(values) for kind, values in VARIATION_VALUES.items()}
DISABLED_VARIATIONS = {kind: set(category['attributes']) - VISIBLE_FIELDS[kind]
                       for kind, category in CATEGORIES.items()}
VISUAL_WEIGHTS = {'color': 2, 'clothing_color': 2}


def canonical_value(key, value):
    """Compare visual meanings rather than spelling or unordered color lists."""
    import re
    text = value.strip().lower()
    aliases = {
        'material': {'cotton fabric': 'cotton', 'cloth': 'fabric', 'textile': 'fabric',
                     'textiles': 'fabric', 'canvas': 'fabric', 'silk': 'fabric', 'felt': 'fabric', 'nylon': 'fabric', 'wooden': 'wood', 'faux leather': 'leather'},
        'age_group': {'young child': 'child', 'school-age child': 'child',
                      'young adult': 'adult', 'middle-aged adult': 'adult', 'older adult': 'elder',
                      'elderly': 'elder', 'teen': 'teenager', 'adolescent': 'teenager'},
        'time_of_day': {**dict.fromkeys(('early morning', 'morning', 'noon', 'afternoon'), 'daytime'),
                        'late night': 'night', 'evening': 'dusk'},
        'season': {'autumn': 'fall'},
        'weather': {'clear skies': 'sunny with few clouds', 'partly cloudy': 'sunny with many clouds'},
    }
    if key in ('color', 'clothing_color'):
        text = text.replace('grey', 'gray')
        return tuple(sorted(set(v.strip() for v in re.split(r'\s+(?:and|with)\s+|[,/+&]', text) if v.strip())))
    return aliases.get(key, {}).get(text, text)


def current_value(row, key, value):
    """Recover explicit garment colors when an older observer used 'multicolored'."""
    if key == 'clothing_color' and value.lower() in ('multicolored', 'multicoloured'):
        import re
        garment = re.search(r'(?i)\bwearing\b(.*?)(?:\b(?:carrying|holding)\b|[.;]|$)', row['description'])
        if garment:
            colors = re.findall(r'(?i)\b(?:' + '|'.join(COLORS) + r')\b', garment[1])
            if len(set(colors)) > 1:
                return canonical_value(key, ' and '.join(colors))
    return canonical_value(key, value)


def visually_distinct(key, before, after):
    return canonical_value(key, before) != canonical_value(key, after)


def design_key(row, attributes):
    """A complete design combination, ignoring aliases and color-list order."""
    return tuple((key, current_value(row, key, value)) for key, value in sorted(attributes.items()))


def alternatives(row, base=None, *, relative=False, locked=(), tried=()):
    base = specification(row, base)
    result = {}
    for key, spec in fields(row).items():
        if key in locked or key not in VISIBLE_FIELDS[asset_type(row)] or base.attributes[key] in UNOBSERVED:
            continue
        # Model-defined values extend this identity's categorical candidates.
        # Keep the editable-field policy; unknown baselines are still skipped.
        candidates = list(VARIATION_VALUES[asset_type(row)][key])
        if key != 'age_group':
            candidates += row.get('attribute_extensions', {}).get(key, {}).get('values', [])
        values, seen = [], set()
        for value in candidates:
            meaning = canonical_value(key, value)
            if meaning not in seen:
                values.append(value)
                seen.add(meaning)
        if relative:
            values = [v for v in spec.get('ordered_values', []) if v in values]
        if relative:
            current = canonical_value(key, base.attributes[key])
            ordered = [canonical_value(key, value) for value in values]
            if current not in ordered:
                continue
            position = ordered.index(current)
            values = [value for index, value in enumerate(values) if abs(index - position) == 1]
        valid = []
        for value in values:
            if value in UNOBSERVED or current_value(row, key, base.attributes[key]) == canonical_value(key, value):
                continue
            attrs = {**base.attributes, key: value}
            if design_key(row, attrs) in tried:
                continue
            try:
                validate_attributes(row, attrs)
            except ValueError:
                continue
            valid.append(value)
        if valid:
            result[key] = valid
    return result

def alternate(row, description, rng, *, relative=False, locked=(), tried=()):
    base = specification(row, description)
    options = alternatives(row, base, relative=relative, locked=locked, tried=tried)
    if not options:
        raise ValueError('No observed, editable attributes for ' + row['id'])
    key = rng.choices(list(options), weights=[VISUAL_WEIGHTS.get(k, 1) for k in options], k=1)[0]
    value = rng.choice(options[key])
    change = dict(field=key, before=base.attributes[key], after=value)
    return AssetSpec(row, {**base.attributes, key: value}, change=change)

COMPARATIVES = {
    'age_group': ('younger', 'older'), 'hair_length': ('shorter', 'longer'), 'makeup_intensity': ('lighter', 'heavier'),
    'height': ('shorter', 'taller'), 'clothing_fit': ('more fitted', 'looser'),
    'color_lightness': ('darker', 'lighter'), 'color_saturation': ('less saturated', 'more saturated'),
    'size': ('smaller', 'larger'), 'aspect_ratio': ('longer and narrower', 'wider and flatter'),
    'space_size': ('smaller', 'larger'), 'ceiling_height': ('lower', 'higher'),
    'furniture_density': ('less densely furnished', 'more densely furnished'),
    'clutter': ('tidier', 'more cluttered'), 'background_crowd': ('less crowded', 'more crowded'),
    'lighting_level': ('dimmer', 'brighter'), 'camera_distance': ('farther back', 'closer'),
    'vegetation_density': ('sparser', 'denser'), 'background_density': ('lower', 'higher'),
    'visibility': ('clearer', 'hazier'), 'wind_level': ('weaker', 'stronger'),
}

def relative_variation(row, desired, rng, *, locked=(), tried=()):
    desired = specification(row, desired)
    earlier = alternate(row, desired, rng, relative=True, locked=locked, tried=tried)
    key = earlier.change['field']
    order = fields(row)[key]['ordered_values']
    direction = int(order.index(canonical_value(key, desired.attributes[key])) > order.index(canonical_value(key, earlier.attributes[key])))
    adjective = COMPARATIVES[key][direction]
    subject = f"the {fields(row)[key]['label'].lower()} of the {row['name']}"
    request = (f"Make the {row['name']} {adjective}, in the {desired.attributes[key]} age group." if key == 'age_group' else f"Make {subject} a little {adjective}.")
    question = (f"Here is the {adjective} {row['name']} in the {desired.attributes[key]} age group. Is that what you meant?" if key == 'age_group' else f"I made {subject} a little {adjective} in this preview. Is that what you meant?")
    return key, earlier, request, question


def changed_attributes(before, after):
    return {key: dict(before=before[key], after=value)
            for key, value in after.items() if canonical_value(key, value) != canonical_value(key, before[key])}


def edit_request(row, before, after):
    before, after = specification(row, before), specification(row, after)
    changes = changed_attributes(before.attributes, after.attributes)
    commands = [f"set the {fields(row)[key]['label'].lower()} to {change['after']}" for key, change in changes.items()]
    return f"For the {row['name']}, " + " and ".join(commands) if commands else f"Keep the current {row['name']}"
