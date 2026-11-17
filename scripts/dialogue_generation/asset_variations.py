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
    """Compact dialogue text with complete state; description is invariant content."""
    def __new__(cls, row, attributes=None, *, change=None, focus=(), max_fields=7):
        attrs = deepcopy(row['attributes'] if attributes is None else attributes)
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

# Keep observed scale as metadata, but do not manufacture scale differences in
# isolated product images: their framing supplies no reliable size reference.
DISABLED_VARIATIONS = {'object': {'size', 'color_saturation', 'color_lightness', 'condition'},
                       'person': {'height', 'makeup_intensity', 'face_shape', 'eye_shape', 'eyebrow_shape', 'nose_shape', 'lip_shape'}}
VISIBLE_FIELDS = {
    'object': {'color','shape','aspect_ratio','material','surface_finish','texture','pattern','decoration','transparency'},
    'person': {'hair_length','hair_style','hair_color','clothing_type','clothing_color','clothing_fit','accessory'},
    'indoor_environment': {'room_type','interior_style','layout','wall_material','floor_material','palette','furniture_style','window_type','clutter','lighting_source','time_of_day','camera_distance'},
    'outdoor_environment': {'setting_type','geography','season','weather','surface_material','surface_condition','vegetation_density','vegetation_color','water_state','water_color','background_density','lighting_source','time_of_day','camera_distance'},
}
VISUAL_WEIGHTS = {'color': 8, 'shape': 8, 'material': 5, 'pattern': 4,
                  'hair_color': 7, 'hair_style': 5, 'clothing_color': 8,
                  'clothing_type': 5, 'decoration': 3, 'aspect_ratio': 3}
CONTRAST_GROUPS = {
    'shape': [('round', 'oval', 'spherical', 'hemispherical'),
              ('square', 'rectangular', 'rounded rectangle', 'flat'),
              ('triangular', 'conical'), ('cylindrical',),
              ('elongated', 'streamlined', 'spindle-shaped'), ('irregular',)],
    'surface_finish': [('matte','semi-matte','satin finish'), ('glossy','glazed','mirror finish')],
    'hair_style': [('sleek straight hair','voluminous straight hair','side part','center part','slicked back'), ('slightly wavy hair','wavy hair','tousled hair'), ('blunt bangs','wispy bangs','choppy bangs')],
    'weather': [('clear skies','partly cloudy','overcast')],
    'texture': [('smooth','fine-textured'), ('rough','coarse-textured')],
    'material': [('canvas', 'cotton fabric', 'leather', 'faux leather', 'nylon', 'silk', 'felt', 'paper'),
                 ('wood', 'bamboo'), ('ceramic', 'stone'), ('glass', 'plastic'), ('metal',), ('rubber',)],
    'color': [('black', 'gray'), ('white', 'cream', 'beige'), ('brown', 'khaki', 'tan'),
              ('red', 'orange', 'pink'), ('yellow', 'gold'), ('blue', 'cyan'), ('green',), ('purple', 'violet')],
}

def visually_distinct(key, before, after):
    groups = CONTRAST_GROUPS.get('color' if key.endswith('_color') else key, ())
    return not any(before in group and after in group for group in groups)

def alternatives(row, base=None, *, relative=False, locked=()):
    base = specification(row, base)
    result = {}
    for key, spec in fields(row).items():
        if key in locked or key not in VISIBLE_FIELDS[asset_type(row)] or key in DISABLED_VARIATIONS.get(asset_type(row), ()) or base.attributes[key] in UNOBSERVED:
            continue
        values = spec.get('ordered_values', []) if relative else spec['values']
        if relative and base.attributes[key] in values:
            position = values.index(base.attributes[key])
            values = [value for index, value in enumerate(values) if abs(index - position) == 1]
        groups = spec.get('variation_groups')
        if groups:
            values = next((group for group in groups if base.attributes[key] in group), []) if not relative else []
        valid = []
        for value in values:
            if value in UNOBSERVED or value == base.attributes[key] or (not relative and not visually_distinct(key, base.attributes[key], value)):
                continue
            attrs = {**base.attributes, key: value}
            try:
                validate_attributes(row, attrs)
            except ValueError:
                continue
            valid.append(value)
        if valid:
            result[key] = valid
    return result

def alternate(row, description, rng, *, relative=False, locked=()):
    base = specification(row, description)
    options = alternatives(row, base, relative=relative, locked=locked)
    if not options:
        raise ValueError('No observed, editable attributes for ' + row['id'])
    key = rng.choices(list(options), weights=[VISUAL_WEIGHTS.get(k, 1) for k in options], k=1)[0]
    value = rng.choice(options[key])
    change = dict(field=key, before=base.attributes[key], after=value)
    return AssetSpec(row, {**base.attributes, key: value}, change=change)

COMPARATIVES = {
    'hair_length': ('shorter', 'longer'), 'makeup_intensity': ('lighter', 'heavier'),
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

def relative_variation(row, desired, rng, *, locked=()):
    desired = specification(row, desired)
    earlier = alternate(row, desired, rng, relative=True, locked=locked)
    key = earlier.change['field']
    order = fields(row)[key]['ordered_values']
    direction = int(order.index(desired.attributes[key]) > order.index(earlier.attributes[key]))
    adjective = COMPARATIVES[key][direction]
    subject = f"the {fields(row)[key]['label'].lower()} of the {row['name']}"
    request = f"Make {subject} a little {adjective}."
    question = f"I made {subject} a little {adjective} in this preview. Is that what you meant?"
    return key, earlier, request, question


def changed_attributes(before, after):
    return {key: dict(before=before[key], after=value)
            for key, value in after.items() if value != before[key]}


def edit_request(row, before, after):
    before, after = specification(row, before), specification(row, after)
    changes = changed_attributes(before.attributes, after.attributes)
    commands = [f"set the {fields(row)[key]['label'].lower()} to {change['after']}" for key, change in changes.items()]
    return f"For the {row['name']}, " + " and ".join(commands) if commands else f"Keep the current {row['name']}"
