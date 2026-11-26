"""Shared text-writing policy: current structured attributes have one meaning."""
ATTRIBUTE_AUTHORITY = """Before writing, analyze conflicts among the supplied attribute_overrides/current attributes, source descriptions, identity names, action prose and references. The structured CURRENT attributes are authoritative for their named subject. Resolve each conflict in favor of those attributes and omit the incompatible adjective or clause from the final text. Preserve compatible details. Do not concatenate contradictory source and current descriptions or average conflicting values. Unknown attributes supply no new instruction.
Bind attributes to the correct subject: a carried bag has its own current state; a person's source caption does not freeze that bag's old color/material. Use neutral subject names. Keep action and camera information when dropping appearance adjectives.
Use the current age_group exactly: young adult stays young adult, adult stays adult, and older adult stays older adult. A source caption saying girl or boy must not override that category or erase a confirmed age edit. For an age-only edit, explicitly preserve the supplied hair color, hair length/style, clothing colors and identity; youth does not authorize changing white hair to brown hair or changing its length. Analyze these conflicts internally and return only the requested final dialogue or generation prompt."""


def neutral_name(name, kind, attributes=None):
    """Stable identity names cannot encode an editable appearance or age."""
    import re
    if kind == 'person':
        # Age/gender descriptors are version attributes, not persistent identity names.
        return 'person'
    words = ('black white grey gray beige brown red orange yellow green cyan blue purple pink gold golden silver copper '
             'dark light burgundy khaki off-white neon multicolored multicoloured wooden wood glass metal paper cardboard '
             'leather cotton fabric cloth textile canvas silk felt nylon plastic ceramic porcelain rubber stone powder spring summer autumn fall winter '
             'sunny cloudy rainy snowy dawn daytime dusk night young old elderly').split()
    for field in ('material','color','size','surface_finish','texture','condition'):
        value=(attributes or {}).get(field)
        if isinstance(value,str) and value not in ('unknown','not applicable',''):
            words.append(value)
    pattern = r'(?i)\b(?:' + '|'.join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r')\b'
    result = re.sub(pattern, '', name)
    result = re.sub(r'\s+', ' ', result).strip(' -_,').lower()
    return result or ('object' if kind == 'object' else 'scene')


def name_assets(assets):
    counts = {}
    aliases = {}
    for asset in assets:
        original = asset['name']
        name = neutral_name(asset['name'], asset['kind'], asset.get('attributes'))
        counts[name] = counts.get(name, 0) + 1
        asset['name'] = name if counts[name] == 1 else f'{name} {counts[name]}'
        if original != asset['name']:
            aliases[original] = asset['name']
    return aliases


def neutral_asset_mentions(text, aliases):
    """Carry identity renames into source staging without changing action/camera prose."""
    import re
    if not aliases:
        return text
    names = {name.casefold(): replacement for name, replacement in aliases.items()}
    pattern = r'(?<!\w)(?:' + '|'.join(re.escape(name) for name in sorted(aliases, key=len, reverse=True)) + r')(?!\w)'
    return re.sub(pattern, lambda match: names[match.group().casefold()], text, flags=re.IGNORECASE)


def neutral_asset_descriptions(assets, shots, aliases=None):
    """Keep observed appearance in attributes, with version-neutral identity/action prose."""
    aliases=dict(aliases or {})
    renamed=name_assets(assets)
    aliases={name:renamed.get(target,target) for name,target in aliases.items()}
    aliases.update(renamed)
    for asset in assets:
        for field in ('material','color','size','surface_finish','texture','condition'):
            value=asset.get('attributes',{}).get(field)
            if isinstance(value,str) and value not in ('unknown','not applicable',''):
                aliases.setdefault(value+' '+asset['name'],asset['name'])
    for row in [*assets,*shots]:
        row['description']=neutral_asset_mentions(row['description'],aliases)
