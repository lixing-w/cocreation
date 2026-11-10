"""Natural-language sampling and temporary reference binding; no inheritance graph."""
import re

# All media-looking tokens are checked, so invented/noncanonical labels cannot pass.
MEDIA_LABEL = re.compile(r'<(?:Image|Video|Picture|Audio|Subject)\b[^<>]*>', re.I)
SOURCE = re.compile(r'\(from (<(?:Image|Video) [1-9]\d*>)\)')


def description_text(value, *, sourced=False):
    """Normalize whitespace only; never repair or decode model-generated JSON."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Expected a nonempty natural-language description')
    text = value.strip()
    if text in ('NO_OBSERVATION', 'CANNOT_FULFILL'):
        raise ValueError(text)
    if text.startswith(('{', '[', '"')) or any(x in text for x in ('```', '{{artifact:', '<|')):
        raise ValueError('Expected direct prose without envelopes or transport tokens')
    if any(re.match(r'^\s*(?:[-*#]|\d+[.)]\s)', line) for line in text.splitlines()):
        raise ValueError('Expected unnumbered sentences, not headings or lists')
    if not sourced and MEDIA_LABEL.search(text):
        raise ValueError('Pool observations must not contain temporary media labels')
    return '\n'.join(line.strip() for line in text.splitlines())


def media_catalog(references):
    counters = {'image': 0, 'video': 0}
    catalog = {}
    for ref in references:
        if ref.artifact_id in catalog or ref.kind not in counters:
            raise ValueError('Duplicate or unsupported media reference')
        counters[ref.kind] += 1
        catalog[ref.artifact_id] = f'<{ref.kind.title()} {counters[ref.kind]}>'
    return catalog


def validate_sources(text, references):
    """Every media label must occur in an exact (from <Kind N>) annotation."""
    text = description_text(text, sourced=True)
    allowed = set(media_catalog(references).values())
    labels = set(SOURCE.findall(text))
    if not labels <= allowed or MEDIA_LABEL.search(SOURCE.sub('', text)):
        raise ValueError('Unknown media label or malformed source annotation')
    for line in text.splitlines():
        if SOURCE.search(line) and not SOURCE.sub('', line).strip():
            raise ValueError('Source annotation must accompany a description')
    return text


def prune_references(text, references):
    """Drop unused media and renumber surviving labels in one simultaneous pass."""
    text = validate_sources(text, references)
    old = media_catalog(references)
    used = set(SOURCE.findall(text))
    retained = [ref for ref in references if old[ref.artifact_id] in used]
    new = media_catalog(retained)
    mapping = {old[ref.artifact_id]: new[ref.artifact_id] for ref in retained}
    text = SOURCE.sub(lambda m: '(from ' + mapping[m[1]] + ')', text)
    return text, retained


def source_description(text, label):
    """Natural-language frame query containing only requirements tied to this source."""
    return '\n'.join(SOURCE.sub('', line).strip() for line in text.splitlines()
                     if label in SOURCE.findall(line))


def sample_count(bounds, rng):
    """Linearly decreasing weights: default counts 0/1/2/3 have 40/30/20/10%."""
    if (len(bounds) != 2 or any(type(n) is not int for n in bounds)
            or not 0 <= bounds[0] <= bounds[1]):
        raise ValueError('Sample bounds must be an inclusive nonnegative integer range')
    counts = list(range(bounds[0], bounds[1] + 1))
    return rng.choices(counts, weights=range(len(counts), 0, -1), k=1)[0]


def sample_selection(pool, k, rng, *, max_parents=2):
    """Sample lines from (description, artifact) tuples, tracking actual carriers.

    Randomize the full line pool; accept lines until K or the carrier cap prevents
    completion. Repeated text on separate artifacts remains separate evidence.
    One nonempty line is the smallest intuitive paragraph, as instructed to VLMs.
    """
    if type(k) is not int or k < 0 or type(max_parents) is not int or max_parents < 1:
        raise ValueError('k must be nonnegative and max_parents positive')
    if k == 0:
        return '', []
    candidates = [(line, ref) for description, ref in pool
                  for line in description_text(description).splitlines() if line]
    rng.shuffle(candidates)
    chosen, refs, seen = [], [], set()
    for line, ref in candidates:
        if ref.artifact_id not in seen:
            if len(refs) >= max_parents:
                continue
            refs.append(ref)
            seen.add(ref.artifact_id)
        chosen.append((line, ref))
        if len(chosen) == k:
            labels = media_catalog(refs)
            return '\n'.join(line + ' (from ' + labels[ref.artifact_id] + ')'
                             for line, ref in chosen), refs
    return None
