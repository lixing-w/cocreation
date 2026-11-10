"""Extract validated visual CPG attributes from one image/video artifact."""
import argparse
import json
from pathlib import Path
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cocreation import Qwen, Reference
from cocreation.storage import scratch_root
from tools.extract_attribute_nodes_prompts import (
    CATEGORY_RULES, OBJECT_FIELDS, TOP_LEVEL_CATEGORIES,
    extraction_prompt, extraction_system_prompt)


def _unique_object(pairs):
    """Decode JSON key/value pairs while rejecting duplicate object keys."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key {key!r}')
        result[key] = value
    return result



def repair_attributes(result):
    """Repair only unambiguous output-shape/omission issues, in place.

    This is for fresh VLM observations, before validation/atom IDs/provenance.
    Do not use it to silently mutate already indexed graph observations. Metadata,
    meaningful unknown fields, duplicate UIDs and unexplained bad links remain for
    strict validation. No observations, IDs, endpoint roles or timelines are invented.
    """
    def empty(value):
        return (value is None or value == [] or value == {} or
                isinstance(value, str) and value.strip().lower() in ('', 'none', 'null', 'unknown'))

    if not isinstance(result, dict):
        return result
    for key in list(result):
        if key not in ('artifact_id', 'kind', 'attributes') and empty(result[key]):
            del result[key]
    if 'attributes' in result and empty(result['attributes']):
        result['attributes'] = {}
    attrs = result.get('attributes')
    if not isinstance(attrs, dict):
        return result
    removed_uids, removed_categories = set(), set()
    for category in list(attrs):
        value = attrs[category]
        if empty(value):
            removed_categories.add(category)
            del attrs[category]
            continue
        if category not in TOP_LEVEL_CATEGORIES:
            continue
        if category not in OBJECT_FIELDS:
            if isinstance(value, list):
                value[:] = [item for item in value if not empty(item)]
                if not value:
                    del attrs[category]
                    continue
                if len(value) == 1 and isinstance(value[0], str):
                    value = attrs[category] = value[0]
            if isinstance(value, str):
                attrs[category] = value.strip()
            continue
        if not isinstance(value, list):
            continue
        kept = []
        for entry in value:
            if empty(entry):
                continue
            if not isinstance(entry, dict):
                kept.append(entry)  # Nontrivial wrong type: validator rejects.
                continue
            original_uid = entry.get('uid')
            for field in list(entry):
                item = entry[field]
                if empty(item):
                    del entry[field]
                    continue
                expected = OBJECT_FIELDS[category].get(field)
                if field in ('uid', 'links') or expected is None:
                    continue  # Never rewrite identifiers or repair endpoint cardinality.
                if isinstance(item, list):
                    item[:] = [part for part in item if not empty(part)]
                    if not item:
                        del entry[field]
                        continue
                    if expected == 'str' and len(item) == 1 and isinstance(item[0], str):
                        item = entry[field] = item[0]
                if expected == 'list[str]' and isinstance(item, str):
                    item = entry[field] = [item]
                if isinstance(item, str):
                    entry[field] = item.strip()
                elif isinstance(item, list):
                    item[:] = [part.strip() if isinstance(part, str) else part for part in item]
            required = {'uid', 'description'}
            if category in ('identity_relations', 'identity_background_relations'):
                required.add('links')
                # A placeholder endpoint makes the whole relation ungrounded.
                if isinstance(entry.get('links'), list) and any(empty(link) for link in entry['links']):
                    del entry['links']
            if not required <= set(entry):
                if isinstance(original_uid, str):
                    removed_uids.add(original_uid)
                removed_categories.add(category)
                continue
            kept.append(entry)
        value[:] = kept
        if not value:
            removed_categories.add(category)
            del attrs[category]
    # Only cascade from nodes explicitly removed above. Never hide unexplained IDs.
    retained_uids = {entry['uid'] for category, entries in attrs.items()
                     if category in OBJECT_FIELDS and isinstance(entries, list)
                     for entry in entries if isinstance(entry, dict) and isinstance(entry.get('uid'), str)}
    removed_uids -= retained_uids
    for category in ('identity_relations', 'identity_background_relations'):
        entries = attrs.get(category)
        if isinstance(entries, list):
            entries[:] = [entry for entry in entries if not (isinstance(entry, dict)
                and isinstance(entry.get('links'), list)
                and any(isinstance(link, str) and link in removed_uids for link in entry['links']))]
            if not entries:
                del attrs[category]
    for category, change in (('environment', 'environment_change'), ('viewpoint', 'viewpoint_change')):
        if category in removed_categories:
            attrs.pop(change, None)  # Cannot safely rewrite a partially deleted timeline.
    return result


def validate_attributes(result, artifact_id, kind):
    """Validate decoded result for source artifact_id/kind; raise path-specific errors.

    Return the original object without modifying it. Validation checks structure,
    cardinalities and local graph bindings, not observational truth.
    """
    def require(condition, path, message):
        if not condition:
            raise ValueError(f'{path}: {message}')
    # meta checks. if these go wrong, probably fields are, so reject as a whole
    require(kind in ('image', 'video'), '$.kind', 'expected image or video')
    require(isinstance(result, dict), '$', 'expected a JSON object')
    required = {'artifact_id', 'kind', 'attributes'}
    require(set(result) == required, '$',
            f'expected exactly {sorted(required)}; got {sorted(result)}')
    require(result['artifact_id'] == artifact_id, '$.artifact_id', f'expected {artifact_id!r}')
    require(result['kind'] == kind, '$.kind', f'expected {kind!r}')
    def description(value, path):
        require(isinstance(value, str) and bool(value.strip()), path, 'expected a nonempty string')
        require(value.strip().lower() not in ('none', 'null', 'unknown'), path,
                'omit unobserved fields instead of placeholder descriptions')

    def temporal(category, path):
        require(kind != 'image' or CATEGORY_RULES[category]['temporal'] != 'Yes',
                path, 'flow-in-time attributes cannot be established by an image')

    attrs = result['attributes']
    require(isinstance(attrs, dict), '$.attributes', 'expected an object')
    unknown = set(attrs) - set(TOP_LEVEL_CATEGORIES)
    require(not unknown, '$.attributes', f'unknown categories: {sorted(unknown)}')
    nodes = {}
    relations = []
    for category, value in attrs.items():
        category_path = f'$.attributes[{category!r}]'
        temporal(category, category_path)
        if category not in OBJECT_FIELDS:
            description(value, category_path)
            continue
        require(isinstance(value, list) and bool(value), category_path,
                'expected a nonempty array; omit unobserved categories')
        schema = OBJECT_FIELDS[category]
        is_relation = category in ('identity_relations', 'identity_background_relations')
        for index, entry in enumerate(value):
            path = f'{category_path}[{index}]'
            require(isinstance(entry, dict), path, 'expected an object')
            required = {'uid', 'description', 'links'} if is_relation else {'uid', 'description'}
            require(required <= set(entry), path, f'missing required fields: {sorted(required - set(entry))}')
            extra = set(entry) - set(schema)
            require(not extra, path, f'unsupported fields: {sorted(extra)}')
            uid = entry['uid']
            require(isinstance(uid, str) and re.fullmatch(r'[A-Za-z0-9_.-]+', uid),
                    path+'.uid', 'expected a nonempty local identifier using letters/digits/_/./-')
            require(uid not in nodes, path+'.uid', f'duplicate local UID {uid!r}')
            nodes[uid] = category
            for field, item in entry.items():
                field_path = path+'.'+field
                if field == 'uid':
                    continue
                if field == 'links':
                    require(isinstance(item, list) and len(item) == 2 and
                            all(isinstance(link, str) and bool(link) for link in item),
                            field_path, 'expected exactly two nonempty UID strings')
                    relations.append((category, item, field_path))
                    continue
                temporal(category+'.'+field, field_path)
                if schema[field] == 'list[str]':
                    require(isinstance(item, list) and bool(item), field_path,
                            'expected a nonempty array of strings; omit unobserved fields')
                    for i, text in enumerate(item):
                        description(text, f'{field_path}[{i}]')
                else:
                    description(item, field_path)
    for category, links, path in relations:
        for index, link in enumerate(links):
            require(link in nodes, f'{path}[{index}]', f'unresolved UID {link!r}')
        types = [nodes[link] for link in links]
        if category == 'identity_relations':
            require(types == ['identity', 'identity'] and links[0] != links[1], path,
                    'expected exactly two distinct identity endpoints')
        else:
            require(types == ['identity', 'environment'], path,
                    'expected one identity then one environment')

    return result


def parse_attributes(text, artifact_id, kind):
    """Parse text for artifact_id/kind; report failures to stderr and return None.

    Harmless omissions/wrapping errors are repaired in place before validation.
    Syntax errors include line/column; remaining contract errors include a JSON field path.
    CLI stdout remains reserved for the resulting JSON/null.
    """
    try:
        if not isinstance(text, str):
            raise ValueError('$: model response text is missing or is not a string')
        text = text.strip()
        if text.startswith('```json') and text.endswith('```'):
            text = text[7:-3].strip()
        elif text.startswith('```') and text.endswith('```'):
            text = text[3:-3].strip()
        def invalid_constant(value):
            raise ValueError(f'JSON contains non-finite number {value!r}')
        result = json.loads(text, object_pairs_hook=_unique_object,
                            parse_constant=invalid_constant)
        return validate_attributes(repair_attributes(result), artifact_id, kind)
    except json.JSONDecodeError as error:
        print(f'extract_attribute_nodes [{artifact_id}] JSON syntax error at '
              f'line {error.lineno}, column {error.colno} (offset {error.pos}): {error.msg}',
              file=sys.stderr)
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        print(f'extract_attribute_nodes [{artifact_id}] {type(error).__name__}: {error}',
              file=sys.stderr)
    return None


def extract_attribute_nodes(artifact, *, qwen=None, output_dir=None,
                            max_new_tokens=8192, video_fps=2.0, video_max_frames=32,
                            max_pixels=512 * 512, use_system_cache=True):
    """Return conservatively repaired, validated JSON; None for unrecoverable output.

    Args:
        artifact: Reference or mapping with artifact_id, path and kind (image/video).
        qwen: Optional shared Qwen instance; never closed by this tool.
        output_dir: Per-call run root, default scratch/runs/extract_attribute_nodes;
            also applies when reusing a supplied Qwen instance.
        max_new_tokens: Response budget; truncated JSON returns None.
        video_fps: Requested official video sampling rate.
        video_max_frames: Official video sampling frame cap.
        max_pixels: Per-frame/image pixel budget.
        use_system_cache: Reuse the dedicated system prefix; False uses ordinary inference.

    A local model is closed in finally, including on failures. File and inference
    failures propagate; only output parsing/validation failures return None.
    This extracts observations only; it does not write nodes into a CPG database.
    """
    if isinstance(artifact, dict):
        artifact = Reference(artifact['artifact_id'], artifact['path'], artifact['kind'])
    if not isinstance(artifact, Reference) or artifact.kind not in ('image','video'):
        raise ValueError('artifact must be an image/video Reference or artifact mapping')
    path = Path(artifact.path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError('artifact path must be a local file')
    reference = Reference(artifact.artifact_id, path, artifact.kind)
    output_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else (
        scratch_root() / 'runs/extract_attribute_nodes')
    owned = qwen is None
    model = Qwen(output_dir=output_dir) if owned else qwen
    try:
        response = model(extraction_prompt(reference.artifact_id, reference.kind), [reference],
                         system_prompt=extraction_system_prompt(), max_new_tokens=max_new_tokens,
                         output_dir=output_dir,
                         system_cache_path=(scratch_root() / "kvcache/extract_attribute_nodes/system_prompt.kvcache"
                                            if use_system_cache else None),
                         video_fps=video_fps, video_max_frames=video_max_frames,
                         max_pixels=max_pixels)
        text = response.get('text') if isinstance(response,dict) else None
        return parse_attributes(text, reference.artifact_id, reference.kind)
    finally:
        if owned:
            model.close()


def main():
    """Read one local artifact from CLI arguments and print JSON or null."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--artifact-id', required=True)
    parser.add_argument('--kind', choices=('image','video'), required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-new-tokens', type=int, default=8192)
    parser.add_argument('--video-fps', type=float, default=2.0)
    parser.add_argument('--video-max-frames', type=int, default=32)
    parser.add_argument('--no-system-cache', dest='use_system_cache', action='store_false')
    args = vars(parser.parse_args())
    artifact = Reference(args.pop('artifact_id'), args.pop('path'), args.pop('kind'))
    print(json.dumps(extract_attribute_nodes(artifact, **args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
