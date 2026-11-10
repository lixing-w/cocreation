"""Extract validated visual CPG attributes from one image/video artifact."""
import argparse
import json
import math
from pathlib import Path
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cocreation import Qwen, Reference
from cocreation.storage import scratch_root
from tools.extract_attribute_nodes_prompts import (
    CATEGORY_RULES, extraction_prompt, extraction_system_prompt)


def _unique_object(pairs):
    """Decode JSON key/value pairs while rejecting duplicate object keys."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key {key!r}')
        result[key] = value
    return result


def validate_attributes(result, artifact_id, kind):
    """Validate decoded result for source artifact_id/kind; raise path-specific errors.

    Return the original object without modifying it. Validation checks structure,
    cardinalities and local graph bindings, not observational truth.
    """
    def require(condition, path, message):
        if not condition:
            raise ValueError(f'{path}: {message}')

    require(kind in ('image', 'video'), '$.kind', 'expected image or video')
    require(isinstance(result, dict), '$', 'expected a JSON object')
    required = {'artifact_id', 'kind', 'attributes'}
    require(set(result) == required, '$',
            f'expected exactly {sorted(required)}; got {sorted(result)}')
    require(result['artifact_id'] == artifact_id, '$.artifact_id', f'expected {artifact_id!r}')
    require(result['kind'] == kind, '$.kind', f'expected {kind!r}')
    attrs = result['attributes']
    require(isinstance(attrs, dict), '$.attributes', 'expected an object of category arrays')
    unknown = set(attrs) - set(CATEGORY_RULES)
    require(not unknown, '$.attributes', f'unknown categories: {sorted(unknown)}')
    nodes = {}
    for category, entries in attrs.items():
        category_path = f'$.attributes[{category!r}]'
        rule = CATEGORY_RULES[category]
        require(isinstance(entries, list) and bool(entries), category_path,
                'expected a nonempty array; omit unobserved categories')
        require(kind != 'image' or rule['temporal'] != 'Yes', category_path,
                'flow-in-time attributes cannot be established by an image')
        parents = set()
        for index, entry in enumerate(entries):
            path = f'{category_path}[{index}]'
            require(isinstance(entry, dict), path, 'expected an entry object')
            require({'id','text'} <= set(entry), path, 'missing required id/text')
            extra = set(entry) - {'id','text','parent_id','links','time_range'}
            require(not extra, path, f'unsupported fields: {sorted(extra)}')
            for field, value in entry.items():
                require(value is not None, path+'.'+field, 'omit this field instead of null')
            key, text, parent = entry['id'], entry['text'], entry.get('parent_id')
            require(isinstance(key,str) and re.fullmatch(r'[A-Za-z0-9_.-]+',key),
                    path+'.id', 'expected a nonempty local identifier using letters/digits/_/./-')
            require(key not in nodes, path+'.id', f'duplicate local ID {key!r}')
            require(isinstance(text,str), path+'.text', 'expected a string')
            require(0 < len(text.split()) <= 15, path+'.text',
                    f'expected 1–15 words; got {len(text.split())}')
            require(text.strip().lower() not in ('none','null','unknown'), path+'.text',
                    'omit unobserved entries instead of placeholder descriptions')
            require(parent is None or (isinstance(parent,str) and bool(parent)),
                    path+'.parent_id', 'expected a nonempty parent identifier')
            links = entry.get('links', [])
            require(isinstance(links,list) and all(isinstance(x,str) and x for x in links),
                    path+'.links', 'expected an array of nonempty identifiers')
            require('links' not in entry or bool(links), path+'.links', 'omit empty links')
            if not rule['multiple']:
                require(parent not in parents, path,
                        f'at most one {category} entry for parent {parent!r}')
            parents.add(parent)
            span = entry.get('time_range')
            require(kind != 'image' or span is None, path+'.time_range',
                    'time ranges must be omitted for images')
            if span is not None:
                require(isinstance(span,list) and len(span)==2, path+'.time_range',
                        'expected [start_seconds, end_seconds]')
                require(all(type(t) in (int,float) and math.isfinite(t) for t in span),
                        path+'.time_range', 'timestamps must be finite numbers, not booleans')
                require(0 <= span[0] <= span[1], path+'.time_range',
                        'expected 0 <= start <= end')
            nodes[key] = (category, entry, path)
    changes = {'identity.appearance_change':'identity.appearance',
               'identity.expression_change':'identity.expression',
               'identity.pose_change':'identity.body_pose',
               'environment_change':'environment', 'viewpoint_change':'viewpoint'}
    for category, entry, path in nodes.values():
        parent, links = entry.get('parent_id'), entry.get('links', [])
        if '.' in category:
            expected = category.split('.')[0]
            require(parent in nodes, path+'.parent_id', f'missing/unresolved parent {parent!r}')
            require(nodes[parent][0] == expected, path+'.parent_id',
                    f'parent {parent!r} must have category {expected!r}')
        else:
            require(parent is None, path+'.parent_id', 'undotted category cannot have parent_id')
        for index, link in enumerate(links):
            require(link in nodes, f'{path}.links[{index}]', f'unresolved ID {link!r}')
            require(link != entry['id'], f'{path}.links[{index}]', 'self-link is not allowed')
        types = [nodes[link][0] for link in links]
        if category == 'identity_relations':
            require(types == ['identity','identity'] and links[0] != links[1], path+'.links',
                    'expected exactly two distinct identity endpoints')
        elif category == 'identity_background_relations':
            require(types == ['identity','environment'], path+'.links',
                    'expected exactly one identity then one environment')
        elif category in changes:
            require(len(set(links)) >= 2 and all(t == changes[category] for t in types),
                    path+'.links', f'expected at least two distinct {changes[category]} IDs')
            if parent is not None:
                require(all(nodes[link][1].get('parent_id') == parent for link in links),
                        path+'.links', f'all linked states must belong to parent {parent!r}')
        elif category == 'visible_text':
            require(all(t == 'identity' for t in types) and len(set(links)) == len(links),
                    path+'.links', 'optional carrier links must be distinct identities')
        else:
            require(not links, path+'.links', f'links not supported for category {category!r}')
    return result


def parse_attributes(text, artifact_id, kind):
    """Parse text for artifact_id/kind; report failures to stderr and return None.

    Syntax errors include line/column; contract errors include a JSON field path.
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
        return validate_attributes(result, artifact_id, kind)
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
    """Return validated attribute JSON, or None for malformed/invalid model output.

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
