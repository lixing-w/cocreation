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
from tools.extract_attribute_nodes_prompts import (
    CATEGORY_RULES, extraction_prompt, extraction_system_prompt)


def _unique_object(pairs):
    """Decode JSON key/value pairs while rejecting duplicate object keys."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def validate_attributes(result, artifact_id, kind):
    """Validate result structure, source identity, cardinalities and local graph links.

    Args:
        result: Decoded JSON object to validate without modifying it.
        artifact_id: Expected source artifact identifier.
        kind: Expected source modality, image or video.

    Returns:
        The original object, or raises ValueError on an invalid contract.
        Structural validation cannot prove that a visual observation is true.
    """
    def require(condition):
        if not condition:
            raise ValueError('Invalid attribute extraction response')

    require(kind in ('image', 'video'))
    require(isinstance(result, dict) and set(result) == {'artifact_id','kind','attributes'})
    require(result['artifact_id'] == artifact_id and result['kind'] == kind)
    attrs = result['attributes']
    require(isinstance(attrs, dict) and set(attrs) == set(CATEGORY_RULES))
    nodes = {}
    for category, entries in attrs.items():
        if entries is None:
            continue
        rule = CATEGORY_RULES[category]
        require(isinstance(entries, list) and bool(entries))
        require(kind != 'image' or rule['temporal'] != 'Yes')
        parents = set()
        for entry in entries:
            require(isinstance(entry, dict) and set(entry) ==
                    {'id','text','parent_id','links','time_range'})
            key, text, parent = entry['id'], entry['text'], entry['parent_id']
            require(isinstance(key, str) and re.fullmatch(r'[A-Za-z0-9_-]+', key))
            require(key not in nodes)
            require(isinstance(text, str) and 0 < len(text.split()) <= 15)
            require(text.strip().lower() not in ('none','null','unknown'))
            require(parent is None or isinstance(parent, str))
            require(isinstance(entry['links'], list) and
                    all(isinstance(x,str) for x in entry['links']))
            if not rule['multiple']:
                require(parent not in parents)
            parents.add(parent)
            span = entry['time_range']
            require(kind != 'image' or span is None)
            if span is not None:
                require(isinstance(span,list) and len(span)==2)
                require(all(type(t) in (int,float) and math.isfinite(t) for t in span))
                require(0 <= span[0] <= span[1])
            nodes[key] = (category, entry)
    changes = {'identity.appearance_change':'identity.appearance',
               'identity.expression_change':'identity.expression',
               'identity.pose_change':'identity.body_pose',
               'environment_change':'environment', 'viewpoint_change':'viewpoint'}
    for category, entry in nodes.values():
        parent, links = entry['parent_id'], entry['links']
        if '.' in category:
            require(parent in nodes and nodes[parent][0] == category.split('.')[0])
        else:
            require(parent is None)
        require(all(link in nodes and link != entry['id'] for link in links))
        types = [nodes[link][0] for link in links]
        if category == 'identity_relations':
            require(types == ['identity','identity'] and links[0] != links[1])
        elif category == 'identity_background_relations':
            require(types == ['identity','environment'])
        elif category in changes:
            require(len(set(links)) >= 2 and all(t == changes[category] for t in types))
            if parent is not None:
                require(all(nodes[link][1]['parent_id'] == parent for link in links))
        elif category == 'visible_text':
            require(all(t == 'identity' for t in types) and len(set(links)) == len(links))
        else:
            require(not links)
    return result


def parse_attributes(text, artifact_id, kind):
    """Parse model text for expected artifact_id/kind; return validated JSON or None."""
    try:
        if not isinstance(text, str):
            return None
        text = text.strip()
        if text.startswith('```json') and text.endswith('```'):
            text = text[7:-3].strip()
        elif text.startswith('```') and text.endswith('```'):
            text = text[3:-3].strip()
        result = json.loads(text, object_pairs_hook=_unique_object,
                            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        return validate_attributes(result, artifact_id, kind)
    except (ValueError, TypeError, KeyError, RecursionError):
        return None


def extract_attribute_nodes(artifact, *, qwen=None, output_dir=None,
                            max_new_tokens=8192, video_fps=2.0, video_max_frames=32,
                            max_pixels=512 * 512):
    """Return validated attribute JSON, or None for malformed/invalid model output.

    Args:
        artifact: Reference or mapping with artifact_id, path and kind (image/video).
        qwen: Optional shared Qwen instance; never closed by this tool.
        output_dir: Run-log directory for an internally created Qwen instance.
        max_new_tokens: Response budget; truncated JSON returns None.
        video_fps: Requested official video sampling rate.
        video_max_frames: Official video sampling frame cap.
        max_pixels: Per-frame/image pixel budget.

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
    owned = qwen is None
    model = Qwen(output_dir=output_dir) if owned else qwen
    try:
        response = model(extraction_prompt(reference.artifact_id, reference.kind), [reference],
                         system_prompt=extraction_system_prompt(), max_new_tokens=max_new_tokens,
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
    args = vars(parser.parse_args())
    artifact = Reference(args.pop('artifact_id'), args.pop('path'), args.pop('kind'))
    print(json.dumps(extract_attribute_nodes(artifact, **args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
