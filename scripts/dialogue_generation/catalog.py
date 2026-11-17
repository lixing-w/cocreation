"""Readable, directly executable placeholder catalog; full state stays in the plan."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def display_description(text):
    """Keep visual requirements, removing repeated generic instruction boilerplate."""
    return text.removeprefix('Content and design reference: ').split(
        '. Apply these requirements to the entire output frame;', 1)[0]


def placeholder_catalog(result):
    """Project the existing contract without recomputing prompts or relationships.

    references are the ordered expert slots, not the construction DAG. When the
    executable dependencies differ from reference_materials, keep them explicitly;
    otherwise consumers use reference_materials as dependencies.
    """
    catalog = []
    for row in result['placeholders']:
        task = row.get('generation', {})
        item = dict(placeholder=row['placeholder'], kind=row['kind'],
                    description=display_description(row['description']))
        for key in ('purpose', 'asset_ids', 'shot_ids'):
            if row.get(key):
                item[key] = deepcopy(row[key])
        if 'generation_requirements' in row:
            item['generation_requirements'] = deepcopy(row['generation_requirements'])
        item['reference_depth'] = row['reference_depth']
        item['reference_materials'] = list(row['reference_materials'])
        for key in ('operation', 'backend', 'model_id'):
            if task.get(key) is not None:
                item[key] = task[key]
        item['prompt'] = task.get('prompt', row.get('generation_prompt'))
        if task.get('references'):
            item['references'] = [
                {key: deepcopy(value) for key, value in ref.items()
                 if key not in ('description', 'attributes')}
                for ref in task['references']]
        if 'dependencies' in task and task['dependencies'] != row['reference_materials']:
            item['dependencies'] = list(task['dependencies'])
        for key in ('seconds', 'output_seconds', 'shot_id', 'frame_seconds', 'preprocess', 'clips', 'clip_edits', 'render_state', 'presentation'):
            if key in task:
                item[key] = deepcopy(task[key])
        source = row.get('source_video', task.get('source_video'))
        if source is not None:
            item['source_video'] = source
        if not task or task.get('status') != 'ready':
            item['status'] = task.get('status', 'pending')
        catalog.append(item)
    return catalog


def refresh_catalog(run_dir):
    """Refresh only placeholders.json from an existing dialogue; no model calls."""
    directory = Path(run_dir)
    result = json.loads((directory/'dialogue.json').read_text(encoding='utf-8'))
    catalog = placeholder_catalog(result)
    path = directory/'placeholders.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(catalog, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path, nargs='+')
    for directory in parser.parse_args().run_dir:
        catalog = refresh_catalog(directory)
        print(f'{directory}/placeholders.json: {len(catalog)} assets')


if __name__ == '__main__':
    main()
