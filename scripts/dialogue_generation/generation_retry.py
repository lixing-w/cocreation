"""Build an edit from the latest rejected attempt and the original input files."""
import hashlib
from pathlib import Path


def attempt_placeholder(token, number):
    """Keep the initial ID and append a version for each subsequent attempt."""
    return token if number == 1 else f'{token[:-2]}_{number - 1:02d}]]'


def clip_placeholder(token, index):
    """Scope attempts for legacy assemblies that generate multiple hidden clips."""
    return f'{token[:-2]}_clip{index:02d}]]'


def retry_source_path(directory, edit):
    source = edit['references'][-1]
    path = Path(directory) / source['path']
    with path.open('rb') as media:
        matches = hashlib.file_digest(media, 'sha256').hexdigest() == source['sha256']
    if not matches:
        raise RuntimeError('Archived retry edit source changed: ' + source['placeholder'])
    return path


def retry_edit_task(task, token, first, previous):
    originals = first.get('references', [])
    kind = task['kind']
    number = 1 + sum(ref['kind'] == kind for ref in originals)
    label = ('Image' if task['backend'] == 'flux' else 'Picture') if kind == 'image' else 'Video'
    placeholder = attempt_placeholder(token, previous['attempt'])
    if previous.get('placeholder', placeholder) != placeholder:
        raise ValueError('Retry source has an inconsistent version: ' + placeholder)
    source = dict(kind=kind, path=previous['path'], placeholder=placeholder, label=f'<{label} {number}>',
                  attempt=previous['attempt'], sha256=previous['sha256'], role='edit_source')
    # Keep the original prepared files, order and labels. The most recent
    # rejected result is the only extra input, even on the third attempt.
    return dict(task, operation='video_edit' if kind == 'video' else 'local_edit',
                references=[*originals, source], retry_edit=True, retry_seconds=first.get('options', {}).get('seconds'))
