"""CLI adapter shared by the two generation-prompt tools."""
import argparse
import json
from pathlib import Path


def main(tool, target):
    """Run tool using a task JSON path; target selects optional H3 duration flags.

    Task keys: attributes, references (optional), user_prompt (optional).
    Local media paths are resolved relative to the task file, not the shell cwd.
    """
    parser=argparse.ArgumentParser(description=tool.__doc__.split('\n')[0])
    parser.add_argument('task',type=Path,help='Local manifest: attributes is prose; references/user_prompt optional')
    if target == 'h3':
        parser.add_argument('--seconds',type=float,default=5.0)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--no-system-cache',dest='use_system_cache',action='store_false')
    args=vars(parser.parse_args()); path=args.pop('task').resolve()
    task=json.loads(path.read_text(encoding='utf-8'))
    if set(task) - {'attributes','references','user_prompt'}:
        parser.error('Unknown task fields')
    for ref in task.get('references',[]):
        media=Path(ref['path']).expanduser()
        ref['path']=str(media if media.is_absolute() else path.parent/media)
    result=tool(**task,**args)
    print(result if result is not None else 'CANNOT_FULFILL')
