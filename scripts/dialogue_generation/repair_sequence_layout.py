"""Re-export existing media with versioned attempts, preserving the source records."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import quote
import uuid

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.dialogue_generation.formal_sequence import export_sequence, tool_call_segment
from scripts.dialogue_generation.generation_retry import attempt_placeholder


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace(sequence, field):
    return [(turn['id'], value) for turn in sequence['turns']
            for segment in turn['segments']
            for value in (segment.get('tool_calls', []) if field=='calls'
                          else [segment] if segment['type']==field else [])]


def validate(old, new, markdown):
    if old['status'] != new['status']:
        raise ValueError('Repair changed the sequence completion status')
    if [(t['id'],t['role']) for t in old['turns']] != [(t['id'],t['role']) for t in new['turns']]:
        raise ValueError('Repair changed the spoken turn list')
    old_calls = [(turn, tool_call_segment(call)['tool_calls'][0]) for turn, call in trace(old, 'calls')]
    new_calls = [(turn, tool_call_segment(call)['tool_calls'][0]) for turn, call in trace(new, 'calls')]
    new_media = trace(new, 'media')
    if old.get('version', 0) < 4:
        for _, call in new_calls:
            for ref in call['arguments'].get('references', []):
                if ref.get('role') == 'edit_source' and ref.get('placeholder'):
                    ref['placeholder'] = re.sub(r'(\[\[(?:image|video)_\d{4})(?:_clip\d+)?(?:_\d+)?\]\]$',
                                                r'\1]]', ref['placeholder'])
        new_media = [(turn, dict({key: value for key, value in segment.items() if key != 'logical_placeholder'},
                                 placeholder=segment.get('logical_placeholder', segment['placeholder'])))
                     for turn, segment in new_media]
    elif old.get('media_aliases', {}) != new.get('media_aliases', {}):
        raise ValueError('Repair changed a delivered media alias')
    if old_calls != new_calls:
        raise ValueError('Repair changed the ordered calls trace')
    if trace(old, 'media') != new_media or trace(old, 'text') != trace(new, 'text'):
        raise ValueError('Repair changed the ordered media or text trace')
    old_reviews=[(t['id'],s) for t in old['turns'] for s in t['segments'] if s.get('source')=='media_critique']
    new_reviews=[(t['id'],s) for t in new['turns'] for s in t['segments'] if s.get('source')=='media_critique']
    if old_reviews != new_reviews:
        raise ValueError('Repair changed an original critique')
    cursor=0
    for turn in new['turns']:
        segments=turn['segments']
        media_seen = {}
        for index,segment in enumerate(segments):
            if segment['type']=='thinking':
                if '<tool_call>' in segment['text'] or segment.get('tool_calls'):
                    raise ValueError('Tool call remains inside thinking')
                if segment.get('source')=='media_critique' and (not index or segments[index-1]['type']!='media'):
                    raise ValueError('Critique appears before its media result')
            if segment['type']=='tool_call':
                following = segments[index+1] if index + 1 < len(segments) else {}
                is_retry = following.get('attempt', 1) > 1
                if is_retry:
                    preceding = segments[index-1] if index else {}
                    critique = segments[index-2] if index > 1 and preceding['type'] == 'text' else preceding
                    if critique.get('source') != 'media_critique':
                        raise ValueError('Retry tool call lacks its preceding critique')
                    scope = following['placeholder'][:-2].rsplit('_', 1)[0] + ']]'
                    previous_attempt = following['attempt'] - 1
                    expected = attempt_placeholder(scope, previous_attempt)
                    for call in segment['tool_calls']:
                        for ref in call['arguments'].get('references', []):
                            if ref.get('role') == 'edit_source':
                                source = media_seen.get(ref['placeholder'], {})
                                if (ref['placeholder'] != expected or ref.get('attempt') != previous_attempt
                                        or source.get('attempt') != previous_attempt):
                                    raise ValueError('Retry source is not the preceding attempt')
                elif not index or segments[index-1]['type'] != 'thinking':
                    raise ValueError('Tool call lacks its preceding thought/transition')
                if index+1<len(segments) and segments[index+1]['type']!='media':
                    raise ValueError('A segment interrupts the tool call and media result')
                if new['status']=='complete' and (index+1==len(segments) or segments[index+1]['type']!='media'):
                    raise ValueError('Completed tool call lacks its media result')
            if segment['type']=='media':
                media_seen[segment['placeholder']] = segment
                needle='src="'+html.escape(quote(segment['path'],safe='/'),quote=True)+'"'
            else:
                needle=segment['text']
            position=markdown.find(needle,cursor)
            if position<0:
                raise ValueError('Markdown and JSON segment order differ')
            cursor=position+len(needle)
        if any(s.get('source')=='prompt_reasoner' for s in segments):
            if sum(s.get('source')=='prompt_reasoner' for s in segments)>1 and len({s.get('placeholder') for s in segments if s['type']=='media'})>1:
                raise ValueError('Multi-output turn repeats the full analysis')


def repair(directory):
    directory=Path(directory).resolve()
    inputs=[directory/name for name in ('dialogue.json','media_manifest.json','generation_plan.json')
            if (directory/name).is_file()]
    before={p.name:digest(p) for p in inputs}
    exports=[directory/'sequence.json',directory/'dialogue.md']
    old=json.loads(exports[0].read_text())
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    backup=directory/'export_backups'/('sequence-layout-'+stamp)
    backup.mkdir(parents=True)
    for path in exports:
        shutil.copy2(path,backup/path.name)
    result=json.loads((directory/'dialogue.json').read_text())
    manifest=json.loads((directory/'media_manifest.json').read_text())
    try:
        records = manifest['media']
        if pruning := manifest.get('pruning'):
            removed = set(pruning['removed_turn_ids'])
            omitted = set(pruning['exhausted']) | set(pruning['omitted_media'])
            result['turns'] = [turn for turn in result['turns'] if turn['id'] not in removed]
            result['placeholders'] = [row for row in result.get('placeholders', []) if row['placeholder'] not in omitted]
            records = {token: record for token, record in records.items() if token not in omitted}
        new=export_sequence(directory,result,records,status=old['status'])
        validate(old,new,exports[1].read_text())
        if before!={p.name:digest(p) for p in inputs}:
            raise ValueError('Repair changed a raw input/audit artifact')
        report=dict(directory=str(directory),backup=str(backup),version=new['version'],status=new['status'],
            turns=len(new['turns']),tool_calls=len(trace(new,'calls')),
            media=dict(Counter(s['kind'] for _,s in trace(new,'media'))),
            raw_inputs_sha256=before,old_exports_sha256={p.name:digest(backup/p.name) for p in exports},
            new_exports_sha256={p.name:digest(p) for p in exports})
        (backup/'repair_receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        return report
    except BaseException:
        for path in exports:
            shutil.copy2(backup/path.name,path)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dialogue_dirs',type=Path,nargs='+')
    args=parser.parse_args()
    for directory in args.dialogue_dirs:
        print(json.dumps(repair(directory),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
