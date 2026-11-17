"""Measure steady-state end-to-end serving throughput with fixed-length outputs.

Outputs are deliberately forced to the token cap for a comparable workload;
use validate.py and the generation examples for quality, not these forced replies.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cocreation import VLLMQwen
from cocreation.qwen_client import MODEL_ID
from cocreation.storage import scratch_root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=MODEL_ID)
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[1,4,8])
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--output-tokens', type=int, default=256)
    p.add_argument('--video', type=Path, default=scratch_root()/'data/samples/story_example.mp4')
    a = p.parse_args()
    if min(a.batch_sizes) < 1 or a.repetitions < 1 or a.output_tokens < 1:
        p.error('Use positive batch sizes, repetitions and output tokens')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    client = VLLMQwen(a.model, base_url=a.base_url)
    reference = ('The scene contains a traveler carrying a bag through a city, entering a cafe, '
                 'then returning to the street. Preserve the subject identity, requested changes, '
                 'camera movement and reference roles independently. ')
    cases = {
        'text': [dict(role='system', content='Write precise English generation instructions from the supplied requirements.'),
                 dict(role='user', content=reference*32+' Describe the scene and each possible edit in detail.')],
        'video': [dict(role='user', content=[
            dict(type='text', text='Describe the visible subjects, setting, changes and camera movements in detail: '),
            dict(type='video', video=str(a.video)),
            dict(type='text', text='Use only evidence in the video, with detailed descriptions of each visible subject.')])],
    }
    server_file = scratch_root()/'runtime/qwen3.6-35b-a3b/server.json'
    report = dict(model=a.model, server=json.loads(server_file.read_text()) if server_file.exists() else None,
        output_tokens=a.output_tokens, repetitions=a.repetitions,
        methodology='Steady-state HTTP wall time; fixed min/max output tokens; repeated text/sample video with warm token prefix cache; media processor cache disabled. Excludes model startup. This measures reuse, not distinct cold videos. Forced outputs are not quality samples.',
        measurements=[])
    def request(body, directory):
        directory.mkdir()
        (directory/'request.json').write_text(json.dumps(body,indent=2))
        req = urllib.request.Request(client.base_url+'/chat/completions',
            data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        start = time.perf_counter()
        with urllib.request.urlopen(req,timeout=600) as response:
            value = json.load(response)
        elapsed = time.perf_counter()-start
        (directory/'response.json').write_text(json.dumps(value,ensure_ascii=False,indent=2))
        return dict(seconds=elapsed, **value['usage'])
    for case, messages in cases.items():
        body = client.chat(messages, max_new_tokens=a.output_tokens, seed=42,
            temperature=0.1, dry_run=True)['request']
        body.update(min_tokens=a.output_tokens,ignore_eos=True)
        request(body,a.output_dir/(case+'-warmup'))
        for batch_size in a.batch_sizes:
            for repetition in range(a.repetitions):
                folder = a.output_dir/f'{case}-b{batch_size}-r{repetition}'
                folder.mkdir()
                bodies = []
                for index in range(batch_size):
                    current = deepcopy(body)
                    current['seed'] = 42 + index + repetition*100
                    current['messages'].append(dict(role='user', content=f'Independent request {index}, round {repetition}.'))
                    bodies.append(current)
                start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    rows = list(executor.map(lambda item: request(item[1],folder/str(item[0])),enumerate(bodies)))
                elapsed = time.perf_counter()-start
                if any(row['completion_tokens'] != a.output_tokens for row in rows):
                    raise AssertionError('Engine did not honor the fixed output length')
                result = dict(case=case,batch_size=batch_size,repetition=repetition,seconds=elapsed,
                    completion_tokens=sum(row['completion_tokens'] for row in rows),
                    prompt_tokens=sum(row['prompt_tokens'] for row in rows),
                    output_tokens_per_second=sum(row['completion_tokens'] for row in rows)/elapsed,
                    median_request_seconds=statistics.median(row['seconds'] for row in rows))
                report['measurements'].append(result)
                (a.output_dir/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
