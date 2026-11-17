"""Real-model acceptance: multi-turn interleaving, mixed-media batching, video and observation JSON."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cocreation import VLLMQwen
from cocreation.qwen_client import MODEL_ID
from cocreation.storage import scratch_root
from tools.video_observer import observe_video


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=('vllm',), default='vllm')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--video', type=Path, default=scratch_root()/'data/samples/story_example.mp4')
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--model', default=MODEL_ID)
    p.add_argument('--skip-observe', action='store_true')
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    from PIL import Image
    conversations = []
    for code, color in (('ALPINE', 'red'), ('TOKYO', 'blue')):
        path = a.output_dir/(color+'.png')
        Image.new('RGB', (128,128), color).save(path)
        conversations.append([
            dict(role='system', content='Answer concisely. Preserve the code supplied in this conversation.'),
            dict(role='user', content='The code is '+code+'.'),
            dict(role='assistant', content='I will remember that code.'),
            dict(role='user', content=[dict(type='text', text='Inspect this image: '),
                dict(type='image', image=str(path)), dict(type='text', text=' Return the code followed by the dominant color, and nothing else.')])])
    conversations.append([dict(role='user', content=[
        dict(type='text', text='Watch this video before answering: '),
        dict(type='video', video=str(a.video)),
        dict(type='text', text='Describe only the visible main subjects and actions in one sentence.')])])
    (a.output_dir/'messages.json').write_text(json.dumps(conversations, indent=2))
    manifest_file = 'model.json'
    manifest = json.loads((scratch_root()/'runtime/qwen3.6-35b-a3b'/manifest_file).read_text())
    model = VLLMQwen(a.model, base_url=a.base_url, output_dir=a.output_dir/'runtime')
    with model:
        # One native padded batch mixes image/multi-turn conversations and native
        # video; on vLLM the same independent requests share its scheduler.
        batch = model.generate_batch(conversations, max_new_tokens=128, max_pixels=512*512,
                                     video_fps=1.0, video_max_frames=16, seed=17, temperature=0.1)
        (a.output_dir/'batch.json').write_text(json.dumps(batch, ensure_ascii=False, indent=2))
        for row, code, color in zip(batch['results'], ('ALPINE','TOKYO'), ('red','blue')):
            if code.lower() not in row['text'].lower() or color not in row['text'].lower():
                raise AssertionError('Batched context/image association failed: '+row['text'])
        if not batch['results'][2]['text'].strip():
            raise AssertionError('Native video response was empty')
        print('MIXED_BATCH_OK', [row['text'] for row in batch['results']], flush=True)
        schema = dict(type='object', properties=dict(
            code=dict(type='string', enum=['ALPINE','TOKYO']),
            color=dict(type='string', enum=['red','blue'])),
            required=['code','color'], additionalProperties=False)
        structured = deepcopy(conversations[:2])
        for messages in structured:
            messages[-1]['content'][-1]['text'] = 'Return a JSON object with the earlier code in code and the dominant image color in color.'
        schema_batch = model.generate_batch(structured, max_new_tokens=128, seed=17,
                                           temperature=0.1, response_schema=schema)
        (a.output_dir/'schema_batch.json').write_text(json.dumps(schema_batch, ensure_ascii=False, indent=2))
        expected = [dict(code='ALPINE',color='red'),dict(code='TOKYO',color='blue')]
        if [json.loads(row['text']) for row in schema_batch['results']] != expected:
            raise AssertionError('Schema batch did not preserve conversation/image association')
        print('SCHEMA_BATCH_OK', flush=True)
        if not a.skip_observe:
            options = dict(backend='vllm', base_url=a.base_url, model=a.model)
            observation = observe_video(a.video, a.output_dir/'observation', frames=16,
                retries=2, **options)
            print('OBSERVATION_OK', len(observation['assets']), len(observation['shots']), flush=True)
    (a.output_dir/'acceptance.json').write_text(json.dumps(dict(status='passed', backend=a.backend,
        model=manifest, mixed_batch=True, schema_batch=True, observation=not a.skip_observe), indent=2)+'\n')


if __name__ == '__main__':
    main()
