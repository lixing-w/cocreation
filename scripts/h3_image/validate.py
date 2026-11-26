"""Validate Qwen3.6 rewriting -> recommended 4K W4A8 generation -> reference edit."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from cocreation import VLLMQwen
from cocreation.storage import scratch_root
from scripts.dialogue_generation.materialize import materialize
from tools.h3_image_prompter import h3_image_prompter
from tools.placeholder_prompter import rewrite_placeholder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',default='http://127.0.0.1:8000')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--stop-qwen',action='store_true',help='Stop the Qwen service launched for this validation')
    p.add_argument('--resume',action='store_true',help='Reuse saved Qwen prompts and completed image outputs')
    p.add_argument('--prompts-only',action='store_true',help='Write prompts, then release Qwen before image setup finishes')
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        p.error('Run inside a Slurm GPU allocation')
    root=a.output_dir or scratch_root()/'runs'/('h3-image-validation-'+datetime.now().strftime('%Y%m%dT%H%M%S'))
    root.mkdir(parents=True,exist_ok=a.resume)
    if not a.resume:
        model=VLLMQwen(base_url=a.base_url,output_dir=root/'qwen',enable_thinking=False)
        prompt=h3_image_prompter('A single blue ceramic vessel.\nA neutral studio backdrop, natural front three-quarter view and soft contact shadow.',
                                 qwen=model,output_dir=root/'prompter')
        if not prompt:
            raise RuntimeError('Qwen did not return an accepted text-to-image prompt')
        first=dict(placeholder='[[image_0001]]',kind='image',backend='h3_image',
            model_id='MiniMaxAI/MiniMax-H3-Image',operation='text_generation',prompt=prompt,
            status='ready',dependencies=[],references=[],render_state={})
        edit=dict(placeholder='[[image_0002]]',kind='image',backend='h3_image',
            model_id='MiniMaxAI/MiniMax-H3-Image',operation='local_edit',status='ready',layout='product',
            changes=[dict(field='color',after='amber')],dependencies=['[[image_0001]]'],render_state={},
            references=[dict(placeholder='[[image_0001]]',label='<Picture 1>',kind='image',role='base to edit')])
        def request(text,*,system,seed,max_tokens):
            result=model(text,system_prompt=system,seed=seed,max_new_tokens=max_tokens)
            return result['text'],result['finish_reason']
        edit['prompt']=rewrite_placeholder(edit,request,seed=43,output_dir=root/'edit_prompter')
        (root/'placeholders.json').write_text(json.dumps([first,edit],indent=2))
        (root/'dialogue.md').write_text('Original vessel: [[image_0001]]\n\nAmber edit: [[image_0002]]\n')
        print('Accepted Qwen generation and edit prompts',flush=True)
    if a.stop_qwen:
        subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'qwen/serve.py'),'--stop'],check=True)
    if a.prompts_only:
        print('Saved prompts: '+str(root),flush=True)
        return
    result=materialize(root,resume=a.resume,seed=42)
    if result['completed']!=2 or any(row['options']['width']!=4096 or row['options']['height']!=4096 for row in result['media'].values()):
        raise RuntimeError('Validation did not produce both recommended 4K images')
    print(json.dumps(dict(status=result['status'],output_dir=str(root),images=2,
        generation_seconds=[r['generation_seconds'] for r in result['media'].values()]),indent=2),flush=True)


if __name__=='__main__':
    main()
