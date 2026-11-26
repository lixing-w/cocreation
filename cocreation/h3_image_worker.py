"""Resident isolated ComfyUI worker for W4A8 MiniMax-H3-Image generation and edits."""
import contextlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

from .comfy_worker import api, wait_result, stage_media
from .h3_image import installation, build_graph, COMFY_REVISION
from .storage import runtime_environment, scratch_root


@contextlib.contextmanager
def server(root, runtime, home):
    source, python = root/'ComfyUI', root/'venv/bin/python'
    actual = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if actual != COMFY_REVISION:
        raise RuntimeError('Unexpected image ComfyUI revision: '+actual)
    for folder in ('input','output','user'):
        (home/folder).mkdir(parents=True,exist_ok=True)
    paths = home/'model_paths.json'
    paths.write_text(json.dumps({'h3_image':dict(base_path=str(root/'models'),
        diffusion_models='diffusion_models',text_encoders='text_encoders',vae='vae')}))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port=sock.getsockname()[1]
    bootstrap = Path(__file__).with_name('h3_image_comfy.py')
    command=[str(python),str(bootstrap),str(source/'main.py'),'--listen','127.0.0.1','--port',str(port),
        '--input-directory',str(home/'input'),'--output-directory',str(home/'output'),
        '--user-directory',str(home/'user'),'--extra-model-paths-config',str(paths),
        '--disable-auto-launch','--disable-all-custom-nodes','--offline','--cache-classic',
        '--use-pytorch-cross-attention','--enable-triton-backend','--preview-method','none']
    env = runtime_environment()
    ptxas = shutil.which('ptxas')
    if not ptxas:
        manifest=scratch_root()/'runtime/qwen3.6-35b-a3b/server.json'
        if manifest.is_file():
            candidate=Path(json.loads(manifest.read_text()).get('cuda_home',''))/'bin/ptxas'
            if candidate.is_file():
                ptxas=str(candidate)
    if ptxas:
        env.update(TRITON_PTXAS_PATH=ptxas,TRITON_PTXAS_BLACKWELL_PATH=ptxas)
    process = subprocess.Popen(command,cwd=source,env=env,start_new_session=True)
    base=f'http://127.0.0.1:{port}'
    try:
        deadline=time.monotonic()+runtime['startup_timeout']
        while True:
            if process.poll() is not None:
                raise RuntimeError('H3 Image ComfyUI startup failed; see session.log')
            try:
                api(base,'/system_stats')
                break
            except (OSError,TimeoutError):
                if time.monotonic()>deadline:
                    raise TimeoutError('H3 Image startup timed out')
                time.sleep(.5)
        yield base,process
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()


class H3ImageSession:
    def __init__(self):
        self.stack=contextlib.ExitStack()
        self.endpoint=None
        self.runtime=None

    def execute(self,request,directory):
        if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise RuntimeError('H3 Image inference requires an allocated Slurm GPU')
        runtime=request['runtime']
        if self.runtime is not None and self.runtime!=runtime:
            raise ValueError('Close H3 Image before changing runtime')
        started=time.perf_counter()
        root=Path(runtime['root'])
        info=installation(root)
        if self.endpoint is None:
            self.runtime=dict(runtime)
            self.home=Path(os.environ.get('COCREATION_SESSION_DIR',str(directory)))/'image_comfy_session'
            self.endpoint=self.stack.enter_context(server(root,runtime,self.home))
        names,_=stage_media(request,self.home/'input'/Path(directory).name)
        graph=build_graph(request,[Path(directory).name+'/'+n for n in names])
        graph['save']['inputs']['filename_prefix']=Path(directory).name+'/image'
        (Path(directory)/'workflow.json').write_text(json.dumps(graph,indent=2))
        base,process=self.endpoint
        submitted=api(base,'/prompt',{'prompt':graph})
        if submitted.get('node_errors') or not submitted.get('prompt_id'):
            raise RuntimeError('Invalid H3 Image workflow: '+json.dumps(submitted))
        history=wait_result(base,submitted['prompt_id'],process,runtime['timeout'])
        (Path(directory)/'history.json').write_text(json.dumps(history,indent=2))
        outputs=list((self.home/'output'/Path(directory).name).glob('*.png'))
        if len(outputs)!=1:
            raise RuntimeError('H3 Image did not produce exactly one PNG')
        target=Path(directory)/'image.png'
        shutil.move(outputs[0],target)
        from PIL import Image
        with Image.open(target) as image:
            if image.size != (request['options']['width'],request['options']['height']):
                raise RuntimeError('H3 Image returned an unexpected canvas')
        return dict(status='ok',path=str(target),kind='image',model_id=request['model_id'],
            model_revision=info['revision'],weight_repo=info['weight_repo'],comfy_revision=COMFY_REVISION,
            transformer_precision='W4A8',encoder_precision='NVFP4-AWQ',torch=info['torch'],
            steps=6,guidance_scale=1.,sampler='euler',scheduler='simple',generator_device='cpu',
            attention_backend='pytorch_sdpa',
            comfy_pid=process.pid,time_elapsed=f'{time.perf_counter()-started:.2f}s')

    def close(self):
        self.stack.close()
        self.endpoint=None
