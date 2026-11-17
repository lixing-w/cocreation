"""Serve the Qwen3.6 NVFP4 checkpoint using the scratch vLLM environment, inside a GPU allocation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cocreation.storage import runtime_environment, scratch_root
from cocreation.qwen_client import MODEL_ID


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--max-model-len', type=int, default=32768)
    p.add_argument('--max-num-seqs', type=int, default=8)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.50)
    p.add_argument('--enable-thinking', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--enforce-eager', action='store_true')
    p.add_argument('--video-fps', type=float, default=1.0)
    p.add_argument('--video-max-frames', type=int, default=16)
    p.add_argument('--python', type=Path)
    p.add_argument('--cuda-home', type=Path, help='CUDA 12.9 toolkit; alternatively load cuda/12.9.0-cinr')
    p.add_argument('--stop', action='store_true', help='Stop this deployment\'s recorded vLLM process group')
    a = p.parse_args()
    runtime = scratch_root()/'runtime/qwen3.6-35b-a3b'
    if a.stop:
        manifest = runtime/'server.json'
        if not manifest.is_file():
            print('No recorded Qwen vLLM server')
            return
        pid = int(json.loads(manifest.read_text())['pid'])
        process = Path('/proc')/str(pid)/'cmdline'
        if not process.is_file():
            print('Recorded Qwen vLLM server is not running')
            return
        command = process.read_bytes().replace(b'\0', b' ').decode()
        if 'vllm.entrypoints.openai.api_server' not in command or MODEL_ID not in command:
            raise RuntimeError('Recorded PID belongs to another process; leaving it untouched')
        if os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
        print('Stopping Qwen vLLM server', pid)
        return
    env = runtime_environment()
    env.update(OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    env.setdefault('MAX_JOBS', env.get('SLURM_CPUS_PER_TASK', '6'))
    cuda_home = a.cuda_home or (Path(env['CUDA_HOME']) if env.get('CUDA_HOME') else None)
    if cuda_home is None and shutil.which('nvcc'):
        cuda_home = Path(shutil.which('nvcc')).resolve().parents[1]
    if cuda_home is None or not (cuda_home/'bin/nvcc').is_file():
        raise RuntimeError('FlashInfer needs the CUDA toolkit: module load cuda/12.9.0-cinr, or pass --cuda-home')
    env.update(CUDA_HOME=str(cuda_home), CUDA_PATH=str(cuda_home),
        TRITON_PTXAS_PATH=str(cuda_home/'bin/ptxas'),
        TRITON_PTXAS_BLACKWELL_PATH=str(cuda_home/'bin/ptxas'),
        FLASHINFER_WORKSPACE_BASE=str(runtime),
        FLASHINFER_CUBIN_DIR=str(scratch_root()/'cache/flashinfer/cubins'),
        TORCHINDUCTOR_CACHE_DIR=str(scratch_root()/'cache/torchinductor'),
        VLLM_CACHE_ROOT=str(scratch_root()/'cache/vllm'))
    model = json.loads((runtime/'model.json').read_text())
    checkpoint = Path(model['path'])
    expected_id = MODEL_ID
    if model['model_id'] != expected_id or not checkpoint.is_dir():
        raise ValueError('Run scripts/setup/setup_qwen.py to prepare the NVFP4 checkpoint')
    python = a.python or runtime/'vllm/bin/python'
    # Invoking a venv interpreter directly does not activate its bin directory.
    # FP8 JIT compilation also launches the venv's ninja executable.
    env['PATH'] = os.pathsep.join((str(cuda_home/'bin'), str(python.parent), env['PATH']))
    command = [str(python), '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(checkpoint), '--served-model-name', model['model_id'],
        '--dtype', 'bfloat16', '--tensor-parallel-size', '1',
        '--host', a.host, '--port', str(a.port),
        '--max-model-len', str(a.max_model_len), '--max-num-seqs', str(a.max_num_seqs),
        '--max-num-batched-tokens', '8192',
        '--gpu-memory-utilization', str(a.gpu_memory_utilization),
        '--enable-prefix-caching', '--mamba-cache-mode', 'align',
        '--reasoning-parser', 'qwen3',
        '--default-chat-template-kwargs', json.dumps(dict(enable_thinking=a.enable_thinking)),
        '--attention-backend', 'FLASHINFER',
        '--max-cudagraph-capture-size', '8', '--allowed-local-media-path', str(scratch_root()),
        # Avoid a second media-cache layer while vLLM owns token prefix caching.
        '--mm-processor-cache-gb', '0',
        '--limit-mm-per-prompt', json.dumps(dict(image=32, video=1)),
        '--media-io-kwargs', json.dumps(dict(video=dict(num_frames=a.video_max_frames, fps=a.video_fps))),
        '--mm-processor-kwargs', json.dumps(dict(max_pixels=512*512, fps=a.video_fps, do_sample_frames=False))]
    if a.enforce_eager:
        command.append('--enforce-eager')
    (runtime/'server.json').write_text(json.dumps(dict(pid=os.getpid(), command=command,
        model=model, host=a.host, port=a.port, dtype='bfloat16', precision='nvfp4', enable_thinking=a.enable_thinking,
        cuda_home=str(cuda_home),
        build_jobs=env['MAX_JOBS'],
        video_fps=a.video_fps, video_max_frames=a.video_max_frames,
        slurm_job_id=os.environ.get('SLURM_JOB_ID')), indent=2)+'\n')
    print('Launching NVFP4 vLLM:', ' '.join(command), flush=True)
    os.execve(str(python), command, env)


if __name__ == '__main__':
    main()
