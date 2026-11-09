"""Private persistent worker. Each job builds new conditioning/messages."""
import json
import os
from pathlib import Path
import signal
import sys
import threading
import traceback
from multiprocessing.connection import Connection


def main():
    backend, fd, parent = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    def terminate(*_):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    done = threading.Event()
    def watch():
        while not done.wait(1):
            if os.getppid() != parent:
                os.kill(os.getpid(), signal.SIGTERM)
                return
    threading.Thread(target=watch, daemon=True).start()
    if backend in ('qwen', 'flux'):
        from .worker import ModelSession
        engine = ModelSession(backend)
    elif backend == 'minimax_h3':
        from .comfy_worker import ComfySession
        engine = ComfySession()
    elif backend == 'vdn_h3':
        from .vdn_worker import VDNSession
        engine = VDNSession()
    elif backend == 'vdn_native':
        from .vdn_worker import NativeVDNSession
        engine = NativeVDNSession()
    else:
        raise ValueError('Unknown session backend')
    try:
        with Connection(fd) as connection:
            while True:
                try:
                    job = Path(json.loads(connection.recv_bytes())['job'])
                except EOFError:
                    break
                request = json.loads((job / 'request.json').read_text())
                try:
                    result = engine.execute(request, job)
                    result['worker_pid'] = os.getpid()
                    (job / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
                    connection.send_bytes(b'{"ok":true}')
                except Exception as error:
                    traceback.print_exc()
                    message = f'{type(error).__name__}: {error}'
                    (job / 'error.json').write_text(json.dumps({'error': message}))
                    connection.send_bytes(json.dumps({'ok': False, 'error': message}).encode())
                    break  # Failed CUDA state is never reused.
    finally:
        done.set()
        engine.close()


if __name__ == '__main__':
    main()
