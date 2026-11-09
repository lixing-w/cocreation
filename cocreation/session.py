"""Object-owned lazy worker; private IPC is independent of inference stdout."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import weakref
from multiprocessing.connection import Connection


def _dispose(state):
    connection = state.pop('connection', None)
    process = state.pop('process', None)
    if connection is not None:
        connection.close()
    if process is not None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        else:
            process.wait()
    stream = state.pop('log', None)
    if stream is not None:
        stream.close()


class ProcessSession:
    def __init__(self, backend, python, environment, directory):
        self.backend = backend
        self.python = str(python)
        self.environment = environment
        self.directory = Path(directory)
        self.state = {}
        self.lock = threading.RLock()
        self.finalizer = weakref.finalize(self, _dispose, self.state)

    def _start(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        left, right = socket.socketpair()
        connection = Connection(left.detach())
        log = (self.directory / 'session.log').open('ab', buffering=0)
        self.state.update(connection=connection, log=log)
        try:
            command = [self.python, '-u', '-m', 'cocreation.session_worker',
                       self.backend, str(right.fileno()), str(os.getpid())]
            self.state['process'] = subprocess.Popen(command, pass_fds=(right.fileno(),),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env=self.environment, start_new_session=True)
        except BaseException:
            _dispose(self.state)
            raise
        finally:
            right.close()

    def call(self, job, timeout=7200):
        with self.lock:
            if not self.finalizer.alive:
                raise RuntimeError('Session is closed')
            if not self.state:
                self._start()
            try:
                connection = self.state['connection']
                connection.send_bytes(json.dumps({'job': str(job)}).encode())
                if not connection.poll(timeout):
                    raise TimeoutError('Model request timed out')
                reply = json.loads(connection.recv_bytes())
                if not reply['ok']:
                    raise RuntimeError(reply['error'])
            except BaseException:
                self.close()
                raise

    def close(self):
        with self.lock:
            self.finalizer()
