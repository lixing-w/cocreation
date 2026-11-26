"""Queue media batches and deferred dependencies between FIFO dispatch waves."""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import time
from typing import Any, Callable


@dataclass(frozen=True)
class AssetEvent:
    token: str
    operation: str
    path: str
    seconds: float
    status: str = 'accepted'
    discarded: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReviewResult:
    token: str
    seconds: float
    value: Any = None
    error: Exception | None = None

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value


@dataclass(frozen=True)
class ReviewRequest:
    token: str
    call: Callable[[], Any]

    def run(self):
        started = time.monotonic()
        try:
            value = self.call()
        except Exception as error:
            return ReviewResult(self.token, time.monotonic() - started, error=error)
        return ReviewResult(self.token, time.monotonic() - started, value=value)


def review_call(session, token, directory, method, *args, deferred=False, **kwargs):
    def call():
        with session(directory) as reviewer:
            return getattr(reviewer, method)(*args, **kwargs)

    if deferred:
        # The worker reads a snapshot; the dispatcher owns the mutable manifest.
        args, kwargs = deepcopy((args, kwargs))
        return (yield ReviewRequest(token, call))
    return call()


def generation_time(receipt):
    seconds = receipt.get('generation_seconds', receipt.get('elapsed_seconds', 0.0))
    return max(0.0, seconds - receipt.get('model_reload_seconds', 0.0))


class AssetRejected(RuntimeError):
    def __init__(self, event, message):
        super().__init__(message)
        self.event = event


class GenerationQueue:
    def __init__(self, units, schedule):
        self.units = {}
        self.schedule = schedule
        self.round = 0
        self.main = deque()
        self.buffer = {}
        self.enqueue(units)

    def __bool__(self):
        return bool(self.main or self.buffer)

    def enqueue(self, units):
        self.units.update(units)
        self.buffer.update(dict.fromkeys(units))

    def promote(self, current_model):
        if self.main:
            raise RuntimeError('Cannot promote media while the main queue has work')
        waiting = {
            key: dict(self.units[key], dependencies=[parent for parent in self.units[key]['dependencies']
                                                    if parent in self.buffer])
            for key in self.buffer
        }
        self.main.extend(self.schedule(waiting, current_model=current_model))
        self.buffer.clear()
        self.round += 1

    def pop_ready(self, in_flight):
        unaccepted = set(self.main).union(in_flight, self.buffer)
        ready = next((key for key in self.main if unaccepted.isdisjoint(self.units[key]['dependencies'])), None)
        if ready is not None:
            self.main.remove(ready)
        return ready

    def defer(self, rejected):
        self.buffer[rejected] = None
        retained = deque()
        for key in self.main:
            # The queue is already topological. A newly held node can also
            # block its descendants later in the same scan.
            if any(parent in self.buffer for parent in self.units[key]['dependencies']):
                self.buffer[key] = None
            else:
                retained.append(key)
        self.main = retained

    def discard(self, keys):
        self.main = deque(key for key in self.main if key not in keys)
        for key in keys:
            self.buffer.pop(key, None)

    def discard_case(self, case_id):
        self.discard({key for key in self.units if key[0] == case_id})
