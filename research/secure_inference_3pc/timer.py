from dataclasses import dataclass, field
import time
from typing import Callable, ClassVar, Dict, Optional

# Copied from: https://realpython.com/python-timer/#using-the-python-timer-context-manager

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""
#TODO: should be like :@Timer("ShareConvertClient")

@dataclass
class Timer:
    timers: ClassVar[Dict[str, float]] = {}
    counter: ClassVar[Dict[str, int]] = {}
    is_avg: ClassVar[Dict[str, bool]] = {}

    avg: bool = False
    name: Optional[str] = None
    text: str = "Elapsed time: {:0.4f} seconds. Total elapsed time: {:0.4f} seconds. "
    _start_time: Optional[float] = field(default=None, init=False, repr=False)


    def __post_init__(self) -> None:
        """Add timer to dict of timers after initialization"""
        if self.name is not None:
            self.timers.setdefault(self.name, 0)
            self.counter.setdefault(self.name, 0)
            self.is_avg[self.name] = self.avg

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time

        if self.name:
            self.timers[self.name] += elapsed_time
            self.counter[self.name] += 1

        return elapsed_time

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.stop()


def timer(name, avg=False):
    def inner_timer(func):
        def wrapper(*args, **kwargs):
            with Timer(name=name, avg=avg):
                return func(*args, **kwargs)
        return wrapper
    return inner_timer
#
def print_timers():
    for name, elapsed in Timer.timers.items():
        counter = Timer.counter[name]
        is_avg = Timer.is_avg[name]
        if is_avg:
            elapsed = elapsed / counter
        print(f"{name} Elapsed time: {elapsed:0.4f} seconds")

import atexit
atexit.register(print_timers)
