import os
import resource
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class DeadlineReached(Exception):
    pass


def ncpu() -> int:
    """Cores this process may use.

    `os.cpu_count()` reports the machine, not the cgroup, so under Slurm it reads the whole node
    however few `--cpus-per-task` were granted. Both dask's default threadpool and arosics' `CPUs`
    size themselves from it, so left alone a small allocation spawns a node's worth of workers.
    """
    return len(os.sched_getaffinity(0))


@dataclass
class Workspace:
    dir: Path
    deadline: float | None = None
    log: Callable[[str], None] = lambda msg: print(msg, flush=True)

    def time_left(self) -> float:
        return np.inf if self.deadline is None else self.deadline - time.monotonic()

    def path(self, name: str) -> Path:
        return self.dir / name

    def check_deadline(self, need_s: float, what: str) -> None:
        if self.time_left() < need_s:
            msg = f"{what}: {self.time_left() / 60:.0f} min left < ~{need_s / 60:.0f} min needed"
            self.log(msg)
            raise DeadlineReached(msg)


class phase:
    """Time one phase and log wall, CPU and the mean cores it actually kept busy.

    `cores` is the number that answers "would more `--cpus-per-task` help here": a stage pinned near
    1.0 is serial (GIL-bound Python, or waiting on disk) and more cores buy it nothing, while one
    that tracks `ncpu()` is already scaling. CPU time sums this process *and* its waited-for
    children, so a subprocess stage is measured on the same footing as an in-process one, not
    reported as idle.
    """

    def __init__(self, ws: Workspace, name: str):
        self.ws, self.name = ws, name

    def _cpu_s(self) -> float:
        return sum(
            r.ru_utime + r.ru_stime
            for r in (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN))
        )

    def __enter__(self):
        self.t0, self.c0 = time.monotonic(), self._cpu_s()
        return self

    def __exit__(self, *exc):
        self.wall, self.cpu = time.monotonic() - self.t0, self._cpu_s() - self.c0
        self.cores = self.cpu / max(self.wall, 1e-9)
        self.ws.log(f"{self.name}: {self.wall:.0f}s wall, {self.cpu:.0f}s cpu, {self.cores:.1f} cores of {ncpu()}")
        return False
