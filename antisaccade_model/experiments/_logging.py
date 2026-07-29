"""Run logging helpers for experiment entry points."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout


class _TeeStream:
    """Write a text stream to a terminal stream and a log file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextmanager
def tee_run_output(log_path: str, *, level: int = logging.INFO, fmt: str = "%(asctime)s %(levelname)s %(message)s"):
    """Mirror console output to ``log_path`` while preserving terminal output."""

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    previous_stdout = sys.stdout
    previous_stderr = sys.stderr
    previous_handlers = logging.getLogger().handlers[:]
    previous_level = logging.getLogger().level

    with open(log_path, "w", encoding="utf-8") as log_file:
        stdout_tee = _TeeStream(previous_stdout, log_file)
        stderr_tee = _TeeStream(previous_stderr, log_file)
        handler = logging.StreamHandler(stderr_tee)
        handler.setFormatter(logging.Formatter(fmt))

        root_logger = logging.getLogger()
        root_logger.handlers = [handler]
        root_logger.setLevel(level)

        try:
            with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
                yield
        finally:
            logging.getLogger().handlers = previous_handlers
            logging.getLogger().setLevel(previous_level)
            log_file.flush()