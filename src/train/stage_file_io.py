"""Stage-only JSON IPC adapter; unrelated training and handshake files stay unchanged."""
import os
from pathlib import Path
import threading

from src.train.cache_release_watchdog import append_durable
from src.train.windows_json_ipc import read_json, write_json_atomic


def stage_evidence(directory):
    # Each process/thread owns its log, so two readers never append the same file.
    path = Path(directory) / f'stage_ipc_{os.getpid()}_{threading.get_ident()}.jsonl'
    def record(row):
        with path.open('a', encoding='utf-8') as stream:
            append_durable(stream, row)
    return record


def read_stage(directory):
    return read_json(Path(directory) / 'stage.json', missing_ok=True,
                     evidence=stage_evidence(directory))


def write_stage(directory, value):
    return write_json_atomic(Path(directory) / 'stage.json', value,
                             evidence=stage_evidence(directory))
