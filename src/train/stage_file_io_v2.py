"""Serialize stage reads (through close) and complete atomic publication."""
from pathlib import Path

from src.train.stage_file_io import stage_evidence
from src.train.windows_json_ipc import read_json, write_json_atomic
from src.train.windows_stage_mutex import locked_stage


def read_stage(directory, *, wait_seconds=1.0):
    evidence = stage_evidence(directory)
    # read_json closes its file before returning, therefore before mutex release.
    with locked_stage(directory, wait_seconds=wait_seconds, evidence=evidence):
        return read_json(Path(directory) / 'stage.json', missing_ok=True, evidence=evidence)


def write_stage(directory, value, *, wait_seconds=1.0):
    evidence = stage_evidence(directory)
    # Keep serialization/fsync inside this small critical section for minimal change.
    with locked_stage(directory, wait_seconds=wait_seconds, evidence=evidence):
        return write_json_atomic(Path(directory) / 'stage.json', value, evidence=evidence)
