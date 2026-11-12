"""Freeze v2 sources and preserve the previous failed CPU/preflight attempt."""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / 'reports/sft_round2_pilot_v2_freeze.json'
SNAPSHOT = ROOT / 'outputs/checks/sft_round2_pilot_v2_freeze'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DESTINATION.exists() or SNAPSHOT.exists():
        raise FileExistsError('Freeze already exists; no overwrite')
    sources = {Path(__file__).relative_to(ROOT).as_posix()}
    for name in ('src/train/sft_round2_pilot_v2.py', 'src/train/test_round2_pilot_cpu_v2.py'):
        tree = ast.parse((ROOT / name).read_text(encoding='utf-8-sig'))
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'SOURCES' for t in node.targets))
        sources.update(ast.literal_eval(assignment.value))
    old = set()
    for folder in ('outputs/checks/sft_round2_pilot_cpu_v1',
                   'outputs/checks/sft_round2_pilot_preflight_failure_v1'):
        location = ROOT / folder
        assert location.is_dir(), folder
        old.update(path.relative_to(ROOT).as_posix() for path in location.rglob('*') if path.is_file())
    old.update(('configs/sft_round2_pilot_v1.json', 'src/train/sft_round2_pilot_v1.py',
                'src/train/sft_round2_pilot_v1_worker.py', 'src/train/test_round2_pilot_cpu_v1.py',
                'reports/sft_round2_pilot_cpu_v1_checks.json',
                'reports/sft_round2_pilot_v1_preflight.json', 'reports/sft_round2_pilot_v1_walkthrough.md'))
    record = {'version': 'sft_round2_pilot_v2', 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': {name: digest(ROOT / name) for name in sorted(sources)},
              'previous_attempt_sha256': {name: digest(ROOT / name) for name in sorted(old)},
              'policy': 'No protected source edits after freeze; one CPU acceptance then at most one 2+2 GPU pilot.'}
    SNAPSHOT.mkdir(parents=True, exist_ok=False)
    for mapping in ('source_sha256', 'previous_attempt_sha256'):
        for name, expected in record[mapping].items():
            target = SNAPSHOT / mapping / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
            assert digest(target) == digest(ROOT / name) == expected
    with DESTINATION.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'frozen', 'sources': len(sources), 'previous_files': len(old),
                      'manifest_sha256': digest(DESTINATION)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
