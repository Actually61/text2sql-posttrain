"""Freeze the formal weighted run before its one CPU acceptance; preserve history."""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / 'reports/sft_round2_formal_v1_freeze.json'
SNAPSHOT = ROOT / 'outputs/checks/sft_round2_formal_v1_freeze'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if DESTINATION.exists() or SNAPSHOT.exists():
        raise FileExistsError('Formal freeze already exists; no overwrite')
    sources = {Path(__file__).relative_to(ROOT).as_posix()}
    for name in ('src/train/sft_round2_formal_v1.py', 'src/train/test_round2_formal_cpu_v1.py'):
        tree = ast.parse((ROOT / name).read_text(encoding='utf-8-sig'))
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'SOURCES' for t in node.targets))
        sources.update(ast.literal_eval(assignment.value))
    previous = {}
    def merge(mapping):
        for name, sha in mapping.items():
            path = Path(name)
            path = path if path.is_absolute() else ROOT / path
            relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
            if relative in previous and previous[relative] != sha:
                raise ValueError(f'Conflicting historical digest: {relative}')
            previous[relative] = sha
    pilot_preflight = read(ROOT / 'reports/sft_round2_pilot_v2_preflight.json')
    merge(pilot_preflight['protected_sha256'])
    run = ROOT / 'outputs/runs/sft_round2_pilot_v2'
    merge({str(run / name): sha for name, sha in read(run / 'artifact_sha256.json').items()})
    additional = [run / 'artifact_sha256.json']
    additional += [ROOT / 'reports' / name for name in (
        'sft_round2_pilot_v2_preflight.json', 'sft_round2_pilot_v2_result.json',
        'sft_round2_pilot_v2_analysis.json', 'sft_round2_pilot_v2_walkthrough.md',
        'sft_round2_pilot_v2_start_resources.json')]
    additional += [ROOT / 'configs/sft_round1_resource_v3.json']
    merge({str(path): digest(path) for path in additional})
    for name, expected in previous.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Historical file changed before freeze: {name}')
    record = {'version': 'sft_round2_formal_v1', 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': {name: digest(ROOT / name) for name in sorted(sources)},
              'previous_attempt_sha256': previous,
              'policy': 'No protected source changes after freeze. One CPU acceptance and one formal 796-update attempt. No automatic retry.'}
    SNAPSHOT.mkdir(parents=True, exist_ok=False)
    for name, expected in record['source_sha256'].items():
        target = SNAPSHOT / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        assert digest(target) == digest(ROOT / name) == expected
    # Prior large artifacts remain in their original directories with verified hashes.
    with DESTINATION.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    with (SNAPSHOT / 'freeze.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'status': 'frozen', 'source_files': len(sources),
                      'preserved_files': len(previous), 'manifest_sha256': digest(DESTINATION)}))


if __name__ == '__main__':
    main()
