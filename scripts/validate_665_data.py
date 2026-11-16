"""Fail early for a wrong physical schema; never edit labels or load credentials."""
import argparse
import json
from pathlib import Path


def validate(root):
    root = Path(root)
    info = json.loads((root / 'meta/info.json').read_text())
    if info.get('codebase_version') != 'v2.1':
        raise ValueError('Expected a LeRobot v2.1 dataset')
    for key in ('action', 'observation.state'):
        if info['features'][key]['shape'] != [17]:
            raise ValueError(f'{key} must contain exactly 17 SDK-user-radian channels')
    for key in ('observation.images.head', 'observation.images.wrist'):
        if key not in info['features']:
            raise ValueError(f'Missing RGB video feature: {key}')
    if info['total_episodes'] < 1 or info['total_frames'] < 1:
        raise ValueError('Dataset is empty')
    # When available, reject a legacy domain instead of double-flipping signs.
    provenance = root / 'meta/source_provenance.jsonl'
    if provenance.exists():
        for line in provenance.read_text().splitlines():
            row = json.loads(line)
            domain = row.get('action_hand_domain')
            if domain is not None and domain != 'sdk_user_rad':
                raise ValueError(f'Unsupported hand action domain: {domain}')
    return {'frames': info['total_frames'], 'episodes': info['total_episodes'], 'fps': info['fps']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Existing LeRobot v2.1 root')
    print(json.dumps(validate(parser.parse_args().dataset)))
