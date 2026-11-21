"""Run one indexed experiment and publish only a complete attempt."""

import json
import errno
import os
from pathlib import Path
import re
import subprocess


def run_worker(source, root, index, uid):
    plan = json.loads((source / 'discovery.json').read_text())
    if not 0 <= index < len(plan['experiments']):
        raise ValueError('Experiment index is outside the plan.')
    name = plan['experiments'][index]['name']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or not re.fullmatch(r'[A-Za-z0-9-]+', uid):
        raise ValueError('Invalid experiment name or pod UID.')
    target = root / name
    if target.exists():
        if json.loads((target / 'run.json').read_text())['status'] == 'complete':
            return
        raise FileExistsError(f'Incomplete canonical result: {target}')
    attempt_root = root.with_name(root.name + '-attempts') / uid
    env = {**os.environ, 'OUTPUT_ROOT': str(attempt_root), 'JOB_COMPLETION_INDEX': str(index)}
    subprocess.run(['/bin/sh', str(source / 'discovery-multiscene.sh')], env=env, check=True)
    attempt = attempt_root / name
    if json.loads((attempt / 'run.json').read_text())['status'] != 'complete':
        raise ValueError('The experiment did not complete.')
    root.mkdir(parents=True, exist_ok=True)
    try:
        attempt.rename(target)
    except OSError as error:
        if (error.errno not in (errno.EEXIST, errno.ENOTEMPTY)
                or json.loads((target / 'run.json').read_text())['status'] != 'complete'):
            raise
    print(json.dumps({'published_experiment': name, 'pod_uid': uid}), flush=True)


if __name__ == '__main__':
    run_worker(Path('/source'), Path(os.environ['OUTPUT_ROOT']),
               int(os.environ['JOB_COMPLETION_INDEX']), os.environ['POD_UID'])
