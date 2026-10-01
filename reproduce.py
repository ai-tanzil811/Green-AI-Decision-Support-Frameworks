#!/usr/bin/env python
"""
One command to regenerate every derived artifact in this repository (workflow section 26).

Runs, in dependency order:

    1. surrogate/validate.py                  grouped CV -> results/surrogate/
    2. analysis/build_training_envelope.py    envelope -> models/model_metadata.json
    3. analysis/benchmark_analysis.py         summary tables -> results/benchmark/
    4. analysis/pareto_analysis.py            frontiers -> results/pareto/
    5. analysis/recommendation_scenarios.py   Table 4 -> results/recommendations/
    6. analysis/figures.py                    Figures A and E
    7. experiments/validate_recommendation.py predicted vs actual back-test
    8. pytest                                 the test suite

Nothing here touches the input dataset or the .joblib bundle; both are read-only inputs.

Usage:
    python reproduce.py            # run everything
    python reproduce.py --check    # verify recorded artifacts match the data, change nothing
    python reproduce.py --list     # show the steps without running them
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# (label, argv, runs_in_check_mode)
STEPS = [
    # The envelope is rebuilt AFTER the surrogate audit: validate.py regenerates
    # models/model_metadata.json, and the envelope is a derived block written into it.
    ('surrogate validation', [PY, 'surrogate/validate.py'], False),
    ('training envelope', [PY, 'analysis/build_training_envelope.py'], False),
    ('training envelope (check)', [PY, 'analysis/build_training_envelope.py', '--check'], True),
    ('benchmark tables', [PY, 'analysis/benchmark_analysis.py'], False),
    ('pareto frontiers', [PY, 'analysis/pareto_analysis.py'], False),
    ('recommendation scenarios', [PY, 'analysis/recommendation_scenarios.py'], False),
    ('figures', [PY, 'analysis/figures.py'], False),
    ('recommendation back-test',
     [PY, 'experiments/validate_recommendation.py', '--from-benchmark', '--replace'], False),
    ('tests', [PY, '-m', 'pytest', 'green_peft_cli/green_peft_pkg/tests', '-q'], True),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true',
                    help='Only run steps that verify without writing.')
    ap.add_argument('--list', action='store_true', help='List the steps and exit.')
    args = ap.parse_args()

    steps = [s for s in STEPS if s[2]] if args.check else [s for s in STEPS if s[0] !=
                                                           'training envelope (check)']

    if args.list:
        for i, (label, argv, _) in enumerate(steps, 1):
            print(f'{i}. {label:28s} {" ".join(argv[1:])}')
        return 0

    failures = []
    for i, (label, argv, _) in enumerate(steps, 1):
        print(f'\n{"=" * 78}\n[{i}/{len(steps)}] {label}\n{"=" * 78}', flush=True)
        started = time.time()
        proc = subprocess.run(argv, cwd=ROOT)
        elapsed = time.time() - started
        if proc.returncode != 0:
            failures.append(label)
            print(f'--- FAILED ({label}, exit {proc.returncode}, {elapsed:.1f}s)')
        else:
            print(f'--- ok ({elapsed:.1f}s)')

    print(f'\n{"=" * 78}')
    if failures:
        print(f'{len(failures)} step(s) failed: {", ".join(failures)}')
        return 1
    print(f'All {len(steps)} steps completed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
