"""
green-peft: command-line recommender built on the GreenPEFT surrogate models.

Usage:
    green-peft recommend --artifacts-dir ./peft_bench_export --vram 16 \\
        --carbon 0.005 --accuracy 0.90 --profile balanced

    green-peft recommend --artifacts-dir ./peft_bench_export --vram 16 \\
        --weights 0.5,0.2,0.2,0.1 --top-k 5 --json

    green-peft list-zoo --artifacts-dir ./peft_bench_export

The artifacts directory is whatever you exported from the notebook -- it must contain
results/surrogate_models.joblib (from Cell 13) and configs/backbones.yaml +
configs/methods/*.yaml (from Cell 3). Zip peft_bench/results and peft_bench/configs
together and unzip them here; nothing else from the run is needed.
"""

from __future__ import annotations
import argparse
import json
import sys

import pandas as pd

from .recommender import (
    GreenPEFTArtifacts, Constraints, GEI_PROFILES, recommend, explain, build_candidates,
)


def _parse_weights(s: str) -> tuple:
    parts = [float(x) for x in s.split(',')]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            '--weights needs exactly 4 comma-separated numbers: accuracy,memory,carbon,time')
    return tuple(parts)


def cmd_recommend(args):
    try:
        artifacts = GreenPEFTArtifacts.load(args.artifacts_dir)
    except FileNotFoundError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2

    constraints = Constraints(
        max_vram_gb=args.vram,
        max_carbon_kgco2eq=args.carbon,
        min_accuracy=args.accuracy,
        max_time_seconds=args.time,
    )
    weights = _parse_weights(args.weights) if args.weights else None
    profile = args.profile if weights is None else None

    result = recommend(
        artifacts, constraints,
        profile=profile or 'balanced', custom_weights=weights,
        gpu_vram_gb=args.vram,
        backbones=args.backbones.split(',') if args.backbones else None,
        methods=args.methods.split(',') if args.methods else None,
        top_k=args.top_k,
    )

    if args.json:
        payload = {
            'n_candidates': result['n_candidates'], 'n_feasible': result['n_feasible'],
            'profile': result['profile'], 'weights': result['weights'],
        }
        if result['ranked'] is not None:
            payload['ranked'] = json.loads(result['ranked'].to_json(orient='records'))
            payload['pareto_only'] = json.loads(result['pareto_only'].to_json(orient='records'))
        else:
            payload['dropped_summary'] = result['dropped_summary']
        print(json.dumps(payload, indent=2))
        return 0

    print(explain(result))
    if result['ranked'] is not None and args.top_k > 1:
        print(f"\nTop {min(args.top_k, len(result['ranked']))} by GEI:")
        cols = ['backbone', 'method', 'pred_accuracy', 'pred_peak_vram_gb',
               'pred_carbon_kgco2eq', 'gei', 'on_pareto_front']
        with pd.option_context('display.width', 120, 'display.float_format', '{:.4f}'.format):
            print(result['ranked'][cols].to_string(index=False))
    return 0


def cmd_list_zoo(args):
    try:
        artifacts = GreenPEFTArtifacts.load(args.artifacts_dir)
    except FileNotFoundError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    candidates = build_candidates(artifacts,
                                  args.backbones.split(',') if args.backbones else None,
                                  args.methods.split(',') if args.methods else None)
    with pd.option_context('display.width', 120):
        print(candidates[['backbone', 'model_id', 'family', 'params_b', 'method']]
              .drop_duplicates(subset=['backbone', 'method']).to_string(index=False))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog='green-peft',
                                description='Constraint-aware PEFT strategy recommender.')
    sub = p.add_subparsers(dest='command', required=True)

    r = sub.add_parser('recommend', help='Recommend a PEFT config under given constraints.')
    r.add_argument('--artifacts-dir', required=True,
                   help='Directory with results/surrogate_models.joblib and configs/.')
    r.add_argument('--vram', type=float, default=None, help='Max VRAM budget in GB.')
    r.add_argument('--carbon', type=float, default=None, help='Max carbon budget in kgCO2eq.')
    r.add_argument('--accuracy', type=float, default=None, help='Minimum required accuracy.')
    r.add_argument('--time', type=float, default=None, help='Max wall-clock budget in seconds.')
    r.add_argument('--profile', choices=list(GEI_PROFILES), default='balanced',
                   help='Named GEI weight profile (ignored if --weights is given).')
    r.add_argument('--weights', type=str, default=None,
                   help='Custom GEI weights "w_acc,w_mem,w_carbon,w_time", must sum to 1.0.')
    r.add_argument('--backbones', type=str, default=None,
                   help='Comma-separated backbone keys to consider (default: whole model_zoo).')
    r.add_argument('--methods', type=str, default=None,
                   help='Comma-separated method keys to consider (default: all configured methods).')
    r.add_argument('--top-k', type=int, default=3, help='How many ranked candidates to show.')
    r.add_argument('--json', action='store_true', help='Emit machine-readable JSON instead of text.')
    r.set_defaults(func=cmd_recommend)

    z = sub.add_parser('list-zoo', help='List every backbone x method candidate the engine can score.')
    z.add_argument('--artifacts-dir', required=True)
    z.add_argument('--backbones', type=str, default=None)
    z.add_argument('--methods', type=str, default=None)
    z.set_defaults(func=cmd_list_zoo)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:
        # e.g. `green-peft list-zoo ... | head` -- the reader closed early, not an error.
        sys.stderr.close()
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
