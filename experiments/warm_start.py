"""Warm-start continuation of the strongest LightGBM on a disjoint shard block.

The shipped ensemble is built on a LightGBM trained on 48 of the 512 train shards
(295 376 feeds, 23 111 632 candidates). This script continues that exact booster on a
second, disjoint block of shards: 2x training data at the same peak memory, because the
first block is never loaded again.

Protocol, fixed before any late metric is computed:

* trees are added with the original parameters and seed, the initial 160 trees are frozen;
* the number of added trees is chosen by full NDCG on the early val only;
* a blend with the shipped ensemble is chosen by the minimum gain over two early blocks;
* the late val is reported once, after both choices are frozen.

The continuation also dominates its parent booster on both early blocks, so the second
placement tested here is substitution: rebuild the shipped tree ensemble with the
continued booster in place of the one it continues, keeping every weight unchanged.
That comparison can only be measured where the cached val and the component
predictions live.

Usage:
    python experiments/warm_start.py run <cache> <init-model> <out>
    python experiments/warm_start.py assess <run-dir> <reference-val-parquet> <out>
    python experiments/warm_start.py swap <val-cache> <out> --replace <run> --replacement <run> runs... --weights ...
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ranking import COLS, NAMES, evaluate, ndcg_per_feed, prediction_frame, temporal_boundary  # noqa: E402

CATEGORICAL = ('category', 'intent_match_type', 'intent_match_source')
WEIGHTS = (0.025, 0.05, 0.10, 0.20, 0.35, 0.50)
BLOCKS = ((0, 3749), (3749, 5766))


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def params_for(threads, seed):
    return {'objective': 'binary', 'metric': 'None', 'learning_rate': .08,
            'num_leaves': 63, 'min_data_in_leaf': 160, 'lambda_l2': 5,
            'max_bin': 127, 'feature_fraction': .9, 'num_threads': threads,
            'force_col_wise': True, 'verbosity': -1, 'seed': seed}


def per_feed_ndcg(target, score, counts):
    return ndcg_per_feed(target, score, counts)[:, 0]


def block_slices(counts):
    if len(counts) < BLOCKS[-1][1]:
        raise ValueError(f'early blocks need at least {BLOCKS[-1][1]} feeds, got {len(counts)}')
    return [(slice(int(counts[:start].sum()), int(counts[:end].sum())), slice(start, end))
            for start, end in BLOCKS]


def select_weight(base, candidate, counts, targets):
    """Pick the blend weight by the minimum gain over both frozen early blocks."""
    rows = []
    for weight in WEIGHTS:
        blended = (1.0 - weight) * base + weight * candidate
        gains = [float(per_feed_ndcg(targets[rows_slice], blended[rows_slice], counts[feeds]).mean()
                       - per_feed_ndcg(targets[rows_slice], base[rows_slice], counts[feeds]).mean())
                 for rows_slice, feeds in block_slices(counts)]
        minimum = min(gains)
        weighted = float(np.average(gains, weights=[end - start for start, end in BLOCKS]))
        rows.append({'weight': weight, 'block_gains': gains, 'minimum': minimum, 'weighted': weighted})
    admissible = [row for row in rows if row['minimum'] > 0]
    pool = admissible or rows
    chosen = max(pool, key=lambda row: (row['minimum'], row['weighted']))
    return chosen, rows


def paired_bootstrap(base, candidate, counts, targets, rows=None, feeds=None,
                     resamples=2000, seed=2026):
    rows = slice(0, len(targets)) if rows is None else rows
    feeds = slice(0, len(counts)) if feeds is None else feeds
    delta = (per_feed_ndcg(targets[rows], candidate[rows], counts[feeds])
             - per_feed_ndcg(targets[rows], base[rows], counts[feeds]))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(resamples, len(delta)))
    means = delta[draws].mean(axis=1)
    return {'delta': float(delta.mean()),
            'interval': [float(np.quantile(means, .025)), float(np.quantile(means, .975))]}


def standardize(score, counts):
    starts = np.r_[0, np.cumsum(counts[:-1], dtype=np.int64)]
    centered = score - np.repeat(np.add.reduceat(score, starts) / counts, counts)
    spread = np.sqrt(np.add.reduceat(centered ** 2, starts) / counts)
    return centered / np.repeat(np.maximum(spread, 1e-8), counts)


def blend_scores(components, weights):
    """Average per-feed standardized components with the given positive weights."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(components),) or not np.isfinite(weights).all():
        raise ValueError('give one finite weight per component')
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('weights must be nonnegative with a positive sum')
    total = np.zeros(len(components[0]), dtype=np.float64)
    for component, weight in zip(components, weights, strict=True):
        if len(component) != len(total):
            raise ValueError('components must share one candidate alignment')
        total += weight * component
    return total / weights.sum()


def swap_verdict(block_gains, late_gains, late_delta, interval, target=0.001):
    """Apply the journal rule: both early blocks, both late halves, and a real gain."""
    minimum = float(min(block_gains))
    halves = [float(gain) for gain in late_gains]
    accepted = bool(minimum > 0 and min(halves) > 0
                    and (late_delta >= target or (late_delta > 0 and interval[0] > 0)))
    return {'minimum_block_gain': minimum, 'minimum_late_half_gain': min(halves),
            'late_delta': float(late_delta), 'target': float(target), 'accepted': accepted}


def fit(args):
    import lightgbm as lgb

    with np.load(args.val_cache / 'val.npz') as handle:
        val = dict(handle)
    with np.load(args.cache / 'train.npz') as handle:
        x, y, counts = handle['x'], handle['target'], handle['counts']
        shards = handle['shards'].tolist() if 'shards' in handle else []
        prepare_seed = int(handle['prepare_seed']) if 'prepare_seed' in handle else None
    n = COLS[args.kind]
    if x.shape[1] < n:
        raise ValueError(f'{args.kind} needs at least {n} features, cache has {x.shape[1]}')
    if x.shape[1] != n:
        x = np.ascontiguousarray(x[:, :n])
    val_x = np.ascontiguousarray(val['x'][:, :n])
    init = lgb.Booster(model_file=str(args.init_model))
    if init.feature_name() != NAMES[:n]:
        raise ValueError('initial model was trained on different features')
    base_trees = init.num_trees()
    params = params_for(args.threads, args.seed)
    names = NAMES[:n]
    categories = [name for name in CATEGORICAL if name in names]
    # init_model sets a predictor on the constructed dataset, which needs the raw data alive.
    train_data = lgb.Dataset(x, label=y, group=counts, params=params, feature_name=names,
                             categorical_feature=categories, free_raw_data=False)
    print(f'{args.kind}: {len(counts)} train feeds, {len(y)} rows, {n} features, '
          f'{base_trees} initial trees', flush=True)
    train_data.construct()
    del x
    print('training matrix binned', flush=True)

    def progress(state):
        completed = state.iteration + 1
        if completed % args.checkpoint == 0 or completed == args.rounds:
            print(f'trained {completed}/{args.rounds} new trees', flush=True)

    model = lgb.train(params, train_data, num_boost_round=args.rounds,
                      init_model=str(args.init_model), keep_training_booster=True,
                      callbacks=[progress])
    del train_data

    boundary = temporal_boundary(val['timestamp'], .4)
    early_rows = int(val['counts'][:boundary].sum(dtype=np.int64))
    targets = val['target']
    search = {}
    best_score, best_added, best_scores = float('-inf'), 0, None
    for added in range(args.checkpoint, args.rounds + 1, args.checkpoint):
        scores = model.predict(val_x, num_iteration=base_trees + added, num_threads=args.threads,
                               raw_score=True).astype(np.float64)
        early = float(ndcg_per_feed(targets[:early_rows], scores[:early_rows].astype(np.float32),
                                    val['counts'][:boundary])[:, 0].mean())
        search[added] = early
        print(f'added trees={added}, early NDCG={early:.6f}', flush=True)
        if early > best_score:
            best_score, best_added, best_scores = early, added, scores
    direct = best_scores
    repeat = model.predict(val_x, num_iteration=base_trees + best_added, num_threads=args.threads,
                           raw_score=True)
    np.testing.assert_array_equal(direct, repeat.astype(np.float64))

    args.out.mkdir(parents=True, exist_ok=False)
    model.save_model(str(args.out / 'model.txt'), num_iteration=base_trees + best_added)
    prediction_frame(val, direct).write_parquet(args.out / 'val_predictions.parquet')
    metric = evaluate(val, direct, boundary)
    report = {'kind': args.kind, 'params': params, 'init_model': str(args.init_model),
              'init_sha256': sha256(args.init_model), 'base_trees': base_trees,
              'added_trees': best_added, 'total_trees': base_trees + best_added,
              'rounds': args.rounds, 'checkpoint': args.checkpoint,
              'train_feeds': len(counts), 'train_rows': len(y), 'train_shards': shards,
              'prepare_seed': prepare_seed, 'early_search': search,
              'early_feeds': boundary, 'late_feeds': len(val['counts']) - boundary,
              'late_start_timestamp': int(val['timestamp'][boundary]),
              'selected_early_ndcg': best_score, 'metrics': metric}
    (args.out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(metric), flush=True)


def assess(args):
    import polars as pl

    with np.load(args.val_cache / 'val.npz') as handle:
        val = dict(handle)
    boundary = temporal_boundary(val['timestamp'], .4)
    counts, targets = val['counts'], val['target']
    run = json.loads((args.run / 'metrics.json').read_text())
    if run['late_start_timestamp'] != int(val['timestamp'][boundary]):
        raise ValueError('run and assessment use different validation splits')

    keys = prediction_frame(val, np.zeros(len(val['item']))).select(['feed_id', 'item_id'])
    reference = pl.read_parquet(args.reference)
    if not reference.select(['feed_id', 'item_id']).equals(keys):
        raise ValueError('reference predictions are misaligned with the val cache')
    ref = standardize(reference['score'].to_numpy().astype(np.float64), counts)
    new = standardize(pl.read_parquet(args.run / 'val_predictions.parquet')['score']
                      .to_numpy().astype(np.float64), counts)

    chosen, grid = select_weight(ref, new, counts, targets)
    blended = (1.0 - chosen['weight']) * ref + chosen['weight'] * new
    early_rows = int(counts[:boundary].sum())
    blocks = {'early': (slice(0, early_rows), slice(0, boundary)),
              'late': (slice(early_rows, len(targets)), slice(boundary, len(counts)))}

    def metrics_block(scores, rows, feeds):
        values = ndcg_per_feed(targets[rows], scores[rows].astype(np.float32), counts[feeds])
        return {'ndcg': float(values[:, 0].mean()), 'ndcg_at_10': float(values[:, 1].mean()),
                'ndcg_at_30': float(values[:, 2].mean())}

    def summary(scores):
        return {name: metrics_block(scores, rows, feeds) for name, (rows, feeds) in blocks.items()}

    report = {'run': str(args.run), 'reference': str(args.reference),
              'chosen': chosen, 'grid': grid,
              'standalone': summary(new), 'reference_only': summary(ref), 'blend': summary(blended),
              'late_vs_reference': paired_bootstrap(ref, blended, counts, targets,
                                                    *blocks['late'])}
    args.out.mkdir(parents=True, exist_ok=False)
    prediction_frame(val, blended).write_parquet(args.out / 'val_predictions.parquet')
    (args.out / 'assessment.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('chosen', 'blend', 'late_vs_reference')},
                     indent=2), flush=True)


def swap(args):
    import polars as pl

    from ranking import KEYS

    with np.load(args.cache / 'val.npz') as handle:
        val = dict(handle)
    counts, targets = val['counts'], val['target']
    keys = prediction_frame(val, np.zeros(len(val['item']))).select(KEYS)
    paths = [Path(path) for path in args.runs]
    if len(paths) != len(args.weights):
        raise ValueError('give one weight per component')
    if args.replace not in paths:
        raise ValueError('the replaced component is not part of the blend')
    weights = np.asarray(args.weights, dtype=np.float64)
    if weights.shape != (len(paths),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('weights must be finite and nonnegative')

    def read(run):
        frame = pl.read_parquet(Path(run) / 'val_predictions.parquet')
        if not frame.select(KEYS).equals(keys):
            raise ValueError(f'misaligned candidate keys for {run}')
        score = frame['score'].to_numpy().astype(np.float64)
        if not np.isfinite(score).all():
            raise ValueError(f'nonfinite scores for {run}')
        return standardize(score, counts)

    originals = [read(path) for path in paths]
    components = list(originals)
    components[paths.index(args.replace)] = read(args.replacement)
    baseline = blend_scores(originals, weights)
    candidate = blend_scores(components, weights)

    boundary = temporal_boundary(val['timestamp'], .4)
    early_rows = int(counts[:boundary].sum(dtype=np.int64))
    late_feeds = slice(boundary, len(counts))
    late_rows = slice(early_rows, len(targets))
    middle = boundary + (len(counts) - boundary) // 2
    middle_row = int(counts[:middle].sum(dtype=np.int64))

    def gain(rows, feeds):
        return float(per_feed_ndcg(targets[rows], candidate[rows], counts[feeds]).mean()
                     - per_feed_ndcg(targets[rows], baseline[rows], counts[feeds]).mean())

    block_gains = [gain(rows, feeds) for rows, feeds in block_slices(counts)]
    late_gains = [gain(slice(late_rows.start, middle_row), slice(boundary, middle)),
                  gain(slice(middle_row, late_rows.stop), slice(middle, len(counts)))]
    late_delta = gain(late_rows, late_feeds)
    interval = paired_bootstrap(baseline, candidate, counts, targets, late_rows, late_feeds)
    per_feed = (per_feed_ndcg(targets[late_rows], candidate[late_rows], counts[late_feeds])
                - per_feed_ndcg(targets[late_rows], baseline[late_rows], counts[late_feeds]))
    late_sizes = counts[late_feeds]
    size_gains = {f'{lo}-{hi}': (float(per_feed[(late_sizes >= lo) & (late_sizes <= hi)].mean())
                                 if ((late_sizes >= lo) & (late_sizes <= hi)).any() else None)
                  for lo, hi in ((5, 30), (31, 100), (101, 300))}
    verdict = swap_verdict(block_gains, late_gains, late_delta, interval['interval'])
    report = {'cache': str(args.cache), 'components': [str(path) for path in paths],
              'weights': (weights / weights.sum()).tolist(),
              'replaced': str(args.replace), 'replacement': str(args.replacement),
              'block_gains': block_gains, 'late_half_gains': late_gains,
              'late_delta': late_delta, 'late_interval': interval['interval'],
              'late_gain_by_size': size_gains,
              'baseline': evaluate(val, baseline, boundary),
              'candidate': evaluate(val, candidate, boundary), 'verdict': verdict,
              'early_feeds': boundary, 'late_feeds': len(counts) - boundary,
              'late_start_timestamp': int(val['timestamp'][boundary])}
    args.out.mkdir(parents=True, exist_ok=False)
    prediction_frame(val, candidate).write_parquet(args.out / 'val_predictions.parquet')
    (args.out / 'swap.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('verdict', 'late_half_gains', 'late_gain_by_size',
                                                   'candidate')}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    fit_parser = subs.add_parser('run')
    fit_parser.add_argument('cache', type=Path)
    fit_parser.add_argument('init_model', type=Path)
    fit_parser.add_argument('out', type=Path)
    fit_parser.add_argument('--val-cache', type=Path, default=Path('experiments/cache-intent-024'))
    fit_parser.add_argument('--kind', choices=sorted(COLS), default='history-intent')
    fit_parser.add_argument('--rounds', type=int, default=160)
    fit_parser.add_argument('--checkpoint', type=int, default=40)
    fit_parser.add_argument('--seed', type=int, default=2026)
    fit_parser.add_argument('--threads', type=int, default=6)
    assess_parser = subs.add_parser('assess')
    assess_parser.add_argument('run', type=Path)
    assess_parser.add_argument('reference', type=Path)
    assess_parser.add_argument('out', type=Path)
    assess_parser.add_argument('--val-cache', type=Path, default=Path('experiments/cache-intent-024'))
    swap_parser = subs.add_parser('swap')
    swap_parser.add_argument('cache', type=Path)
    swap_parser.add_argument('out', type=Path)
    swap_parser.add_argument('runs', type=Path, nargs='+')
    swap_parser.add_argument('--weights', type=float, nargs='+', required=True)
    swap_parser.add_argument('--replace', type=Path, required=True)
    swap_parser.add_argument('--replacement', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'run':
        fit(args)
    elif args.action == 'assess':
        assess(args)
    else:
        swap(args)


if __name__ == '__main__':
    main()
