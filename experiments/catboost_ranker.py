"""CatBoost member for the shipped tree ensemble on the same cached features.

Every tree model in the shipped ensemble is LightGBM. CatBoost was never trained in
this project (the local Python had no package), so its errors are the cheapest source
of diversity left: same 79 cached features, same train cache, same binary objective,
but symmetric trees, ordered target statistics on the categorical columns and a
different split search.

Protocol, fixed before any late metric is computed:

* the training matrix is one of the existing train caches, so no feature code changes;
* identical parameters and seed for every run, only the tree count is chosen by full
  NDCG on the early val, at checkpoints of ``--checkpoint`` iterations;
* the late val is reported once, for the selected checkpoint only;
* a blend with the shipped ensemble is a separate step (``ranking.py blend`` or
  ``warm_start.py assess``), also chosen on the early blocks.

Memory: the cache is loaded whole, so a 24-shard cache is about 3.7 GiB of float32 and
a 48-shard cache about 7.5 GiB. ``--train-file`` accepts a subsampled cache, for
example ``experiments/feed-width-048/random.npz`` with 3 575 775 rows.

Usage:
    python experiments/catboost_ranker.py train <cache> <out> [--train-file <npz>]
    python experiments/catboost_ranker.py predict <cache> <model> <output>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ranking import (COLS, NAMES, audit_submission, evaluate, ndcg_per_feed,  # noqa: E402
                     prediction_frame, temporal_boundary)

CATEGORICAL = ('category', 'intent_match_type', 'intent_match_source')


def feature_columns(kind):
    """Return the column names and count the cache must provide for this feature set."""
    n = COLS[kind]
    return NAMES[:n], n


def categorical_columns(names):
    """Indices of the columns CatBoost may use as categorical features."""
    return [names.index(name) for name in CATEGORICAL if name in names]


def select_checkpoint(search):
    """Pick the tree count by early NDCG; ties keep the smaller model."""
    if not search:
        raise ValueError('no checkpoint was evaluated')
    return max(sorted(search), key=lambda trees: search[trees])


def prepare_matrix(x, n, categoricals):
    """Slice the cache to the feature count and optionally build mixed-dtype data."""
    if x.shape[1] < n:
        raise ValueError(f'cache has {x.shape[1]} features, the model needs {n}')
    if not categoricals:
        return np.ascontiguousarray(x[:, :n])
    import pandas as pd

    names = NAMES[:n]
    cats = set(categorical_columns(names))
    columns = {}
    for i, name in enumerate(names):
        column = x[:, i]
        # CatBoost accepts categorical features only as integer or string columns.
        columns[name] = (np.nan_to_num(column, nan=-1).astype(np.int32) if i in cats
                         else column.astype(np.float32, copy=False))
    return pd.DataFrame(columns, columns=names)


def read_split(path, n, categoricals):
    with np.load(path) as handle:
        data = {name: handle[name] for name in ('target', 'counts', 'item', 'feed', 'timestamp')}
        data['x'] = prepare_matrix(handle['x'], n, categoricals)
    return data


def train(args):
    import catboost
    from catboost import CatBoostClassifier, Pool

    if args.out.exists():
        raise FileExistsError(args.out)
    names, n = feature_columns(args.kind)
    train_file = args.train_file or args.cache / 'train.npz'
    with np.load(train_file) as handle:
        train_x = prepare_matrix(handle['x'], n, args.categoricals)
        train_y = handle['target']
        train_counts = handle['counts']
        if 'shards' in handle:
            shards, prepare_seed = handle['shards'].tolist(), int(handle['prepare_seed'])
        else:
            shards, prepare_seed = [], None
    val = read_split(args.cache / 'val.npz', n, args.categoricals)
    if int(train_counts.sum()) != len(train_y):
        raise ValueError('train labels and feed sizes do not align')

    cats = categorical_columns(names) if args.categoricals else []
    boundary = temporal_boundary(val['timestamp'], .4)
    early_rows = int(val['counts'][:boundary].sum(dtype=np.int64))
    params = {'iterations': args.rounds, 'depth': args.depth, 'learning_rate': args.lr,
              'l2_leaf_reg': args.l2, 'border_count': args.border_count,
              'random_seed': args.seed, 'thread_count': args.threads,
              'loss_function': 'Logloss', 'eval_metric': 'Logloss',
              'bootstrap_type': 'Bernoulli', 'subsample': .9,
              'allow_writing_files': False, 'verbose': args.verbose}
    print(f'{args.kind}: {len(train_counts)} train feeds, {len(train_y)} rows, '
          f'{n} features, categoricals={bool(cats)}', flush=True)
    model = CatBoostClassifier(**params)
    model.fit(Pool(train_x, label=train_y, cat_features=cats or None))
    del train_x

    early_pool = Pool(val['x'][:early_rows] if not args.categoricals else val['x'].iloc[:early_rows],
                      cat_features=cats or None)
    search = {}
    for trees in range(args.checkpoint, args.rounds + 1, args.checkpoint):
        scores = model.predict_proba(early_pool, ntree_end=trees)[:, 1]
        early = float(ndcg_per_feed(val['target'][:early_rows], scores.astype(np.float32),
                                    val['counts'][:boundary])[:, 0].mean())
        search[trees] = early
        print(f'{args.out.name}: trees={trees}, early NDCG={early:.6f}', flush=True)
    selected = select_checkpoint(search)
    model.shrink(ntree_end=selected)
    val_pool = Pool(val['x'], cat_features=cats or None)
    direct = model.predict_proba(val_pool)[:, 1].astype(np.float64)
    quality = evaluate(val, direct, boundary)

    args.out.mkdir(parents=True, exist_ok=False)
    model.save_model(str(args.out / 'model.cbm'))
    prediction_frame(val, direct).write_parquet(args.out / 'val_predictions.parquet')
    report = {'kind': args.kind, 'catboost': catboost.__version__, 'params': params,
              'categoricals': bool(cats), 'categorical_features': [names[i] for i in cats],
              'train_file': str(train_file), 'train_feeds': len(train_counts),
              'train_rows': len(train_y), 'train_shards': shards, 'prepare_seed': prepare_seed,
              'checkpoint': args.checkpoint, 'selected_trees': selected, 'early_search': search,
              'early_feeds': boundary, 'late_feeds': len(val['counts']) - boundary,
              'late_start_timestamp': int(val['timestamp'][boundary]), 'metrics': quality}
    (args.out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(quality), flush=True)


def predict(args):
    from catboost import CatBoostClassifier, Pool

    if args.output.exists():
        raise FileExistsError(args.output)
    model = CatBoostClassifier()
    model.load_model(str(args.model))
    n = len(model.feature_names_)
    cats = categorical_columns(NAMES[:n]) if args.categoricals else []
    data = read_split(args.cache / 'test.npz', n, args.categoricals)
    score = model.predict_proba(Pool(data['x'], cat_features=cats or None))[:, 1].astype(np.float64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame(data, score).write_parquet(args.output)
    print(json.dumps(audit_submission(args.output)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    fit = subs.add_parser('train')
    fit.add_argument('cache', type=Path)
    fit.add_argument('out', type=Path)
    fit.add_argument('--kind', choices=sorted(COLS), default='history-intent')
    fit.add_argument('--train-file', type=Path)
    fit.add_argument('--rounds', type=int, default=800)
    fit.add_argument('--checkpoint', type=int, default=200)
    fit.add_argument('--depth', type=int, default=8)
    fit.add_argument('--lr', type=float, default=.1)
    fit.add_argument('--l2', type=float, default=3.0)
    fit.add_argument('--border-count', type=int, default=127)
    fit.add_argument('--seed', type=int, default=2026)
    fit.add_argument('--threads', type=int, default=6)
    fit.add_argument('--categoricals', action='store_true')
    fit.add_argument('--verbose', type=int, default=200)
    inf = subs.add_parser('predict')
    inf.add_argument('cache', type=Path)
    inf.add_argument('model', type=Path)
    inf.add_argument('output', type=Path)
    inf.add_argument('--categoricals', action='store_true')
    args = parser.parse_args()
    if args.action == 'train':
        train(args)
    else:
        predict(args)


if __name__ == '__main__':
    main()
