"""Compare a CatBoost member with the project's LightGBM on one real val cache.

The shipped tree members use the 79 ``history-intent`` columns, which need item priors
built from train impressions. When only the public val split is available, this script
runs the same comparison on the columns the val cache does have: it trains both families
on the early val feeds, chooses the tree count and the blend weight on the middle block,
and reports the last block once, untouched by either choice.

Scope note: the "train" block here is early val, not train, so the absolute NDCG is not
comparable with the journal. The relative LightGBM / CatBoost / blend numbers on the same
block and features are what this measures.

Usage:
    python experiments/val_feature_ab.py <cache> <out> [--threads 6]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ranking import NAMES, ndcg_per_feed, standardized_scores, temporal_boundary  # noqa: E402
from warm_start import paired_bootstrap  # noqa: E402

CATEGORICAL = ('category', 'intent_match_type', 'intent_match_source')
BLEND_WEIGHTS = (0.05, 0.10, 0.20, 0.30, 0.50)


def row_slices(counts, feed_slices):
    offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    return [slice(int(offsets[a]), int(offsets[b])) for a, b in feed_slices]


def mean_ndcg(labels, score, counts):
    return float(ndcg_per_feed(labels, score.astype(np.float32), counts)[:, 0].mean())


def lightgbm_member(x, y, counts, names, rows, tune, threads, seed, checkpoints):
    import lightgbm as lgb

    params = {'objective': 'binary', 'metric': 'None', 'learning_rate': .08, 'num_leaves': 63,
              'min_data_in_leaf': 160, 'lambda_l2': 5, 'max_bin': 127, 'feature_fraction': .9,
              'num_threads': threads, 'force_col_wise': True, 'verbosity': -1, 'seed': seed}
    categories = [name for name in CATEGORICAL if name in names]
    data = lgb.Dataset(x[rows], label=y[rows], params=params, feature_name=names,
                       categorical_feature=categories)
    model = lgb.train(params, data, num_boost_round=max(checkpoints))
    search = {trees: mean_ndcg(y[tune], model.predict(x[tune], num_iteration=trees,
                                                     raw_score=True), counts) for trees in checkpoints}
    return model, search, max(sorted(search), key=lambda trees: search[trees])


def catboost_member(x, y, counts, names, rows, tune, threads, seed, iterations, checkpoints):
    from catboost import CatBoostClassifier, Pool

    params = {'iterations': iterations, 'depth': 8, 'learning_rate': .1, 'l2_leaf_reg': 3,
              'border_count': 127, 'random_seed': seed, 'thread_count': threads,
              'loss_function': 'Logloss', 'bootstrap_type': 'Bernoulli', 'subsample': .9,
              'allow_writing_files': False, 'verbose': 100}
    model = CatBoostClassifier(**params)
    model.fit(Pool(x[rows], label=y[rows]))
    tune_pool = Pool(x[tune])
    search = {trees: mean_ndcg(y[tune], model.predict_proba(tune_pool, ntree_end=trees)[:, 1], counts)
              for trees in checkpoints}
    return model, search, max(sorted(search), key=lambda trees: search[trees])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--rounds', type=int, default=160)
    parser.add_argument('--iterations', type=int, default=400)
    parser.add_argument('--checkpoint', type=int, default=100)
    args = parser.parse_args()

    with np.load(args.cache / 'val.npz') as handle:
        x = handle['x'].astype(np.float32)
        y, counts, timestamps = handle['target'], handle['counts'], handle['timestamp']
    names = NAMES[:x.shape[1]]
    boundary = temporal_boundary(timestamps, .4)
    middle = boundary + (len(counts) - boundary) // 2
    rows = row_slices(counts, [(0, boundary), (boundary, middle), (middle, len(counts))])
    train_rows, tune_rows, test_rows = rows
    tune_feeds, test_feeds = counts[boundary:middle], counts[middle:]
    train_feeds = counts[:boundary]
    print(f'blocks: train {boundary} feeds, tune {middle - boundary}, test {len(counts) - middle}', flush=True)
    print(f'rows: train {train_rows.stop}, tune {tune_rows.stop - tune_rows.start}, '
          f'test {test_rows.stop - test_rows.start}', flush=True)

    lgb_checkpoints = sorted(set(range(40, args.rounds + 1, 40)))
    cb_checkpoints = sorted(set(range(args.checkpoint, args.iterations + 1, args.checkpoint)))
    lgb_model, lgb_search, lgb_trees = lightgbm_member(
        x, y, tune_feeds, names, train_rows, tune_rows, args.threads, args.seed, lgb_checkpoints)
    print('lightgbm tuned', flush=True)
    cb_model, cb_search, cb_trees = catboost_member(
        x, y, tune_feeds, names, train_rows, tune_rows, args.threads, args.seed,
        args.iterations, cb_checkpoints)
    print('catboost tuned', flush=True)

    lgb_score = {'tune': lgb_model.predict(x[tune_rows], num_iteration=lgb_trees, raw_score=True),
                 'test': lgb_model.predict(x[test_rows], num_iteration=lgb_trees, raw_score=True)}
    cb_score = {'tune': cb_model.predict_proba(x[tune_rows], ntree_end=cb_trees)[:, 1],
                'test': cb_model.predict_proba(x[test_rows], ntree_end=cb_trees)[:, 1]}

    def normalise(block):
        feeds = tune_feeds if block == 'tune' else test_feeds
        return (standardized_scores(lgb_score[block].astype(np.float64), feeds),
                standardized_scores(cb_score[block].astype(np.float64), feeds))

    grid = []
    for weight in BLEND_WEIGHTS:
        tune_lgb, tune_cb = normalise('tune')
        blended = (1 - weight) * tune_lgb + weight * tune_cb
        value = mean_ndcg(y[tune_rows], blended, tune_feeds)
        grid.append({'weight': weight, 'tune_ndcg': value})
        print(json.dumps(grid[-1]), flush=True)
    chosen = max(grid, key=lambda row: row['tune_ndcg'])
    test_lgb, test_cb = normalise('test')
    test_blend = (1 - chosen['weight']) * test_lgb + chosen['weight'] * test_cb
    interval = paired_bootstrap(test_lgb, test_blend, test_feeds, y[test_rows], resamples=2000, seed=args.seed)
    report = {'scope': 'single public val cache; train block is early val, so absolute NDCG is not '
                        'comparable with runs trained on train',
              'cache': str(args.cache), 'features': names, 'feature_count': len(names),
              'blocks': {'train_feeds': int(boundary), 'tune_feeds': int(middle - boundary),
                         'test_feeds': int(len(counts) - middle),
                         'train_rows': int(train_rows.stop),
                         'tune_rows': int(tune_rows.stop - tune_rows.start),
                         'test_rows': int(test_rows.stop - test_rows.start)},
              'selected_trees': {'lightgbm': int(lgb_trees), 'catboost': int(cb_trees)},
              'tune_search': {'lightgbm': lgb_search, 'catboost': cb_search},
              'tune': {'lightgbm': mean_ndcg(y[tune_rows], lgb_score['tune'], tune_feeds),
                       'catboost': mean_ndcg(y[tune_rows], cb_score['tune'], tune_feeds),
                       'blend': chosen['tune_ndcg']},
              'test': {'lightgbm': mean_ndcg(y[test_rows], lgb_score['test'], test_feeds),
                       'catboost': mean_ndcg(y[test_rows], cb_score['test'], test_feeds),
                       'blend': mean_ndcg(y[test_rows], test_blend, test_feeds)},
              'blend_grid': grid, 'chosen_weight': chosen['weight'],
              'test_blend_minus_lightgbm': interval}
    report['test']['catboost_minus_lightgbm'] = (report['test']['catboost']
                                                 - report['test']['lightgbm'])
    report['test']['blend_minus_lightgbm'] = report['test']['blend'] - report['test']['lightgbm']
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'ab.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('test', 'chosen_weight', 'selected_trees',
                                                   'test_blend_minus_lightgbm')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
