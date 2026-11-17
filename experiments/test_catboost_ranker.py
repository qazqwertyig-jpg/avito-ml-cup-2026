"""Unit tests for the CatBoost member (no CatBoost needed)."""
import unittest

import numpy as np

import catboost_ranker as ranker
from ranking import COLS, NAMES


class FeatureSelectionTest(unittest.TestCase):
    def test_history_intent_width_matches_the_shared_name_list(self):
        names, n = ranker.feature_columns('history-intent')
        self.assertEqual(n, COLS['history-intent'])
        self.assertEqual(names, NAMES[:n])
        self.assertIn('intent_rank_recent_max', names)

    def test_categorical_columns_follow_the_declared_names(self):
        names, n = ranker.feature_columns('history-intent')
        expected = [names.index(name) for name in ('category', 'intent_match_type', 'intent_match_source')]
        self.assertEqual(ranker.categorical_columns(names), expected)

    def test_only_the_category_column_is_categorical_in_the_base_features(self):
        names, _ = ranker.feature_columns('basic')
        self.assertEqual(ranker.categorical_columns(names), [names.index('category')])

    def test_narrow_cache_is_rejected(self):
        with self.assertRaises(ValueError):
            ranker.prepare_matrix(np.zeros((4, 3), dtype=np.float32), COLS['history-intent'], False)

    def test_numeric_mode_slices_to_the_feature_count(self):
        x = np.arange(4 * 90, dtype=np.float32).reshape(4, 90)
        result = ranker.prepare_matrix(x, 79, False)
        self.assertEqual(result.shape, (4, 79))
        np.testing.assert_array_equal(result, x[:, :79])


class CheckpointSelectionTest(unittest.TestCase):
    def test_largest_early_ndcg_wins(self):
        self.assertEqual(ranker.select_checkpoint({200: .31, 400: .33, 600: .32}), 400)

    def test_later_checkpoint_wins_a_tie(self):
        self.assertEqual(ranker.select_checkpoint({200: .33, 400: .33}), 200)

    def test_empty_search_is_rejected(self):
        with self.assertRaises(ValueError):
            ranker.select_checkpoint({})


if __name__ == '__main__':
    unittest.main()
