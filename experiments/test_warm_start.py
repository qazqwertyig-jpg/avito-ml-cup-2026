"""Unit tests for warm-start blend selection (no LightGBM needed)."""
import unittest

import numpy as np

import warm_start


class SelectWeightTest(unittest.TestCase):
    def test_smallest_weight_is_chosen_when_candidate_is_worse(self):
        counts = np.array([3] * 6000, dtype=np.int32)
        targets = np.tile([1, 0, 0], 6000)
        base = np.tile([3.0, 2.0, 1.0], 6000)      # positive already first
        candidate = np.tile([0.0, 1.0, 2.0], 6000)  # positive pushed down
        chosen, grid = warm_start.select_weight(base, candidate, counts, targets)
        self.assertEqual(len(grid), len(warm_start.WEIGHTS))
        self.assertLessEqual(chosen['minimum'], 0.0)
        self.assertEqual(chosen['weight'], warm_start.WEIGHTS[0])

    def test_weight_grows_when_candidate_improves_both_blocks(self):
        counts = np.array([3] * 6000, dtype=np.int32)
        targets = np.tile([1, 0, 0], 6000)
        base = np.zeros(len(targets))                      # every feed is a tie
        candidate = np.tile([5.0, 0.0, 0.0], 6000)         # positive moved to the top
        chosen, _ = warm_start.select_weight(base, candidate, counts, targets)
        self.assertGreater(chosen['minimum'], 0.0)
        self.assertGreater(chosen['weight'], 0.0)

    def test_bootstrap_interval_brackets_the_mean(self):
        counts = np.array([2] * 500, dtype=np.int32)
        targets = np.tile([1, 0], 500)
        base = np.tile([0.0, 1.0], 500)
        candidate = np.tile([1.0, 0.0], 500)
        result = warm_start.paired_bootstrap(base, candidate, counts, targets)
        self.assertGreater(result['delta'], 0)
        self.assertLessEqual(result['interval'][0], result['delta'])
        self.assertGreaterEqual(result['interval'][1], result['delta'])

    def test_standardize_is_centered_per_feed_and_survives_constant_feed(self):
        counts = np.array([2, 3], dtype=np.int32)
        score = np.array([1.0, 3.0, 5.0, 5.0, 5.0])
        result = warm_start.standardize(score, counts)
        self.assertAlmostEqual(result[:2].mean(), 0.0)
        self.assertTrue(np.all(result[2:] == 0.0))


class BlendScoresTest(unittest.TestCase):
    def test_weights_are_normalized_to_a_single_average(self):
        components = [np.array([2.0, 0.0]), np.array([0.0, 4.0])]
        result = warm_start.blend_scores(components, [1.0, 3.0])
        np.testing.assert_allclose(result, [0.5, 3.0])

    def test_negative_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            warm_start.blend_scores([np.zeros(2), np.zeros(2)], [1.0, -1.0])

    def test_misaligned_components_are_rejected(self):
        with self.assertRaises(ValueError):
            warm_start.blend_scores([np.zeros(2), np.zeros(3)], [1.0, 1.0])


class SwapVerdictTest(unittest.TestCase):
    def test_negative_early_block_rejects_any_late_gain(self):
        verdict = warm_start.swap_verdict([-0.0001, 0.0002], [0.0004, 0.0005], 0.0002, [0.0001, 0.0003])
        self.assertFalse(verdict['accepted'])
        self.assertAlmostEqual(verdict['minimum_block_gain'], -0.0001)

    def test_target_gain_with_positive_blocks_is_accepted(self):
        verdict = warm_start.swap_verdict([0.0004, 0.0003], [0.0009, 0.0011], 0.0012, [0.0006, 0.0018])
        self.assertTrue(verdict['accepted'])

    def test_smaller_gain_needs_a_positive_interval_bound(self):
        accepted = warm_start.swap_verdict([0.0002, 0.0001], [0.0003, 0.0002], 0.0003, [0.0001, 0.0005])
        self.assertTrue(accepted['accepted'])
        rejected = warm_start.swap_verdict([0.0002, 0.0001], [0.0003, 0.0002], 0.0003, [-0.0001, 0.0005])
        self.assertFalse(rejected['accepted'])


if __name__ == '__main__':
    unittest.main()
