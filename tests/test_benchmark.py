import unittest

from benchmark import score_predictions


class ScorePredictionsTests(unittest.TestCase):
    def test_reports_capped_and_raw_targets_separately(self):
        report = score_predictions(
            y_capped=[125, 125, 35],
            y_raw=[175, 150, 35],
            prediction=[100, 120, 35],
            baseline=[125, 125, 125],
        )
        self.assertEqual(report["model_mae_cycles"], 10.0)
        self.assertEqual(report["model_raw_target_mae_cycles"], 35.0)
        self.assertEqual(report["median_baseline_raw_target_mae_cycles"], 55.0)
        self.assertEqual(report["test_engines_raw_rul_above_125"], 2)
        self.assertEqual(report["test_engines_raw_rul_max_cycles"], 175)


if __name__ == "__main__":
    unittest.main()
