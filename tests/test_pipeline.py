import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

from transfer_ranking import build_decisions, status_to_label, topk_capture, lift_over_random


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / "data" / "synthetic"
        cls.encounters = pd.read_csv(root / "synthetic_all_disch.csv")
        cls.transfers = pd.read_csv(root / "synthetic_ctal_disch.csv")

    def test_synthetic_identity_and_join(self):
        self.assertTrue(self.encounters.PAT_ID.str.startswith("SYN-P").all())
        self.assertTrue(self.encounters.CSN.str.startswith("SYN-C").all())
        decisions = build_decisions(self.encounters, self.transfers)
        self.assertTrue(decisions.Hospital.str.startswith("Community Hospital ").all())
        self.assertEqual(decisions.CSN.nunique(), len(self.transfers))
        self.assertEqual(set(decisions.y_accept), {0, 1})

    def test_unknown_and_negated_statuses_not_accepted(self):
        for text in [None, "Pending", "Not accepted", "Acceptance pending", "", "not on call"]:
            self.assertTrue(np.isnan(status_to_label(text)))
        self.assertEqual(status_to_label(" Accepted "), 1)
        self.assertEqual(status_to_label("REJECTED"), 0)

    def test_duplicate_encounter_rejected(self):
        duplicate = pd.concat([self.encounters, self.encounters.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "one row per CSN"):
            build_decisions(duplicate, self.transfers)

    def test_mismatched_patient_rejected(self):
        broken = self.transfers.copy()
        broken.loc[0, "PAT_ID"] = "SYN-P-MISMATCH"
        with self.assertRaisesRegex(ValueError, "must match"):
            build_decisions(self.encounters, broken)

    def test_grouped_holdout_and_cv_are_disjoint(self):
        df = build_decisions(self.encounters, self.transfers)
        splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
        train, test = next(splitter.split(df, df.y_accept, df.CSN))
        self.assertFalse(set(df.iloc[train].CSN) & set(df.iloc[test].CSN))
        training = df.iloc[train]
        for fit, validation in GroupKFold(3).split(training, training.y_accept, training.CSN):
            self.assertFalse(set(training.iloc[fit].CSN) & set(training.iloc[validation].CSN))

    def test_capture_and_finite_sample_lift(self):
        labels, scores = [1, 0, 1, 0, 0], [.9, .8, .7, .6, .5]
        self.assertAlmostEqual(topk_capture(labels, scores, .3), .5)
        self.assertAlmostEqual(lift_over_random(labels, scores, .3), 1.25)
        self.assertEqual(topk_capture(labels, scores, 1), 1)
        self.assertTrue(np.isnan(topk_capture([0, 0], [.7, .2], .5)))
        with self.assertRaises(ValueError):
            topk_capture(labels, scores, 0)


if __name__ == "__main__":
    unittest.main()
