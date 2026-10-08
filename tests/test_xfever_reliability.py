import unittest

from analyze_xfever_reliability import reliability


class XfeverReliabilityTest(unittest.TestCase):
    def test_brier_and_ece_use_declared_class_order(self):
        rows = [
            {'file': 'en/test.6h.jsonl', 'row': 0, 'true_label': 'Supported',
             'predicted_label': 'Supported', 'probabilities': [.9, .05, .05]},
            {'file': 'en/test.6h.jsonl', 'row': 1, 'true_label': 'Refuted',
             'predicted_label': 'Supported', 'probabilities': [.8, .1, .1]},
        ]
        result = reliability(rows, bins=2)['en/test.6h.jsonl']
        self.assertAlmostEqual(result['multiclass_brier'], .7375)
        self.assertAlmostEqual(result['top_label_ece'], .35)
        self.assertEqual(result['reliability_bins'][1]['count'], 2)
        self.assertEqual(result['reliability_bins'][1]['correct'], 1)

    def test_rejects_malformed_probabilities_and_duplicate_rows(self):
        row = {'file': 'en/test.6h.jsonl', 'row': 0, 'true_label': 'Supported',
               'predicted_label': 'Supported', 'probabilities': [.9, .05, .05]}
        with self.assertRaises(ValueError):
            reliability([row, row])
        with self.assertRaises(ValueError):
            reliability([{**row, 'probabilities': [.9, .05, .1]}])
        with self.assertRaises(ValueError):
            reliability([{**row, 'predicted_label': 'Refuted'}])


if __name__ == '__main__':
    unittest.main()
