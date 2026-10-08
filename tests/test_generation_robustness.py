import unittest

from apv_rag.generation_robustness import context_variants, explanation_ids


class GenerationRobustnessTests(unittest.TestCase):
    def test_copy_condition_preserves_original_and_adds_three_copies(self):
        evidence = [{"id": 1, "text": "First source"}, {"id": 2, "text": "Second source"}]
        variants = context_variants(evidence)
        self.assertEqual(len(evidence), 2)
        self.assertEqual(len(variants["copies_raw"]), 5)
        self.assertEqual(variants["copies_raw"][:2], evidence)
        self.assertTrue(all(r["text"] == evidence[0]["text"] for r in variants["copies_raw"][2:]))

    def test_collapse_restores_baseline_context(self):
        evidence = [{"id": 1, "text": "First"}, {"id": 2, "text": "Second"}]
        self.assertEqual(context_variants(evidence)["copies_collapsed"], evidence)

    def test_reversal_preserves_source_content(self):
        evidence = [{"id": 1, "text": "First"}, {"id": 2, "text": "Second"}]
        self.assertEqual(context_variants(evidence)["reverse_order"], list(reversed(evidence)))

    def test_explanation_selection_does_not_use_labels_or_row_order(self):
        rows = [{"claim_id": i, "true_label": "Supported"} for i in range(100)]
        altered = [{"claim_id": r["claim_id"], "true_label": "Refuted"} for r in reversed(rows)]
        self.assertEqual(explanation_ids(rows, "cohort"), explanation_ids(altered, "cohort"))
        self.assertEqual(len(explanation_ids(rows, "cohort")), 60)

    def test_empty_context_rejected_before_copy_construction(self):
        with self.assertRaises(ValueError):
            context_variants([])


if __name__ == "__main__":
    unittest.main()
