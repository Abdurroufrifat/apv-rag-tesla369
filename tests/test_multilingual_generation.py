import unittest

from apv_rag.multilingual_generation import explanation_ids, explanation_question, verdict_question


class MultilingualGenerationTests(unittest.TestCase):
    def test_subset_is_order_and_label_independent(self):
        rows = [{"id": i, "label": "SUPPORTS"} for i in range(100)]
        changed = [{"id": r["id"], "label": "REFUTES"} for r in reversed(rows)]
        self.assertEqual(explanation_ids(rows), explanation_ids(changed))
        self.assertEqual(len(explanation_ids(rows)), 60)

    def test_duplicate_claim_rows_do_not_change_subset(self):
        rows = [{"id": i} for i in range(100)]
        self.assertEqual(explanation_ids(rows), explanation_ids(rows + [rows[0]] * 10))

    def test_subset_size_requires_enough_unique_claims(self):
        with self.assertRaises(ValueError):
            explanation_ids([{"id": 1}], count=60)

    def test_verdict_prompt_contains_only_supplied_text(self):
        prompt = verdict_question("声称", "证据")
        self.assertIn("[premise] 证据", prompt)
        self.assertIn("Claim: 声称", prompt)
        self.assertIn("Not Enough Evidence", prompt)

    def test_explanation_uses_the_requested_language(self):
        prompt = explanation_question("Claim", "Evidence", "Refuted", "ja")
        self.assertIn("Explain in Japanese", prompt)
        self.assertIn("Verdict: Refuted", prompt)
        with self.assertRaises(ValueError):
            explanation_question("Claim", "Evidence", "Maybe", "en")


if __name__ == "__main__":
    unittest.main()
