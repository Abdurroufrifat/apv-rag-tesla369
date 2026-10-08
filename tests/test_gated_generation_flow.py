import unittest

from apv_rag.gated_generation_flow import execute_generation


class GenerationFlowTests(unittest.TestCase):
    evidence = [{"id": "source", "text": "The treatment helped."}]

    def forbidden_generation(self, kind, question):
        self.fail("A rejected claim reached generation")

    def generator(self, kind, question):
        answer = "Supported" if kind == "verdict" else "The treatment helped."
        return {"answer": answer, "prompt": question, "prompt_tokens": 20}

    def test_low_gate_prevents_all_generation(self):
        r = execute_generation("The treatment helped.", self.evidence, self.forbidden_generation, .49)
        self.assertIsNone(r["candidate_label"])
        self.assertEqual(r["generation_requests"], [])
        self.assertEqual(r["reasons"], ["gate_below_threshold"])

    def test_threshold_boundary_produces_answer(self):
        r = execute_generation("The treatment helped.", self.evidence, self.generator, .5)
        self.assertEqual(r["candidate_label"], "Supported")
        self.assertEqual(r["generation_requests"], ["verdict", "explanation"])
        self.assertEqual(r["stages"], ["collapse_context", "gate", "verdict", "explanation", "numeric_check"])

    def test_empty_context_prevents_generation(self):
        r = execute_generation("Claim", [], self.forbidden_generation)
        self.assertEqual(r["reasons"], ["no_evidence"])

    def test_numeric_rejection_occurs_after_generation(self):
        def generate(kind, question):
            return {"answer": "Supported" if kind == "verdict" else "It helped 99 people.",
                    "prompt": question, "prompt_tokens": 20}
        r = execute_generation("It helped.", self.evidence, generate, .9)
        self.assertIsNone(r["candidate_label"])
        self.assertEqual(r["reasons"], ["numeric_value_absent"])
        self.assertEqual(r["generation_requests"], ["verdict", "explanation"])

    def test_invalid_verdict_prevents_explanation(self):
        def generate(kind, question):
            self.assertEqual(kind, "verdict")
            return {"answer": "Maybe", "prompt": question, "prompt_tokens": 20}
        r = execute_generation("Claim", self.evidence, generate)
        self.assertEqual(r["reasons"], ["invalid_verdict"])
        self.assertEqual(r["generation_requests"], ["verdict"])

    def test_duplicate_sources_removed_before_prompt(self):
        duplicate = dict(self.evidence[0], id="copy", family_id="source")
        r = execute_generation("Claim", self.evidence + [duplicate], self.generator)
        self.assertEqual(len(r["evidence"]), 1)
        self.assertNotIn("[copy]", r["verdict_prompt"])

    def test_nonfinite_probability_rejected(self):
        with self.assertRaises(ValueError):
            execute_generation("Claim", self.evidence, self.forbidden_generation, float("nan"))


if __name__ == "__main__":
    unittest.main()
