"""Small fixtures for frozen-score selective diagnostics."""
import unittest

from audit_selective_diagnostic import diagnose_rows


class SelectiveDiagnosticTest(unittest.TestCase):
    def test_ranking_uses_ungated_outcome_and_fixed_score(self):
        rows=[]
        for claim_id,p,answer,gold in [("a",.9,"Supported","Supported"),
                                       ("b",.1,"Refuted","Supported")]:
            rows.append(dict(cohort="scifact",claim_id=claim_id,policy="no_gate",
                             true_label=gold,candidate_label=answer,gate_probability=None))
            for policy in ('nli','embedding','combined'):
                rows.append(dict(cohort="scifact",claim_id=claim_id,policy=policy,
                                 true_label=gold,candidate_label=answer if p>=.5 else None,
                                 gate_probability=p))
        result=diagnose_rows(rows)
        self.assertEqual(result['scifact']['nli']['ranking']['top_20_percent']['count'],1)
        self.assertEqual(result['scifact']['nli']['ranking']['top_20_percent']['accuracy'],1.0)
        self.assertEqual(result['scifact']['nli']['fixed_threshold']['accepted'],1)
        self.assertEqual(result['scifact']['nli']['fixed_threshold']['accepted_correct'],1)

    def test_duplicate_and_truth_mismatch_rejected(self):
        a=dict(cohort="scifact",claim_id="a",policy="no_gate",true_label="Supported",
               candidate_label="Supported",gate_probability=None)
        b=dict(a,policy="nli",true_label="Refuted",gate_probability=.6)
        with self.assertRaises(ValueError):diagnose_rows([a,b])
        with self.assertRaises(ValueError):diagnose_rows([a,a])


if __name__=='__main__':unittest.main()
