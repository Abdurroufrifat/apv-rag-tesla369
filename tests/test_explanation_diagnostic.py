import json
import tempfile
import unittest
from pathlib import Path

from run_explanation_diagnostic import select_tasks, pair_specs, quote_check, summarize, write_preflight_receipt
from audit_project_reproducibility import audit_receipt


class ExplanationDiagnosticTest(unittest.TestCase):
    def test_preflight_receipt_is_auditable(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'preflight.json').write_text('{"tasks": 2}\n', encoding='utf-8')
            write_preflight_receipt(folder)
            report = audit_receipt(folder/'audit_manifest.json', folder)
            self.assertNotIn('schema_error', report)
            self.assertEqual(report['files'][0]['status'], 'verified')
            self.assertEqual(json.loads((folder/'audit_manifest.json').read_text())['preflight_sha256'],
                             report['files'][0]['actual_sha256'])

    def test_selection_excludes_labels_and_duplicate_baseline(self):
        row={'cohort':'scifact','claim_id':'2','policy':'no_gate','claim':'Tesla',
             'shown_claim':'Tesla','true_label':'Supported','generated_explanation':'It says three.',
             'evidence':[{'id':42,'text':'Tesla counts three.'}]}
        fresh=[row]
        stress=[{**row,'condition':'baseline'},{**row,'condition':'ocr_noise'}]
        tasks=select_tasks(fresh,stress)
        self.assertEqual(len(tasks),2)
        self.assertNotIn('true_label',str(tasks))
        self.assertEqual(len(pair_specs(tasks)),2)

    def test_quote_check_is_literal(self):
        self.assertEqual(quote_check('The article says “Tesla counts three”.',
                                     'Tesla counts three','Tesla counts three.'),
                         {'quoted':1,'found_in_evidence':1,'found_in_claim_only':0,'unmatched':0})
        self.assertEqual(quote_check('“Unknown words”','Tesla','Tesla counts three.')['unmatched'],1)

    def test_invalid_scores_rejected(self):
        tasks=[{'key':'scifact:2:original','cohort':'scifact','condition':'original',
                'claim':'Tesla','explanation':'It says three.',
                'evidence':[{'id':42,'text':'Tesla counts three.'}]}]
        pair=pair_specs(tasks)[0]
        with self.assertRaises(ValueError):summarize(tasks,{pair['key']:{
            'premise_sha256':pair['premise_sha256'],
            'hypothesis_sha256':pair['hypothesis_sha256'],
            'scores_cen':[.2,.2,.2],'skip':None}})

    def test_valid_scores_and_budget_skip_are_distinct(self):
        tasks=[{'key':'scifact:2:original','cohort':'scifact','condition':'original',
                'claim':'Tesla','explanation':'“Tesla” is named.',
                'evidence':[{'id':42,'text':'Tesla was named.'},{'id':43,'text':'A second passage.'}]}]
        specs=pair_specs(tasks)
        cache={p['key']:{'premise_sha256':p['premise_sha256'],
                          'hypothesis_sha256':p['hypothesis_sha256'],
                          'scores_cen':[.1,.8,.1] if i==0 else None,
                          'skip':None if i==0 else 'token_budget'} for i,p in enumerate(specs)}
        result=summarize(tasks,cache)
        self.assertEqual(result['pairs'],2)
        self.assertEqual(result['groups']['scifact:original']['pairs_skipped'],1)
        self.assertAlmostEqual(result['details'][tasks[0]['key']]['max_entailment'],.8)


if __name__=='__main__':unittest.main()
