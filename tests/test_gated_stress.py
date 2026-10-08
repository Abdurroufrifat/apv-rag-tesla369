import copy
import unittest

from apv_rag.fresh_pipeline import feature_entry, validate_feature_entry
from apv_rag.generation_robustness import context_variants
from apv_rag.gated_stress import transform_entry, execute_stress_policy


class GatedStressTests(unittest.TestCase):
    def setUp(self):
        self.evidence = [{'id': 1, 'text': 'The treatment helped.'}, {'id': 2, 'text': 'Birds fly.'}]
        self.claim = 'The treatment helped.'
        self.entry = feature_entry(self.claim, self.evidence, [[.1, .8, .1], [.2, .1, .7]], [.8, .2])
        self.model = {'columns': [1], 'mean': [0], 'scale': [1], 'coefficients': [[1]], 'intercept': [-.5]}

    def generate(self, kind, question):
        return {'answer': 'Supported' if kind == 'verdict' else 'The treatment helped.',
                'prompt': question, 'prompt_tokens': 30}

    def test_copied_features_change_weights_and_collapsed_features_restore_original(self):
        variants = context_variants(self.evidence)
        raw = transform_entry(self.claim, self.evidence, self.entry, variants['copies_raw'])
        self.assertEqual(len(raw['scores_cen']), 5)
        self.assertAlmostEqual(raw['features'][1], .66)
        collapsed = transform_entry(self.claim, self.evidence, self.entry, variants['copies_collapsed'])
        self.assertEqual(collapsed, self.entry)
        validate_feature_entry(raw, self.claim, variants['copies_raw'])

    def test_reversed_scores_follow_their_source_and_preserve_aggregate(self):
        variant = context_variants(self.evidence)['reverse_order']
        entry = transform_entry(self.claim, self.evidence, self.entry, variant)
        self.assertEqual(entry['scores_cen'], list(reversed(self.entry['scores_cen'])))
        self.assertEqual(entry['features'], self.entry['features'])

    def test_changed_text_unknown_parent_and_corrupt_base_are_rejected(self):
        bad = copy.deepcopy(self.evidence);bad[0]['text'] += ' changed'
        with self.assertRaises(ValueError):transform_entry(self.claim, self.evidence, self.entry, bad)
        bad = copy.deepcopy(self.evidence);bad[0]['id'] = 99
        with self.assertRaises(ValueError):transform_entry(self.claim, self.evidence, self.entry, bad)
        bad = copy.deepcopy(self.entry);bad['features'][0] += 1
        with self.assertRaises(ValueError):transform_entry(self.claim, self.evidence, bad, self.evidence)

    def test_gate_rejects_before_requesting_any_saved_response(self):
        def forbidden(*args):self.fail('Rejected policy requested a response')
        r = execute_stress_policy(self.claim, self.evidence, self.entry, self.model, forbidden, True)
        self.assertEqual(r['reasons'], ['gate_below_threshold'])
        self.assertEqual(r['generation_requests'], [])
        self.assertIsNone(r['gate_verdict_label'])

    def test_copy_weighting_can_cross_frozen_threshold(self):
        evidence = context_variants(self.evidence)['copies_raw']
        entry = transform_entry(self.claim, self.evidence, self.entry, evidence)
        r = execute_stress_policy(self.claim, evidence, entry, self.model, self.generate, True)
        self.assertGreater(r['gate_probability'], .5)
        self.assertEqual(r['gate_verdict_label'], 'Supported')
        self.assertEqual(r['explanation_guarded_label'], 'Supported')

    def test_unselected_claim_has_verdict_only_without_explanation_metrics(self):
        r = execute_stress_policy(self.claim, self.evidence, self.entry, None, self.generate, False)
        self.assertEqual(r['generation_requests'], ['verdict'])
        self.assertEqual(r['gate_verdict_label'], 'Supported')
        self.assertIsNone(r['explanation_guarded_label'])
        self.assertIsNone(r['numeric_provenance'])

    def test_numeric_failure_does_not_relabel_gate_only_verdict(self):
        def generate(kind, question):
            return {'answer': 'Supported' if kind == 'verdict' else 'It helped 99 people.',
                    'prompt': question, 'prompt_tokens': 30}
        r = execute_stress_policy(self.claim, self.evidence, self.entry, None, generate, True)
        self.assertEqual(r['gate_verdict_label'], 'Supported')
        self.assertIsNone(r['explanation_guarded_label'])
        self.assertEqual(r['reasons'], ['numeric_value_absent'])


if __name__ == '__main__':unittest.main()

class StressPayloadTests(unittest.TestCase):
    def test_export_rejects_changed_gate_response_and_duplicate_records(self):
        from verify_gated_stress_replay import verify_payloads
        from unittest.mock import patch
        row = {'cohort':'scifact','claim_id':'1','condition':'baseline','policy':'nli','gate_probability':.4}
        rows=[row];features={'bound':{'features':[.1]}};responses={'prompt':{'answer':'Supported'}}
        actual={'predictions':copy.deepcopy(rows),'features':copy.deepcopy(features),'responses':copy.deepcopy(responses),'summary':{}}
        with patch('verify_gated_stress_replay.summarize_replay',return_value={}):
            verify_payloads(actual,(rows,features,responses))
            actual['predictions'][0]['gate_probability']=.6
            with self.assertRaisesRegex(ValueError,'mismatch'):verify_payloads(actual,(rows,features,responses))
            actual['predictions']=rows*2
            with self.assertRaisesRegex(ValueError,'Duplicate'):verify_payloads(actual,(rows,features,responses))
            actual['predictions']=copy.deepcopy(rows)
            actual['responses']['prompt']['answer']='Refuted'
            with self.assertRaisesRegex(ValueError,'binding'):verify_payloads(actual,(rows,features,responses))
