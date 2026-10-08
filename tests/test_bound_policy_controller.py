import copy
import unittest

from apv_rag.bound_policy_controller import execute_bound_policies
from apv_rag.fresh_pipeline import FreshRetriever, execute_policies, feature_entry
from replay_bound_policies import same_engine_output


class BoundPolicyControllerTest(unittest.TestCase):
    def setUp(self):
        self.corpus = [{'doc_id': 42, 'abstract': ['The treatment helped 10 people.']}]
        self.prepared = FreshRetriever(self.corpus).prepare(
            'The treatment helped 10 people.', lambda text, limit: text)
        self.entry = feature_entry(self.prepared['shown_claim'], self.prepared['evidence'],
                                   [[.1, .8, .1]], [.7])
        self.models = {name: {'columns': [0], 'mean': [0], 'scale': [1],
                              'coefficients': [[0]], 'intercept': [value]}
                       for name, value in [('nli', -1), ('embedding', 0), ('combined', 1)]}
        self.snapshot = {'42': self.corpus[0]}
        self.calls = []

    def generate(self, kind, question):
        self.calls.append(kind)
        return {'answer': 'Supported' if kind == 'verdict' else 'The treatment helped 10 people.',
                'prompt': question, 'prompt_tokens': 30}

    def test_bound_context_preserves_four_policies(self):
        rows = execute_bound_policies(self.prepared, self.entry, self.models,
                                      self.snapshot, self.generate)
        ordinary = execute_policies(self.prepared, self.entry, self.models, self.generate)
        self.assertEqual([{k: v for k, v in row.items() if k not in
                          ('snapshot_source_bound', 'source_trace')} for row in rows], ordinary)
        self.assertTrue(all(row['snapshot_source_bound'] for row in rows))

    def test_changed_passage_blocks_every_policy_before_generation(self):
        changed = copy.deepcopy(self.prepared)
        changed['retrieved_evidence'][0]['text'] = 'FORGED ' + changed['retrieved_evidence'][0]['text']
        changed['evidence'][0]['text'] = 'FORGED ' + changed['evidence'][0]['text']
        rows = execute_bound_policies(changed, self.entry, self.models,
                                      self.snapshot, self.generate)
        self.assertEqual(self.calls, [])
        self.assertEqual([r['policy'] for r in rows], ['no_gate', 'nli', 'embedding', 'combined'])
        self.assertTrue(all(r['reasons'] == ['snapshot_source_mismatch'] and
                            r['candidate_label'] is None and not r['generation_requests']
                            for r in rows))

    def test_empty_context_remains_abstention(self):
        empty = {'claim': 'Unmatched', 'shown_claim': 'Unmatched',
                 'retrieved_evidence': [], 'evidence': []}
        rows = execute_bound_policies(empty, None, self.models, self.snapshot, self.generate)
        self.assertEqual(self.calls, [])
        self.assertTrue(all(r['reasons'] == ['no_evidence'] for r in rows))

    def test_saved_probability_roundoff_only(self):
        reference = {'gate_probability': .3906306584126826, 'candidate_label': None}
        self.assertTrue(same_engine_output({**reference, 'gate_probability': .3906306584126827},
                                           reference, ('gate_probability', 'candidate_label')))
        self.assertFalse(same_engine_output({**reference, 'gate_probability': .41},
                                            reference, ('gate_probability', 'candidate_label')))
        self.assertFalse(same_engine_output({**reference, 'candidate_label': 'Supported'},
                                            reference, ('gate_probability', 'candidate_label')))


if __name__ == '__main__':
    unittest.main()
