import copy
import unittest

from apv_rag.fresh_pipeline import FreshRetriever, feature_entry, validate_feature_entry, execute_policies


class FreshPipelineTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            {'doc_id': 3, 'abstract': ['Moon temperatures vary.', 'The treatment helped 10 people.']},
            {'doc_id': 1, 'abstract': ['The treatment helped 10 people.']},
            {'doc_id': 2, 'abstract': ['Birds fly.']},
        ]
        self.prepared = FreshRetriever(self.corpus).prepare('The treatment helped 10 people.', lambda s, n: s)
        self.entry = feature_entry(self.prepared['shown_claim'], self.prepared['evidence'], [[.1, .8, .1]], [.7])
        self.models = {n: {'columns': [0], 'mean': [0], 'scale': [1], 'coefficients': [[0]],
                           'intercept': [z]} for n, z in [('nli', -1), ('embedding', 0), ('combined', 1)]}

    def generator(self, kind, question):
        return {'answer': 'Supported' if kind == 'verdict' else 'The treatment helped 10 people.',
                'prompt': question, 'prompt_tokens': 30}

    def test_fresh_retrieval_selects_sentences_and_collapses_identical_contexts(self):
        self.assertEqual([e['id'] for e in self.prepared['retrieved_evidence']], [1, 3])
        self.assertEqual(len(self.prepared['evidence']), 1)
        self.assertEqual(self.prepared['retrieved_evidence'][1]['selected_sentence_indices'], [1])

    def test_clipping_is_applied_to_claim_and_selected_text(self):
        calls = []
        def clip(s, n):
            calls.append((s, n))
            return s[:n]
        r = FreshRetriever(self.corpus).prepare('treatment ' * 20, clip)
        self.assertEqual(len(r['shown_claim']), 64)
        self.assertEqual([n for _, n in calls], [96, 96, 64])

    def test_gate_precedes_generation_with_fresh_features(self):
        calls = []
        def generate(kind, question):
            calls.append(kind)
            return self.generator(kind, question)
        rows = execute_policies(self.prepared, self.entry, self.models, generate)
        self.assertEqual(rows[1]['reasons'], ['gate_below_threshold'])
        self.assertEqual(rows[1]['generation_requests'], [])
        self.assertEqual(calls, ['verdict', 'explanation'] * 3)
        self.assertEqual(rows[2]['gate_probability'], .5)

    def test_empty_retrieval_abstains_without_feature_or_generation_calls(self):
        r = FreshRetriever(self.corpus).prepare('unmatchedword', lambda s, n: s)
        def forbidden(*args):
            self.fail('Empty retrieval reached generation')
        rows = execute_policies(r, None, self.models, forbidden)
        self.assertTrue(all(x['reasons'] == ['no_evidence'] for x in rows))

    def test_cache_binding_rejects_changed_text_and_recomputed_digest(self):
        changed = copy.deepcopy(self.prepared['evidence'])
        changed[0]['text'] += ' edited'
        with self.assertRaisesRegex(ValueError, 'binding'):
            validate_feature_entry(self.entry, self.prepared['shown_claim'], changed)
        bad = copy.deepcopy(self.entry)
        bad['features'][0] += .01
        with self.assertRaisesRegex(ValueError, 'aggregate'):
            validate_feature_entry(bad, self.prepared['shown_claim'], self.prepared['evidence'])

    def test_malformed_neural_scores_are_rejected(self):
        for scores, cosines in [([[.1, .8, .2]], [.7]), ([[float('nan'), .8, .1]], [.7]),
                                ([[.1, .8, .1]], []), ([[.1, .8, .1]], [1.1])]:
            with self.subTest(scores=scores, cosines=cosines), self.assertRaises(ValueError):
                feature_entry(self.prepared['shown_claim'], self.prepared['evidence'], scores, cosines)

    def test_gold_labels_do_not_affect_retrieval_or_gate(self):
        corpus = [dict(d, label='Refuted', evidence={'gold': [100]}) for d in self.corpus]
        r = FreshRetriever(corpus).prepare('The treatment helped 10 people.', lambda s, n: s)
        self.assertEqual(r, self.prepared)
        self.assertEqual(execute_policies(r, self.entry, self.models, self.generator),
                         execute_policies(self.prepared, self.entry, self.models, self.generator))


if __name__ == '__main__':
    unittest.main()
