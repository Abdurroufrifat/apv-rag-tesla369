import copy
import json
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import patch

from apv_rag.fresh_pipeline import FreshRetriever, feature_entry, execute_policies
from run_fresh_pipeline import make_backend, response_key
from run_gated_generation import qwen_prompt
from verify_fresh_pipeline import verify_rows


class FreshExportTests(unittest.TestCase):
    def setUp(self):
        self.claim = 'The treatment helped 10 people.'
        self.prepared = FreshRetriever([{'doc_id': 1, 'abstract': [self.claim]}]).prepare(self.claim, lambda t, n: t)
        self.cache = {'scifact:1': feature_entry(self.claim, self.prepared['evidence'], [[.1, .8, .1]], [.7])}
        self.models = {n: {'columns': [0], 'mean': [0], 'scale': [1], 'coefficients': [[0]], 'intercept': [1]}
                       for n in ('nli', 'embedding', 'combined')}
        self.responses = {}
        self.baseline = {'claim_id': '1', 'true_label': 'Supported'}
        def generate(kind, question):
            prompt = qwen_prompt(question)
            response = {'answer': 'Supported' if kind == 'verdict' else self.claim, 'prompt': prompt, 'prompt_tokens': 20}
            self.responses[response_key(kind, prompt)] = {**response, 'kind': kind, 'origin': 'prior_exact_prompt'}
            self.baseline.update({kind + '_prompt': prompt, 'generated_' + kind: response['answer'], kind + '_prompt_tokens': 20})
            return response
        self.rows = execute_policies(self.prepared, self.cache['scifact:1'], self.models, generate)
        for row in self.rows:
            row.update(cohort='scifact', claim_id='1', claim=self.claim, shown_claim=self.claim,
                       retrieved_evidence=self.prepared['retrieved_evidence'],
                       context_sha256=self.cache['scifact:1']['context_sha256'], true_label='Supported')

    def verify(self, rows=None, responses=None):
        return verify_rows(self.rows if rows is None else rows, {'scifact:1': self.prepared}, self.cache,
                           self.models, self.responses if responses is None else responses,
                           {'scifact': {'1': self.baseline}})

    def test_export_replays_all_policies_and_exact_prompt_responses(self):
        self.assertEqual(self.verify(), self.rows)

    def test_changed_policy_label_or_evidence_is_rejected(self):
        for field, value in [('candidate_label', 'Refuted'), ('shown_claim', 'Changed claim'),
                             ('true_label', 'Refuted'), ('generation_requests', [])]:
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'binding'):
                self.verify(rows=rows)

    def test_duplicate_missing_or_unrequested_response_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.verify(rows=self.rows + [self.rows[0]])
        with self.assertRaisesRegex(ValueError, 'Missing'):
            self.verify(rows=self.rows[:-1])
        responses = copy.deepcopy(self.responses)
        record = {'kind': 'explanation', 'answer': 'extra', 'prompt': 'extra', 'prompt_tokens': 1,
                  'origin': 'current_run_live_or_resume'}
        responses[response_key('explanation', 'extra')] = record
        with self.assertRaisesRegex(ValueError, 'Unrequested'):
            self.verify(responses=responses)

    def test_cache_backend_reuses_exact_prompt_without_loading_model(self):
        seeds = {k: {n: v[n] for n in ('answer', 'prompt', 'prompt_tokens')} for k, v in self.responses.items()}
        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE TABLE answers (key TEXT PRIMARY KEY, response TEXT)')
            with patch('run_fresh_pipeline.live_backend', side_effect=AssertionError('Model should not load')):
                generate, used = make_backend(Path('.'), {}, db, seeds)
                question = self.rows[0]['verdict_prompt'].split('<|im_start|>user\n', 1)[1].split('<|im_end|>', 1)[0]
                self.assertEqual(generate('verdict', question)['answer'], 'Supported')
                self.assertEqual(next(iter(used.values()))['origin'], 'prior_exact_prompt')

    def test_existing_corrupt_seed_cache_is_rejected_without_overwrite(self):
        seeds = {k: {n: v[n] for n in ('answer', 'prompt', 'prompt_tokens')} for k, v in self.responses.items()}
        key = next(iter(seeds))
        altered = dict(seeds[key], answer='Refuted')
        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE TABLE answers (key TEXT PRIMARY KEY, response TEXT)')
            db.execute('INSERT INTO answers VALUES (?,?)', (key, json.dumps(altered)))
            with self.assertRaisesRegex(ValueError, 'conflict'):
                make_backend(Path('.'), {}, db, seeds)
            self.assertEqual(json.loads(db.execute('SELECT response FROM answers WHERE key=?', (key,)).fetchone()[0]), altered)

    def test_new_prompt_loads_live_backend_once_then_reuses_current_run_answer(self):
        calls = []
        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE TABLE answers (key TEXT PRIMARY KEY, response TEXT)')
            def backend(root, meta, connection):
                calls.append('load')
                def generate(kind, question):
                    prompt = qwen_prompt(question)
                    response = {'answer': 'Supported', 'prompt': prompt, 'prompt_tokens': 10}
                    connection.execute('INSERT INTO answers VALUES (?,?)', (response_key(kind, prompt), json.dumps(response)))
                    connection.commit()
                    return response
                return generate
            with patch('run_fresh_pipeline.live_backend', side_effect=backend):
                generate, used = make_backend(Path('.'), {}, db, {})
                first = generate('verdict', 'new claim')
                self.assertEqual(generate('verdict', 'new claim'), first)
                self.assertEqual(calls, ['load'])
                self.assertEqual(next(iter(used.values()))['origin'], 'current_run_live_or_resume')


if __name__ == '__main__':
    unittest.main()
