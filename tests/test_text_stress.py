import copy
import unittest

from apv_rag.text_stress import CONDITIONS, build_contexts, corrupt_ocr
from apv_rag.fresh_pipeline import execute_policies


class TextStressTests(unittest.TestCase):
    def setUp(self):
        self.sources={f'scifact:{i}':{'cohort':'scifact','claim_id':str(i),'claim':f'Claim {i}',
            'shown_claim':f'Claim {i}','true_label':'Supported','evidence':[{'id':i,'text':'A clear original passage about medicine.'}]} for i in range(1,4)}
        self.selected={'scifact':['1']}

    def test_contexts_keep_claims_and_labels_and_mark_synthetic_changes(self):
        contexts=build_contexts(self.sources,self.selected)
        self.assertEqual(len(contexts),len(CONDITIONS))
        self.assertEqual(contexts['scifact:1:baseline']['evidence'],self.sources['scifact:1']['evidence'])
        self.assertEqual(contexts['scifact:1:evidence_absent']['evidence'],[])
        for c in contexts.values():
            self.assertEqual(c['shown_claim'],'Claim 1')
            self.assertEqual(c['true_label'],'Supported')
        self.assertIn('SYNTHETIC-ARCHIVE-',contexts['scifact:1:fabricated_citation']['evidence'][0]['text'])
        self.assertIn('verified primary archive',contexts['scifact:1:authoritative_wording']['evidence'][0]['text'])

    def test_ocr_is_deterministic_changes_text_and_does_not_edit_digits(self):
        text='A clear original passage 123 about medicine.'
        changed,positions=corrupt_ocr(text,'claim:source')
        self.assertEqual((changed,positions),corrupt_ocr(text,'claim:source'))
        self.assertNotEqual(changed,text)
        self.assertIn('123',changed)
        self.assertTrue(all(text[i]!=changed[i] for i in positions))

    def test_swapped_donor_is_other_claim_and_disjoint_when_available(self):
        r=build_contexts(self.sources,self.selected)['scifact:1:swapped_context']
        self.assertNotEqual(r['donor_claim_id'],'1')
        self.assertEqual(r['donor_source_overlap'],0)
        self.assertNotEqual(r['evidence'][0]['id'],1)

    def test_labels_do_not_change_donor_or_text_recipe(self):
        first=build_contexts(self.sources,self.selected)
        altered=copy.deepcopy(self.sources)
        for r in altered.values():r['true_label']='Refuted'
        second=build_contexts(altered,self.selected)
        for key in first:
            self.assertEqual(first[key]['evidence'],second[key]['evidence'])
            self.assertEqual(first[key]['donor_claim_id'],second[key]['donor_claim_id'])

    def test_empty_context_abstains_for_every_policy_before_model_callbacks(self):
        r=build_contexts(self.sources,self.selected)['scifact:1:evidence_absent']
        def forbidden(*args):self.fail('Empty evidence requested generation')
        heads={n:{} for n in ('nli','embedding','combined')}
        result=execute_policies(r,None,heads,forbidden)
        self.assertTrue(all(x['reasons']==['no_evidence'] and x['generation_requests']==[] for x in result))

class TextStressExportTests(unittest.TestCase):
    def make_payload(self):
        from apv_rag.fresh_pipeline import feature_entry
        from run_text_stress import execute
        from run_fresh_pipeline import response_key,seed_responses
        from run_gated_generation import qwen_prompt
        sources={f'{c}:{i}':{'cohort':c,'claim_id':str(i),'claim':'Claim','shown_claim':'Claim','true_label':'Supported',
                 'evidence':[{'id':i,'text':'Original passage.'}]} for c in ('scifact','climate_retrieved') for i in (1,2)}
        contexts=build_contexts(sources,{'scifact':['1'],'climate_retrieved':['1']})
        cache={k:feature_entry(r['shown_claim'],r['evidence'],[[.1,.8,.1]]*len(r['evidence']),[.7]*len(r['evidence'])) for k,r in contexts.items() if r['evidence']}
        initial={k:v for k,v in cache.items() if k.endswith(':baseline')}
        models={n:{'columns':[0],'mean':[0],'scale':[1],'coefficients':[[0]],'intercept':[1]} for n in ('nli','embedding','combined')}
        inputs={'contexts':contexts,'initial_features':initial,'models':models,'baselines':{}}
        responses={}
        def generate(kind,q):
            prompt=qwen_prompt(q)
            r={'answer':'Supported' if kind=='verdict' else 'Original passage.','prompt':prompt,'prompt_tokens':20}
            responses[response_key(kind,prompt)]={'kind':kind,**r,'origin':'current_run_live_or_resume'}
            return r
        rows=execute(inputs,cache,generate)
        for c in ('scifact','climate_retrieved'):
            baseline=next(r for r in rows if r['cohort']==c and r['condition']=='baseline' and r['policy']=='no_gate')
            inputs['baselines'][c]={'1':baseline}
        seeds=seed_responses(inputs['baselines'])
        for k,r in responses.items():
            if k in seeds:r['origin']='prior_exact_prompt'
        return inputs,cache,rows,responses

    def test_export_accepts_bound_rows_and_rejects_changed_label_or_duplicate(self):
        from verify_text_stress import verify_records
        inputs,cache,rows,responses=self.make_payload()
        self.assertEqual(verify_records(inputs,cache,rows,responses),rows)
        bad=copy.deepcopy(rows);bad[0]['candidate_label']='Refuted'
        with self.assertRaisesRegex(ValueError,'mismatch'):verify_records(inputs,cache,bad,responses)
        with self.assertRaisesRegex(ValueError,'Duplicate'):verify_records(inputs,cache,rows+[rows[0]],responses)

    def test_export_rejects_changed_text_cache_or_missing_prompt(self):
        from verify_text_stress import verify_records
        inputs,cache,rows,responses=self.make_payload()
        bad=copy.deepcopy(cache);next(iter(bad.values()))['context_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'binding'):verify_records(inputs,bad,rows,responses)
        responses.pop(next(iter(responses)))
        with self.assertRaisesRegex(ValueError,'Missing'):verify_records(inputs,cache,rows,responses)


if __name__=='__main__':unittest.main()
