import unittest

from audit_source_trace import passage_trace, explanation_trace, snapshot_gate


class SourceTraceTest(unittest.TestCase):
    def test_exact_source_and_tampering(self):
        corpus={'42':{'abstract':['Alpha about Tesla.','Beta sentence.']}}
        row={'claim':'Tesla','retrieved_evidence':[{'id':42,'text':'Alpha about',
                'selected_sentence_indices':[0]}]}
        self.assertEqual(passage_trace(row,corpus),{'passages':1,'known_ids':1,
                         'matching_indices':1,'matching_prefixes':1,'bound_passages':1})
        self.assertTrue(snapshot_gate(passage_trace(row,corpus)))
        row['retrieved_evidence'][0]['text']='Verified archive: Alpha about'
        self.assertEqual(passage_trace(row,corpus)['bound_passages'],0)
        self.assertFalse(snapshot_gate(passage_trace(row,corpus)))
        row['retrieved_evidence'][0]['id']=999
        self.assertEqual(passage_trace(row,corpus)['known_ids'],0)
        self.assertFalse(snapshot_gate(passage_trace(row,corpus)))

    def test_references_are_structural_only(self):
        row={'claim':'A claim','shown_claim':'A claim','evidence':[{'id':42,
             'text':'Archive http://example.org/doc'}],
             'generated_explanation':'See [42] and [99], https://fake.invalid/',
             'raw_candidate_label':'Supported'}
        result=explanation_trace(row)
        self.assertEqual(result['known_ids'],1)
        self.assertEqual(result['unknown_ids'],1)
        self.assertEqual(result['unseen_urls'],1)
        self.assertEqual(result['uncited_decisive_verdict'],0)
        row['generated_explanation']='This proves it.'
        self.assertEqual(explanation_trace(row)['uncited_decisive_verdict'],1)


if __name__=='__main__':unittest.main()
