import unittest

from apv_rag.source_snapshot_guard import execute_snapshot_guarded


class SnapshotGuardControllerTest(unittest.TestCase):
    def setUp(self):
        self.corpus={'42':{'abstract':['A sentence about Tesla.']}}
        self.context=[{'id':42,'text':'A sentence about Tesla.',
                       'selected_sentence_indices':[0]}]
        self.calls=[]

    def generate(self,kind,question):
        self.calls.append(kind)
        return {'answer':'Supported' if kind=='verdict' else 'It says Tesla.',
                'prompt':question,'prompt_tokens':9}

    def run_context(self,evidence):
        return execute_snapshot_guarded('Tesla','Tesla',evidence,evidence,
                                        self.corpus,self.generate)

    def test_matches_before_generation(self):
        output=self.run_context(self.context)
        self.assertTrue(output['snapshot_source_bound'])
        self.assertEqual(self.calls,['verdict','explanation'])
        self.assertEqual(output['candidate_label'],'Supported')

    def test_changed_text_abstains_without_model_calls(self):
        evidence=[{**self.context[0],'text':'SYNTHETIC-ARCHIVE A sentence about Tesla.'}]
        output=self.run_context(evidence)
        self.assertFalse(output['snapshot_source_bound'])
        self.assertEqual(output['reasons'],['snapshot_source_mismatch'])
        self.assertIsNone(output['candidate_label'])
        self.assertEqual(self.calls,[])

    def test_unknown_id_and_no_evidence_abstain(self):
        unknown=[{**self.context[0],'id':99}]
        self.assertEqual(self.run_context(unknown)['reasons'],['snapshot_source_mismatch'])
        self.assertEqual(self.run_context([])['reasons'],['no_evidence'])
        self.assertEqual(self.calls,[])


if __name__=='__main__':unittest.main()
