import unittest

from run_multilingual_agreement_replay import decide,align,metrics


class MultilingualAgreementTest(unittest.TestCase):
    def row(self,row,q,n,gold='Supported'):
        base={'file':'en/test.6h.jsonl','row':row,'claim_id':row,'english_page':'Page',
              'true_label':gold}
        return ({**base,'raw_candidate_label':q,'language':'en','translation_origin':'original'},
                {**base,'predicted_label':n,'probabilities':[.8,.1,.1]})

    def test_agreement_abstains_before_truth_is_scored(self):
        self.assertEqual(decide('Supported','Supported'),'Supported')
        self.assertIsNone(decide('Refuted','Supported'))
        a,b=self.row(1,'Supported','Supported')
        c,d=self.row(2,'Refuted','Supported')
        joined=align([a,c],[b,d])
        self.assertEqual([r['agreement_label'] for r in joined],['Supported',None])
        result=metrics(joined)
        self.assertEqual(result['en/test.6h.jsonl']['accepted'],1)
        self.assertEqual(result['en/test.6h.jsonl']['accepted_correct'],1)

    def test_duplicate_and_truth_mismatch_refused(self):
        a,b=self.row(1,'Supported','Supported')
        with self.assertRaises(ValueError):align([a,a],[b])
        with self.assertRaises(ValueError):align([a],[{**b,'true_label':'Refuted'}])


if __name__=='__main__':unittest.main()
