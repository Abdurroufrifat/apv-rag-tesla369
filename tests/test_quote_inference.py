import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

class InferenceContract(unittest.TestCase):
 def test_input_contract(self):
  from apv_rag.quote_inference import validate_input
  self.assertEqual(validate_input({'tokens':['Alice','said','hello'], 'quote_span':[2,3]}),(['Alice','said','hello'],(2,3)))
  for row in ({'tokens':['x'],'quote_span':[-1,1]}, {'tokens':['x'],'quote_span':[0,2]}, {'tokens':['x'],'quote_span':[True,1]}, {'tokens':['x'],'quote_span':[0,1],'speaker':'Alice'}, {'tokens':[''],'quote_span':[0,1]}):
   with self.assertRaises(ValueError):validate_input(row)
 def test_out_of_scope_abstains_before_model_use(self):
  from apv_rag.quote_inference import infer
  result=infer({'tokens':['x']*181,'quote_span':[0,6]},None)
  self.assertEqual(result['status'],'abstain')
  self.assertEqual(result['reason'],'outside_training_scope')
  self.assertIsNone(result['speaker_candidate'])
if __name__=='__main__':unittest.main()
