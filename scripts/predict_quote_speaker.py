"""Usage: python scripts/predict_quote_speaker.py input.json output.json"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from apv_rag.quote_inference import infer,load_model

def main():
 parser=argparse.ArgumentParser(description='Frozen supplied-quote speaker attribution; no authentication verdict.')
 parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
 args=parser.parse_args()
 try:
  row=json.loads(args.input.read_text(encoding='utf-8'))
  saved=load_model(ROOT/'artifacts/trained_quote_relation_v1/model.joblib')
  result=infer(row,saved)
  with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
 except (ValueError,OSError) as exc:parser.exit(2,str(exc)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
