"""Exploratory selective speaker attribution; thresholds selected on development only."""
import hashlib,json,re,sys,joblib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from prepare_directquote_study import spans,normalize
from apv_rag.quote_relation import token_features,decode_speaker

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def metrics(rows,cut):
 answered=[x for x in rows if x['candidate'] is not None and x['confidence']>=cut]
 correct=sum(normalize(x['candidate'])==normalize(x['speaker'] or '') for x in answered)
 unknown=[x for x in rows if x['speaker'] is None]
 return {'cases':len(rows),'answered':len(answered),'coverage':len(answered)/len(rows),'correct_answers':correct,'answer_accuracy':correct/len(answered) if answered else None,'errors':len(answered)-correct,'unknown_cases':len(unknown),'unknown_false_attributions':sum(x['candidate'] is not None and x['confidence']>=cut for x in unknown)}

def main():
 base=ROOT/'artifacts/trained_quote_relation_v1';out=ROOT/'artifacts/quote_abstention_v1'
 if out.exists():raise FileExistsError(out)
 receipt=json.loads((base/'receipt.json').read_text())
 for name,h in receipt.items():assert sha(base/name)==h,name
 source=ROOT/'data/external/directquote/frozen_v1/truecased.txt'
 assert sha(source)==json.loads((base/'protocol.json').read_text())['source_sha256']
 split=json.loads((base/'split_ids.json').read_text());wanted=set(split['development']+split['test']);lookup={}
 for block in re.split(r'\n\s*\n',source.read_text().strip()):
  pairs=[x.rsplit(None,1) for x in block.splitlines() if x.strip()];tokens=[x[0] for x in pairs];key=hashlib.sha256(normalize(' '.join(tokens)).encode()).hexdigest()
  if key not in wanted or key in lookup:continue
  ranges=spans([x[1] for x in pairs]);quotes=[x for x in ranges if x[0]!='Speaker'];speakers=[x for x in ranges if x[0]=='Speaker']
  if len(quotes)!=1:continue
  kind,a,b=quotes[0]
  if kind=='Unknown':
   if speakers:continue
   gold=None
  else:
   if len(speakers)!=1:continue
   _,sa,sb=speakers[0]
   if not ((kind=='LeftSpeaker' and sb<=a) or (kind=='RightSpeaker' and sa>=b)):continue
   gold=' '.join(tokens[sa:sb])
  lookup[key]=(tokens,(a,b),gold)
 assert set(lookup)==wanted
 saved=joblib.load(base/'model.joblib');out.mkdir()
 protocol={'scope':'Exploratory post-hoc selective attribution on supplied quotes. No source authentication. Reused previously observed test cohort; not independent confirmation.', 'confidence':'Mean token score of the span selected by the frozen decoder; uncalibrated score, not probability of correct attribution.', 'grid':[.9,.95,.97,.98,.99,.995,.999,1.01], 'selection':'Maximum development coverage with empirical answer error <= 5% and >=30 answers; ties choose higher threshold. If infeasible abstain on all cases.', 'source_sha256':sha(source),'model_sha256':sha(base/'model.joblib'),'code_sha256':sha(Path(__file__))}
 write(out/'protocol.json',protocol)
 def predict(name):
  rows=[]
  for key in split[name]:
   tokens,quote,gold=lookup[key];p=saved['model'].predict_proba(saved['vectorizer'].transform(token_features(tokens,quote)))[:,1]
   candidate=decode_speaker(tokens,quote,p,saved['threshold']);runs=[];start=None
   for i in range(len(tokens)+1):
    keep=i<len(tokens) and not quote[0]<=i<quote[1] and p[i]>=saved['threshold']
    if keep and start is None:start=i
    if not keep and start is not None:runs.append((float(p[start:i].mean()),start,i));start=None
   confidence=max(runs,key=lambda x:(x[0],-x[1]))[0] if runs else 0.
   rows.append({'id':key,'speaker':gold,'candidate':candidate,'confidence':confidence})
  return rows
 dev=predict('development');choices=[{'threshold':t,**metrics(dev,t)} for t in protocol['grid']]
 eligible=[x for x in choices if x['answered']>=30 and x['errors']/x['answered']<=.05]
 threshold=max(eligible,key=lambda x:(x['coverage'],x['threshold']))['threshold'] if eligible else 1.01
 write(out/'selection.json',{'threshold':threshold,'target_feasible':bool(eligible),'development':choices})
 test=predict('test');old={x['id']:x for x in json.loads((base/'predictions.json').read_text())}
 assert all(x['candidate']==old[x['id']]['predicted_speaker'] and x['speaker']==old[x['id']]['speaker'] for x in test)
 summary={'threshold':threshold,'baseline':metrics(test,0),'selected':metrics(test,threshold),'frozen_test_predictions_replayed':len(test),'full_suite_rerun':False}
 write(out/'development_predictions.json',dev);write(out/'test_predictions.json',test);write(out/'summary.json',summary)
 write(out/'receipt.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
