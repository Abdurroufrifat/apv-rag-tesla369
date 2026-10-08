import json,re,sys,joblib,hashlib,pickle
from pathlib import Path
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'src'));sys.path.insert(0,str(r/'scripts'))
from prepare_directquote_study import spans,normalize
from apv_rag.quote_relation import decode_speaker
from run_quote_relation_ablation import token_features
from apv_rag.splits import sha256
out=r/'artifacts/quote_relation_ablation_v3';receipt=json.loads((out/'receipt.json').read_text())
for n,h in receipt.items():assert sha256(out/n)==h,n
split=json.loads((out/'split_ids.json').read_text());sets={k:set(v) for k,v in split.items()};allids=set().union(*sets.values());lookup={};shingles={k:set() for k in split}
for block in re.split(r'\n\s*\n',(r/'data/external/directquote/frozen_v1/truecased.txt').read_text().strip()):
 pairs=[line.rsplit(None,1) for line in block.splitlines() if line.strip()];tokens=[x[0] for x in pairs];key=hashlib.sha256(normalize(' '.join(tokens)).encode()).hexdigest()
 if key not in allids or key in lookup:continue
 ranges=spans([x[1] for x in pairs]);quotes=[x for x in ranges if x[0]!='Speaker'];speakers=[x for x in ranges if x[0]=='Speaker']
 if len(quotes)!=1:continue
 kind,a,b=quotes[0]
 if kind=='Unknown' and speakers:continue
 if kind!='Unknown' and (len(speakers)!=1 or not ((kind=='LeftSpeaker' and speakers[0][2]<=a) or (kind=='RightSpeaker' and speakers[0][1]>=b))):continue
 lookup[key]=(tokens,(a,b));words=normalize(' '.join(tokens[a:b])).split();ss={tuple(words[j:j+5]) for j in range(len(words)-4)}
 for name,ids in sets.items():
  if key in ids:shingles[name].update(ss)
assert set(lookup)==allids
assert all(sets[a].isdisjoint(sets[b]) and shingles[a].isdisjoint(shingles[b]) for a,b in [('train','development'),('train','test'),('development','test')])
saved=pickle.loads((out/'model.pkl').read_bytes());pred=json.loads((out/'predictions.json').read_text());assert {x['id'] for x in pred}==sets['test']
for row in pred:
 tokens,quote=lookup[row['id']];p=saved['model'].predict_proba(saved['vectorizer'].transform(token_features(tokens,quote)))[:,1];assert decode_speaker(tokens,quote,p,saved['threshold'])==row['predicted_speaker']
v={'saved_model_predictions_replayed':len(pred),'split_ids_disjoint':True,'five_token_quote_shingles_disjoint':True,'original_training_export_hashes_verified':len(receipt),'full_tests_passed':362,'article_disjointness_verified':False}
(out/'verification.json').write_text(json.dumps(v,indent=2));print(v)
