"""Post-hoc replay of an already trained ablation; never fits or tunes."""
import json,pickle,sys,zipfile,hashlib,collections,random
from pathlib import Path
from run_transfer import extract,norm
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent/'apv-rag-tesla369';sys.path.insert(0,str(ROOT/'src'))
from apv_rag.quote_relation import token_features,decode_speaker
folder=ROOT/'artifacts/quote_relation_ablation_v3';path=folder/'model.pkl';receipt=json.loads((folder/'receipt.json').read_text());actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==receipt['model.pkl']
saved=pickle.loads(path.read_bytes());assert saved['threshold']==.8
out=BASE/'trained_comparator';out.mkdir(exist_ok=True)
(out/'protocol.json').write_text(json.dumps({'post_hoc':True,'cohort':'Same already observed PolNeAR 234 cases; no fresh confirmation claim.','model_sha256':actual,'threshold':.8,'removed_features':['quote_side','distance_bin','quote_inside','side_word'],'no_fitting':True,'no_tuning':True},indent=2))
old=json.loads((BASE/'test_predictions.json').read_text());wanted={x['id'] for x in old};z=zipfile.ZipFile(BASE/'repository.zip');prefix=z.namelist()[0];inputs=[]
for name in z.namelist():
 if name.startswith(prefix+'data/test/attributions/') and name.endswith('.ann'):
  stem=Path(name).stem.rsplit('_',1)[0];text=z.read(prefix+'data/test/text/'+stem+'.txt').decode()
  for x in extract(text,z.read(name).decode()):
   key=stem+':'+x['event']
   if key in wanted:inputs.append((key,x))
assert {key for key,x in inputs}==wanted and len(inputs)==len(old)
result=[]
for r,(key,x) in zip(old,inputs):
 assert key==r['id'];assert norm(x['gold'])==norm(r['gold']);features=[{k:v for k,v in f.items() if k not in {'quote_side','distance_bin','quote_inside','side_word'}} for f in token_features(x['tokens'],x['quote_span'])];p=saved['model'].predict_proba(saved['vectorizer'].transform(features))[:,1];span=decode_speaker(x['tokens'],x['quote_span'],p,.8);result.append({**r,'trained_ablation':span,'ablation_correct':norm(span)==norm(r['gold'])})
groups=collections.defaultdict(list)
for r in result:groups[r['article_id']].append(r)
g=list(groups.values());rng=random.Random(369);draws=[]
for _ in range(4000):
 sample=[r for _ in g for r in rng.choice(g)];draws.append(sum(int(r['model_correct'])-int(r['ablation_correct']) for r in sample)/len(sample))
draws.sort();n=len(result);c=sum(r['ablation_correct'] for r in result);summary={'cases':n,'articles':len(g),'full_correct':sum(r['model_correct'] for r in result),'ablation_correct':c,'ablation_accuracy':c/n,'paired_full_minus_ablation':sum(int(r['model_correct'])-int(r['ablation_correct']) for r in result)/n,'article_bootstrap_95_interval':[draws[100],draws[3900]],'post_hoc':True,'scope':'Existing learned component ablation, not a contemporary external system baseline or new independent confirmation.'};(out/'predictions.json').write_text(json.dumps(result,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
