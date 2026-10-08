"""Exploratory lexical attribution classifier on grouped existing ratings."""
import json,re,hashlib,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score
ROOT=Path(__file__).resolve().parents[1]
def h(s):return hashlib.sha256(s.encode()).hexdigest()
def norm(s):return ' '.join(re.findall(r'\w+',s.casefold()))
def features(row):
 q=set(norm(row['question']).split());a=set(norm(row['answer']).split());p=set(norm(row['passage']).split());an=norm(row['answer']);pn=norm(row['passage']);nums=lambda s:set(re.findall(r'\d+',s))
 sentences=re.split(r'[.!?]',row['passage']);best=max((len(q&set(norm(s).split()))/max(1,len(q)) for s in sentences),default=0)
 return [float(bool(an) and ' '+an+' ' in ' '+pn+' '),len(a&p)/max(1,len(a)),len(q&p)/max(1,len(q)),len(q&p)/max(1,len(q|p)),best,len(nums(row['answer'])&nums(row['passage']))/max(1,len(nums(row['answer']))),float(bool(nums(row['answer']))),np.log1p(len(p)),np.log1p(len(a))]
def main():
 base=ROOT/'artifacts/attributed_qa_reference_v1';out=ROOT/'artifacts/trained_attribution_v1'
 if out.exists():raise FileExistsError(out)
 rows=json.loads((base/'inputs.json').read_text());gold={x['id']:x['human_supported'] for x in json.loads((base/'gold.json').read_text())};refs={x['id']:x for x in json.loads((base/'reference_predictions.json').read_text())}
 observed=[]
 for name in ('attributed_qa_nli_v1','attribution_format_fresh_v1'):observed+=json.loads((ROOT/'artifacts'/name/'inputs.json').read_text())
 oldq={norm(x['question']) for x in observed};oldpages={x['attribution'].split('#')[0] for x in observed}
 available=[x for x in rows if norm(x['question']) not in oldq and x['attribution'].split('#')[0] not in oldpages]
 # Remove conflicting duplicate payloads and count identical payloads once.
 payload={}
 for x in available:payload.setdefault((norm(x['question']),norm(x['answer']),norm(x['passage'])),[]).append(x)
 conflicts=sum(len({gold[x['id']] for x in group})>1 for group in payload.values());clean=[min(g,key=lambda x:x['id']) for g in payload.values() if len({gold[x['id']] for x in g})==1]
 parent=list(range(len(clean)));owner={}
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 for i,x in enumerate(clean):
  for key in ('q:'+norm(x['question']),'page:'+x['attribution'].split('#')[0],'passage:'+h(norm(x['passage']))):
   if key in owner:parent[find(i)]=find(owner[key])
   else:owner[key]=i
 grouped={}
 for i,x in enumerate(clean):grouped.setdefault(find(i),[]).append(x)
 groups=sorted(grouped.values(),key=lambda g:h('369'+min(x['id'] for x in g)));n=len(groups)
 split={name:[x for g in gs for x in g] for name,gs in [('train',groups[:int(.7*n)]),('development',groups[int(.7*n):int(.85*n)]),('test',groups[int(.85*n):])]}
 for a,b in [('train','development'),('train','test'),('development','test')]:
  for getter in (lambda x:norm(x['question']),lambda x:x['attribution'].split('#')[0],lambda x:norm(x['passage'])):assert {getter(x) for x in split[a]}.isdisjoint(getter(x) for x in split[b])
 assert all(len({gold[x['id']] for x in data})==2 for data in split.values())
 out.mkdir()
 def save(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
 protocol={'scope':'New supervised lexical component, exploratory within an already inspected benchmark. Not neural fine-tuning, independent confirmation, source authentication, or full APV-RAG. Semantic/pretraining and all prior-project isolation not guaranteed.','excluded_observed_questions':len(oldq),'excluded_observed_source_pages':len(oldpages),'available_rows':len(available),'deduplicated_rows':len(clean),'conflicting_payloads_excluded':conflicts,'groups':n,'split':'70/15/15 SHAseed369 connected question/page/exact-passage groups. Labels only for training, development selection and test scoring.','features':'Nine fixed lexical/numeric overlap and length features; no human rating, system identity, AutoAIS score, or attribution URL features.','model':'StandardScaler and LogisticRegression C=1 balanced weights max_iter1000 seed369; no hyperparameter search','threshold_grid':[.3,.4,.5,.6,.7],'selection':'Development macro F1, ties prefer threshold nearest .5 then higher threshold','input_sha256':h((base/'inputs.json').read_text()),'code_sha256':h(Path(__file__).read_text())}
 save('protocol.json',protocol);save('split_ids.json',{k:[x['id'] for x in v] for k,v in split.items()})
 x=np.array([features(r) for r in split['train']]);y=[gold[r['id']] for r in split['train']];scale=StandardScaler();model=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=369);model.fit(scale.fit_transform(x),y)
 dev=model.predict_proba(scale.transform([features(r) for r in split['development']]))[:,1];dy=[gold[r['id']] for r in split['development']];choices=[{'threshold':t,'macro_f1':float(f1_score(dy,dev>=t,average='macro'))} for t in protocol['threshold_grid']];threshold=max(choices,key=lambda c:(c['macro_f1'],-abs(c['threshold']-.5),c['threshold']))['threshold'];save('selection.json',{'threshold':threshold,'development':choices})
 saved={'mean':scale.mean_.tolist(),'scale':scale.scale_.tolist(),'coefficients':model.coef_[0].tolist(),'intercept':float(model.intercept_[0]),'threshold':threshold};save('model.json',saved)
 test=split['test'];prob=model.predict_proba(scale.transform([features(r) for r in test]))[:,1];ty=[gold[r['id']] for r in test];metrics={}
 policies={'trained_lexical':prob>=threshold,'released_autoais':[refs[r['id']]['released_autoais'] for r in test],'answer_containment':[refs[r['id']]['answer_containment'] for r in test],'always_supported':[True]*len(test)}
 for name,pred in policies.items():metrics[name]={'accuracy':float(accuracy_score(ty,pred)),'macro_f1':float(f1_score(ty,pred,average='macro',zero_division=0))}
 save('predictions.json',[{'id':r['id'],'gold':gold[r['id']],'probability':float(p),'supported':bool(p>=threshold)} for r,p in zip(test,prob)])
 # Verify serialized coefficients reproduce predictions without sklearn objects.
 z=(np.array([features(r) for r in test])-saved['mean'])/saved['scale'];replay=1/(1+np.exp(-(z@np.array(saved['coefficients'])+saved['intercept'])));assert np.allclose(prob,replay,atol=1e-12)
 summary={'rows':{k:len(v) for k,v in split.items()},'threshold':threshold,'metrics':metrics,'serialized_predictions_replayed':len(test),'question_page_passage_splits_disjoint':True,'scope':protocol['scope']};save('summary.json',summary);save('receipt.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()});print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
