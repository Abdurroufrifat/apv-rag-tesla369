"""Post-hoc NLI controls and paired declarative-format diagnostic; no tuning."""
import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_attributed_qa_nli import digest,load,save
from apv_rag.multilingual_nli import cen_order,normalize_model_scores

def reformulate(question,answer):
 q=question.strip().rstrip('?');a=answer.strip()
 if not a:return None
 if re.search(r'\b(and|or)\b',q):return None
 if re.match(r'^who (is|was|are|were) ',q):return a+' '+q[4:]+'.'
 if re.match(r'^who (wrote|played|sings|added|said|carried|won|invented|founded|discovered|directed|created|built) ',q):return a+' '+q[4:]+'.'
 m=re.fullmatch(r'when (was|were) (.+) (built|introduced|released|founded|born|established|invented|opened|completed)',q)
 if m:return m[2]+' '+m[1]+' '+m[3]+' in '+a+'.'
 return None

def main():
 base=ROOT/'artifacts/attribution_format_fresh_v1';out=ROOT/'artifacts/attribution_format_fresh_results_v1'
 rows=load(base/'inputs.json');protocol=load(base/'protocol.json')
 if digest(ROOT/'scripts/run_attribution_format_diagnostic.py')!=protocol['rules_sha256']:raise ValueError('Frozen rules changed')
 if digest(base/'inputs.json')!=protocol['inputs_sha256']:raise ValueError('Original input mismatch')
 tasks=[]
 for row in rows:
  text=reformulate(row['question'],row['answer'])
  if text:tasks.append({'id':row['id'],'kind':'paired_format','premise':row['passage'],'original_hypothesis':'Question: '+row['question']+' Answer: '+row['answer'],'declarative_hypothesis':text})
 expected_files=load(base/'model_identity.json');modelpath=ROOT/'models/nli-deberta-v3-small'
 for name,h in expected_files.items():
  if digest(modelpath/name)!=h:raise ValueError('Model mismatch: '+name)
 import torch
 from transformers import AutoTokenizer,AutoModelForSequenceClassification
 torch.set_num_threads(4);torch.manual_seed(369);torch.use_deterministic_algorithms(True)
 tokenizer=AutoTokenizer.from_pretrained(modelpath,local_files_only=True);model=AutoModelForSequenceClassification.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.float32).eval();order=cen_order(model.config.id2label)
 out.mkdir(exist_ok=True)
 if (out/'summary.json').exists():raise FileExistsError('Completed diagnostic exists')
 from importlib.metadata import version
 identity={'runner_sha256':digest(Path(__file__)),'input_sha256':digest(base/'inputs.json'),'normalizer_sha256':digest(ROOT/'src/apv_rag/multilingual_nli.py'),'model_files':expected_files,'packages':{n:version(n) for n in ('torch','transformers','numpy')},'scope':'Post-hoc diagnosis on observed cases. Rule outputs are unreviewed and may change meaning. No new efficacy or independent confirmation claim.'}
 ip=out/'identity.json'
 if ip.exists() and load(ip)!=identity:raise ValueError('Resume identity changed')
 save(ip,identity);cp=out/'predictions.json';cache=load(cp) if cp.exists() else {}
 wanted={t['id']+':'+format for t in tasks for format in ('original_hypothesis','declarative_hypothesis') if format in t}
 if not set(cache)<=wanted:raise ValueError('Unexpected cached cases')
 for t in tasks:
  for format in ('original_hypothesis','declarative_hypothesis'):
   if format not in t:continue
   key=t['id']+':'+format
   if key in cache:continue
   length=len(tokenizer.encode(t['premise'],t[format],truncation=False))
   encoded=tokenizer(t['premise'],t[format],return_tensors='pt',truncation='only_first',max_length=512)
   with torch.inference_mode():scores=normalize_model_scores(model(**encoded).logits.float().softmax(-1).cpu().numpy()[:,order].tolist())[0]
   cache[key]={'scores_cen':scores,'hypothesis':t[format],'truncated':length>512};save(cp,cache)
  print('Completed',t['id'],flush=True)
 gold={x['id']:x['human_supported'] for x in load(base/'gold.json')};refs={x['id']:x for x in load(base/'references.json')}
 from sklearn.metrics import accuracy_score,f1_score
 paired=[t for t in tasks if t['kind']=='paired_format'];metrics={}
 for format in ('original_hypothesis','declarative_hypothesis'):
  y=[gold[t['id']] for t in paired];p=[cache[t['id']+':'+format]['scores_cen'][1]>=.5 for t in paired]
  metrics[format]={'cases':len(y),'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,labels=[False,True],average='macro',zero_division=0))}
 for policy in ('released_autoais','answer_containment'):
  y=[gold[t['id']] for t in paired];p=[refs[t['id']][policy] for t in paired]
  metrics[policy]={'cases':len(y),'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,labels=[False,True],average='macro',zero_division=0))}
 controls=[]
 for t in tasks:
  if t['kind']=='synthetic_control':
   scores=cache[t['id']+':declarative_hypothesis']['scores_cen'];label=('contradiction','entailment','neutral')[max(range(3),key=lambda i:scores[i])];controls.append({'id':t['id'],'expected':t['expected'],'predicted':label,'correct':label==t['expected']})
 summary={'paired_metrics':metrics,'controls':controls,'control_correct':sum(x['correct'] for x in controls),'skipped_questions':len(rows)-len(paired),'scope':protocol['scope']};save(out/'summary.json',summary);save(out/'tasks.json',tasks);save(out/'receipt.json',{p.name:digest(p) for p in out.glob('*.json') if p.name!='receipt.json'});print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
