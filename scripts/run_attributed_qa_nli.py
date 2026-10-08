"""Resumable frozen NLI transfer on 200 supplied question-answer-passage cases."""
import argparse,hashlib,json,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from apv_rag.multilingual_nli import cen_order,normalize_model_scores

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
 temporary=p.with_suffix(p.suffix+'.tmp');temporary.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8');os.replace(temporary,p)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
 out=ROOT/'artifacts/attributed_qa_nli_v1';protocol=load(out/'protocol.json');rows=load(out/'inputs.json');expected=load(out/'model_identity.json')
 if digest(out/'inputs.json')!=protocol['inputs_sha256']:raise ValueError('Input hash mismatch')
 if len(rows)!=200 or len({x['id'] for x in rows})!=200 or len({' '.join(x['question'].casefold().split()) for x in rows})!=200:raise ValueError('Invalid cohort')
 if any(set(x)!={'id','question','answer','passage','attribution','system_name'} for x in rows):raise ValueError('Unexpected input fields')
 if args.preflight:print('200 label-free inputs validated. Neural model execution not run.');return
 modelpath=ROOT/'models/nli-deberta-v3-small'
 print('Checking existing NLI model files; no download.',flush=True)
 for name,h in expected.items():
  if digest(modelpath/name)!=h:raise ValueError('Model file mismatch: '+name)
 if (out/'summary.json').exists():raise FileExistsError('Completed evaluation exists; refusing overwrite')
 from importlib.metadata import version
 identity={'protocol_sha256':digest(out/'protocol.json'),'runner_sha256':digest(Path(__file__)),'normalizer_sha256':digest(ROOT/'src/apv_rag/multilingual_nli.py'),'packages':{n:version(n) for n in ('torch','transformers','numpy')},'model_files':expected}
 ip=out/'execution_identity.json'
 if ip.exists() and load(ip)!=identity:raise ValueError('Resume identity changed')
 save(ip,identity);cachepath=out/'predictions.json';cache=load(cachepath) if cachepath.exists() else {}
 if not set(cache)<=set(x['id'] for x in rows):raise ValueError('Unexpected cached IDs')
 import torch
 from transformers import AutoTokenizer,AutoModelForSequenceClassification
 torch.set_num_threads(4);torch.manual_seed(369);torch.use_deterministic_algorithms(True)
 tokenizer=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
 model=AutoModelForSequenceClassification.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.float32).eval();order=cen_order(model.config.id2label)
 for i,row in enumerate(rows):
  if row['id'] in cache:continue
  hypothesis='Question: '+row['question']+' Answer: '+row['answer']
  length=len(tokenizer.encode(row['passage'],hypothesis,add_special_tokens=True,truncation=False))
  encoded=tokenizer(row['passage'],hypothesis,return_tensors='pt',truncation='only_first',max_length=512)
  with torch.inference_mode():scores=normalize_model_scores(model(**encoded).logits.float().softmax(-1).cpu().numpy()[:,order].tolist())[0]
  cache[row['id']]={'scores_cen':scores,'supported':scores[1]>=.5,'pair_tokens_before_truncation':length,'truncated':length>512}
  save(cachepath,cache)
  if (i+1)%10==0 or i+1==len(rows):print(f'Completed {len(cache)}/200',flush=True)
 # Existing labels are read only after all inference has finished.
 from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
 gold={x['id']:x['human_supported'] for x in load(out/'gold.json')};refs={x['id']:x for x in load(out/'references.json')}
 if set(gold)!=set(cache) or set(refs)!=set(cache):raise ValueError('Evaluation ID mismatch')
 y=[gold[x['id']] for x in rows];metrics={}
 for name in ('frozen_nli','released_autoais','answer_containment'):
  pred=[cache[x['id']]['supported'] if name=='frozen_nli' else refs[x['id']][name] for x in rows]
  metrics[name]={'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,labels=[False,True],average='macro',zero_division=0)),'confusion_matrix_unsupported_supported':confusion_matrix(y,pred,labels=[False,True]).tolist()}
 summary={'cases':len(rows),'supported_gold':sum(y),'truncated_pairs':sum(x['truncated'] for x in cache.values()),'metrics':metrics,'scope':protocol['scope']}
 save(out/'summary.json',summary);save(out/'output_receipt.json',{p.name:digest(p) for p in out.glob('*.json') if p.name!='output_receipt.json'});print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
