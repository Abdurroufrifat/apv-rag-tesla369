"""Pinned MiniCheck single-passage comparison on the fixed 131-case cohort."""
import argparse,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_attributed_qa_nli import digest,load,save
from run_attribution_format_diagnostic import reformulate
REPO='lytang/MiniCheck-Flan-T5-Large';REV='96eafd01cee2d16cf81aaa2fb226b14f422a37b3'
FILES=['config.json','generation_config.json','pytorch_model.bin','tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','spiece.model','README.md']
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true');args=parser.parse_args();path=ROOT/'models/minicheck-flan-t5-large'
 if args.download:
  from huggingface_hub import snapshot_download
  snapshot_download(repo_id=REPO,revision=REV,local_dir=path,allow_patterns=FILES)
  save(path/'apv_model_manifest.json',{'repo':REPO,'revision':REV,'files':{n:digest(path/n) for n in FILES}})
  print('Pinned model download complete. Run again without --download.');return
 manifest=load(path/'apv_model_manifest.json')
 if manifest['repo']!=REPO or manifest['revision']!=REV:raise ValueError('Unexpected model revision')
 for name,h in manifest['files'].items():
  if digest(path/name)!=h:raise ValueError('Model checksum mismatch: '+name)
 base=ROOT/'artifacts/attribution_format_fresh_v1';rows=load(base/'inputs.json');protocol=load(base/'protocol.json')
 if digest(base/'inputs.json')!=protocol['inputs_sha256'] or len(rows)!=131:raise ValueError('Cohort mismatch')
 if digest(ROOT/'scripts/run_attribution_format_diagnostic.py')!=protocol['rules_sha256']:raise ValueError('Reformulation rules changed')
 out=ROOT/'artifacts/minicheck_attribution_v1';out.mkdir(exist_ok=True)
 if (out/'summary.json').exists():raise FileExistsError('Completed comparison exists')
 import torch
 from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
 from importlib.metadata import version
 identity={'model':manifest,'input_sha256':digest(base/'inputs.json'),'runner_sha256':digest(Path(__file__)),'rules_sha256':protocol['rules_sha256'],'packages':{n:version(n) for n in ('torch','transformers','numpy')},'scope':'Exploratory pretrained-model comparison on the previously observed 131-case reformulation cohort. Single supplied passage; no upstream chunk aggregation. Not independent confirmation, original method efficacy or source authentication.'}
 ip=out/'identity.json'
 if ip.exists() and load(ip)!=identity:raise ValueError('Resume identity changed')
 save(ip,identity);cp=out/'predictions.json';cache=load(cp) if cp.exists() else {}
 if not set(cache)<=set(x['id'] for x in rows):raise ValueError('Unexpected cache IDs')
 torch.set_num_threads(4);torch.manual_seed(369);torch.use_deterministic_algorithms(True)
 tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True);model=AutoModelForSeq2SeqLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32).eval()
 for i,row in enumerate(rows):
  if row['id'] in cache:continue
  claim=reformulate(row['question'],row['answer'])
  if not claim:raise ValueError('Missing fixed declarative claim')
  text='predict: '+row['passage']+tokenizer.eos_token+claim
  encoded=tokenizer(text,return_tensors='pt',truncation=False)
  if encoded['input_ids'].shape[1]>2048:raise ValueError('Input exceeds 2048 tokens; no silent truncation')
  with torch.inference_mode():
   logits=model(**encoded,decoder_input_ids=torch.zeros((1,1),dtype=torch.long)).logits[0,0,[3,209]]
   probability=float(logits.float().softmax(-1)[1].item())
  cache[row['id']]={'support_probability':probability,'supported':probability>=.5,'claim':claim,'input_tokens':encoded['input_ids'].shape[1]};save(cp,cache)
  print(f'Completed {len(cache)}/131',flush=True)
 gold={x['id']:x['human_supported'] for x in load(base/'gold.json')};refs={x['id']:x for x in load(base/'references.json')}
 from sklearn.metrics import accuracy_score,f1_score
 y=[gold[x['id']] for x in rows];metrics={}
 for name in ('minicheck','released_autoais','answer_containment'):
  pred=[cache[x['id']]['supported'] if name=='minicheck' else refs[x['id']][name] for x in rows]
  metrics[name]={'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,labels=[False,True],average='macro',zero_division=0))}
 summary={'cases':131,'metrics':metrics,'scope':identity['scope']};save(out/'summary.json',summary);save(out/'receipt.json',{p.name:digest(p) for p in out.glob('*.json') if p.name!='receipt.json'});print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
