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
 base=ROOT/'artifacts/minicheck_selective_inputs_v1';rows=load(base/'inputs.json');protocol=load(base/'protocol.json')
 if digest(base/'inputs.json')!=protocol['inputs_sha256'] or len(rows)!=208:raise ValueError('Cohort mismatch')
 out=ROOT/'artifacts/minicheck_selective_v1';out.mkdir(exist_ok=True)
 if (out/'summary.json').exists():raise FileExistsError('Completed comparison exists')
 import torch
 from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
 from importlib.metadata import version
 identity={'model':manifest,'input_sha256':digest(base/'inputs.json'),'runner_sha256':digest(Path(__file__)),'packages':{n:version(n) for n in ('torch','transformers','numpy')},'scope':'Exploratory pretrained-model comparison on the previously observed 131-case reformulation cohort. Single supplied passage; no upstream chunk aggregation. Not independent confirmation, original method efficacy or source authentication.'}
 ip=out/'identity.json'
 if ip.exists() and load(ip)!=identity:
  previous=load(ip)
  expected_previous=dict(identity);expected_previous['runner_sha256']='a65a9e5675641f1e5cf001a7cf5c8d306917ee6bd8729286731694e625631287'
  if previous!=expected_previous:raise ValueError('Resume identity changed beyond the documented length fix')
  save(out/'length_fix_migration.json',{'previous_identity':previous,'new_identity':identity,'policy':'Inputs above 2048 tokens abstain. Existing predictions preserved. Post-start protocol amendment before test summary.'})
 save(ip,identity);cp=out/'predictions.json';cache=load(cp) if cp.exists() else {}
 if not set(cache)<=set(x['id'] for x in rows):raise ValueError('Unexpected cache IDs')
 torch.set_num_threads(4);torch.manual_seed(369);torch.use_deterministic_algorithms(True)
 tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True);model=AutoModelForSeq2SeqLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32).eval()
 for i,row in enumerate(rows):
  if row['id'] in cache:continue
  claim='The answer to the question '+json.dumps(row['question'])+' is '+json.dumps(row['answer'])+'.'
  if not claim:raise ValueError('Missing fixed declarative claim')
  text='predict: '+row['passage']+tokenizer.eos_token+claim
  encoded=tokenizer(text,return_tensors='pt',truncation=False)
  if encoded['input_ids'].shape[1]>2048:
   cache[row['id']]={'support_probability':None,'supported':None,'claim':claim,'input_tokens':int(encoded['input_ids'].shape[1]),'status':'abstain','reason':'input_exceeds_2048_tokens'}
   save(cp,cache);print(f"Completed {len(cache)}/208 (length abstention)",flush=True)
   continue
  with torch.inference_mode():
   logits=model(**encoded,decoder_input_ids=torch.zeros((1,1),dtype=torch.long)).logits[0,0,[3,209]]
   probability=float(logits.float().softmax(-1)[1].item())
  cache[row['id']]={'support_probability':probability,'supported':probability>=.5,'claim':claim,'input_tokens':encoded['input_ids'].shape[1]};save(cp,cache)
  print(f'Completed {len(cache)}/208',flush=True)
 def metric(data,gold,cut):
  kept=[x for x in data if cache[x['id']]['support_probability'] is not None and max(cache[x['id']]['support_probability'],1-cache[x['id']]['support_probability'])>=cut]
  correct=sum(cache[x['id']]['supported']==gold[x['id']] for x in kept)
  return {'cases':len(data),'answered':len(kept),'correct':correct,'errors':len(kept)-correct,'coverage':len(kept)/len(data),'answer_accuracy':correct/len(kept) if kept else None,'length_abstentions':sum(cache[x['id']]['support_probability'] is None for x in data)}
 dev=[x for x in rows if x['split']=='development'];dg={x['id']:x['human_supported'] for x in load(base/'development_gold.json')}
 choices=[{'cutoff':t,**metric(dev,dg,t)} for t in protocol['thresholds']];eligible=[x for x in choices if x['answered']>=20 and x['errors']/x['answered']<=.1]
 cut=max(eligible,key=lambda x:(x['coverage'],x['cutoff']))['cutoff'] if eligible else 1.01
 save(out/'selection.json',{'cutoff':cut,'target_feasible':bool(eligible),'development':choices})
 test=[x for x in rows if x['split']=='test'];tg={x['id']:x['human_supported'] for x in load(base/'test_gold.json')}
 summary={'development_cases':len(dev),'test_cases':len(test),'cutoff':cut,'target_feasible':bool(eligible),'test_baseline':metric(test,tg,.5),'test_selective':metric(test,tg,cut),'scope':protocol['scope'],'protocol_amendment':'Oversized inputs abstain; 2048-token limit preserved. Amendment made after partial inference; exploratory results.'}
 save(out/'summary.json',summary);save(out/'receipt.json',{p.name:digest(p) for p in out.glob('*.json') if p.name!='receipt.json'});print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
