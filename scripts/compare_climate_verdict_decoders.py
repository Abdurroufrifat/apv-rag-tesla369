"""Post-hoc fixed-context label-likelihood comparison; no explanation regeneration."""
import json
from importlib.metadata import version
from pathlib import Path
import numpy as np
from apv_rag.generative_rag import LABELS
from apv_rag.nli_comparison import _model_files
from apv_rag.splits import sha256, write_json_atomic


def summarize(rows, field):
    from sklearn.metrics import accuracy_score, f1_score
    return {'accuracy': float(accuracy_score([r['true_label'] for r in rows], [r[field] for r in rows])),
            'macro_f1': float(f1_score([r['true_label'] for r in rows], [r[field] for r in rows], labels=list(LABELS), average='macro', zero_division=0))}


def main():
    root = Path(__file__).resolve().parents[1]
    output = root/'artifacts/climate_decoder_comparison_v1'
    if (output/'output_manifest.json').exists():
        raise FileExistsError('Completed comparison exists; refusing overwrite')
    model_path = root/'models/qwen2.5-1.5b-instruct'
    meta = json.loads((root/'models/instruction_model_manifest.json').read_text(encoding='utf-8'))
    if _model_files(model_path) != meta['files']:
        raise ValueError('Generator checksum mismatch')
    cohorts = {}
    source_hashes = {}
    for name, folder_name in [('retrieved','climate_rag_frozen_v1'),('supplied','climate_supplied_evidence_v1')]:
        folder = root/'artifacts'/folder_name
        manifest = json.loads((folder/'output_manifest.json').read_text(encoding='utf-8'))
        for filename in ('predictions.json','input_manifest.json'):
            if sha256(folder/filename) != manifest[filename]:
                raise ValueError('Source output checksum mismatch')
        identity = json.loads((folder/'input_manifest.json').read_text(encoding='utf-8'))
        if identity['models']['generator'] != meta:
            raise ValueError('Source generator differs')
        cohorts[name] = json.loads((folder/'predictions.json').read_text(encoding='utf-8'))
        if len(cohorts[name]) != 300 or len({r['claim_id'] for r in cohorts[name]}) != 300:
            raise ValueError('Unexpected cohort')
        source_hashes[name] = manifest['predictions.json']
    if {r['claim_id'] for r in cohorts['retrieved']} != {r['claim_id'] for r in cohorts['supplied']}:
        raise ValueError('Cohorts differ')
    frozen = {'source_hashes':source_hashes,'generator':meta,'code_sha256':sha256(Path(__file__)),
              'protocol_sha256':sha256(root/'docs/CLIMATE_DECODER_COMPARISON.md'),
              'packages':{n:version(n) for n in ('torch','transformers','numpy','scikit-learn')},
              'settings':{'dtype':'float32','seed':369,'threads':4,'primary':'sum token log likelihood including EOS','secondary':'mean token log likelihood including EOS','label_order':list(LABELS)}}
    output.mkdir(exist_ok=True)
    receipt = output/'input_manifest.json'
    if receipt.exists() and json.loads(receipt.read_text(encoding='utf-8')) != frozen:
        raise ValueError('Comparison identity changed; cache reuse refused')
    write_json_atomic(receipt,frozen)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_num_threads(4)
    torch.manual_seed(369)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.float32).eval()
    labels = [tokenizer.encode(label,add_special_tokens=False)+[tokenizer.eos_token_id] for label in LABELS]
    rows=[]
    cache=output/'score_cache.json'
    cached=json.loads(cache.read_text(encoding='utf-8')) if cache.exists() else {}
    for cohort, sources in cohorts.items():
        for position, row in enumerate(sources):
            key=f"{cohort}:{row['claim_id']}"
            prompt=tokenizer.encode(row['verdict_prompt'],add_special_tokens=False)
            if len(prompt)!=row['verdict_prompt_tokens']:
                raise ValueError('Rendered prompt token count mismatch')
            if key in cached:
                token_scores=cached[key]
            else:
                token_scores=[]
                for label_ids in labels:
                    ids=torch.tensor([prompt+label_ids],dtype=torch.long)
                    with torch.inference_mode():
                        logits=model(input_ids=ids,attention_mask=torch.ones_like(ids)).logits
                        log_probs=logits[0,len(prompt)-1:len(prompt)+len(label_ids)-1].float().log_softmax(-1)
                        scores=log_probs.gather(1,torch.tensor(label_ids).unsqueeze(1)).squeeze(1).tolist()
                    if not all(np.isfinite(scores)):
                        raise ValueError('Nonfinite label score')
                    token_scores.append(scores)
                cached[key]=token_scores
                write_json_atomic(cache,cached)
            sums=[sum(s) for s in token_scores]
            means=[sum(s)/len(s) for s in token_scores]
            rows.append({'cohort':cohort,'claim_id':row['claim_id'],'true_label':row['true_label'],
                         'greedy_label':row['raw_candidate_label'], 'sum_label':LABELS[int(np.argmax(sums))],
                         'mean_label':LABELS[int(np.argmax(means))], 'label_token_log_probs':token_scores,
                         'sum_log_likelihoods':sums,'mean_log_likelihoods':means})
            if position%10==0: print(f'{cohort}: {position+1}/300',flush=True)
    summary={name:{field:summarize([r for r in rows if r['cohort']==name],field) for field in ('greedy_label','sum_label','mean_label')} for name in cohorts}
    write_json_atomic(output/'predictions.json',rows)
    write_json_atomic(output/'decoder_summary.json',summary)
    write_json_atomic(output/'output_manifest.json',{p.name:sha256(p) for p in output.glob('*.json') if p.name!='output_manifest.json'})
    print(output)


if __name__=='__main__':
    main()
