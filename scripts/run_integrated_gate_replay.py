"""Fixed-context experimental gate integration; existing generations preserved."""
import json
from importlib.metadata import version
from pathlib import Path
import numpy as np
from apv_rag.integrated_gate import collapse_context,probability_complete,apply_gate
from apv_rag.numeric_integrity_v2 import numeric_provenance_v2
from apv_rag.multilingual_nli import cen_order,normalize_model_scores
from apv_rag.nli_comparison import _model_files
from apv_rag.splits import sha256,write_json_atomic
from run_semantic_sufficiency import aggregate_features
from run_sentence_rag_scifact import summarize


def main():
    root=Path(__file__).resolve().parents[1];output=root/'artifacts/integrated_gate_replay_v1'
    if (output/'output_manifest.json').exists():raise FileExistsError('Completed integration exists')
    learned=root/'artifacts/semantic_sufficiency_v1'
    receipt=json.loads((learned/'output_manifest.json').read_text(encoding='utf-8'))
    for name in ('models.json','input_manifest.json'):
        if sha256(learned/name)!=receipt[name]:raise ValueError('Learned component checksum mismatch')
    models=json.loads((learned/'models.json').read_text(encoding='utf-8'))
    if set(models)!={'nli','embedding','combined'}:raise ValueError('Unexpected gate models')
    learned_meta=json.loads((learned/'input_manifest.json').read_text(encoding='utf-8'))
    paths={'nli':root/'models/nli-deberta-v3-small','embedding':root/'models/all-MiniLM-L6-v2'}
    for name,path in paths.items():
        if _model_files(path)!=learned_meta['models'][name]:raise ValueError('Feature model checksum mismatch')
    sources={};source_hashes={}
    for cohort,dirname in [('scifact','sentence_rag_scifact_v3'),('climate_retrieved','climate_rag_frozen_v1'),('climate_supplied','climate_supplied_evidence_v1')]:
        folder=root/'artifacts'/dirname
        hashes=json.loads((folder/'output_manifest.json').read_text(encoding='utf-8'))
        if sha256(folder/'predictions.json')!=hashes['predictions.json']:raise ValueError('Generation checksum mismatch')
        sources[cohort]=json.loads((folder/'predictions.json').read_text(encoding='utf-8'));source_hashes[cohort]=hashes['predictions.json']
        rows=sources[cohort]
        if not rows or len({str(r['claim_id']) for r in rows})!=len(rows):raise ValueError('Empty or duplicate cohort')
        if any(not collapse_context(r['evidence']) for r in rows):raise ValueError('Empty evidence context')
    identity={'sources':source_hashes,'learned_models_sha256':sha256(learned/'models.json'),'feature_models':learned_meta['models'],'code_sha256':{n:sha256(root/n) for n in ('scripts/run_integrated_gate_replay.py','src/apv_rag/integrated_gate.py','src/apv_rag/numeric_integrity_v2.py','scripts/run_semantic_sufficiency.py')},'protocol_sha256':sha256(root/'docs/INTEGRATED_GATE_REPLAY.md'),'packages':{n:version(n) for n in ('torch','transformers','sentence-transformers','numpy','scikit-learn')},'settings':{'seed':369,'threads':4,'dtype':'float32','nli_tokens':256,'embedding_tokens':384,'gate_threshold':.5,'copy_count':5}}
    output.mkdir(exist_ok=True);p=output/'input_manifest.json'
    if (output/'feature_cache.json').exists() and not p.exists():raise ValueError('Cache without identity; reuse refused')
    if p.exists() and json.loads(p.read_text(encoding='utf-8'))!=identity:raise ValueError('Integration identity changed')
    write_json_atomic(p,identity)
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4);torch.manual_seed(369);torch.use_deterministic_algorithms(True)
    nt=AutoTokenizer.from_pretrained(paths['nli'],local_files_only=True)
    nli=AutoModelForSequenceClassification.from_pretrained(paths['nli'],local_files_only=True,torch_dtype=torch.float32).eval()
    order=cen_order(nli.config.id2label)
    embedding=SentenceTransformer(str(paths['embedding']),device='cpu',local_files_only=True);embedding.float();embedding.max_seq_length=384
    cachepath=output/'feature_cache.json';cache=json.loads(cachepath.read_text(encoding='utf-8')) if cachepath.exists() else {}
    predictions=[];copy_audit=[]
    for cohort,rows in sources.items():
        for position,r in enumerate(rows):
            key=f"{cohort}:{r['claim_id']}";evidence=collapse_context(r['evidence'])
            if key not in cache:
                text=[e['text'] for e in evidence]
                tokens=nt(text,[r['shown_claim']]*len(text),return_tensors='pt',padding=True,truncation=True,max_length=256)
                with torch.inference_mode():scores=nli(**tokens).logits.float().softmax(-1).cpu().numpy()[:,order]
                scores=normalize_model_scores(scores.tolist())
                v=embedding.encode([r['shown_claim']]+text,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
                cache[key]={'scores_cen':scores,'cosines':(v[1:]@v[0]).tolist()}
                write_json_atomic(cachepath,cache)
            scores=cache[key]['scores_cen'];cosines=cache[key]['cosines']
            features=aggregate_features(scores,cosines)
            numeric=numeric_provenance_v2(r['generated_explanation'] or '',r['shown_claim'],r['evidence'])
            reasons=['numeric_value_absent'] if numeric['absent_from_inputs'] else []
            if not r['raw_candidate_label'] or not r['generated_explanation']:reasons.append('invalid_or_empty_generation')
            row={'cohort':cohort,'claim_id':r['claim_id'],'true_label':r['true_label'],'no_gate_label':r['raw_candidate_label'] if not reasons else None,'context_families':[str(e.get('family_id',e['id'])) for e in evidence],'gate_probabilities':{},'numeric_reasons':reasons,'features':features}
            # Duplicate only the first source to expose copy weighting effects.
            copies=[dict(evidence[0],id=f"copy{i}",family_id=evidence[0].get('family_id',evidence[0]['id'])) for i in range(5)]
            assert collapse_context(evidence+copies)==evidence
            duplicate_features=aggregate_features(scores+[scores[0]]*5,cosines+[cosines[0]]*5)
            audit={'cohort':cohort,'claim_id':r['claim_id'],'collapsed_context_unchanged':True,'without_collapse_probability_shift':{}}
            for name,m in models.items():
                prob=probability_complete(features,m);row['gate_probabilities'][name]=prob
                row[name+'_label']=apply_gate(r['raw_candidate_label'],reasons,prob)
                audit['without_collapse_probability_shift'][name]=probability_complete(duplicate_features,m)-prob
            predictions.append(row);copy_audit.append(audit)
            if position%20==0:print(f'{cohort}: {position+1}/{len(rows)}',flush=True)
    summary={cohort:{field:summarize([r for r in predictions if r['cohort']==cohort],field) for field in ('no_gate_label','nli_label','embedding_label','combined_label')} for cohort in sources}
    write_json_atomic(output/'predictions.json',predictions);write_json_atomic(output/'copy_audit.json',copy_audit);write_json_atomic(output/'metrics.json',summary)
    write_json_atomic(output/'output_manifest.json',{p.name:sha256(p) for p in output.glob('*.json') if p.name!='output_manifest.json'})
    print(output)


if __name__=='__main__':main()
