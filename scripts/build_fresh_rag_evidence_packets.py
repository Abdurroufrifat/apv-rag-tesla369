"""Reuse frozen-snapshot evidence displays for verified fresh climate RAG rows."""
import copy
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from apv_rag.evidence_display import build_evidence_display,verify_evidence_display
from apv_rag.integrated_gate import collapse_context
from apv_rag.splits import sha256,write_json_atomic


def main():
    received=ROOT/'artifacts/fresh_climate_rag_received'
    verification=ROOT/'artifacts/fresh_climate_rag_verification_v1'
    previous=json.loads((verification/'receipt.json').read_text(encoding='utf-8'))
    for name,digest in previous['received_files'].items():
        if sha256(received/name)!=digest:raise ValueError('Verified received file changed: '+name)
    if sha256(verification/'verification.json')!=previous['verification_sha256']:
        raise ValueError('Prior verification changed')
    rows=json.loads((received/'predictions.json').read_text(encoding='utf-8'))
    source=ROOT/'data/external/climate_fever/frozen_v1/corpus.jsonl'
    identity=json.loads((received/'input_manifest.json').read_text(encoding='utf-8'))
    if sha256(source)!=identity['source_sha256']['corpus.jsonl']:
        raise ValueError('Pinned corpus changed')
    corpus={str(r['doc_id']):r for r in map(json.loads,source.read_text(encoding='utf-8').splitlines())}
    packets=[];mutations=0;entailed=[];bracketed=0
    for original in rows:
        # Explicit allowlist: gold labels and free-form explanations never enter builder.
        row={'claim_id':original['claim_id'],'claim':original['claim'],
             'retrieved_evidence':original['evidence'],'evidence':collapse_context(original['evidence']),
             'candidate_label':original['structural_candidate_label']}
        packet=build_evidence_display(row,corpus)
        verify_evidence_display(packet,row,corpus)
        if not packet['source_snapshot_bound']:
            raise ValueError('A verified fresh context failed snapshot binding')
        if packet['status']=='machine_candidate':
            if packet['candidate_label']!=original['structural_candidate_label']:
                raise ValueError('Evidence output changed upstream verdict')
            for field,value in (('quote','FABRICATED EXCERPT NEVER IN SNAPSHOT'),('citation_id','unknown_document'),('excerpt_sha256','0'*64),('source_doc_sha256','0'*64)):
                changed=copy.deepcopy(packet);changed['items'][0][field]=value
                try:verify_evidence_display(changed,row,corpus)
                except ValueError:mutations+=1
                else:raise ValueError('Modified evidence packet accepted: '+field)
        packets.append(packet)
        bracketed+=bool(re.findall(r'\[([^\[\]]+)\]',original['generated_explanation'] or ''))
        entailed.extend(a['max_entailment'] for a in original['explanation_nli_audit'])
    result={'claims':len(rows),'snapshot_bound_contexts':sum(p['source_snapshot_bound'] for p in packets),
        'machine_candidate_packets':sum(p['status']=='machine_candidate' for p in packets),
        'upstream_abstentions':sum(p['status']=='abstain' for p in packets),
        'literal_cited_excerpts':sum(len(p['items']) for p in packets),
        'modified_packet_controls_rejected':mutations,
        'generated_explanations_with_bracketed_references':bracketed,
        'bracketed_reference_scope':'Frozen prompt did not require references; absence is descriptive, not an accuracy failure.',
        'nli_sentences':len(entailed),'mean_max_sentence_entailment':sum(entailed)/len(entailed),
        'minimum_max_sentence_entailment':min(entailed),'maximum_max_sentence_entailment':max(entailed),
        'nli_scope':'Unthresholded saved machine scores; not factual truth labels.',
        'freeform_explanations_in_packets':False,'verdicts_changed':0,'new_model_calls':0,
        'publisher_authentication_verified':False,'factual_explanation_truth_verified':False,
        'scope':'Optional literal evidence display on saved outputs; no accuracy improvement or completed original charter claimed.'}
    output=ROOT/'artifacts/fresh_rag_evidence_packets_v1';output.mkdir(exist_ok=True)
    write_json_atomic(output/'packets.json',packets);write_json_atomic(output/'summary.json',result)
    inputs=[source,received/'predictions.json',received/'input_manifest.json',verification/'receipt.json',verification/'verification.json']
    code=[Path(__file__).resolve(),ROOT/'src/apv_rag/evidence_display.py',ROOT/'src/apv_rag/source_snapshot_guard.py',ROOT/'src/apv_rag/integrated_gate.py',ROOT/'src/apv_rag/sentence_context.py']
    write_json_atomic(output/'receipt.json',{'inputs':{p.relative_to(ROOT).as_posix():sha256(p) for p in inputs},
        'code':{p.relative_to(ROOT).as_posix():sha256(p) for p in code},
        'outputs':{n:sha256(output/n) for n in ('packets.json','summary.json')}})
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
