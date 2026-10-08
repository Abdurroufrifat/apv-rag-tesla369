"""Verify exact-copy input invariance against saved control predictions."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from apv_rag.exact_copy_removal import remove_exact_source_copies
from apv_rag.repetition_stress import inject_derivative_copies
from apv_rag.splits import sha256,write_json_atomic
from run_fresh_copy_confirmation import check_freeze


def main():
    check_freeze()
    prior=ROOT/'artifacts/fresh_copy_confirmation_v1';out=ROOT/'artifacts/exact_copy_removal_v1'
    if out.exists():raise FileExistsError('Completed diagnostic exists; refusing overwrite')
    def read(path):return json.loads(path.read_text(encoding='utf-8'))
    receipt=read(prior/'evaluation_receipt.json')
    for n,d in receipt['outputs'].items():
        if sha256(prior/n)!=d:raise ValueError('Prior evaluation changed')
    inputs=read(prior/'model_inputs.json');predictions=read(prior/'predictions.json');gold=read(prior/'gold.json')
    counts=(0,1,5,10,25);source=[r['record'] for r in inputs]
    out.mkdir()
    deps=[prior/n for n in ('model_inputs.json','predictions.json','gold.json','evaluation_receipt.json')]
    deps += [Path(__file__).resolve(),ROOT/'src/apv_rag/exact_copy_removal.py',ROOT/'src/apv_rag/repetition_stress.py']
    write_json_atomic(out/'protocol.json',{'scope':'Post-hoc exact-copy input-identity diagnostic on 48 already observed supplied-evidence claims; not independent confirmation.',
        'rule':'Within each question, preserve first identical complete answer object with a nonempty source URL and text. Preserve missing-source answers, distinct text, distinct metadata and other questions.',
        'copies':list(counts),'fitting':False,'neural_inference':False,'promotion':False,
        'inputs_and_code':{p.relative_to(ROOT).as_posix():sha256(p) for p in deps}})
    for count in counts:
        restored=[remove_exact_source_copies(inject_derivative_copies(row,count)) for row in source]
        if restored!=source:raise ValueError('Clean evidence not exactly restored')
    clean=[r for r in predictions if r['model']=='control' and r['copies']==0]
    if [r['claim_id'] for r in clean]!=[r['claim_id'] for r in inputs]:raise ValueError('Saved control alignment differs')
    # Input identity proves the deterministic frozen control receives the original
    # records. Its saved outputs are reused explicitly, not presented as new inference.
    inferred=[]
    for count in counts:
        inferred.extend(dict(r,model='control_exact_copy_removal',copies=count) for r in clean)
    truth={r['claim_id']:r['label'] for r in gold};labels=('Supported','Refuted','Not Enough Evidence','Conflicting Evidence/Cherrypicking')
    fp=0;correct=0;cm=[[0]*4 for _ in range(4)]
    for r in clean:
        pred=max(range(4),key=lambda i:r['probabilities'][i]);actual=labels.index(truth[r['claim_id']]);cm[actual][pred]+=1
        correct+=pred==actual;fp+=actual!=0 and pred==0
    f1=[]
    for i in range(4):
        denominator=sum(cm[i])+sum(row[i] for row in cm);f1.append(2*cm[i][i]/denominator if denominator else 0)
    summary={'claims':len(source),'copy_counts_checked':list(counts),'record_identity_checks':len(source)*len(counts),
        'original_evidence_answers':sum(len(q['answers']) for row in source for q in row['questions']),
        'original_answers_lost':0,'injected_25_copy_answers_removed':25*len(source),
        'false_supported_answers':fp,'non_support_reference_claims':sum(r['label']!='Supported' for r in gold),
        'correct':correct,'accuracy':correct/len(source),'macro_f1':sum(f1)/4,
        'inference_status':'Saved deterministic clean-control outputs reused after exact model-input equality checks; no new prediction computation.',
        'scope':'Exact same-source copying only; paraphrases, different mirror URLs, publisher authentication and factual explanation truth are not addressed.'}
    write_json_atomic(out/'predictions_by_input_identity.json',inferred);write_json_atomic(out/'summary.json',summary)
    write_json_atomic(out/'receipt.json',{n:sha256(out/n) for n in ('protocol.json','predictions_by_input_identity.json','summary.json')})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
