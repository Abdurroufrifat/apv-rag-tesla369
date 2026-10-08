"""Post-hoc Qwen/multilingual-NLI agreement policy on fixed supplied evidence."""
import json
import math
from collections import Counter
from pathlib import Path

from apv_rag.splits import sha256,write_json_atomic

ROOT=Path(__file__).resolve().parents[1]
LABELS=('Supported','Refuted','Not Enough Evidence')


def require(condition,message):
    if not condition:raise ValueError(message)


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def receipt(folder):
    output=read(folder/'output_manifest.json')
    for name,digest in output.items():
        require(Path(name).name==name and sha256(folder/name)==digest,'Frozen output digest mismatch')
    return sha256(folder/'output_manifest.json')


def decide(generated,nli):
    if generated not in LABELS or nli not in LABELS:
        raise ValueError('Unexpected verdict label')
    return generated if generated==nli else None


def align(generated,nli_rows):
    left={};right={}
    for rows,target in ((generated,left),(nli_rows,right)):
        for row in rows:
            key=(row['file'],row['row'])
            if key in target:raise ValueError('Duplicate aligned row')
            target[key]=row
    require(set(left)==set(right),'Row coverage differs')
    joined=[]
    for file,row_id in sorted(left):
        q=left[file,row_id];n=right[file,row_id]
        for field in ('claim_id','english_page','true_label'):
            require(q[field]==n[field],f'Aligned source or label changed: {file}/{row_id}')
        probabilities=n['probabilities']
        require(type(probabilities) is list and len(probabilities)==3 and all(
            type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in probabilities)
            and abs(sum(probabilities)-1)<1e-6,'Invalid NLI probabilities')
        require(n['predicted_label']==LABELS[max(range(3),key=lambda i:probabilities[i])],
                'NLI label differs from declared Supported/Refuted/NEI probability order')
        a=decide(q['raw_candidate_label'],n['predicted_label'])
        joined.append({'file':file,'row':row_id,'claim_id':q['claim_id'],
            'english_page':q['english_page'],'language':q['language'],
            'translation_origin':q['translation_origin'],'true_label':q['true_label'],
            'qwen_label':q['raw_candidate_label'],'multilingual_nli_label':n['predicted_label'],
            'agreement_label':a})
    return joined


def metrics(rows):
    files={}
    for file in sorted({r['file'] for r in rows}):
        group=[r for r in rows if r['file']==file]
        n=len(group);accepted=[r for r in group if r['agreement_label'] is not None]
        accepted_correct=sum(r['agreement_label']==r['true_label'] for r in accepted)
        q_correct=sum(r['qwen_label']==r['true_label'] for r in group)
        nli_correct=sum(r['multilingual_nli_label']==r['true_label'] for r in group)
        files[file]={'rows':n,'language':group[0]['language'],
            'translation_origin':group[0]['translation_origin'],
            'accepted':len(accepted),'coverage':len(accepted)/n,
            'accepted_correct':accepted_correct,
            'accepted_wrong':len(accepted)-accepted_correct,
            'selective_accuracy':accepted_correct/len(accepted) if accepted else None,
            'all_claim_accuracy_abstentions_as_errors':accepted_correct/n,
            'qwen_correct':q_correct,'qwen_accuracy':q_correct/n,
            'multilingual_nli_correct':nli_correct,'multilingual_nli_accuracy':nli_correct/n,
            'rejected_qwen_correct':q_correct-accepted_correct}
    return files


def main():
    root=ROOT/'artifacts'
    qfolder=root/'xfever_generation_received';nfolder=root/'xfever_multilingual_received'
    qdigest=receipt(qfolder);ndigest=receipt(nfolder)
    qinput=read(qfolder/'input_manifest.json');ninput=read(nfolder/'input_manifest.json')
    require(qinput['files']==ninput['files'] and len(qinput['files'])==11,
            'Different frozen multilingual source files')
    previous=read(root/'xfever_generation_verification/audit_manifest.json')
    require(previous['source_output_manifest_sha256']==qdigest and
            previous['baseline_output_manifests']['xfever_multilingual_received']==ndigest,
            'Prior multilingual verifier binding differs')
    rows=align(read(qfolder/'predictions.json'),read(nfolder/'predictions.json'))
    outcome=metrics(rows)
    require(len(rows)==6600 and len(outcome)==11 and all(v['rows']==600 for v in outcome.values()),
            'Unexpected multilingual cohort coverage')
    out=root/'multilingual_agreement_replay_v1';out.mkdir(exist_ok=True)
    write_json_atomic(out/'predictions.json',rows)
    write_json_atomic(out/'summary.json',{'scope':'Post-hoc agreement on observed supplied-evidence multilingual rows; no new inference, retrieval, fitting or language calibration.',
        'source_output_manifests_sha256':{'generation':qdigest,'multilingual_nli':ndigest},
        'source_files_sha256':qinput['files'],'results':outcome})
    lines=['# Multilingual agreement replay','',
        'A fixed rule outputs Qwen’s raw verdict only when the existing multilingual NLI verdict matches it; otherwise it abstains. This is a counterfactual policy replay of the previously observed eleven supplied-evidence files, with 600 rows per file. Rule decisions use model outputs only; benchmark labels enter the reported metrics afterward. No model was fit, no new neural inference occurred and the result is exploratory because the models and multilingual cohorts have already been examined.','',
        '| File | Agreement coverage | Accepted correct / accepted | All-claim accuracy, abstentions errors | Qwen accuracy | NLI accuracy | Rejected correct Qwen answers |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for file,v in outcome.items():
        lines.append(f"| {file} | {v['accepted']}/600 | {v['accepted_correct']}/{v['accepted']} | {v['all_claim_accuracy_abstentions_as_errors']:.3f} | {v['qwen_accuracy']:.3f} | {v['multilingual_nli_accuracy']:.3f} | {v['rejected_qwen_correct']} |")
    lines+=['','Accepted accuracy uses only the agreed subset; its change from overall accuracy follows partly from abstaining. Correct Qwen answers are also rejected. The NLI model used supplied claim/evidence text with a 256-token pair limit; Qwen used a different clipping and prompt budget. Agreement is not independent source support, a trained evidence-sufficiency estimate or calibrated confidence. This does not establish multilingual retrieval, publisher authentication or factual explanation quality.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json_atomic(out/'audit_manifest.json',{
        'source_output_manifests_sha256':{'generation':qdigest,'multilingual_nli':ndigest},
        'script_sha256':sha256(Path(__file__)),
        'files':{p.name:sha256(p) for p in out.iterdir() if p.name!='audit_manifest.json'}})
    print('Multilingual agreement replay passed: 6600 aligned rows, 11 files.')


if __name__=='__main__':main()
