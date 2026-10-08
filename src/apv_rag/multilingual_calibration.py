"""Exploratory claim-group cross-fitting of final accepted-answer correctness."""
import hashlib
import math
import numpy as np

from apv_rag.fever_nli import LABEL_MAP
from apv_rag.fever_pipeline import POLICIES,fit_correctness,correctness_probability,reliability


def require(ok,message):
    if not ok:raise ValueError(message)


def claim_folds(rows):
    ids={r['claim_id'] for r in rows}
    require(len(ids)>=5 and all(type(i) is int for i in ids),'At least five integer claim IDs required')
    ordered=sorted(ids,key=lambda i:(hashlib.sha256(f'apv-multilingual-calibration:369:{i}'.encode()).hexdigest(),i))
    return {str(i):position%5 for position,i in enumerate(ordered)}


def validate(rows,gold,folds):
    require(rows and gold,'Nonempty predictions and gold required')
    targets={g['key']:g for g in gold}
    require(len(targets)==len(gold) and {r['key'] for r in rows}==set(targets),'Gold/prediction keys differ')
    require(len(rows)==len(gold)*len(POLICIES) and
        len({(r['key'],r['policy']) for r in rows})==len(rows) and
        all(r['policy'] in POLICIES for r in rows),'Unique four-policy records required')
    require(set(folds)=={str(g['claim_id']) for g in gold} and set(folds.values())==set(range(5)),'Fold coverage differs')
    parallel={}
    for g in gold:
        require(g['label'] in LABEL_MAP,'Unknown benchmark label')
        require(g['claim_id'] not in parallel or parallel[g['claim_id']]==g['label'],'Parallel labels differ')
        parallel[g['claim_id']]=g['label']
    for r in rows:
        require(r['claim_id']==targets[r['key']]['claim_id'],'Claim ID/gold binding differs')
        score=r['uncalibrated_english_nli_label_score']
        if r['candidate_label'] is None:require(score is None,'Abstention has a correctness score')
        else:
            require(r['candidate_label'] in LABEL_MAP.values() and isinstance(score,(int,float)) and
                np.isfinite(score) and 0<=score<=1,'Accepted answer lacks finite raw score')
    return targets


def fit_folds(rows,gold,folds):
    targets=validate(rows,gold,folds);fits={}
    for fold in range(5):
        development=sorted(int(i) for i,f in folds.items() if f!=fold)
        heldout=sorted(int(i) for i,f in folds.items() if f==fold)
        models={};controls={}
        for policy in POLICIES:
            train=[r for r in rows if r['policy']==policy and r['candidate_label'] is not None and
                   folds[str(r['claim_id'])]!=fold]
            models[policy]=fit_correctness([r['uncalibrated_english_nli_label_score'] for r in train],
                [r['candidate_label']==LABEL_MAP[targets[r['key']]['label']] for r in train])
            model=models[policy];n=model['accepted_development_answers']
            controls[policy]={'method':'constant_beta_1_1_prevalence' if n else 'unavailable_no_accepted_development_answers',
                'accepted_development_answers':n,'correct_development_answers':model['correct_development_answers'],
                'probability':(model['correct_development_answers']+1)/(n+2) if n else None}
        fits[str(fold)]={'development_claim_ids':development,'held_out_claim_ids':heldout,'policies':models,
            'prevalence_controls':controls}
    return fits


def predict_folds(rows,folds,fits):
    require(set(fits)=={str(i) for i in range(5)},'Expected five calibrator folds')
    predictions=[]
    for r in rows:
        fold=folds[str(r['claim_id'])];fit=fits[str(fold)]
        require(r['claim_id'] in fit['held_out_claim_ids'] and r['claim_id'] not in fit['development_claim_ids'],
                'Held-out claim leaked into calibrator fitting')
        raw=r['uncalibrated_english_nli_label_score']
        p=None if r['candidate_label'] is None else correctness_probability(raw,fit['policies'][r['policy']])
        constant=None if r['candidate_label'] is None else fit['prevalence_controls'][r['policy']]['probability']
        predictions.append({n:r[n] for n in ('key','file','claim_id','policy','candidate_label')}|
            {'fold':fold,'raw_generated_label_nli_score':raw,'correctness_confidence':p,
             'prevalence_control_confidence':constant})
    return predictions


def metrics(rows,targets):
    accepted=[r for r in rows if r['candidate_label'] is not None]
    evaluable=[r for r in accepted if r['correctness_confidence'] is not None]
    correct=lambda r:r['candidate_label']==LABEL_MAP[targets[r['key']]['label']]
    result={'queries':len(rows),'accepted_answers':len(accepted),'correct_answers':sum(correct(r) for r in accepted),
        'coverage':len(accepted)/len(rows),'accuracy_all_queries_abstentions_as_errors':sum(correct(r) for r in accepted)/len(rows),
        'evaluable_answers':len(evaluable),'unavailable_accepted_confidences':len(accepted)-len(evaluable)}
    if not evaluable:return result|{'status':'no_evaluable_accepted_confidences'}
    y=np.array([correct(r) for r in evaluable]);raw=np.array([r['raw_generated_label_nli_score'] for r in evaluable])
    calibrated=np.array([r['correctness_confidence'] for r in evaluable])
    a=reliability(raw,y);b=reliability(calibrated,y)
    constant=reliability([r['prevalence_control_confidence'] for r in evaluable],y)
    selected={}
    for name,values in [('raw',raw),('calibrated',calibrated)]:
        order=np.argsort(-values,kind='stable');selected[name]={}
        for fraction in (.5,.8,1.0):
            n=math.ceil(len(evaluable)*fraction)
            selected[name][str(fraction)]={'selected_answers':n,'accuracy':float(y[order[:n]].mean()),
                'coverage_all_queries':n/len(rows)}
    return result|{'status':'evaluated','raw_score_metrics':a,'cross_fitted_metrics':b,
        'calibrated_minus_raw':{n:b[n]-a[n] for n in ('binary_brier','binary_nll','ece_15_bins')},
        'cross_fitted_prevalence_control_metrics':constant,
        'calibrated_minus_prevalence':{n:b[n]-constant[n] for n in ('binary_brier','binary_nll','ece_15_bins')},
        'confidence_ranking_among_evaluable_accepted':selected}


def summarize(predictions,gold):
    targets={g['key']:g for g in gold}
    require(len(targets)==len(gold) and len(predictions)==len(gold)*4 and
        len({(r['key'],r['policy']) for r in predictions})==len(predictions),'Output/gold coverage differs')
    files=list(dict.fromkeys(r['file'] for r in predictions))
    perfile={name:{p:metrics([r for r in predictions if r['file']==name and r['policy']==p],targets)
        for p in POLICIES} for name in files}
    return {'scope':'exploratory five-fold claim-group correctness calibration on observed multilingual outputs',
        'underlying_claim_ids':len({g['claim_id'] for g in gold}),'variant_queries':len(gold),
        'policy_records':len(predictions),'folds':5,'per_file':perfile,
        'pooled_descriptive':{p:metrics([r for r in predictions if r['policy']==p],targets) for p in POLICIES},
        'neural_inference_run':False,'verdicts_gates_or_thresholds_changed':False,
        'independent_confirmation':False,'deployment_calibrator_fitted':False,
        'limitations':['All variants of a claim are excluded together from the fit producing its evaluated confidence.',
            'Each policy uses one pooled logistic C=1 model across languages, without target-language tuning.',
            'A Beta(1,1) development-correctness prevalence control uses the same excluded-claim folds.',
            'The sixty claims and model outcomes were previously observed; fold validation is exploratory, not new independent confirmation.',
            'Pooled counts and losses summarize variant rows and do not make the eleven translations independent observations.',
            'Raw label NLI scores estimate entailment of generated labels, not calibrated answer correctness.',
            'Reliability and ranking are evaluated only among accepted answers with available confidence; abstention behavior is unchanged.',
            'No confidence threshold, policy or model is selected from these outcomes. There is no production fit on all sixty claims.',
            'The target-derived closed excerpt pool, English gate features and lexical numeric guard limitations remain.',
            'Shared excerpt dependence across different claim IDs is not separated by these folds; no significance tests or confidence intervals are claimed.',
            'Cross-fitted confidence does not establish calibrated deployment, publisher authentication or explanation truth.']}
