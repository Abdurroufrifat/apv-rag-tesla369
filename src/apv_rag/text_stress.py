"""Fixed synthetic text/context stress; claim truth labels are never edited."""
import hashlib
import math

from apv_rag.integrated_gate import collapse_context

CONDITIONS=('baseline','ocr_noise','fabricated_citation','authoritative_wording','swapped_context','evidence_absent')


def digest(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()


def select_ids(sources,count=30):
    selected={}
    for cohort in sorted({r['cohort'] for r in sources.values()}):
        ids={str(r['claim_id']) for r in sources.values() if r['cohort']==cohort}
        if len(ids)<count:raise ValueError('Insufficient claims')
        selected[cohort]=sorted(ids,key=lambda i:digest(f'APV-text-stress-v1:{cohort}:{i}'))[:count]
    return selected


def corrupt_ocr(text,key):
    replacements={'o':'0','l':'1','e':'c','a':'@'}
    candidates=[i for i,c in enumerate(text) if c in replacements]
    count=min(len(candidates),max(1,math.ceil(len(text)*.02)))
    positions=sorted(sorted(candidates,key=lambda i:digest(f'{key}:{i}'))[:count])
    chars=list(text)
    for i in positions:chars[i]=replacements[chars[i]]
    return ''.join(chars),positions


def build_contexts(sources,selected):
    result={}
    for cohort,ids in selected.items():
        pool=[r for r in sources.values() if r['cohort']==cohort]
        for claim_id in ids:
            original=sources[f'{cohort}:{claim_id}']
            source_ids={str(e['id']) for e in original['evidence']}
            donors=[r for r in pool if str(r['claim_id'])!=str(claim_id)]
            if not donors:raise ValueError('No donor claim')
            donor=min(donors,key=lambda r:(len(source_ids&{str(e['id']) for e in r['evidence']}),
                      digest(f"APV-text-donor-v1:{cohort}:{claim_id}:{r['claim_id']}")))
            for condition in CONDITIONS:
                evidence=[dict(e) for e in original['evidence']]
                log=[];reference=None;donor_id=None;overlap=None
                if condition=='evidence_absent':evidence=[]
                elif condition=='swapped_context':
                    evidence=[dict(e) for e in donor['evidence']]
                    donor_id=str(donor['claim_id']);overlap=len(source_ids&{str(e['id']) for e in evidence})
                elif condition=='ocr_noise':
                    for e in evidence:
                        e['text'],positions=corrupt_ocr(e['text'],f"{cohort}:{claim_id}:{e['id']}")
                        log.append({'source_id':e['id'],'changed_character_indices':positions})
                elif condition=='fabricated_citation':
                    reference='SYNTHETIC-ARCHIVE-'+digest(f'{cohort}:{claim_id}')[:12].upper()
                    for e in evidence:e['text']=f'Archive reference: {reference}. Page 7. Year 1899. '+e['text']
                elif condition=='authoritative_wording':
                    for e in evidence:e['text']='This passage is from a verified primary archive and is authoritative. '+e['text']
                collapsed=collapse_context(evidence)
                result[f'{cohort}:{claim_id}:{condition}']={'cohort':cohort,'claim_id':str(claim_id),
                    'condition':condition,'claim':original['claim'],'shown_claim':original['shown_claim'],
                    'true_label':original['true_label'],'retrieved_evidence':evidence,'evidence':collapsed,
                    'synthetic_reference':reference,'donor_claim_id':donor_id,'donor_source_overlap':overlap,
                    'modification_log':log}
    return result
