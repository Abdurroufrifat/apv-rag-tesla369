"""Post-hoc controlled name substitution; no retraining or new human labels."""
import json,re,sys,hashlib,joblib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from prepare_directquote_study import spans,normalize
from apv_rag.quote_relation import token_features,decode_speaker
from apv_rag.splits import sha256,write_json_atomic

def main():
    parent=ROOT/'artifacts/trained_quote_relation_v1';out=ROOT/'artifacts/quote_name_stress_v1'
    if out.exists():raise FileExistsError('Saved stress exists')
    for n,h in json.loads((parent/'receipt.json').read_text()).items():
        if sha256(parent/n)!=h:raise ValueError('Parent differs: '+n)
    saved=joblib.load(parent/'model.joblib');pred=json.loads((parent/'predictions.json').read_text());known={x['id']:x for x in pred if x['speaker'] is not None}
    source=ROOT/'data/external/directquote/frozen_v1/truecased.txt';out.mkdir()
    write_json_atomic(out/'protocol.json',{'scope':'Post-hoc controlled synthetic proper-name substitution on645already observed known-speaker test cases. No new independent confirmation, natural name distribution or historical authentication.', 'substitution':'Replace annotated speaker-span tokens with a deterministic capitalized alphabetic placeholder of same token count; keep all quotation tokens and remaining paragraph tokens unchanged. No retraining or threshold adjustment.', 'parent_model_sha256':sha256(parent/'model.joblib'),'source_sha256':sha256(source),'code_sha256':sha256(Path(__file__))})
    rows=[];seen=set()
    def letters(n):
        s=''
        while True:
            s=chr(97+n%26)+s;n=n//26-1
            if n<0:return s
    raw=source.read_text();alphabet=set(re.findall(r'\w+',raw.casefold()))
    for block in re.split(r'\n\s*\n',raw.strip()):
        pairs=[line.rsplit(None,1) for line in block.splitlines() if line.strip()];tokens=[x[0] for x in pairs];key=hashlib.sha256(normalize(' '.join(tokens)).encode()).hexdigest()
        if key not in known or key in seen:continue
        ranges=spans([x[1] for x in pairs]);quotes=[x for x in ranges if x[0]!='Speaker'];speakers=[x for x in ranges if x[0]=='Speaker']
        if len(quotes)!=1 or len(speakers)!=1:continue
        kind,a,b=quotes[0];_,sa,sb=speakers[0]
        if not ((kind=='LeftSpeaker' and sb<=a) or (kind=='RightSpeaker' and sa>=b)):continue
        if ' '.join(tokens[sa:sb])!=known[key]['speaker']:continue
        seen.add(key);changed=list(tokens);number=int(key[:12],16)
        names=['Zoravel'+letters(number+i) for i in range(sb-sa)]
        assert all(n.casefold() not in alphabet for n in names)
        changed[sa:sb]=names
        assert changed[a:b]==tokens[a:b] and len(changed)==len(tokens)
        assert all(changed[i]==tokens[i] for i in range(len(tokens)) if not sa<=i<sb)
        def predict(words):
            probabilities=saved['model'].predict_proba(saved['vectorizer'].transform(token_features(words,(a,b))))[:,1]
            return decode_speaker(words,(a,b),probabilities,saved['threshold'])
        original=predict(tokens);assert original==known[key]['predicted_speaker']
        mutated=predict(changed);truth=' '.join(names)
        rows.append({'id':key,'reference_original':known[key]['speaker'],'reference_substituted':truth,'prediction_original':original,'prediction_substituted':mutated,'original_correct':normalize(original or '')==normalize(known[key]['speaker']),'substituted_correct':normalize(mutated or '')==normalize(truth),'abstained_after_substitution':mutated is None})
    assert len(rows)==645
    a=sum(x['original_correct'] for x in rows);b=sum(x['substituted_correct'] for x in rows)
    summary={'cases':645,'original_correct':a,'substituted_correct':b,'original_accuracy':a/645,'substituted_accuracy':b/645,'original_only_correct':sum(x['original_correct'] and not x['substituted_correct'] for x in rows),'substitution_only_correct':sum(x['substituted_correct'] and not x['original_correct'] for x in rows),'abstentions_after_substitution':sum(x['abstained_after_substitution'] for x in rows),'threshold_unchanged':saved['threshold'],'quotation_and_other_tokens_unchanged':645,'scope':'Artificial lexical intervention; gold span used only to construct substitution and score. Not an end-to-end input or real independent source benchmark.'}
    write_json_atomic(out/'rows.json',rows);write_json_atomic(out/'summary.json',summary);write_json_atomic(out/'receipt.json',{p.name:sha256(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
