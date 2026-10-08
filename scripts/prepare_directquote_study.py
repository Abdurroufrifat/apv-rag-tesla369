"""Freeze an existing-label quotation attribution component study, no inference."""
import json,hashlib,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda b:hashlib.sha256(b).hexdigest()
def spans(tags):
    out=[];current=None
    for i,tag in enumerate(tags+['Out']):
        kind=tag[2:] if tag!='Out' else None
        if current and (kind!=current[0] or tag.startswith('B-')):
            out.append((current[0],current[1],i));current=None
        if kind and current is None:current=(kind,i)
    return out
def normalize(text):return ' '.join(re.findall(r'\w+',text.casefold()))
def main():
    source=ROOT/'data/external/directquote/frozen_v1/truecased.txt';out=ROOT/'artifacts/directquote_preflight_v1'
    if (out/'protocol.json').exists():raise FileExistsError('Frozen study exists')
    assert spans(['I-LeftSpeaker','I-LeftSpeaker','Out','B-Speaker'])==[('LeftSpeaker',0,2),('Speaker',3,4)]
    assert spans(['B-Unknown','I-Unknown','B-Unknown'])==[('Unknown',0,2),('Unknown',2,3)]
    # Existing claim text across every available benchmark source and saved model input.
    prior=set();sources=[]
    def walk(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in ('claim','claim_text','canonical_claim','text') and isinstance(v,str):prior.add(normalize(v))
                elif isinstance(v,(dict,list)):walk(v)
        elif isinstance(value,list):
            for v in value:walk(v)
    paths=list((ROOT/'data/external').rglob('*.json'))+list((ROOT/'data/external').rglob('*.jsonl'))+list((ROOT/'artifacts').rglob('model_inputs.json'))
    for p in paths:
        if 'directquote' in p.parts:continue
        try:
            if p.suffix=='.jsonl':
                for line in p.read_text(encoding='utf-8').splitlines():walk(json.loads(line))
            else:walk(json.loads(p.read_text(encoding='utf-8')))
            sources.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p.read_bytes())})
        except (ValueError,UnicodeError):continue
    eligible=[];seen=set();overlap=0;valid={'Out','B-Speaker','I-Speaker','B-LeftSpeaker','I-LeftSpeaker','B-RightSpeaker','I-RightSpeaker','B-Unknown','I-Unknown'}
    blocks=re.split(r'\n\s*\n',source.read_text(encoding='utf-8').strip())
    for block in blocks:
        rows=[line.rsplit(None,1) for line in block.splitlines() if line.strip()]
        assert all(len(row)==2 and row[1] in valid for row in rows)
        tokens=[x[0] for x in rows];tags=[x[1] for x in rows];ranges=spans(tags)
        quotes=[x for x in ranges if x[0]!='Speaker'];speakers=[x for x in ranges if x[0]=='Speaker']
        if len(quotes)!=1 or len(tokens)>180:continue
        kind,a,b=quotes[0]
        if b-a<6:continue
        if kind=='Unknown':
            if speakers:continue
            gold=None
        else:
            if len(speakers)!=1:continue
            _,sa,sb=speakers[0]
            if not ((kind=='LeftSpeaker' and sb<=a) or (kind=='RightSpeaker' and sa>=b)):continue
            gold=' '.join(tokens[sa:sb])
        paragraph=' '.join(tokens);quote=' '.join(tokens[a:b]);key=sha(normalize(paragraph).encode())
        if key in seen:continue
        if normalize(paragraph) in prior or normalize(quote) in prior:overlap+=1;continue
        seen.add(key);eligible.append({'id':'DQ-'+key[:20],'paragraph':paragraph,'quote':quote,'speaker':gold,'reference_type':kind,'group':key})
    selected=[]
    for unknown in (False,True):
        pool=sorted([x for x in eligible if (x['speaker'] is None)==unknown],key=lambda x:sha(('369'+x['group']).encode()))
        if len(pool)<50:raise ValueError('Insufficient eligible stratum')
        selected+=pool[:50]
    selected.sort(key=lambda x:x['id']);out.mkdir(exist_ok=True)
    inputs=[{'id':x['id'],'paragraph':x['paragraph'],'quote':x['quote']} for x in selected]
    gold=[{'id':x['id'],'speaker':x['speaker'],'reference_type':x['reference_type'],'group':x['group']} for x in selected]
    for n,value in [('inputs.json',inputs),('gold.json',gold),('prior_source_inventory.json',sources)]:
        (out/n).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    protocol={'scope':'New quotation-to-speaker attribution component using published token labels. Not source-family independence, historical authenticity or full APV-RAG confirmation. No model results inspected.', 'records':100,'selection':'One quote per paragraph; known quote has one speaker on annotated side; unknown quote has no annotated speaker. Six or more quote tokens, at most180 paragraph tokens. Deterministic hash selection50known+50unknown; balanced filtered diagnostic, not native dataset distribution.', 'official_split':False,'training_records':0,'inputs':'paragraph and supplied quotation only; speaker labels excluded','primary_metrics':['exact normalized speaker accuracy including null abstention','false attribution on Unknown','known speaker accuracy'],'overlap':'Exact normalized paragraph/quote excluded against all available prior source claims/text. No semantic overlap, speaker or article-disjointness established; article IDs absent from this released token file. No full provenance label claim. Pretrained-model exposure unknown.', 'prior_unique_texts':len(prior),'prior_sources_scanned':len(sources),'exact_overlap_excluded':overlap,'eligible_paragraphs':len(eligible),'source_sha256':sha(source.read_bytes()),'code_sha256':sha(Path(__file__).read_bytes())}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (out/'receipt.json').write_text(json.dumps({p.name:sha(p.read_bytes()) for p in out.iterdir() if p.is_file() and p.name!='receipt.json'},indent=2))
    print(json.dumps(protocol,indent=2))
if __name__=='__main__':main()
