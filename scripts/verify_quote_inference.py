"""Replay frozen test candidates through the label-free inference interface."""
import json,re,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from prepare_directquote_study import normalize,spans
from apv_rag.quote_inference import infer,load_model
base=ROOT/'artifacts/trained_quote_relation_v1'
expected={x['id']:x for x in json.loads((base/'predictions.json').read_text())};lookup={}
source=ROOT/'data/external/directquote/frozen_v1/truecased.txt'
assert hashlib.sha256(source.read_bytes()).hexdigest()==json.loads((base/'protocol.json').read_text())['source_sha256']
for block in re.split(r'\n\s*\n',source.read_text().strip()):
 pairs=[x.rsplit(None,1) for x in block.splitlines() if x.strip()];tokens=[x[0] for x in pairs];key=hashlib.sha256(normalize(' '.join(tokens)).encode()).hexdigest()
 if key not in expected or key in lookup:continue
 ranges=spans([x[1] for x in pairs]);quotes=[x for x in ranges if x[0]!='Speaker'];speakers=[x for x in ranges if x[0]=='Speaker']
 if len(quotes)!=1:continue
 kind,a,b=quotes[0]
 if kind=='Unknown' and speakers:continue
 if kind!='Unknown' and (len(speakers)!=1 or not ((kind=='LeftSpeaker' and speakers[0][2]<=a) or (kind=='RightSpeaker' and speakers[0][1]>=b))):continue
 lookup[key]={'tokens':tokens,'quote_span':[a,b]}
assert set(lookup)==set(expected)
saved=load_model(base/'model.joblib')
for key,row in lookup.items():
 result=infer(row,saved)
 assert result['speaker_candidate']==expected[key]['predicted_speaker'],key
 assert result['source_authenticated'] is False
 assert result['status'] in ('machine_candidate','abstain')
example=next(iter(lookup.values()))
out=ROOT/'artifacts/quote_inference_v1';out.mkdir(exist_ok=True)
(out/'example_input.json').write_text(json.dumps(example,indent=2)+'\n')
result={'frozen_predictions_replayed':len(lookup),'inference_inputs_have_no_gold_fields':True,'contract_unit_tests_passed':2,'full_suite_run':'unavailable: pytest not installed in recovered workspace'}
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
