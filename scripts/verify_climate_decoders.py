"""Replay fixed-context decoder arithmetic and scores without model inference."""
import json
import math
from pathlib import Path
import numpy as np
from compare_climate_verdict_decoders import summarize
from apv_rag.generative_rag import LABELS
from apv_rag.splits import sha256, write_json_atomic


def main():
    root=Path(__file__).resolve().parents[1]
    folder=root/'artifacts/climate_decoder_received'
    load=lambda n:json.loads((folder/n).read_text(encoding='utf-8'))
    manifest=load('output_manifest.json')
    for name,digest in manifest.items():
        assert sha256(folder/name)==digest,name
    identity=load('input_manifest.json')
    assert sha256(root/'scripts/compare_climate_verdict_decoders.py')==identity['code_sha256']
    assert sha256(root/'docs/CLIMATE_DECODER_COMPARISON.md')==identity['protocol_sha256']
    originals={}
    for cohort,name in [('retrieved','climate_rag_received'),('supplied','climate_supplied_received')]:
        path=root/'artifacts'/name/'predictions.json'
        assert sha256(path)==identity['source_hashes'][cohort]
        originals[cohort]={r['claim_id']:r for r in json.loads(path.read_text(encoding='utf-8'))}
    rows=load('predictions.json');cache=load('score_cache.json')
    assert len(rows)==600 and len({(r['cohort'],r['claim_id']) for r in rows})==600
    for r in rows:
        old=originals[r['cohort']][r['claim_id']]
        assert r['true_label']==old['true_label'] and r['greedy_label']==old['raw_candidate_label']
        scores=r['label_token_log_probs']
        assert scores==cache[f"{r['cohort']}:{r['claim_id']}"] and len(scores)==3
        assert all(s and all(math.isfinite(x) and x<=0 for x in s) for s in scores)
        sums=[sum(s) for s in scores];means=[sum(s)/len(s) for s in scores]
        assert sums==r['sum_log_likelihoods'] and means==r['mean_log_likelihoods']
        assert r['sum_label']==LABELS[int(np.argmax(sums))]
        assert r['mean_label']==LABELS[int(np.argmax(means))]
    summary={name:{field:summarize([r for r in rows if r['cohort']==name],field) for field in ('greedy_label','sum_label','mean_label')} for name in originals}
    assert summary==load('decoder_summary.json')
    out=root/'artifacts/climate_decoder_verification';out.mkdir(exist_ok=True)
    write_json_atomic(out/'verified_summary.json',summary)
    (out/'RESULTS.md').write_text('''# Fixed-context decoder diagnostic

600 records across two300-claim contexts passed output hashes, source/code/protocol identity checks, source-label matching, score-cache consistency, finite log-probability checks, arithmetic/argmax replay and metric recomputation. Neural logits, tokenizer alignment and inference were not independently rerun.

| Context | Greedy accuracy | Sum log-likelihood | Mean log-likelihood |
|---|---:|---:|---:|
| Retrieved | 38.00% | 35.67% | 36.33% |
| Supplied evidence | 40.00% | 35.67% | 36.33% |

Primary summed likelihood and secondary mean likelihood both underperform the original greedy policy on both contexts. This diagnostic does not support switching decoders as a remedy for weak transfer. These are raw verdict metrics; old explanations cannot be attached to changed verdicts.

The comparison is post-hoc on already observed claims. Likelihood uses raw logits while generation applies its generation policy, so it is not a search-only causal test. No independent confirmation, calibrated probabilities, source authentication or explanation verification claim. Original frozen scores preserved. Stop further decoder tuning on this cohort; retain the failure in the experiment record. No manuscript or GitHub push.
''',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
