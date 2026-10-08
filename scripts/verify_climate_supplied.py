"""Replay received RAG integrity, retrieval and scoring without neural inference."""
import json
import math
from collections import Counter
from pathlib import Path
from run_climate_supplied_evidence import summarize
from apv_rag.generative_rag import LABELS
from apv_rag.numeric_integrity_v2 import numeric_provenance_v2 as numeric_provenance
from apv_rag.splits import sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / 'artifacts/climate_supplied_received'
    def load(name):
        return json.loads((folder / name).read_text(encoding="utf-8"))
    hashes = load('output_manifest.json')
    for name in ('predictions.json', 'generation_summary.json', 'input_manifest.json'):
        assert sha256(folder / name) == hashes[name], name
    identity = load('input_manifest.json')
    for name, digest in identity['code_sha256'].items():
        assert sha256(root / name) == digest, name
    assert sha256(root / 'docs/CLIMATE_SUPPLIED_EVIDENCE.md') == identity['protocol_sha256']
    source = root / 'data/external/climate_fever/frozen_v1'
    for name, digest in identity['source_sha256'].items():
        assert sha256(source / name) == digest, name
    assert sha256(source / 'manifest.json') == identity['cohort_sha256']
    claims = {r['id']: r for r in map(json.loads, (source / 'claims_dev.jsonl').read_text(encoding="utf-8").splitlines())}
    raw = {r['claim_id']:r for r in map(json.loads, (source/'climate-fever.jsonl').read_text(encoding='utf-8').splitlines())}
    rows = load('predictions.json')
    assert len(rows) == 300 and len({r['claim_id'] for r in rows}) == 300
    assert {r['claim_id'] for r in rows} == set(claims)
    for r in rows:
        assert r['claim'] == claims[r['claim_id']]['claim']
        assert r['true_label'] == claims[r['claim_id']]['frozen_label']
        supplied = raw[r['claim_id']]['evidences']
        assert len(r['evidence']) == len(supplied) == 5
        for e, original in zip(r['evidence'], supplied, strict=True):
            assert e['id'] == original['evidence_id'] and e['article'] == original['article']
        assert r['context_source_ids'] == [e['id'] for e in r['evidence']]
        verdict, explanation = r['generated_verdict'], r['generated_explanation']
        reasons = []
        if not r['evidence']:
            reasons.append('no_evidence')
        elif verdict not in LABELS:
            reasons.append('invalid_verdict')
        elif not explanation:
            reasons.append('empty_explanation')
        numeric = numeric_provenance(explanation or '', r['shown_claim'], r['evidence'])
        assert numeric == r['numeric_provenance']
        if numeric['absent_from_inputs']:
            reasons.append('novel_numeric_value')
        assert reasons == r['reasons']
        assert r['raw_candidate_label'] == (verdict if verdict in LABELS else None)
        assert r['structural_candidate_label'] == (verdict if not reasons else None)
        for audit in r['explanation_nli_audit']:
            assert len(audit['scores_cen']) == len(r['evidence'])
            for p in audit['scores_cen']:
                assert len(p) == 3 and all(math.isfinite(v) and 0 <= v <= 1 for v in p)
                assert math.isclose(sum(p), 1, abs_tol=1e-6)
            assert math.isclose(audit['max_entailment'], max(p[1] for p in audit['scores_cen']))
    summary = load('generation_summary.json')
    for field, key in [('raw_candidate_label', 'raw_verdict_metrics'), ('structural_candidate_label', 'structural_guard_metrics')]:
        assert summarize(rows, field) == summary[key]
    assert summary['sentences_audited'] == sum(len(r['explanation_nli_audit']) for r in rows)
    result = {'rows': len(rows), 'metrics': summary, 'rejection_counts': dict(Counter(reason for r in rows for reason in r['reasons'])), 'checks': 'hashes, cohort, gold labels, supplied evidence IDs/articles/order, numeric guards, abstentions, NLI score consistency and metrics replayed', 'limitations': 'Neural inference, tokenizer clipping and chat rendering not rerun; NLI does not establish semantic grounding.'}
    out = root / 'artifacts/climate_supplied_verification'
    out.mkdir(exist_ok=True)
    write_json_atomic(out / 'verification.json', result)
    lines = ['# Constrained RAG results', '', '300 received records passed non-neural checks.', '', '| Output | Coverage | Correct / 300 | Covered accuracy | Macro F1 |', '|---|---:|---:|---:|---:|']
    for field, key in [('raw_candidate_label', 'raw_verdict_metrics'), ('structural_candidate_label', 'structural_guard_metrics')]:
        m = summary[key]
        correct = sum(r[field] == r['true_label'] for r in rows)
        lines.append(f"| {field} | {m['coverage']:.2%} | {correct} | {m['covered_accuracy']:.2%} | {m['macro_f1_all_claims_abstentions_as_errors']:.4f} |")
    lines += ['', f"Rejections: {result['rejection_counts']}.", '', 'The numeric guard lowers coverage and overall accuracy. Constrained verdicts are syntactically valid, which does not establish factual correctness.', '', result['checks'] + '.', '', result['limitations'], '', 'Exploratory results on an already observed benchmark. Retrieved context IDs are not verified explanation citations. Source authentication, actual learned evidence sufficiency and multilingual generation remain unresolved. No manuscript or GitHub push.']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
