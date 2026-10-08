"""Build a deterministic evaluation cohort and pooled evidence corpus."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from apv_rag.splits import sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / 'data/external/climate_fever/frozen_v1'
    raw = folder / 'climate-fever.jsonl'
    rows = list(map(json.loads, raw.read_text(encoding="utf-8").splitlines()))
    labels = {'SUPPORTS': 'Supported', 'REFUTES': 'Refuted', 'NOT_ENOUGH_INFO': 'Not Enough Evidence'}
    groups = defaultdict(dict)
    for row in rows:
        for e in row['evidences']:
            groups[e['article']][e['evidence_id']] = e['evidence']
    corpus = [{'doc_id': i, 'title': article, 'abstract': [text for _, text in sorted(evidence.items())]}
              for i, (article, evidence) in enumerate(sorted(groups.items()))]
    eligible = [r for r in rows if r['claim_label'] in labels]
    selected = sorted(eligible, key=lambda r: hashlib.sha256(('APV-RAG-climate-v1:'+r['claim_id']).encode()).hexdigest())[:300]
    claims = [{'id': r['claim_id'], 'claim': r['claim'], 'frozen_label': labels[r['claim_label']]} for r in selected]
    for name, records in [('claims_dev.jsonl', claims), ('corpus.jsonl', corpus)]:
        (folder/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    manifest = {'upstream_revision': '03de61617b10a5c1935f8e08bb0e8ac1ee775356', 'upstream_url': 'https://github.com/tdiggelm/climate-fever-dataset', 'raw_sha256': sha256(raw), 'raw_claims': len(rows), 'eligible_claims': len(eligible), 'excluded_disputed': len(rows)-len(eligible), 'evaluation_claims': len(claims), 'pool_articles': len(corpus), 'cohort': [c['id'] for c in claims], 'files': {n: sha256(folder/n) for n in ('corpus.jsonl','claims_dev.jsonl')}, 'scope': 'Different-domain pooled-evidence retrieval evaluation; disputed claims excluded before inference. No claim-specific gold evidence provided to retrieval; global pool is annotation-derived, not full Wikipedia. Model pretraining exposure unknown.'}
    write_json_atomic(folder/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('cohort','files')},indent=2))


if __name__ == '__main__':
    main()
