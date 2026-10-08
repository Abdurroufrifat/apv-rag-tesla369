"""Retrieve from raw frozen corpora, infer fresh features, then execute the gate."""
import argparse
import gc
import hashlib
import json
import sqlite3
from importlib.metadata import version
from pathlib import Path

import numpy as np

from apv_rag.fresh_pipeline import FreshRetriever, POLICIES, feature_entry, validate_feature_entry, execute_policies
from apv_rag.integrated_gate import probability_complete
from apv_rag.multilingual_nli import cen_order, normalize_model_scores
from apv_rag.nli_comparison import _model_files
from apv_rag.scifact import target_label
from apv_rag.splits import sha256, write_json_atomic
from run_gated_generation import live_backend, qwen_prompt
from run_sentence_rag_scifact import summarize

ROOT = Path(__file__).resolve().parents[1]
COHORTS = {'scifact': ('data/external/scifact/sealed_v1', 'sentence_rag_received'),
           'climate_retrieved': ('data/external/climate_fever/frozen_v1', 'climate_rag_received')}
CODE = ('scripts/run_fresh_pipeline.py', 'scripts/verify_fresh_pipeline.py',
        'src/apv_rag/fresh_pipeline.py', 'scripts/run_gated_generation.py',
        'scripts/run_sentence_rag_scifact.py', 'src/apv_rag/gated_generation_flow.py',
        'src/apv_rag/integrated_gate.py', 'src/apv_rag/numeric_integrity_v2.py',
        'src/apv_rag/input_numeric_integrity.py', 'src/apv_rag/generative_interface.py',
        'src/apv_rag/generative_rag.py', 'src/apv_rag/retrieval.py',
        'src/apv_rag/sentence_context.py', 'src/apv_rag/multilingual_nli.py',
        'src/apv_rag/nli_comparison.py', 'src/apv_rag/scifact.py', 'src/apv_rag/splits.py')
SETTINGS = {'seed': 369, 'threads': 4, 'dtype': 'float32', 'top_k': 3,
            'sentence_top_k': 3, 'claim_tokens': 64, 'passage_tokens': 96,
            'nli_tokens': 256, 'embedding_tokens': 384, 'gate_threshold': .5,
            'max_input_tokens': 1024, 'max_new_tokens': 128, 'do_sample': False,
            'num_beams': 1, 'response_reuse': 'exact pinned baseline prompt only; live fallback',
            'feature_reuse': 'resume within this run only; no prior feature cache imported'}
PACKAGES = ('torch', 'transformers', 'sentence-transformers', 'numpy', 'scikit-learn')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(folder, name):
    return json.loads((folder / name).read_text(encoding='utf-8'))


def check_receipt(folder):
    hashes = load(folder, 'output_manifest.json')
    for name, digest in hashes.items():
        require(Path(name).name == name and ':' not in name and '\\' not in name, 'Unsafe receipt path')
        require(sha256(folder / name) == digest, f'Checksum mismatch: {folder.name}/{name}')
    return hashes


def prepare_sources(root=ROOT):
    files, sources, baselines = {}, {}, {}
    generators = []
    for cohort, (relative, prior_name) in COHORTS.items():
        source, prior = root / relative, root / 'artifacts' / prior_name
        check_receipt(prior)
        manifest = load(source, 'manifest.json')
        for name in ('claims_dev.jsonl', 'corpus.jsonl'):
            require(sha256(source / name) == manifest['files'][name], f'Dataset mismatch: {cohort}/{name}')
        claims = [json.loads(x) for x in (source / 'claims_dev.jsonl').read_text(encoding='utf-8').splitlines()]
        corpus = [json.loads(x) for x in (source / 'corpus.jsonl').read_text(encoding='utf-8').splitlines()]
        require(len(claims) == 300 and len({str(r['id']) for r in claims}) == 300, 'Unexpected raw cohort')
        baseline = {str(r['claim_id']): r for r in load(prior, 'predictions.json')}
        require(len(baseline) == 300 and set(baseline) == {str(r['id']) for r in claims}, 'Baseline cohort mismatch')
        for r in claims:
            label = target_label(r) if cohort == 'scifact' else r['frozen_label']
            require(baseline[str(r['id'])]['claim'] == r['claim'] and
                    baseline[str(r['id'])]['true_label'] == label, 'Original claim/label mismatch')
        sources[cohort] = {'claims': claims, 'retriever': FreshRetriever(corpus)}
        baselines[cohort] = baseline
        meta = load(prior, 'input_manifest.json')
        for name, digest in meta['source_sha256'].items():
            require(sha256(source / name) == digest, 'Original dataset identity mismatch')
        generators.append(meta['models']['generator'])
        for p in (source / 'manifest.json', source / 'claims_dev.jsonl', source / 'corpus.jsonl',
                  prior / 'output_manifest.json', prior / 'input_manifest.json', prior / 'predictions.json'):
            files[p.relative_to(root).as_posix()] = sha256(p)
    require(generators[0] == generators[1], 'Different original generators')
    learned = root / 'artifacts/semantic_sufficiency_received'
    replay = root / 'artifacts/integrated_gate_received'
    for folder in (learned, replay):
        check_receipt(folder)
        for p in folder.glob('*.json'):
            files[p.relative_to(root).as_posix()] = sha256(p)
    learned_meta = load(learned, 'input_manifest.json')
    replay_meta = load(replay, 'input_manifest.json')
    require(learned_meta['models'] == replay_meta['feature_models'], 'Feature model declarations differ')
    models = load(learned, 'models.json')
    require(set(models) == set(POLICIES[1:]), 'Unexpected learned heads')
    return {'sources': sources, 'baselines': baselines, 'models': models, 'files': files,
            'generator': generators[0], 'feature_models': learned_meta['models'],
            'packages': learned_meta['packages']}


def check_context(prepared, original):
    require(prepared['claim'] == original['claim'] and prepared['shown_claim'] == original['shown_claim'] and
            prepared['retrieved_evidence'] == original['evidence'],
            'Fresh tokenizer/retrieval context differs from the pinned baseline; inspect before continuing')


def infer_missing(prepared, cache, cachepath, root=ROOT):
    pending = [(k, r) for k, r in prepared.items() if r['evidence'] and k not in cache]
    if not pending:
        return 0
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    torch.manual_seed(369)
    torch.use_deterministic_algorithms(True)
    nt = AutoTokenizer.from_pretrained(root / 'models/nli-deberta-v3-small', local_files_only=True)
    nli = AutoModelForSequenceClassification.from_pretrained(
        root / 'models/nli-deberta-v3-small', local_files_only=True, torch_dtype=torch.float32).eval()
    order = cen_order(nli.config.id2label)
    embedding = SentenceTransformer(str(root / 'models/all-MiniLM-L6-v2'), device='cpu', local_files_only=True)
    embedding.float()
    embedding.max_seq_length = 384
    for position, (key, r) in enumerate(pending):
        text = [e['text'] for e in r['evidence']]
        tokens = nt(text, [r['shown_claim']] * len(text), return_tensors='pt', padding=True,
                    truncation=True, max_length=256)
        with torch.inference_mode():
            scores = nli(**tokens).logits.float().softmax(-1).cpu().numpy()[:, order]
        scores = normalize_model_scores(scores.tolist())
        vectors = embedding.encode([r['shown_claim']] + text, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
        cache[key] = feature_entry(r['shown_claim'], r['evidence'], scores, (vectors[1:] @ vectors[0]).tolist())
        write_json_atomic(cachepath, cache)
        if position % 10 == 0 or position + 1 == len(pending):
            print(f'Fresh NLI + embedding: {position + 1}/{len(pending)}', flush=True)
    # The model objects leave scope before Qwen is loaded by the lazy backend.
    return len(pending)


def response_key(kind, prompt):
    require(kind in ('verdict', 'explanation'), 'Unknown response kind')
    return kind + ':' + hashlib.sha256(prompt.encode('utf-8')).hexdigest()


def seed_responses(baselines):
    seeds = {}
    for rows in baselines.values():
        for r in rows.values():
            for kind in ('verdict', 'explanation'):
                if r[kind + '_prompt'] is None:
                    continue
                response = {'answer': r['generated_' + kind], 'prompt': r[kind + '_prompt'],
                            'prompt_tokens': r[kind + '_prompt_tokens']}
                key = response_key(kind, response['prompt'])
                require(key not in seeds or seeds[key] == response, 'Conflicting prior prompt responses')
                seeds[key] = response
    return seeds


def make_backend(root, generator, db, seeds):
    for key, response in seeds.items():
        existing = db.execute('SELECT response FROM answers WHERE key=?', (key,)).fetchone()
        require(not existing or json.loads(existing[0]) == response, 'Pinned response cache conflict')
        if not existing:
            db.execute('INSERT INTO answers VALUES (?,?)', (key, json.dumps(response, ensure_ascii=False)))
    db.commit()
    live = None
    used = {}
    def generate(kind, question):
        nonlocal live
        prompt = qwen_prompt(question)
        key = response_key(kind, prompt)
        cached = db.execute('SELECT response FROM answers WHERE key=?', (key,)).fetchone()
        if cached:
            response = json.loads(cached[0])
        else:
            if live is None:
                print('Loading Qwen for a new prompt; feature models have been released.', flush=True)
                live = live_backend(root, generator, db)
            response = live(kind, question)
        require(response['prompt'] == prompt and isinstance(response['prompt_tokens'], int) and
                0 < response['prompt_tokens'] <= 1024, 'Invalid generation response')
        origin = 'prior_exact_prompt' if key in seeds else 'current_run_live_or_resume'
        record = {'kind': kind, **response, 'origin': origin}
        require(key not in used or used[key] == record, 'Shared response changed')
        used[key] = record
        return response
    return generate, used


def build_summary(rows, feature_cache, prior_cache, models, responses):
    metrics = {}
    differences = {'max_absolute_feature_difference': 0., 'max_absolute_probability_difference': 0.,
                   'threshold_decision_differences': {p: 0 for p in POLICIES[1:]}}
    for key, entry in feature_cache.items():
        old = prior_cache[key]
        # Prior cache stores individual scores, not their aggregates.
        original = feature_entry('', [{'id': i, 'text': ''} for i in range(len(old['cosines']))],
                                 old['scores_cen'], old['cosines'])['features']
        differences['max_absolute_feature_difference'] = max(differences['max_absolute_feature_difference'],
            float(np.max(np.abs(np.array(original) - entry['features']))))
        for policy in POLICIES[1:]:
            a, b = probability_complete(original, models[policy]), probability_complete(entry['features'], models[policy])
            differences['max_absolute_probability_difference'] = max(differences['max_absolute_probability_difference'], abs(a - b))
            differences['threshold_decision_differences'][policy] += int((a >= .5) != (b >= .5))
    for cohort in COHORTS:
        metrics[cohort] = {}
        for policy in POLICIES:
            subset = [r for r in rows if r['cohort'] == cohort and r['policy'] == policy]
            metrics[cohort][policy] = {'metrics': summarize(subset, 'candidate_label'),
                'gate_rejected_before_generation': sum('gate_below_threshold' in r['reasons'] for r in subset),
                'verdict_requests': sum('verdict' in r['generation_requests'] for r in subset),
                'explanation_requests': sum('explanation' in r['generation_requests'] for r in subset)}
    return {'records': len(rows), 'claims': len(rows) // len(POLICIES), 'cohorts': metrics,
            'fresh_context_feature_records': len(feature_cache), 'prior_feature_comparison': differences,
            'distinct_responses_used': len(responses),
            'distinct_prior_responses_reused': sum(r['origin'] == 'prior_exact_prompt' for r in responses.values()),
            'distinct_current_run_responses': sum(r['origin'] != 'prior_exact_prompt' for r in responses.values()),
            'scope': 'Engineering integration on already observed frozen cohorts; cached responses are not fresh generation or new performance evidence.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true', help='Check inputs without neural packages or weights')
    args = parser.parse_args()
    inputs = prepare_sources()
    if args.preflight:
        out = ROOT / 'artifacts/fresh_pipeline_preflight_v1'
        out.mkdir(exist_ok=True)
        write_json_atomic(out / 'preflight.json', {'status': 'input_checks_passed', 'claims': 600,
            'files': inputs['files'], 'neural_inference_run': False, 'model_bytes_checked': False,
            'code_sha256': {n: sha256(ROOT / n) for n in CODE}})
        write_json_atomic(out / 'audit_manifest.json', {'files': {'preflight.json': sha256(out / 'preflight.json')}})
        print('Fresh pipeline input preflight passed: 600 claims. Neural execution not run.')
        return
    out = ROOT / 'artifacts/fresh_pipeline_v1'
    require(not (out / 'output_manifest.json').exists(), 'Completed run exists; refusing overwrite. Verify/export the existing run.')
    packages = {n: version(n) for n in PACKAGES}
    require(packages == inputs['packages'], 'Use the original feature environment; package versions differ')
    paths = {'generator': ROOT / 'models/qwen2.5-1.5b-instruct',
             'nli': ROOT / 'models/nli-deberta-v3-small', 'embedding': ROOT / 'models/all-MiniLM-L6-v2'}
    print('Checking pinned local model bytes; no downloads.', flush=True)
    for name, path in paths.items():
        expected = inputs['generator']['files'] if name == 'generator' else inputs['feature_models'][name]
        require(_model_files(path) == expected, f'Model checksum mismatch: {name}')
    identity = {'schema_version': 1, 'files': inputs['files'], 'generator': inputs['generator'],
        'feature_models': inputs['feature_models'], 'packages': packages, 'policies': list(POLICIES),
        'settings': SETTINGS, 'code_sha256': {n: sha256(ROOT / n) for n in CODE},
        'protocol_sha256': sha256(ROOT / 'docs/FRESH_PIPELINE.md')}
    out.mkdir(exist_ok=True)
    receipt = out / 'input_manifest.json'
    if not receipt.exists():
        require(not any((out / n).exists() for n in ('feature_cache.json', 'generation_cache.sqlite')), 'Cache without identity')
    else:
        require(load(out, receipt.name) == identity, 'Run identity changed; resume refused')
    write_json_atomic(receipt, identity)
    from transformers import AutoTokenizer
    gt = AutoTokenizer.from_pretrained(paths['generator'], local_files_only=True)
    def clip(text, limit):
        return gt.decode(gt.encode(text, add_special_tokens=False)[:limit], skip_special_tokens=True)
    prepared = {}
    for cohort, source in inputs['sources'].items():
        for claim in source['claims']:
            r = source['retriever'].prepare(claim['claim'], clip)
            check_context(r, inputs['baselines'][cohort][str(claim['id'])])
            prepared[f"{cohort}:{claim['id']}"] = r
    print('Fresh retrieval + tokenizer clipping matched all 600 pinned contexts.', flush=True)
    cachepath = out / 'feature_cache.json'
    cache = load(out, cachepath.name) if cachepath.exists() else {}
    require(set(cache) <= set(prepared), 'Unexpected feature cache keys')
    for key, entry in cache.items():
        validate_feature_entry(entry, prepared[key]['shown_claim'], prepared[key]['evidence'])
    count = infer_missing(prepared, cache, cachepath)
    del gt
    gc.collect()
    write_json_atomic(out / 'execution_progress.json', {'new_feature_records_this_invocation': count,
        'total_feature_records': len(cache), 'retrieval_claims': len(prepared)})
    rows = []
    with sqlite3.connect(out / 'generation_cache.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, response TEXT)')
        generate, used = make_backend(ROOT, inputs['generator'], db, seed_responses(inputs['baselines']))
        for position, (key, prepared_row) in enumerate(prepared.items()):
            cohort, claim_id = key.split(':', 1)
            for r in execute_policies(prepared_row, cache.get(key), inputs['models'], generate):
                r.update(cohort=cohort, claim_id=claim_id, claim=prepared_row['claim'],
                         shown_claim=prepared_row['shown_claim'], retrieved_evidence=prepared_row['retrieved_evidence'],
                         context_sha256=cache[key]['context_sha256'] if key in cache else None,
                         true_label=inputs['baselines'][cohort][claim_id]['true_label'])
                rows.append(r)
            if position % 20 == 0:
                print(f'Gate + generation controller: {position + 1}/{len(prepared)}', flush=True)
    prior = load(ROOT / 'artifacts/integrated_gate_received', 'feature_cache.json')
    summary = build_summary(rows, cache, prior, inputs['models'], used)
    write_json_atomic(out / 'predictions.json', rows)
    write_json_atomic(out / 'responses.json', used)
    write_json_atomic(out / 'summary.json', summary)
    write_json_atomic(out / 'output_manifest.json', {p.name: sha256(p) for p in out.glob('*.json') if p.name != 'output_manifest.json'})
    print('Fresh retrieval/features/controller run completed. Existing exact responses are reported as reused.')
    print(out)


if __name__ == '__main__':
    main()
