"""Recompute fixed gate feature masks on saved English and multilingual confirmation."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from apv_rag.fresh_pipeline import validate_feature_entry
from apv_rag.frozen_component_ablation import VARIANTS, masked_probability, summarize
from apv_rag.fever_nli import LABEL_MAP, read_jsonl
from run_clean_confirmation import digest, load, save
from run_fever_pipeline import load_inputs as english_inputs
from verify_fever_pipeline import audit as english_audit
from run_multilingual_confirmation import load_inputs as multi_inputs, audit as multi_audit, OUT as MULTI_PREP
from verify_fresh_pipeline import equal

OUT = ROOT / 'artifacts/frozen_component_ablation_v1'
CODE = ('scripts/analyze_frozen_components.py', 'src/apv_rag/frozen_component_ablation.py',
        'src/apv_rag/integrated_gate.py', 'src/apv_rag/fresh_pipeline.py', 'docs/FROZEN_COMPONENT_ABLATION.md')
METADATA = {'source_url': ('source_url', 'url'), 'source_type': ('source_type', 'source_tier', 'source_quality'),
            'publication_date': ('publication_date', 'published_date', 'published_at', 'date', 'year'),
            'provenance_family': ('family_id',)}


def cohort_inputs():
    english = ROOT / 'artifacts/fever_pipeline_received_v1/fever_pipeline_v1'
    multi = ROOT / 'artifacts/multilingual_confirmation_received_v1/multilingual_confirmation_v1'
    english_audit(english)
    _, em, er, _ = english_inputs()
    mm, rows, mr, fits = multi_inputs()
    multi_audit(multi, mm, rows, mr, fits)
    e = english / 'confirmation'
    inputs = [('english_confirmation', load(e / 'predictions.json'), load(e / 'prepared_contexts.json'),
               load(e / 'feature_cache.json'), er['gate_models']['combined'], 'claim_id', None)]
    predictions, prepared, cache = (load(multi / name) for name in
                                    ('predictions.json', 'prepared_contexts.json', 'feature_cache.json'))
    for name in sorted({r['file'] for r in predictions}):
        inputs.append((name, [r for r in predictions if r['file'] == name], prepared, cache,
                       mr['gate_models']['combined'], 'key', name))
    identity = {'english_receipt_sha256': digest(english / 'output_manifest.json'),
                'multilingual_receipt_sha256': digest(multi / 'output_manifest.json'),
                'english_head': er['gate_models']['combined'], 'multilingual_head': mr['gate_models']['combined'],
                'code_sha256': {n: digest(ROOT / n) for n in CODE}, 'neural_inference': False,
                'fitting': False, 'threshold': 0.5, 'specified_after_original_outcomes': True}
    return inputs, identity


def decide(inputs):
    decisions, inventory = [], {}
    for name, rows, prepared, cache, head, keyfield, _ in inputs:
        control = [r for r in rows if r['policy'] == 'no_gate']
        original = [r for r in rows if r['policy'] == 'combined']
        if len(control) != len(original) or not control:
            raise ValueError('Missing aligned original policies')
        meta = {'queries': len(control), 'retrieved_documents': 0, 'retained_documents': 0,
                'explicit_field_counts': {k: 0 for k in METADATA}, 'observed_document_fields': set(),
                'neural_gate_source_type_input': False, 'neural_gate_publication_date_input': False,
                'neural_gate_provenance_family_input': False}
        seen = set()
        for base, full in zip(control, original, strict=True):
            key = str(base[keyfield])
            if key in seen or base[keyfield] != full[keyfield]:
                raise ValueError('Original policy ordering or unique query IDs differ')
            seen.add(key)
            context = prepared[key]
            docs = context['retrieved_evidence']
            meta['retrieved_documents'] += len(docs)
            meta['retained_documents'] += len(context['evidence'])
            for doc in docs:
                meta['observed_document_fields'].update(doc)
                for group, fields in METADATA.items():
                    meta['explicit_field_counts'][group] += int(any(doc.get(f) not in (None, '') for f in fields))
            features = validate_feature_entry(cache[key], context['shown_claim'], context['evidence']) if context['evidence'] else None
            for variant in VARIANTS:
                probability = masked_probability(features, head, variant) if features is not None else None
                candidate = base['candidate_label'] if probability is not None and probability >= 0.5 else None
                if variant == 'full':
                    if candidate != full['candidate_label'] or ((probability is None) != (full['gate_probability'] is None)):
                        raise ValueError('Full policy does not reproduce the frozen combined policy')
                    if probability is not None and not math.isclose(probability, full['gate_probability'], rel_tol=0, abs_tol=1e-12):
                        raise ValueError('Full gate probability differs')
                decisions.append({'cohort': name, 'query': key, 'variant': variant,
                                  'candidate_label': candidate, 'gate_probability': probability})
        meta['observed_document_fields'] = sorted(meta['observed_document_fields'])
        inventory[name] = meta
    return decisions, inventory


def score(decisions):
    # Scoring follows frozen masked decisions; preliminary original-export replay checks gold separately.
    english = read_jsonl(ROOT / 'data/external/fever/pipeline_v1/confirmation/gold.jsonl')
    gold = {('english_confirmation', str(r['id'])): LABEL_MAP[r['label']] for r in english}
    gold.update({(r['file'], r['key']): LABEL_MAP[r['label']] for r in load(MULTI_PREP / 'gold.json')})
    results = {}
    for cohort in sorted({r['cohort'] for r in decisions}):
        subsets = {v: [r for r in decisions if r['cohort'] == cohort and r['variant'] == v] for v in VARIANTS}
        full = subsets['full']
        results[cohort] = {}
        for variant, rows in subsets.items():
            if [r['query'] for r in rows] != [r['query'] for r in full]:
                raise ValueError('Variant query order differs')
            truth = [gold[(cohort, r['query'])] for r in rows]
            m = summarize([r['candidate_label'] for r in rows], truth)
            for kind, transition in [('added', lambda a,b: a is None and b is not None),
                                     ('removed', lambda a,b: a is not None and b is None)]:
                selected = [i for i,(a,b) in enumerate(zip(full,rows,strict=True))
                            if transition(a['candidate_label'],b['candidate_label'])]
                compare = rows if kind == 'added' else full
                correct = sum(compare[i]['candidate_label'] == truth[i] for i in selected)
                m[kind + '_correct_vs_full'] = correct
                m[kind + '_incorrect_vs_full'] = len(selected) - correct
            results[cohort][variant] = m
    return {'cohorts': results, 'exploratory_post_result_analysis': True, 'training_performed': False,
            'new_model_outputs': False, 'date_source_family_training_ablation_completed': False,
            'full_charter_complete': False}


def report(result, inventory):
    lines = ['# Frozen learned-feature sensitivity', '',
             'Exploratory mean masking of the original combined head. No fitting, neural inference or policy replacement.', '',
             '| Cohort | Variant | Accepted | Correct | All-query accuracy | Accepted accuracy |',
             '|---|---|---:|---:|---:|---:|']
    for cohort, variants in result['cohorts'].items():
        for variant, m in variants.items():
            covered = f"{m['accepted_accuracy']:.2%}" if m['accepted_accuracy'] is not None else 'undefined'
            lines.append(f"| {cohort} | {variant} | {m['accepted']}/{m['queries']} | {m['correct']} | {m['all_query_accuracy']:.2%} | {covered} |")
    lines += ['', 'Removing a feature group means replacing it with its training mean in a frozen head. '
              'This removes its standardized logit contribution; it does not retrain the remaining model. '
              'The no-gate candidates and their numeric/structural checks are retained. Both groups removed gives an intercept-only control. '
              'Do not select a new deployed policy from these already-observed outcomes.', '',
              '| Cohort | Retrieved docs | Explicit source URL | Explicit source type | Explicit publication date | Explicit family ID |',
              '|---|---:|---:|---:|---:|---:|']
    for cohort, m in inventory.items():
        counts = m['explicit_field_counts']
        lines.append(f"| {cohort} | {m['retrieved_documents']} | {counts['source_url']} | {counts['source_type']} | {counts['publication_date']} | {counts['provenance_family']} |")
    lines += ['', 'The current neural head uses nine NLI and four cosine statistics. It has no explicit source-type, '
              'date or family features. The contexts use document IDs and matching-text collapse; these do not '
              'establish true upstream provenance families. Source/date head ablations are unavailable in this controller. '
              'Earlier AVeriTeC provenance models and separate capture/snapshot guards remain distinct components.', '',
              'The six multilingual rows share 100 claim IDs; no pooled independence or significance claim is made. '
              'Historical authentication, explanation truth, trained provenance/temporal/robustness ablations and '
              'broader retrieval transfer remain open. This analysis does not complete the charter or establish gate superiority.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    inputs, identity = cohort_inputs()
    decisions, inventory = decide(inputs)
    if args.verify:
        if not equal(load(OUT / 'decisions.json'), decisions) or load(OUT / 'input_manifest.json') != identity:
            raise ValueError('Saved decisions or producer identity differ')
    else:
        if OUT.exists():
            raise ValueError('Output exists; use --verify to preserve the analysis')
        OUT.mkdir()
        save(OUT / 'decisions.json', decisions)
        save(OUT / 'input_manifest.json', identity)
    result = score(decisions)
    products = {'analysis.json': result, 'metadata_inventory.json': inventory}
    text = report(result, inventory)
    if args.verify:
        if any(not equal(load(OUT / name), value) for name,value in products.items()) or (OUT / 'RESULTS.md').read_text(encoding='utf-8') != text:
            raise ValueError('Saved metrics, inventory or report differs')
        receipts = load(OUT / 'audit_manifest.json')['files']
        if set(receipts) != {p.name for p in OUT.iterdir() if p.is_file() and p.name != 'audit_manifest.json'}:
            raise ValueError('Analysis receipt coverage differs')
        if any(digest(OUT / name) != h for name,h in receipts.items()):
            raise ValueError('Analysis receipt mismatch')
    else:
        for name,value in products.items(): save(OUT / name,value)
        (OUT / 'RESULTS.md').write_text(text,encoding='utf-8')
        save(OUT / 'audit_manifest.json', {'files': {p.name: digest(p) for p in OUT.iterdir() if p.is_file()}})
    print('Frozen component analysis verified:', len(decisions), 'counterfactual decisions; no model run.')
    print(text)


if __name__ == '__main__': main()
