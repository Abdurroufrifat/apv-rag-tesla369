"""Post-hoc gate policies on exact copies/order with frozen per-document scores."""
from apv_rag.fresh_pipeline import feature_entry, validate_feature_entry
from apv_rag.generation_robustness import question
from apv_rag.generative_rag import LABELS
from apv_rag.integrated_gate import probability_complete
from apv_rag.numeric_integrity_v2 import numeric_provenance_v2


def transform_entry(claim, original, entry, variant):
    validate_feature_entry(entry, claim, original)
    parents = {str(e['id']): i for i, e in enumerate(original)}
    if len(parents) != len(original):
        raise ValueError('Duplicate parent IDs')
    scores, cosines = [], []
    for evidence in variant:
        parent = str(evidence.get('family_id', evidence['id']))
        if parent not in parents:
            raise ValueError('Unknown score parent')
        i = parents[parent]
        if evidence['text'] != original[i]['text']:
            raise ValueError('Changed text cannot reuse neural scores')
        scores.append(entry['scores_cen'][i])
        cosines.append(entry['cosines'][i])
    return feature_entry(claim, variant, scores, cosines)


def execute_stress_policy(claim, evidence, entry, model, generate, explanation_selected):
    """Caller chooses context normalization; model=None is the no-gate ablation."""
    features = validate_feature_entry(entry, claim, evidence)
    probability = probability_complete(features, model) if model is not None else None
    result = {'gate_probability': probability, 'threshold': .5, 'generation_requests': [],
              'gate_verdict_label': None, 'explanation_guarded_label': None,
              'verdict_prompt': None, 'explanation_prompt': None, 'generated_verdict': None,
              'generated_explanation': None, 'verdict_prompt_tokens': None,
              'explanation_prompt_tokens': None, 'numeric_provenance': None, 'reasons': [],
              'stages': ['context_policy', 'gate'], 'explanation_selected': explanation_selected}
    if probability is not None and probability < .5:
        result['reasons'].append('gate_below_threshold')
        return result
    result['generation_requests'].append('verdict')
    result['stages'].append('verdict')
    verdict = generate('verdict', question(claim, evidence))
    result.update(generated_verdict=verdict['answer'], verdict_prompt=verdict['prompt'],
                  verdict_prompt_tokens=verdict['prompt_tokens'])
    if verdict['answer'] not in LABELS:
        raise ValueError('Invalid saved constrained verdict')
    result['gate_verdict_label'] = verdict['answer']
    if not explanation_selected:
        return result
    result['generation_requests'].append('explanation')
    result['stages'].append('explanation')
    explanation = generate('explanation', question(claim, evidence, verdict['answer']))
    result.update(generated_explanation=explanation['answer'], explanation_prompt=explanation['prompt'],
                  explanation_prompt_tokens=explanation['prompt_tokens'])
    if not explanation['answer']:
        result['reasons'].append('empty_explanation')
    result['stages'].append('numeric_check')
    result['numeric_provenance'] = numeric_provenance_v2(explanation['answer'], claim, evidence)
    if result['numeric_provenance']['absent_from_inputs']:
        result['reasons'].append('numeric_value_absent')
    if not result['reasons']:
        result['explanation_guarded_label'] = verdict['answer']
    return result
