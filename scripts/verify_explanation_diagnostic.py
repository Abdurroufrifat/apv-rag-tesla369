"""Bind diagnostic NLI pairs and descriptive statistics to frozen explanations."""
import argparse
import json

from apv_rag.splits import sha256,write_json_atomic
from run_explanation_diagnostic import ROOT,prepare,pair_specs,summarize,require,receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',default='artifacts/explanation_diagnostic_v1');args=parser.parse_args()
    folder=(ROOT/args.folder).resolve()
    require(folder.is_relative_to((ROOT/'artifacts').resolve()) and folder!=(ROOT/'artifacts').resolve(),
            'Output must be inside artifacts')
    manifest=json.loads((folder/'output_manifest.json').read_text(encoding='utf-8'))
    require(set(manifest)=={'input_manifest.json','pairs.json','summary.json'},'Output coverage mismatch')
    source=receipt(folder)
    tasks,identity=prepare()
    require(json.loads((folder/'input_manifest.json').read_text(encoding='utf-8'))==identity,
            'Input, protocol, model declaration or code changed')
    cache=json.loads((folder/'pairs.json').read_text(encoding='utf-8'))
    summary=summarize(tasks,cache)
    require(json.loads((folder/'summary.json').read_text(encoding='utf-8'))==summary,'Summary mismatch')
    out=ROOT/'artifacts/explanation_diagnostic_verification_v1';out.mkdir(exist_ok=True)
    write_json_atomic(out/'verification.json',{
        'status':'saved_explanation_nli_scores_bound','source_output_manifest_sha256':source,
        'tasks':len(tasks),'pairs':len(pair_specs(tasks)),
        'independent_neural_inference':False,'independent_tokenizer_recount':False,
        'scope':'English saved development explanations; NLI model scores are not factual gold labels.'})
    (out/'RESULTS.md').write_text('# Saved explanation diagnostic\n\n'
        f'{len(tasks)} saved English explanations and {len(pair_specs(tasks))} premise–hypothesis pairs passed source, score-shape, probability and summary checks. '
        'Weights, model inference and token counts were not independently repeated. '
        'The diagnostic cannot assign factual explanation correctness or publisher authenticity. '
        'Numeric aggregates and quote counts are in the source summary.json.\n',encoding='utf-8')
    write_json_atomic(out/'audit_manifest.json',{'source_output_manifest_sha256':source,
        'verifier_sha256':sha256(ROOT/'scripts/verify_explanation_diagnostic.py'),
        'files':{p.name:sha256(p) for p in out.iterdir() if p.name!='audit_manifest.json'}})
    print(f'Explanation export verified: {len(tasks)} explanations, {len(pair_specs(tasks))} pairs.')


if __name__=='__main__':main()
