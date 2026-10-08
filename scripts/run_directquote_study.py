"""Frozen quote attribution evaluation with local Qwen; existing labels only."""
import json,sys,re,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from apv_rag.splits import sha256,write_json_atomic
from apv_rag.nli_comparison import _model_files
from download_instruction_model import MODEL_ID,REVISION

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def norm(s):return ' '.join(re.findall(r'\w+',s.casefold())) if s else None
def main():
    frozen=ROOT/'artifacts/directquote_preflight_v1';out=ROOT/'artifacts/directquote_evaluation_v1'
    for n,h in read(frozen/'receipt.json').items():
        if sha256(frozen/n)!=h:raise ValueError('Frozen input differs: '+n)
    if (out/'summary.json').exists():raise FileExistsError('Completed run exists')
    modelpath=ROOT/'models/qwen2.5-1.5b-instruct';meta=read(ROOT/'models/instruction_model_manifest.json')
    if meta['model_id']!=MODEL_ID or meta['revision']!=REVISION or _model_files(modelpath)!=meta['files']:raise ValueError('Pinned model differs')
    identity={'protocol':sha256(frozen/'protocol.json'),'model_manifest':sha256(ROOT/'models/instruction_model_manifest.json'),'runner':sha256(Path(__file__)),'seed':369,'dtype':'float32','decoding':'greedy','max_input_tokens':1024,'max_new_tokens':48}
    out.mkdir(exist_ok=True)
    if (out/'input_manifest.json').exists() and read(out/'input_manifest.json')!=identity:raise ValueError('Resume differs')
    write_json_atomic(out/'input_manifest.json',identity)
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(4);torch.manual_seed(369);torch.use_deterministic_algorithms(True)
    tokenizer=AutoTokenizer.from_pretrained(modelpath,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.float32).eval()
    inputs=read(frozen/'inputs.json')
    for position,row in enumerate(inputs):
        path=out/(row['id']+'.json')
        if path.exists():continue
        prompt='Identify who said the supplied quotation using only the paragraph. Return only the exact speaker name as written in the paragraph, or UNKNOWN if the speaker cannot be identified. Do not explain.\nParagraph: '+row['paragraph']+'\nQuotation: '+row['quote']+'\nSpeaker:'
        tokens=tokenizer.apply_chat_template([{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
        if tokens.input_ids.shape[1]>1024:raise ValueError('Prompt too long; refusing truncation')
        with torch.inference_mode():generated=model.generate(**tokens,max_new_tokens=48,do_sample=False,num_beams=1,pad_token_id=tokenizer.eos_token_id)
        answer=tokenizer.decode(generated[0,tokens.input_ids.shape[1]:],skip_special_tokens=True).strip()
        pred=None if answer=='UNKNOWN' else answer
        write_json_atomic(path,{'id':row['id'],'raw_answer':answer,'predicted_speaker':pred,'prompt_tokens':int(tokens.input_ids.shape[1])})
        print(str(position+1)+'/100',flush=True)
    gold={x['id']:x for x in read(frozen/'gold.json')};rows=[read(out/(x['id']+'.json')) for x in inputs]
    correct=sum(norm(x['predicted_speaker'])==norm(gold[x['id']]['speaker']) for x in rows)
    known=[x for x in rows if gold[x['id']]['speaker'] is not None];unknown=[x for x in rows if gold[x['id']]['speaker'] is None]
    summary={'cases':100,'correct_exact_normalized':correct,'accuracy':correct/100,'known_correct':sum(norm(x['predicted_speaker'])==norm(gold[x['id']]['speaker']) for x in known),'known_cases':len(known),'unknown_false_attributions':sum(x['predicted_speaker'] is not None for x in unknown),'unknown_cases':len(unknown),'scope':'Filtered balanced quotation attribution component; not historical authentication, publisher verification or full-method confirmation.'}
    write_json_atomic(out/'summary.json',summary);write_json_atomic(out/'output_manifest.json',{p.name:sha256(p) for p in out.iterdir() if p.is_file()})
    export=ROOT/'directquote_study_outputs.zip'
    with zipfile.ZipFile(export,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():z.write(p,p.name)
    print(summary);print('Send:',export)
if __name__=='__main__':main()
