"""Remove exact same-source answer copies within a question, preserving order."""
import copy
import json


def remove_exact_source_copies(record):
    output=copy.deepcopy(record)
    for question in output.get('questions') or []:
        seen=set();kept=[]
        for answer in question.get('answers') or []:
            url=answer.get('source_url');text=answer.get('answer')
            known=isinstance(url,str) and bool(url.strip()) and isinstance(text,str) and bool(text.strip())
            key=json.dumps(answer,sort_keys=True,ensure_ascii=False) if known else None
            if key is not None:
                if key in seen:continue
                seen.add(key)
            kept.append(answer)
        question['answers']=kept
    return output
