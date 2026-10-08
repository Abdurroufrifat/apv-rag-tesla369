"""Supervised token-to-supplied-quote relation baseline; no annotated features."""
import numpy as np

def token_features(tokens,quote):
    a,b=quote;output=[]
    for i,word in enumerate(tokens):
        distance=a-i if i<a else i-b+1 if i>=b else 0
        side='before' if i<a else 'after' if i>=b else 'inside'
        row={'word':word.casefold(),'capitalized':word[:1].isupper(),'all_caps':word.isupper(),'quote_side':side,'distance_bin':str(min(distance//3,20)), 'quote_inside':a<=i<b}
        for offset in (-2,-1,1,2):row[('prev' if offset<0 else 'next')+str(abs(offset))]=tokens[i+offset].casefold() if 0<=i+offset<len(tokens) else '<BOUNDARY>'
        row['side_word']=side+':'+word.casefold();output.append(row)
    return output

def decode_speaker(tokens,quote,scores,threshold):
    values=np.asarray(scores,float)
    if len(values)!=len(tokens) or not np.isfinite(values).all():raise ValueError('Invalid token scores')
    a,b=quote;spans=[];start=None
    for i in range(len(tokens)+1):
        keep=i<len(tokens) and not a<=i<b and values[i]>=threshold
        if keep and start is None:start=i
        if not keep and start is not None:
            spans.append((float(values[start:i].mean()),start,i));start=None
    if not spans:return None
    _,start,end=max(spans,key=lambda x:(x[0],-x[1]))
    return ' '.join(tokens[start:end])
