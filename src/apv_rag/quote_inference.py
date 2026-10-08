"""Label-free supplied-quote attribution using the frozen component."""
import hashlib,json
from pathlib import Path
import joblib
from .quote_relation import token_features,decode_speaker

MODEL_SHA256='5f63ad6d959edf88dd794446a17ed4a46622eae1c24f62ebe79dd3f08aa83a00'

def validate_input(row):
 if not isinstance(row,dict) or set(row)!={'tokens','quote_span'}:raise ValueError('Input must contain only tokens and quote_span')
 tokens=row['tokens'];quote=row['quote_span']
 if not isinstance(tokens,list) or not tokens or any(not isinstance(t,str) or not t or any(c.isspace() for c in t) for t in tokens):raise ValueError('tokens must be nonempty strings without whitespace')
 if not isinstance(quote,list) or len(quote)!=2 or any(type(v) is not int for v in quote):raise ValueError('quote_span must be two integer token offsets')
 a,b=quote
 if not 0<=a<b<=len(tokens):raise ValueError('quote_span must be a valid half-open token interval')
 return tokens,(a,b)

def load_model(path):
 path=Path(path)
 if hashlib.sha256(path.read_bytes()).hexdigest()!=MODEL_SHA256:raise ValueError('Frozen model SHA-256 mismatch')
 saved=joblib.load(path)
 if saved['threshold']!=.9:raise ValueError('Unexpected frozen decoder threshold')
 return saved

def infer(row,saved):
 tokens,quote=validate_input(row)
 base={'status':'abstain','speaker_candidate':None,'source_authenticated':False,'scope':'Supplied-quote speaker attribution only'}
 if len(tokens)>180 or quote[1]-quote[0]<6:return {**base,'reason':'outside_training_scope'}
 p=saved['model'].predict_proba(saved['vectorizer'].transform(token_features(tokens,quote)))[:,1]
 speaker=decode_speaker(tokens,quote,p,saved['threshold'])
 return {**base,'status':'machine_candidate' if speaker else 'abstain','speaker_candidate':speaker,'reason':'model_candidate' if speaker else 'no_span_above_frozen_threshold'}
