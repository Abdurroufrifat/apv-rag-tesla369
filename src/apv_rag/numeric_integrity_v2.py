"""Numeric text provenance v2; does not verify units, meaning or truth."""
import re
from decimal import Decimal

# Explicit units only: digits inside identifiers such as NFAT4 remain excluded.
UNITS = r'(?:mmol/L|mol/L|mg/dL|mg/mL|ng/mL|pg/mL|µg/mL|ug/mL|mmHg|mL|µL|uL|mg|µg|ug|ng|kg|cm|mm|nm|km|ms|Hz|kHz|MHz|GHz|°C|°F|mol|mmol|g|L|m|s|h|%)'
PATTERN = re.compile(r'(?<![\w.])([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?![\d.])(?:(?=' + UNITS + r'(?!\w))|(?!\w))')


def numbers(text):
    text = re.sub(r'^([ \t]*)\d+[.)][ \t]+', r'\1', text, flags=re.MULTILINE)
    text = re.sub(r'\[[^\]]*\]', '', text)
    return {str(Decimal(m.group(1)).normalize()) for m in PATTERN.finditer(text)}


def numeric_provenance_v2(explanation, claim, evidence):
    used = numbers(explanation)
    source = numbers(' '.join(d['text'] for d in evidence))
    claim_values = numbers(claim)
    return {'evidence_values': sorted(used & source),
            'claim_only_values': sorted((used & claim_values) - source),
            'absent_from_inputs': sorted(used - source - claim_values)}
