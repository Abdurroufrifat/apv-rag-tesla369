"""Numeric extraction edge cases that caused observed guard failures."""
import unittest
from apv_rag.numeric_integrity_v2 import numbers, numeric_provenance_v2


class NumericV2Tests(unittest.TestCase):
    def test_list_markers_and_decimal(self):
        self.assertEqual(numbers('1. Finding\n2) Other\n1.5 mmol/L'), {'1.5'})

    def test_units_and_identifiers(self):
        self.assertEqual(numbers('5mmol/L NFAT4 IP3R PKM2'), {'5'})

    def test_normalization_and_sign(self):
        self.assertEqual(numbers('1.50 1.5 -5 +5 1e-3'), {'1.5', '-5', '5', '0.001'})

    def test_citations(self):
        self.assertEqual(numbers('[123] 20°C'), {'2E+1'})

    def test_claim_provenance(self):
        p = numeric_provenance_v2('5 mmol/L and 7 mmol/L', 'above 5mmol/L', [])
        self.assertEqual(p['claim_only_values'], ['5'])
        self.assertEqual(p['absent_from_inputs'], ['7'])


if __name__ == '__main__':
    unittest.main()
