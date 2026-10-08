import unittest

from prepare_fever_heldout import select_claims


class FeverHeldoutTest(unittest.TestCase):
    def setUp(self):
        labels = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
        self.rows = [{'id': index, 'claim': f'Claim {index}',
                      'label': labels[index % 3], 'evidence': []}
                     for index in range(10)]

    def test_selection_is_order_and_label_independent(self):
        first = select_claims(self.rows, 4)
        reversed_rows = [{**row, 'label': 'REFUTES'} for row in reversed(self.rows)]
        second = select_claims(reversed_rows, 4)
        self.assertEqual([r['id'] for r in first], [r['id'] for r in second])
        self.assertEqual(len(first), 4)

    def test_reject_duplicate_or_malformed_rows(self):
        with self.assertRaises(ValueError):
            select_claims(self.rows + [self.rows[0]], 4)
        with self.assertRaises(ValueError):
            select_claims([{**self.rows[0], 'label': 'OTHER'}], 1)
        with self.assertRaises(ValueError):
            select_claims(self.rows, 11)

    def test_excludes_previously_observed_ids_and_duplicate_text(self):
        rows = self.rows + [{'id': 20, 'claim': '  CLAIM 1  ',
                             'label': 'SUPPORTS', 'evidence': []}]
        selected = select_claims(rows, 9, exclude_ids={0})
        self.assertNotIn(0, [r['id'] for r in selected])
        self.assertEqual(len({' '.join(r['claim'].casefold().split()) for r in selected}), 9)
        with self.assertRaises(ValueError):
            select_claims(rows, 10, exclude_ids={0})


if __name__ == '__main__':
    unittest.main()
