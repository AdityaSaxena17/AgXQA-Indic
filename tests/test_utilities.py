import copy
import unittest
from shared.metrics import exact_match, token_f1, best_scores
from shared.validate import read_jsonl, validate
from shared.evaluate import evaluate

class UtilitiesTest(unittest.TestCase):
    def setUp(self):
        self.gold = read_jsonl('data/examples/qa.jsonl')
        self.predictions = read_jsonl('data/examples/predictions.jsonl')

    def test_unicode_and_punctuation(self):
        self.assertEqual(exact_match(' CAFÉ!', 'cafe\u0301'), 1)
        self.assertEqual(exact_match('धान।', 'धान'), 1)
        self.assertNotEqual(exact_match('क', 'का'), 1)

    def test_multiset_and_empty(self):
        self.assertAlmostEqual(token_f1('rice rice', 'rice'), 2/3)
        self.assertEqual(token_f1('', ''), 1)
        self.assertEqual(token_f1('', 'धान'), 0)
        self.assertEqual(best_scores('paddy', ['rice', 'paddy'])['em'], 1)

    def test_valid_fixture(self):
        self.assertEqual(validate(self.gold)['records'], 2)
        result = evaluate(self.gold, self.predictions)
        self.assertEqual(result['micro']['f1'], 100)
        self.assertEqual(result['consistency']['percent'], 100)
        self.assertEqual(result['language_gap']['hi']['f1_gap_points'], 0)

    def test_invalid_span(self):
        self.gold[1]['answers'][0]['answer_start'] += 1
        with self.assertRaises(ValueError): validate(self.gold)

    def test_duplicate_and_leakage(self):
        with self.assertRaises(ValueError): validate(self.gold + [self.gold[0]])
        self.gold[1]['split'] = 'train'
        with self.assertRaises(ValueError): validate(self.gold)

    def test_document_leakage(self):
        self.gold[1]['id'] = 'different'
        self.gold[1]['split'] = 'train'
        with self.assertRaises(ValueError): validate(self.gold)

    def test_bad_types_and_empty(self):
        for field, value in [('answers', []), ('language', 'xx'), ('id', '')]:
            rows = copy.deepcopy(self.gold)
            rows[0][field] = value
            with self.assertRaises(ValueError): validate(rows)
        with self.assertRaises(ValueError): validate([])
        self.gold[0]['answers'][0]['answer_start'] = True
        with self.assertRaises(ValueError): validate(self.gold)

    def test_prediction_coverage(self):
        for predictions in [self.predictions[:1], self.predictions + [self.predictions[0]], self.predictions + [{'id':'extra','language':'en','prediction_text':''}]]:
            with self.assertRaises(ValueError): evaluate(self.gold, predictions)

    def test_partial_consistency(self):
        self.gold[1]['id'] = 'unaligned'
        self.predictions[1]['id'] = 'unaligned'
        result = evaluate(self.gold, self.predictions)
        self.assertIsNone(result['consistency']['percent'])
        self.assertIsNone(result['language_gap']['hi']['f1_gap_points'])

    def test_wrong_prediction_and_single_language(self):
        self.predictions[1]['prediction_text'] = ''
        result = evaluate(self.gold, self.predictions)
        self.assertEqual(result['micro']['em'], 50)
        self.assertEqual(result['consistency']['percent'], 0)
        self.assertEqual(result['language_gap']['hi']['f1_gap_points'], 100)
        self.assertIsNone(evaluate(self.gold[:1], self.predictions[:1])['consistency']['percent'])

if __name__ == '__main__':
    unittest.main()
