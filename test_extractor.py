import json
import unittest

from extractor import baseline, evaluate, extract, validate

EXAMPLES = [
    ("Can I move my appointment to 10/14? Call me at 949-555-0142", {"intent": "appointment_change", "date": "10/14", "phone": "9495550142", "amount": None}),
    ("How much does the premium plan cost? I saw $1,200 on the site", {"intent": "pricing_question", "date": None, "phone": None, "amount": "1200"}),
    ("This is unacceptable, I want a refund", {"intent": "complaint", "date": None, "phone": None, "amount": None}),
    ("Thanks!", {"intent": "other", "date": None, "phone": None, "amount": None}),
]


def good_llm(prompt):
    text = prompt.split("Message: ", 1)[1]
    return json.dumps(baseline(text))


class ExtractorTests(unittest.TestCase):
    def test_baseline_extracts_fields(self):
        got = baseline(EXAMPLES[0][0])
        self.assertEqual(got, EXAMPLES[0][1])

    def test_valid_llm_output_is_used(self):
        result, src = extract("anything", llm=lambda p: json.dumps({"intent": "other", "date": None, "phone": None, "amount": None}))
        self.assertEqual((result["intent"], src), ("other", "llm"))

    def test_invalid_json_falls_back_to_rules(self):
        result, src = extract("How much is it?", llm=lambda p: "sure! the intent is pricing")
        self.assertEqual(src, "fallback")
        self.assertEqual(result["intent"], "pricing_question")

    def test_schema_violation_falls_back(self):
        bad = json.dumps({"intent": "angry_customer", "date": None, "phone": None, "amount": None})
        _, src = extract("hello", llm=lambda p: bad)
        self.assertEqual(src, "fallback")
        self.assertIsNone(validate({"intent": "other", "date": 5}))

    def test_evaluate_reports_field_scores_and_sources(self):
        report = evaluate(EXAMPLES, lambda t: extract(t, llm=good_llm))
        self.assertEqual(report["intent_accuracy"], 1.0)
        self.assertEqual(report["sources"], {"llm": 4})
        self.assertEqual(report["phone"]["f1"], 1.0)

    def test_wrong_extraction_lowers_precision(self):
        examples = [("hi", {"intent": "other", "date": None, "phone": None, "amount": None})]
        wrong = lambda t: ({"intent": "other", "date": "1/1", "phone": None, "amount": None}, "llm")
        report = evaluate(examples, wrong)
        self.assertEqual(report["date"]["precision"], 0.0)


if __name__ == "__main__":
    unittest.main()
