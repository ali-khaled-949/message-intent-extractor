# message-intent-extractor

Text classification and information extraction with an LLM, where the model's output is checked before it is used.

- **Classify** a customer message into `pricing_question`, `appointment_change`, `complaint` or `other`.
- **Extract** `date`, `phone` and `amount` as structured fields.
- **Validate**: the model must return JSON that passes the schema (allowed intents, string-or-null fields). Anything else is discarded and the rules baseline answers, and the result is tagged `fallback` so you can see how often it happens.
- **Evaluate**: `evaluate()` reports intent accuracy and per-field precision, recall and F1, plus how many answers came from the model, the rules, or the fallback.

The model is a function argument, so the tests run offline with stubs. In production it would wrap a real model call.

## Limits

- The rules baseline is small regex and keyword logic, and the test set has a handful of examples. This demonstrates the pipeline and evaluation, not accuracy on real data.
- No third-party packages, Python 3.10 or newer.

## Run

```bash
python3 -m unittest
```

Files: `extractor.py`, `test_extractor.py`
