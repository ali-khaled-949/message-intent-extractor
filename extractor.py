"""Text classification and information extraction with an LLM, validated, with a rules fallback and field-level scoring.

`llm` is any function that takes a prompt and returns text. In production it wraps a model call; in tests it is a stub.
Model output is never trusted: it must parse as JSON and pass the schema, or the rules baseline answers instead.
"""
import json
import re

INTENTS = {"pricing_question", "appointment_change", "complaint", "other"}
FIELDS = ("date", "phone", "amount")

PHONE = re.compile(r"\(?\b(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})\b")
DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b")
AMOUNT = re.compile(r"\$\s?(\d[\d,]*(?:\.\d{2})?)")

PROMPT = (
    "Classify the message into one of: pricing_question, appointment_change, complaint, other. "
    "Extract date, phone, amount if present (null otherwise). "
    'Reply with JSON only: {"intent": ..., "date": ..., "phone": ..., "amount": ...}\nMessage: '
)


def baseline(text: str) -> dict:
    low = text.lower()
    if re.search(r"\b(reschedul|cancel|move (it|my)|change (my )?(appointment|time))", low):
        intent = "appointment_change"
    elif re.search(r"\b(price|pricing|cost|how much|quote)\b", low):
        intent = "pricing_question"
    elif re.search(r"\b(refund|terrible|unacceptable|angry|complain|worst)\b", low):
        intent = "complaint"
    else:
        intent = "other"
    phone = PHONE.search(text)
    date = DATE.search(text)
    amount = AMOUNT.search(text)
    return {
        "intent": intent,
        "date": date.group(1) if date else None,
        "phone": "".join(phone.groups()) if phone else None,
        "amount": amount.group(1).replace(",", "") if amount else None,
    }


def validate(obj) -> dict | None:
    """Return a clean dict, or None if the model's output breaks the schema."""
    if not isinstance(obj, dict) or obj.get("intent") not in INTENTS:
        return None
    clean = {"intent": obj["intent"]}
    for f in FIELDS:
        v = obj.get(f)
        if v is not None and not isinstance(v, str):
            return None
        clean[f] = v
    return clean


def extract(text: str, llm=None) -> tuple[dict, str]:
    """Return (result, source). source is 'llm', 'rules', or 'fallback' (the model answered but failed validation)."""
    if llm is None:
        return baseline(text), "rules"
    try:
        parsed = validate(json.loads(llm(PROMPT + text)))
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if parsed is None:
        return baseline(text), "fallback"
    return parsed, "llm"


def evaluate(examples: list[tuple[str, dict]], fn) -> dict:
    """examples: (text, expected). fn(text) -> (result, source). Reports intent accuracy and per-field precision/recall/F1."""
    correct = 0
    counts = {f: {"tp": 0, "fp": 0, "fn": 0} for f in FIELDS}
    sources: dict[str, int] = {}
    for text, expected in examples:
        got, src = fn(text)
        sources[src] = sources.get(src, 0) + 1
        correct += got["intent"] == expected["intent"]
        for f in FIELDS:
            g, e = got.get(f), expected.get(f)
            if g is not None and g == e:
                counts[f]["tp"] += 1
            else:
                counts[f]["fp"] += g is not None
                counts[f]["fn"] += e is not None
    report = {"intent_accuracy": correct / len(examples), "sources": sources}
    for f, c in counts.items():
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        report[f] = {"precision": p, "recall": r, "f1": (2 * p * r / (p + r)) if p + r else 0.0}
    return report
