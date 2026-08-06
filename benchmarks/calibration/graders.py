"""Mechanical pass/fail grading for calibration responses.

The calibration sweep needs a *label* for every (prompt, device) pair: did
this device answer correctly, yes or no. Reading a few hundred answers by
hand is what turns calibration from an afternoon into a week, so every
calibration item carries a grader that decides mechanically.

That constraint is why the calibration set is built from tasks with a single
checkable answer (arithmetic/word problems, multiple choice, short factual
recall) rather than open-ended prose. The trade-off is explicit and worth
stating when reporting results: the fitted heads learn "which tier can do
verifiable reasoning on this prompt", not "which tier writes better prose".

``ecorouter``'s existing ``benchmarks/score_logs.py`` compares the *entire*
response to the expected string, which fails on any real model output ("The
answer is 84." != "84"). These graders extract the answer first.
"""
from __future__ import annotations

import re
from typing import Callable

# Appended to every calibration prompt by run_sweep.py so all three tiers are
# asked for the same output shape. Without it a 0.6B model's rambling tail is
# graded against a large model's tidy one-liner, and the label measures
# formatting compliance rather than capability.
ANSWER_HINT = "\n\nEnd your reply with a final line of the form: Answer: <answer>"

_ANSWER_LINE = re.compile(r"answer\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NUMBER = re.compile(r"-?\$?\d[\d,]*\.?\d*")
_CHOICE_TOKEN = re.compile(r"\b([A-Da-d])\b")


def _answer_segment(response: str) -> str:
    """Return the text after the last ``Answer:`` marker, else the whole reply.

    Falling back to the whole reply matters: a weak model often ignores the
    format instruction but still states the right answer. Grading it ``False``
    purely for that would teach head A that the phone fails prompts it
    actually got right.
    """

    matches = _ANSWER_LINE.findall(response)
    return matches[-1] if matches else response


def _to_float(token: str) -> float | None:
    cleaned = token.replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def grade_number(response: str, expected: str) -> bool:
    """True when the response's final number matches ``expected`` numerically.

    Compares as numbers, not strings, so "18", "18.0" and "$18" all match --
    formatting differences are not capability differences.
    """

    target = _to_float(expected)
    if target is None:
        return False

    segment = _answer_segment(response)
    candidates = _NUMBER.findall(segment) or _NUMBER.findall(response)
    if not candidates:
        return False

    value = _to_float(candidates[-1])
    if value is None:
        return False
    return abs(value - target) <= max(1e-6, abs(target) * 1e-6)


def grade_choice(response: str, expected: str) -> bool:
    """True when the response's final A-D selection matches ``expected``."""

    segment = _answer_segment(response)
    candidates = _CHOICE_TOKEN.findall(segment) or _CHOICE_TOKEN.findall(response)
    if not candidates:
        return False
    return candidates[-1].upper() == expected.strip().upper()


def grade_contains(response: str, expected: str) -> bool:
    """True when every ``|``-separated required phrase appears in the response.

    Used for short factual recall where the answer is a word rather than a
    number. Alternatives for one slot are written ``a/b`` inside a segment.
    """

    haystack = " ".join(response.lower().split())
    for required in expected.split("|"):
        alternatives = [alt.strip().lower() for alt in required.split("/") if alt.strip()]
        if not any(alt in haystack for alt in alternatives):
            return False
    return True


GRADERS: dict[str, Callable[[str, str], bool]] = {
    "number": grade_number,
    "choice": grade_choice,
    "contains": grade_contains,
}


def grade(response: str, expected: str, grader: str) -> bool:
    if grader not in GRADERS:
        raise ValueError(f"unknown grader {grader!r}; expected one of {sorted(GRADERS)}")
    if not response or not response.strip():
        return False
    return GRADERS[grader](response, expected)
