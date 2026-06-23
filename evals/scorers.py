"""Scoring functions for the proxy eval suite.

Exact/normalized match for short-answer categories (qa/classification/
extraction/reasoning) and token-level F1 for free-text (summarization). These
mirror the kind of objective metric the hackathon leaderboard uses, so the
operating point you pick here transfers.
"""
from __future__ import annotations

import re
from typing import List


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, drop leading articles."""
    t = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def normalized_match(pred: str, gold: str) -> bool:
    """True if normalized gold and pred are equal or one contains the other.

    Handles "Shakespeare" vs "William Shakespeare" and a model that wraps the
    answer in a short sentence.
    """
    np_, ng = normalize(pred), normalize(gold)
    if not ng:
        return False
    return np_ == ng or ng in np_ or np_ in ng


def _tokens(text: str) -> List[str]:
    return normalize(text).split()


def token_f1(pred: str, gold: str) -> float:
    """Standard SQuAD-style token-overlap F1 in [0, 1]."""
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return 1.0 if p == g else 0.0
    common: dict = {}
    for tok in p:
        if tok in g:
            common[tok] = min(p.count(tok), g.count(tok))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def score(category: str, pred: str, gold: str) -> float:
    """Score one prediction in [0, 1], picking the metric by category."""
    if gold is None:
        return 0.0
    if category == "summarization":
        return token_f1(pred, gold)
    return 1.0 if normalized_match(pred, gold) else token_f1(pred, gold)
