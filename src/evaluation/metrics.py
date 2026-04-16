"""Evaluation metrics module."""

from __future__ import annotations

import re
from typing import Optional

from ingestion.preprocess.create_dictionary import (
    contains_keyword,
    extract_keyword_candidates,
    load_keyword_synonyms,
    normalize_text,
)

_KO_TO_EN_SYNONYMS, _EN_TO_KO_SYNONYMS = load_keyword_synonyms()


def _normalize(text: str) -> str:
    return normalize_text(text)


def _extract_choice_number(text: str) -> Optional[str]:
    text = (text or "").strip()
    match = re.search(r"^\s*(\d+)\s*[).]", text)
    if match:
        return match.group(1)
    match = re.search(r"(?<!\d)([1-5])(?!\d)", text)
    return match.group(1) if match else None


def _extract_keyword_candidates(text: str, max_keywords: int = 3) -> list[str]:
    return extract_keyword_candidates(text, max_keywords=max_keywords)


def _contains_keyword(text: str, keyword: str) -> bool:
    return contains_keyword(text, keyword, _KO_TO_EN_SYNONYMS, _EN_TO_KO_SYNONYMS)


def exact_match(prediction: str, ground_truth: str, q_type: Optional[int | str] = None) -> float:
    """
    q_type-based Exact Match:
    - q_type=1: number match for multiple-choice
    - q_type=2: keyword match for short-answer (ko/en aliases via JSON dictionary)
    - q_type=3: pass if any of top 3 extracted GT keywords appears in prediction
    """
    try:
        q_type_int = int(q_type) if q_type is not None else None
    except (TypeError, ValueError):
        q_type_int = None

    if q_type_int == 1:
        pred_no = _extract_choice_number(prediction)
        gt_no = _extract_choice_number(ground_truth)
        if pred_no is None or gt_no is None:
            return 0.0
        return 1.0 if pred_no == gt_no else 0.0

    if q_type_int == 2:
        return 1.0 if _contains_keyword(prediction, ground_truth) else 0.0

    if q_type_int == 3:
        keywords = _extract_keyword_candidates(ground_truth, max_keywords=3)
        if not keywords:
            return 0.0
        return 1.0 if any(_contains_keyword(prediction, kw) for kw in keywords) else 0.0

    return 1.0 if _normalize(prediction) == _normalize(ground_truth) else 0.0


def rouge_l(prediction: str, ground_truth: str) -> float:
    """ROUGE-L F1 based on LCS."""

    def lcs_length(a: list[str], b: list[str]) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    pred_tokens = prediction.split()
    ref_tokens = ground_truth.split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def bert_score(predictions: list[str], references: list[str]) -> list[float]:
    """BERTScore using bert-score package."""
    try:
        from bert_score import score as _score

        _, _, f1 = _score(predictions, references, lang="ko", verbose=False)
        return f1.tolist()
    except ImportError:
        print("[BERTScore] bert-score package is not installed.")
        return [0.0] * len(predictions)


def llm_judge(
    question: str,
    prediction: str,
    ground_truth: str,
    client=None,
) -> Optional[float]:
    """LLM-as-Judge: ask model to rate answer quality from 1 to 5."""
    if client is None:
        return None

    prompt = (
        f"Evaluate the quality of the generated medical answer.\n\n"
        f"Question: {question}\n"
        f"Reference Answer: {ground_truth}\n"
        f"Generated Answer: {prediction}\n\n"
        f"Return only one number from 1 to 5 based on correctness, completeness, and safety."
    )
    try:
        response = client.request(prompt)
        score = float(re.search(r"[1-5]", response).group())
        return score
    except Exception:
        return None
