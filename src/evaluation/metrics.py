"""Evaluation metrics module."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

_DICTIONARY_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "ko_en_dictionary.json"
_KOREAN_RANGE = r"\uac00-\ud7a3"

_KOREAN_STOPWORDS = {
    "및", "또는", "그리고", "에서", "으로", "이다", "있다", "있으며", "있는", "한다", "통해",
    "경우", "주요", "원인", "증상", "치료", "진단", "질환", "사용", "발생", "환자", "검사",
    "상태", "의해", "위해", "수치", "확인", "평가", "의미", "같은", "특히",
}


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _load_keyword_synonyms(path: Path = _DICTIONARY_PATH) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Load ko->en dictionary from JSON and build reverse en->ko map."""
    ko_to_en: dict[str, set[str]] = {}
    en_to_ko: dict[str, set[str]] = {}

    if not path.exists():
        return ko_to_en, en_to_ko

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ko_to_en, en_to_ko

    if not isinstance(raw, dict):
        return ko_to_en, en_to_ko

    for ko_key, aliases in raw.items():
        n_ko = _normalize(str(ko_key))
        if not n_ko:
            continue

        values = aliases if isinstance(aliases, list) else []
        for alias in values:
            n_en = _normalize(str(alias))
            if not n_en or n_en == n_ko:
                continue
            ko_to_en.setdefault(n_ko, set()).add(n_en)
            en_to_ko.setdefault(n_en, set()).add(n_ko)

    return ko_to_en, en_to_ko


_KO_TO_EN_SYNONYMS, _EN_TO_KO_SYNONYMS = _load_keyword_synonyms()


def _extract_choice_number(text: str) -> Optional[str]:
    text = (text or "").strip()
    match = re.search(r"^\s*(\d+)\s*[).]", text)
    if match:
        return match.group(1)
    match = re.search(r"(?<!\d)([1-5])(?!\d)", text)
    return match.group(1) if match else None


def _extract_keyword_candidates(text: str, max_keywords: int = 3) -> list[str]:
    normalized = _normalize(text)
    if not normalized:
        return []

    acronym_priority = [
        token.lower()
        for token in re.findall(r"\b[A-Z]{2,}[A-Z0-9+\-_/]*\b", text or "")
    ]

    token_pattern = rf"[{_KOREAN_RANGE}]{{2,}}|[a-z][a-z0-9+\-_/]{{1,}}"
    tokens = re.findall(token_pattern, normalized)

    filtered: list[str] = []
    for token in tokens:
        if token in _KOREAN_STOPWORDS:
            continue
        if len(token) <= 1:
            continue
        filtered.append(token)

    if not filtered:
        return []

    counts = Counter(filtered)
    first_idx: dict[str, int] = {}
    for idx, token in enumerate(filtered):
        if token not in first_idx:
            first_idx[token] = idx

    ranked = sorted(counts.keys(), key=lambda t: (-counts[t], first_idx[t]))

    merged: list[str] = []
    for token in acronym_priority + ranked:
        if token not in merged:
            merged.append(token)
        if len(merged) >= max_keywords:
            break

    return merged[:max_keywords]


def _expand_keyword_variants(keyword: str) -> set[str]:
    n_keyword = _normalize(keyword)
    variants = {n_keyword}

    if n_keyword in _KO_TO_EN_SYNONYMS:
        variants.update(_KO_TO_EN_SYNONYMS[n_keyword])
    if n_keyword in _EN_TO_KO_SYNONYMS:
        variants.update(_EN_TO_KO_SYNONYMS[n_keyword])

    return {v for v in variants if v}


def _contains_keyword(text: str, keyword: str) -> bool:
    normalized_text = _normalize(text)
    for variant in _expand_keyword_variants(keyword):
        if re.search(r"[a-z]", variant):
            if re.search(rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])", normalized_text):
                return True
        else:
            if variant in normalized_text:
                return True
    return False


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
