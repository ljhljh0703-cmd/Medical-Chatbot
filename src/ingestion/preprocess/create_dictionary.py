"""Dictionary utilities shared by ingestion, evaluation, and retrieval."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

DICTIONARY_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "ko_en_dictionary.json"
STOPWORDS_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "korean_stopwords.json"
RAW_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
KOREAN_RANGE = r"\uac00-\ud7a3"

# Base stopwords used for lightweight keyword matching (evaluation/retrieval path).
KOREAN_STOPWORDS = {
    "및",
    "또는",
    "그리고",
    "에서",
    "으로",
    "이다",
    "있다",
    "있으며",
    "있는",
    "한다",
    "통해",
    "경우",
    "주요",
    "원인",
    "증상",
    "치료",
    "진단",
    "질환",
    "사용",
    "발생",
    "환자",
    "검사",
    "상태",
    "의해",
    "위해",
    "수치",
    "확인",
    "평가",
    "의미",
    "같은",
    "특히",
}

# Regex-first extraction for medical terms/acronyms/units.
MEDICAL_KEYWORD_PATTERNS = [
    re.compile(r"\b[A-Z]{2,}[A-Z0-9+\-_/]*\b"),
    re.compile(r"\b\d+(?:\.\d+)?\s?(?:mmhg|bpm|mg/dl|mmol/l|iu/l|mg|g|kg|mcg|ml|l|%)\b", re.IGNORECASE),
    re.compile(rf"[{KOREAN_RANGE}]{{2,}}(?:증후군|질환|염|암|부전|결핍|과다|저하|항진|장애|통|병|증)"),
    re.compile(rf"[{KOREAN_RANGE}]{{2,}}(?:검사|수술|시술|요법|치료|약|주사|판독|소견)"),
    re.compile(r"\b[a-z][a-z0-9+\-_/]{2,}\b"),
]


@lru_cache(maxsize=1)
def _get_kiwi_instance():
    """Lazy-load Kiwi tokenizer to keep import dependency optional until used."""
    try:
        from kiwipiepy import Kiwi
    except ImportError as exc:
        raise RuntimeError(
            "Kiwi tokenizer가 필요합니다. `pip install kiwipiepy` 후 다시 실행하세요."
        ) from exc
    return Kiwi()


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_token(token: str) -> str:
    token = normalize_text(token)
    token = re.sub(r"^[^0-9a-z가-힣]+|[^0-9a-z가-힣]+$", "", token)
    return token


def _iter_records(raw: object) -> Iterable[dict]:
    if isinstance(raw, list):
        for row in raw:
            if isinstance(row, dict):
                yield row
    elif isinstance(raw, dict):
        for value in raw.values():
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        yield row
            elif isinstance(value, dict):
                yield value


def _extract_text_fields(record: dict) -> str:
    parts: list[str] = []
    for key in ("question", "answer", "content", "title"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n".join(parts)


def load_default_corpus_paths(raw_dir: Path = RAW_DATA_DIR) -> list[Path]:
    paths = [
        raw_dir / "TL_내과_통합.json",
        raw_dir / "VL_내과_통합.json",
    ]
    paths.extend(sorted(raw_dir.glob("TS_*.json")))
    return paths


def load_corpus_documents(paths: Optional[Iterable[Path]] = None) -> list[str]:
    """Load 5 raw JSON files (TL, VL, TS_* x3) and return text documents."""
    targets = list(paths) if paths is not None else load_default_corpus_paths()

    documents: list[str] = []
    for path in targets:
        if not path.exists():
            continue

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for record in _iter_records(raw):
            text = _extract_text_fields(record)
            if text:
                documents.append(text)

    return documents


def load_additional_stopwords(path: Path = STOPWORDS_PATH) -> set[str]:
    if not path.exists():
        return set()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()

    if isinstance(raw, list):
        return {_clean_token(str(item)) for item in raw if _clean_token(str(item))}
    if isinstance(raw, dict):
        values: set[str] = set()
        for key, value in raw.items():
            values.add(_clean_token(str(key)))
            if isinstance(value, list):
                values.update(_clean_token(str(v)) for v in value if _clean_token(str(v)))
        return {v for v in values if v}
    return set()


def build_stopwords(path: Path = STOPWORDS_PATH) -> set[str]:
    """Merge built-in KOREAN_STOPWORDS and data/raw/korean_stopwords.json."""
    merged = {_clean_token(w) for w in KOREAN_STOPWORDS if _clean_token(w)}
    merged.update(load_additional_stopwords(path))
    return merged


def _is_keyword_candidate(token: str, stopwords: set[str]) -> bool:
    if not token or token in stopwords:
        return False
    if len(token) <= 1:
        return False
    return bool(re.fullmatch(rf"[{KOREAN_RANGE}]{{2,}}|[a-z][a-z0-9+\-_/]{{1,}}|\d+(?:\.\d+)?", token))


def extract_medical_regex_keywords(text: str, stopwords: set[str]) -> tuple[list[str], str]:
    """Extract regex-based medical keywords first, and return remaining text."""
    if not text:
        return [], ""

    all_matches: list[tuple[int, int, str]] = []
    for pattern in MEDICAL_KEYWORD_PATTERNS:
        for match in pattern.finditer(text):
            token = _clean_token(match.group(0))
            if _is_keyword_candidate(token, stopwords):
                all_matches.append((match.start(), match.end(), token))

    all_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    selected: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, token in all_matches:
        if start < last_end:
            continue
        selected.append((start, end, token))
        last_end = end

    keywords = [token for _, _, token in selected]

    remaining_parts: list[str] = []
    cursor = 0
    for start, end, _ in selected:
        if cursor < start:
            remaining_parts.append(text[cursor:start])
        cursor = end
    if cursor < len(text):
        remaining_parts.append(text[cursor:])

    remaining_text = " ".join(remaining_parts)
    remaining_text = re.sub(r"\s+", " ", remaining_text).strip()
    return keywords, remaining_text


def tokenize_with_kiwi(text: str, stopwords: set[str]) -> list[str]:
    """Tokenize remaining text with Kiwi tokenizer only."""
    if not text:
        return []

    kiwi = _get_kiwi_instance()
    tokens: list[str] = []

    for token in kiwi.tokenize(text):
        term = _clean_token(token.form)
        if _is_keyword_candidate(term, stopwords):
            tokens.append(term)

    return tokens


def extract_keywords_for_document(text: str, stopwords: set[str]) -> list[str]:
    regex_tokens, remaining = extract_medical_regex_keywords(text, stopwords)
    kiwi_tokens = tokenize_with_kiwi(remaining, stopwords)
    return regex_tokens + kiwi_tokens


def compute_tf_idf(keyword_docs: list[list[str]]) -> tuple[dict[str, int], dict[str, float], dict[str, float], dict[str, float]]:
    """Return tf_count, tf_score, idf_score, tfidf_score."""
    tf_counter: Counter[str] = Counter()
    df_counter: Counter[str] = Counter()

    for tokens in keyword_docs:
        tf_counter.update(tokens)
        df_counter.update(set(tokens))

    total_terms = sum(tf_counter.values()) or 1
    n_docs = len(keyword_docs) or 1

    tf_score = {token: count / total_terms for token, count in tf_counter.items()}
    idf_score = {
        token: math.log((1 + n_docs) / (1 + df_counter[token])) + 1.0
        for token in tf_counter
    }
    tfidf_score = {token: tf_score[token] * idf_score[token] for token in tf_counter}

    return dict(tf_counter), tf_score, idf_score, tfidf_score


def build_keyword_statistics(
    paths: Optional[Iterable[Path]] = None,
    stopwords_path: Path = STOPWORDS_PATH,
) -> list[dict[str, float | int | str]]:
    """End-to-end pipeline for dictionary expansion from raw corpus.

    1) TL/VL/TS raw JSON 로드
    2) 내장 stopwords + json stopwords 병합
    3) regex 의학 키워드 우선 추출 후, 남은 텍스트 Kiwi 토큰화
    4) TF/IDF/TF-IDF 계산
    """
    documents = load_corpus_documents(paths=paths)
    stopwords = build_stopwords(stopwords_path)

    keyword_docs = [extract_keywords_for_document(doc, stopwords) for doc in documents if doc]
    tf_count, tf_score, idf_score, tfidf_score = compute_tf_idf(keyword_docs)

    rows: list[dict[str, float | int | str]] = []
    for keyword, count in sorted(tf_count.items(), key=lambda kv: (-kv[1], kv[0])):
        rows.append(
            {
                "keyword": keyword,
                "tf_count": count,
                "tf": tf_score[keyword],
                "idf": idf_score[keyword],
                "tfidf": tfidf_score[keyword],
            }
        )
    return rows


def load_keyword_synonyms(path: Path = DICTIONARY_PATH) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
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
        n_ko = normalize_text(str(ko_key))
        if not n_ko:
            continue

        values = aliases if isinstance(aliases, list) else []
        for alias in values:
            n_en = normalize_text(str(alias))
            if not n_en or n_en == n_ko:
                continue
            ko_to_en.setdefault(n_ko, set()).add(n_en)
            en_to_ko.setdefault(n_en, set()).add(n_ko)

    return ko_to_en, en_to_ko


def extract_keyword_candidates(text: str, max_keywords: int = 3) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    acronym_priority = [
        token.lower()
        for token in re.findall(r"\b[A-Z]{2,}[A-Z0-9+\-_/]*\b", text or "")
    ]

    token_pattern = rf"[{KOREAN_RANGE}]{{2,}}|[a-z][a-z0-9+\-_/]{{1,}}"
    tokens = re.findall(token_pattern, normalized)

    filtered: list[str] = []
    for token in tokens:
        if token in KOREAN_STOPWORDS:
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


def expand_keyword_variants(
    keyword: str,
    ko_to_en: dict[str, set[str]],
    en_to_ko: dict[str, set[str]],
) -> set[str]:
    n_keyword = normalize_text(keyword)
    variants = {n_keyword}

    if n_keyword in ko_to_en:
        variants.update(ko_to_en[n_keyword])
    if n_keyword in en_to_ko:
        variants.update(en_to_ko[n_keyword])

    return {v for v in variants if v}


def contains_keyword(
    text: str,
    keyword: str,
    ko_to_en: dict[str, set[str]],
    en_to_ko: dict[str, set[str]],
) -> bool:
    normalized_text = normalize_text(text)
    for variant in expand_keyword_variants(keyword, ko_to_en, en_to_ko):
        if re.search(r"[a-z]", variant):
            if re.search(rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])", normalized_text):
                return True
        else:
            if variant in normalized_text:
                return True
    return False
