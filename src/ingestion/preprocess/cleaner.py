"""
원천 데이터 전처리 모듈.

수행 작업:
1. 번호 접두사(\'1. \', \'2. \') 씁의 공백 정리
2. 중복 공백/줄바꿼 정제
3. 제어문자(상위 ASCII 제어 코드) 제거
4. 면단이 없는 삭제
"""

import re


class TextCleaner:
    """원천 JSON content 필드를 정제하는 클래스."""

    # 번호+점+공백 패턴: "1. ", "2. " 등
    _NUMBER_PREFIX = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
    # 2칸 이상 공백 통합
    _MULTI_SPACE = re.compile(r" {2,}")
    # 3줄 이상 연속 줄바꿼 통합
    _MULTI_NEWLINE = re.compile(r"\n{3,}")
    # 제어문자 (소수 ASCII 0x00-0x1F, 0x7F)
    _CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")

    def clean(self, text: str) -> str:
        """단일 content 문자열을 정제하여 반환."""
        if not text or not text.strip():
            return ""

        # 제어문자 제거 (줌바꿼 \n, \t 는 보존)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # 번호 접두사 정리: "1. ", "2. " → 내용만 보존
        # (\n1. 로 시작되는 라인의 번호 제거 안 함 — 청크 분할에 활용)
        # 중복 공백 정제
        text = self._MULTI_SPACE.sub(" ", text)
        # 중복 줄바꿼 정제
        text = self._MULTI_NEWLINE.sub("\n\n", text)
        return text.strip()

    def clean_doc(self, doc_dict: dict) -> dict:
        """원천 데이터 dict의 content 필드만 정제하여 새 dict 반환."""
        cleaned = dict(doc_dict)
        cleaned["content"] = self.clean(doc_dict.get("content", ""))
        return cleaned
