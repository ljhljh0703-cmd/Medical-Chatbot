"""
쿼리 정규화 모듈.

수행 작업:
1. 앞뒤 공백 제거 및 중복 공백 정제
2. 의료 약어 → 풀네임 확장 (룰 기반, 확장 가능)
3. 자주 혼동되는 오타 보정 (룰 기반, 확장 가능)
"""

import re


class QueryNormalizer:
    """
    사용자 질문을 정규화하는 클래스.

    ABBREVIATIONS, TYPO_MAP은 클래스 변수로 관리하여
    외부에서 QueryNormalizer.ABBREVIATIONS.update({...}) 로 쉽게 확장 가능.
    """

    # 의료 약어 → 풀네임 (대소문자 무시 매칭)
    ABBREVIATIONS: dict[str, str] = {
        r"\bFEV1\b": "1초간 강제날숨유량",
        r"\bFVC\b":  "강제 폐활량",
        r"\bBP\b":   "혈압",
        r"\bHR\b":   "맥박",
        r"\bRR\b":   "호흡수",
        r"\bBT\b":   "체온",
        r"\bHb\b":   "혈색소",
        r"\bWBC\b":  "백혈구",
        r"\bPlt\b":  "혈소판",
        r"\bCRP\b":  "C-반응단백질",
        r"\bCT\b":   "컴퓨터단층촬영",
        r"\bMRI\b":  "자기공명영상",
        r"\bECG\b":  "심전도",
        r"\bPT\b":   "프로트롬빈시간",
        r"\baPTT\b": "활성화부분트롬보플라스틴시간",
        r"\bBMI\b":  "체질량지수",
        # 향후 추가 시 여기에 등록
    }

    # 오타 → 정답 (단순 문자열 치환)
    TYPO_MAP: dict[str, str] = {
        "혈압약":    "혈압강하제",
        "당뇨약":    "혈당강하제",
        "심장약":    "심혈관계 약물",
        "위내시경":  "상부 위장관 내시경",
        # 향후 추가 시 여기에 등록
    }

    def normalize(self, query: str) -> str:
        """쿼리 문자열을 정규화하여 반환."""
        if not query or not query.strip():
            return query

        text = query.strip()

        # 1. 중복 공백 정제
        text = re.sub(r" {2,}", " ", text)

        # 2. 의료 약어 확장 (대소문자 무시)
        for pattern, expansion in self.ABBREVIATIONS.items():
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)

        # 3. 오타 보정
        for typo, correct in self.TYPO_MAP.items():
            text = text.replace(typo, correct)

        return text.strip()
