"""
Red Flag 감지 모듈.

사용자 질의 또는 챗봇 답변에 위험 키워드가 포함되면
증상을 Red Flag로 타깅하여 빠른 의사 상담을 권고한다.

RED_FLAG_KEYWORDS는 클래스 변수로 선언하여
    RedFlagDetector.RED_FLAG_KEYWORDS.append("신규 키워드")로 쉽게 확장 가능.
"""

import re
from dataclasses import dataclass, field


@dataclass
class RedFlagResult:
    """Red Flag 탐지 결과."""
    triggered: bool
    matched_keywords: list[str] = field(default_factory=list)
    bypass: bool = False  # True이면 LLM/RAG 파이프라인 즉시 중단


class RedFlagDetector:
    """
    입력 텍스트에서 Red Flag 키워드를 탐지하는 엔진 클래스.

    키워드는 부분 일치(substring)로 탐지하며,
    대소문자 무시 매칭을 적용한다.
    """

    # ── 초기 Red Flag 키워드 목록 ───────────────────────────────────
    # 사용자가 .append() 또는 .extend()로 쉽게 확장 가능
    RED_FLAG_KEYWORDS: list[str] = [
        # 의식 장애
        "의식소실", "의식장애", "실신",
        # 호흡 위기
        "호흡공난", "숨이 막힙", "숨을 못 쉬썬", "숫자마퀴리는",
        # 혈관 위기
        "흘흐견", "혈당 초저하", "혈압 강하하여", "심정지", "심박수",
        "흑통", "대량 출혁", "첿허나오는 혈",
        # 심프
        "흥통증", "심한 심장 통증",
        # 신경 이상
        "마비", "경련", "발작", "눈이 보이지 않음", "시야 장애",
        "머리가 턱지는 듯한", "갑지기 심한 두통",
        # 정신 위기
        "자살", "스스로 해치다", "죽고 싶다", "생이 무의미",
        # 약물 과다복용
        "약물 과다복용", "개의 약 먹음", "수면제 과다복용",
        # 아나필라틱스
        "아나필라틱스", "전신 두드러기", "혼떨림",
        # 고열 / 저체온
        "체온 40도", "저체온증", "고열 지속",
    ]

    # ── 즉각 bypass를 트리거하는 정규표현식 패턴 ───────────────────────────────
    # 이 패턴에 매칭되면 RAG/LLM 파이프라인을 건너뛰고 즉시 응급 메시지 반환
    BYPASS_PATTERNS: list[str] = [
        r"피를?\s*토",              # 피를 토, 피 토해요
        r"의식\s*을?\s*잃",         # 의식을 잃, 의식 잃
        r"가슴\s*이?\s*찢어",       # 가슴이 찢어질
        r"말\s*이?\s*어눌",         # 말이 어눌, 말 어눌
        r"쓰러\s*[졌진지]",         # 쓰러졌, 쓰러진, 쓰러지
        r"숨\s*[을를]?\s*못\s*쉬",  # 숨을 못 쉬, 숨 못 쉬
        r"(팔|다리|얼굴).{0,6}마비",# 팔/다리/얼굴 마비
        r"갑자기\s*심한\s*두통",    # 갑자기 심한 두통
        r"(전신|온몸)\s*(마비|경련)",# 전신 마비, 온몸 경련
        r"눈\s*[이가]?\s*안\s*보",  # 눈이 안 보여요
        r"(기절|졸도)\s*[했할]",    # 기절했, 졸도했
        r"심장\s*이?\s*멈",         # 심장이 멈
        r"119\s*불러",              # 119 불러야 해요? (능동 확인)
    ]

    def detect(self, text: str) -> RedFlagResult:
        """
        텍스트에서 Red Flag 키워드를 탐지하여 RedFlagResult 반환.

        - BYPASS_PATTERNS에 매칭되면 bypass=True (LLM/RAG 즉시 중단)
        - RED_FLAG_KEYWORDS에 매칭되면 triggered=True (파이프라인 계속, 배너만 추가)

        text: 사용자 질의 또는 챗봇 답변
        """
        if not text:
            return RedFlagResult(triggered=False)

        lower = text.lower()

        # 1. bypass 패턴 검사 (우선순위 높음)
        for pattern in self.BYPASS_PATTERNS:
            if re.search(pattern, lower):
                return RedFlagResult(
                    triggered=True,
                    matched_keywords=[pattern],
                    bypass=True,
                )

        # 2. 일반 키워드 검사 (파이프라인 유지, 배너만)
        matched = [kw for kw in self.RED_FLAG_KEYWORDS if kw.lower() in lower]
        return RedFlagResult(triggered=bool(matched), matched_keywords=matched)
