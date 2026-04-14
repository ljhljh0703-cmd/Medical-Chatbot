"""
답변 후처리 필터.

1. 단정형 어미("입니다", "합니다" 등)를 권유형으로 소프트 보정
2. 면체 조항 자동 체부
3. Red Flag 감지 시 긴급 상담 권고 배너 삽입
"""

import re

DISCLAIMER = (
    "\n\n---\n"
    "⚠️ 이 답변은 참고용이며, 정확한 진단과 치료는 나버 반드시 의료 전문가와 상담하시기 바랍니다."
)

RED_FLAG_BANNER = (
    "🚨 **위급 증상이 의심됩니다.** "
    "지체 없이 응급실을 방문하거나 119에 신고하시기 바랍니다.\n\n"
)


class PostFilter:
    """
    답변 텍스트를 안전하게 가공하는 필터 클래스.

    add_disclaimer: 면체 조항 체부 여부 (True)
    soften_tone: 단정형 어미 권유형으로 보정 여부 (True)
    """

    # 단정형 → 권유형 매핑 (순서 중요: 긴 패턴 먼저)
    _TONE_MAP: list[tuple[str, str]] = [
        (r"해야 합니다\.?",         "하시는 것을 권장드립니다."),
        (r"해야 합니다",             "하시는 것을 권장드립니다"),
        (r"을 해야합니다",          "을 고려하시기 바랍니다"),
        (r"를 해야합니다",          "를 고려하시기 바랍니다"),
        (r"입니다\.?",                   "수 있습니다."),
        (r"합니다\.?",                   "을 권장드립니다."),
    ]

    def __init__(self, add_disclaimer: bool = True, soften_tone: bool = True):
        self.add_disclaimer = add_disclaimer
        self.soften_tone = soften_tone

    def _apply_tone(self, text: str) -> str:
        """단정형 어미를 권유형으로 소프트 보정."""
        for pattern, replacement in self._TONE_MAP:
            text = re.sub(pattern, replacement, text)
        return text

    def filter(
        self,
        response: str,
        red_flag_triggered: bool = False,
    ) -> str:
        """
        response: 생성된 답변 텍스트
        red_flag_triggered: SafetyService에서 전달받은 Red Flag 여부
        """
        result = response

        # 1. Red Flag 배너 삽입 (답변 앞단에 위치)
        if red_flag_triggered:
            result = RED_FLAG_BANNER + result

        # 2. 단정형 어미 소프트 보정
        if self.soften_tone:
            result = self._apply_tone(result)

        # 3. 면체 조항 체부
        if self.add_disclaimer:
            # 중복 체부 방지
            if DISCLAIMER.strip() not in result:
                result += DISCLAIMER

        return result
