"""
Safety 오케스트레이터.

1. 사용자 질의에서 Red Flag 키워드 탐지
2. 첫봇 답변에 PostFilter 적용 (톤 소프트화 + 면체 조항)
3. ChatResult.red_flag_triggered 업데이트
"""

from safety.red_flag.detector import RedFlagDetector, RedFlagResult
from safety.filters.post_filter import PostFilter
from observability.logger import logger


class SafetyService:
    """
    질의 안전성 검증과 답변 후처리를 수행하는 서비스.
    RedFlagDetector와 PostFilter은 외부 주입이 가능하여 테스트 용이성을 확보한다.
    """

    def __init__(
        self,
        detector: RedFlagDetector | None = None,
        post_filter: PostFilter | None = None,
    ):
        self.detector = detector or RedFlagDetector()
        self.post_filter = post_filter or PostFilter()

    def check_query(self, query: str) -> RedFlagResult:
        """사용자 질의에서 Red Flag를 탐지하여 결과 반환."""
        result = self.detector.detect(query)
        if result.triggered:
            logger.warning(
                f"[SafetyService] Red Flag 감지 — 키워드: {result.matched_keywords}"
            )
        return result

    def process_response(
        self,
        response: str,
        red_flag_triggered: bool = False,
    ) -> str:
        """
        답변에 PostFilter를 적용하여 톤 소프트화 + 면체 조항을 처리하여 반환.

        response: 생성된 답변 원문
        red_flag_triggered: 쿼리에서 Red Flag가 감지된 경우 True
        """
        filtered = self.post_filter.filter(
            response=response,
            red_flag_triggered=red_flag_triggered,
        )
        return filtered

    def validate(self, text: str) -> bool:
        """포워드 호환성 유지 메서드. Red Flag가 없으면 True 반환."""
        return not self.detector.detect(text).triggered
