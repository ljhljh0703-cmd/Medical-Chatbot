"""
평가 오케스트레이터.

조건 A / B / C 각각에 대해 테스트셋을 실행하고 메트릭을 산출.

사용 예시:
    evaluator = Evaluator(chat_service=..., test_data_path="data/test.json")
    results = evaluator.run_all()
"""

import json
from evaluation.metrics import exact_match, rouge_l, bert_score, llm_judge
from domain.models.qa_pair import QAPair


class Evaluator:
    def __init__(self, chat_service, test_data_path: str, llm_judge_client=None):
        """
        chat_service: ChatService 인스턴스 (mode 인자를 지원해야 함)
        test_data_path: 테스트셋 JSON 파일 경로
        llm_judge_client: LLM-as-Judge용 클라이언트 (None이면 스킵)
        """
        self.chat_service = chat_service
        self.test_data_path = test_data_path
        self.llm_judge_client = llm_judge_client
        self._test_data: list[QAPair] = []

    def load_test_data(self) -> None:
        """테스트셋 JSON 로드."""
        with open(self.test_data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = [raw]
        self._test_data = [QAPair(**item) for item in raw]

    def evaluate_mode(self, mode: str) -> dict:
        """
        단일 모드(A/B/C) 평가 실행.
        반환: { "mode": str, "em": float, "rouge_l": float,
                "bert_score": float, "llm_judge": float | None }
        """
        if not self._test_data:
            self.load_test_data()

        predictions, references, questions = [], [], []
        for qa in self._test_data:
            result = self.chat_service.handle(qa.question, mode=mode)
            predictions.append(result if isinstance(result, str) else result.chatbot_answer)
            references.append(qa.answer)
            questions.append(qa.question)

        em_scores = [exact_match(p, r) for p, r in zip(predictions, references)]
        rl_scores = [rouge_l(p, r) for p, r in zip(predictions, references)]
        bs_scores = bert_score(predictions, references)
        lj_scores = [
            llm_judge(q, p, r, self.llm_judge_client)
            for q, p, r in zip(questions, predictions, references)
        ]
        lj_valid = [s for s in lj_scores if s is not None]

        return {
            "mode": mode,
            "em": sum(em_scores) / len(em_scores),
            "rouge_l": sum(rl_scores) / len(rl_scores),
            "bert_score": sum(bs_scores) / len(bs_scores),
            "llm_judge": sum(lj_valid) / len(lj_valid) if lj_valid else None,
            "n_samples": len(self._test_data),
        }

    def run_all(self) -> list[dict]:
        """조건 A / B / C 전체 평가 실행."""
        return [self.evaluate_mode(mode) for mode in ["A", "B", "C"]]
