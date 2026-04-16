"""
평가 오케스트레이터.

조건 A / B / C 각각에 대해 테스트셋을 실행하고 메트릭을 산출.
각 QA 건별 상세 결과를 기록하여 CSV/JSON으로 내보내기 가능.

사용 예시:
    evaluator = Evaluator(chat_service=..., test_data_path="data/test.json")
    results = evaluator.run_all()
    evaluator.save_details("eval_results")   # eval_results/ 폴더에 CSV 저장
"""

import json
import os
from typing import Optional
from evaluation.metrics import exact_match, rouge_l, bert_score, llm_judge
from domain.models.qa_pair import QAPair
from observability.logger import logger


class Evaluator:
    def __init__(
        self,
        chat_service,
        test_data_path: str,
        llm_judge_client=None,
    ):
        """
        chat_service: ChatService 인스턴스 (handle(query, mode, ground_truth) 지원)
        test_data_path: 테스트셋 JSON 파일 경로
        llm_judge_client: LLM-as-Judge용 클라이언트 (None이면 스킵)
        """
        self.chat_service = chat_service
        self.test_data_path = test_data_path
        self.llm_judge_client = llm_judge_client
        self._test_data: list[QAPair] = []
        self._details: dict[str, list[dict]] = {}  # mode → [개별 결과]

    def load_test_data(self) -> None:
        """테스트셋 JSON 로드."""
        with open(self.test_data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = [raw]
        self._test_data = [QAPair(**item) for item in raw]
        logger.info(f"[Evaluator] 테스트셋 로드: {len(self._test_data)}건")

    def evaluate_mode(self, mode: str) -> dict:
        """
        단일 모드(A/B/C) 평가 실행.

        반환: {
            "mode": str, "em": float, "rouge_l": float,
            "bert_score": float, "llm_judge": float | None,
            "n_samples": int
        }
        """
        if not self._test_data:
            self.load_test_data()

        predictions, references, questions = [], [], []
        detail_rows: list[dict] = []
        total = len(self._test_data)

        for idx, qa in enumerate(self._test_data, 1):
            # ChatService 호출 — ChatResult 반환
            chat_result = self.chat_service.handle(
                query=qa.question,
                mode=mode,
                ground_truth=qa.answer,
            )
            pred = chat_result.chatbot_answer if hasattr(chat_result, "chatbot_answer") else str(chat_result)
            predictions.append(pred)
            references.append(qa.answer)
            questions.append(qa.question)

            # 개별 건 상세 기록
            detail_rows.append({
                "qa_id": qa.qa_id,
                "question": qa.question[:80],
                "ground_truth": qa.answer,
                "prediction": pred[:200],
                "red_flag": getattr(chat_result, "red_flag_triggered", False),
                "n_sources": len(getattr(chat_result, "retrieved_sources", [])),
            })

            if idx % 20 == 0 or idx == total:
                logger.info(f"[Evaluator] mode={mode} 진행: {idx}/{total}")

        # 메트릭 산출
        em_scores = [exact_match(p, r) for p, r in zip(predictions, references)]
        rl_scores = [rouge_l(p, r) for p, r in zip(predictions, references)]
        bs_scores = bert_score(predictions, references)
        lj_scores = [
            llm_judge(q, p, r, self.llm_judge_client)
            for q, p, r in zip(questions, predictions, references)
        ]
        lj_valid = [s for s in lj_scores if s is not None]

        # 개별 건에 메트릭 추가
        for i, row in enumerate(detail_rows):
            row["em"] = em_scores[i]
            row["rouge_l"] = rl_scores[i]
            row["bert_score"] = bs_scores[i]
            row["llm_judge"] = lj_scores[i]

        self._details[mode] = detail_rows

        summary = {
            "mode": mode,
            "em": sum(em_scores) / len(em_scores),
            "rouge_l": sum(rl_scores) / len(rl_scores),
            "bert_score": sum(bs_scores) / len(bs_scores),
            "llm_judge": sum(lj_valid) / len(lj_valid) if lj_valid else None,
            "n_samples": total,
        }
        logger.info(f"[Evaluator] mode={mode} 완료 — EM={summary['em']:.3f}, ROUGE-L={summary['rouge_l']:.3f}")
        return summary

    def run_all(self) -> list[dict]:
        """조건 A / B / C 전체 평가 실행."""
        return [self.evaluate_mode(mode) for mode in ["A", "B", "C"]]

    def save_details(self, output_dir: str = "eval_results") -> None:
        """
        모드별 개별 QA 결과를 JSON으로 저장.
        output_dir/mode_A_details.json 등
        """
        os.makedirs(output_dir, exist_ok=True)
        for mode, rows in self._details.items():
            path = os.path.join(output_dir, f"mode_{mode}_details.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
            logger.info(f"[Evaluator] 상세 저장: {path} ({len(rows)}건)")

