"""
평가 메트릭 모듈.

지원 메트릭:
- EM (Exact Match): 객관식 정답 번호 일치 여부
- ROUGE-L: 긴 서술형 답변의 겹침 측정
- BERTScore: 의미적 유사도
- LLM-as-Judge: GPT-4 등을 활용한 품질 평가 (1~5점)
"""

import re
from typing import Optional


def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Exact Match: 객관식 정답 번호(예: '4)')만 추출하여 비교.
    일치하면 1.0, 불일치하면 0.0 반환.
    """
    def extract_choice(text: str) -> str:
        match = re.search(r"^\s*(\d+)[).）]", text.strip())
        return match.group(1) if match else text.strip()

    return 1.0 if extract_choice(prediction) == extract_choice(ground_truth) else 0.0


def rouge_l(prediction: str, ground_truth: str) -> float:
    """
    ROUGE-L: LCS(최장 공통 부분 수열) 기반 F1 점수.
    외부 의존성 없이 순수 Python으로 구현.
    """
    def lcs_length(a: list, b: list) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
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
    """
    BERTScore: transformers 기반 의미적 유사도.
    bert_score 패키지 필요 (requirements.txt 참고).
    """
    try:
        from bert_score import score as _score
        _, _, F1 = _score(predictions, references, lang="ko", verbose=False)
        return F1.tolist()
    except ImportError:
        print("[BERTScore] bert-score 패키지가 설치되지 않았습니다.")
        return [0.0] * len(predictions)


def llm_judge(
    question: str,
    prediction: str,
    ground_truth: str,
    client=None,  # OpenAI / Qwen 클라이언트 주입
) -> Optional[float]:
    """
    LLM-as-Judge: LLM에게 정확성·완전성·안전성을 1~5점으로 채점 요청.
    client가 None이면 스킵하고 None 반환.
    향후 client 인터페이스 확장 가능.
    """
    if client is None:
        return None

    prompt = (
        f"다음 의료 질문에 대한 챗봇 답변을 평가해 주세요.\n\n"
        f"질문: {question}\n"
        f"정답: {ground_truth}\n"
        f"챗봇 답변: {prediction}\n\n"
        f"정확성, 완전성, 안전성을 종합하여 1~5점으로만 답하세요. 숫자만 출력하세요."
    )
    try:
        response = client.request(prompt)
        score = float(re.search(r"[1-5]", response).group())
        return score
    except Exception:
        return None
