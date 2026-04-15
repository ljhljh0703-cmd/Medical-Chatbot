"""
평가 결과 리포트 생성 모듈.

터미널 출력, Streamlit 렌더링, JSON 내보내기 지원.
"""

import json
import os
from typing import Optional


def _delta_str(current: float, baseline: float) -> str:
    """퍼센트포인트 차이 문자열 생성."""
    diff = (current - baseline) * 100
    return f"(Δ{diff:+.1f}%p)"


def format_report(results: list[dict]) -> str:
    """
    조건 A/B/C 비교 리포트를 터미널 텍스트로 포맷.

    예시 출력:
    ─────────────────────────────────────────────────────────────────
     Mode  |   EM    | ROUGE-L  | BERTScore  | LLM-Judge  |   N
    ─────────────────────────────────────────────────────────────────
      A    |  0.420  |  0.380   |   0.710    |    3.20    |  200
      B    |  0.570  |  0.530   |   0.810    |    3.80    |  200   EM: (Δ+15.0%p)  RL: (Δ+15.0%p)
      C    |  0.630  |  0.590   |   0.850    |    4.10    |  200   EM: (Δ+21.0%p)  RL: (Δ+21.0%p)
    ─────────────────────────────────────────────────────────────────

    → RAG 추가 (A→B): EM +15.0%p
    → LoRA 추가 (A→C): EM +21.0%p
    """
    header = f"{'Mode':^6} | {'EM':^7} | {'ROUGE-L':^8} | {'BERTScore':^10} | {'LLM-Judge':^10} | {'N':^5}"
    sep = "─" * len(header)
    lines = [sep, header, sep]

    baseline = results[0] if results else None
    for r in results:
        lj = f"{r['llm_judge']:.2f}" if r["llm_judge"] is not None else " N/A "
        delta = ""
        if baseline and r is not baseline:
            delta = (
                f"  EM: {_delta_str(r['em'], baseline['em'])}"
                f"  RL: {_delta_str(r['rouge_l'], baseline['rouge_l'])}"
            )
        lines.append(
            f"{r['mode']:^6} | {r['em']:^7.3f} | {r['rouge_l']:^8.3f} | "
            f"{r['bert_score']:^10.3f} | {lj:^10} | {r['n_samples']:^5}{delta}"
        )

    lines.append(sep)

    # 요약 코멘트
    if len(results) >= 2:
        lines.append("")
        for i in range(1, len(results)):
            em_delta = (results[i]["em"] - results[0]["em"]) * 100
            label = "RAG 추가" if results[i]["mode"] == "B" else "LoRA 추가"
            lines.append(f"→ {label} (A→{results[i]['mode']}): EM {em_delta:+.1f}%p")

    return "\n".join(lines)


def print_report(results: list[dict]) -> None:
    """터미널에 리포트 출력."""
    print("\n" + format_report(results) + "\n")


def save_report_json(results: list[dict], path: str = "eval_results/summary.json") -> None:
    """평가 결과를 JSON 파일로 저장."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Report] 저장 완료: {path}")


def streamlit_report(results: list[dict]) -> None:
    """
    Streamlit 대시보드에 리포트 렌더링.
    테이블 + 메트릭 카드 + 바 차트.
    """
    try:
        import streamlit as st
        import pandas as pd

        df = pd.DataFrame(results)
        df = df.rename(columns={
            "mode": "모드", "em": "EM", "rouge_l": "ROUGE-L",
            "bert_score": "BERTScore", "llm_judge": "LLM-Judge", "n_samples": "샘플수"
        })

        st.subheader("📊 조건별 정확도 비교 (A / B / C)")
        st.dataframe(df, use_container_width=True)

        # EM 기준 delta 메트릭 카드
        if len(results) >= 2:
            cols = st.columns(len(results) - 1)
            for i, col in enumerate(cols, 1):
                delta_em = (results[i]["em"] - results[0]["em"]) * 100
                label = "RAG 추가" if results[i]["mode"] == "B" else "LoRA 추가"
                col.metric(
                    label=f"{label} (A→{results[i]['mode']})",
                    value=f"{results[i]['em']:.1%}",
                    delta=f"{delta_em:+.1f}%p",
                )

        # 바 차트
        metric_cols = ["EM", "ROUGE-L", "BERTScore"]
        chart_df = df.set_index("모드")[metric_cols]
        st.bar_chart(chart_df)

    except ImportError:
        print_report(results)
