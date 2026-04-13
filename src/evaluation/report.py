"""
평가 결과 리포트 생성 모듈.

터미널 출력 및 Streamlit 렌더링 두 가지 방식 지원.
"""


def format_report(results: list[dict]) -> str:
    """
    조건 A/B/C 비교 리포트를 터미널 텍스트로 포맷.

    예시 출력:
    ┌──────────┬────────┬────────┬────────────┬────────────┐
    │  모드    │   EM   │ ROUGE-L│ BERTScore  │ LLM-Judge  │
    ├──────────┼────────┼────────┼────────────┼────────────┤
    │    A     │ 0.42   │ 0.38   │   0.71     │    3.2     │
    │    B     │ 0.57   │ 0.53   │   0.81     │    3.8     │
    │    C     │ 0.63   │ 0.59   │   0.85     │    4.1     │
    └──────────┴────────┴────────┴────────────┴────────────┘
    """
    header = f"{'모드':^6} | {'EM':^6} | {'ROUGE-L':^8} | {'BERTScore':^10} | {'LLM-Judge':^10} | {'샘플수':^6}"
    sep = "-" * len(header)
    lines = [sep, header, sep]

    baseline_em = None
    for r in results:
        lj = f"{r['llm_judge']:.3f}" if r["llm_judge"] is not None else "  N/A  "
        delta = ""
        if baseline_em is not None:
            diff = (r["em"] - baseline_em) * 100
            delta = f"  (EM Δ{diff:+.1f}%)"
        lines.append(
            f"{r['mode']:^6} | {r['em']:^6.3f} | {r['rouge_l']:^8.3f} | "
            f"{r['bert_score']:^10.3f} | {lj:^10} | {r['n_samples']:^6}{delta}"
        )
        if baseline_em is None:
            baseline_em = r["em"]

    lines.append(sep)
    return "\n".join(lines)


def print_report(results: list[dict]) -> None:
    """터미널에 리포트 출력."""
    print(format_report(results))


def streamlit_report(results: list[dict]) -> None:
    """Streamlit 대시보드에 리포트 렌더링."""
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

        # EM 기준 delta 표시
        if len(results) >= 2:
            for i in range(1, len(results)):
                delta_em = (results[i]["em"] - results[0]["em"]) * 100
                st.metric(
                    label=f"조건 {results[i]['mode']} vs A (EM 기준)",
                    value=f"{results[i]['em']:.1%}",
                    delta=f"{delta_em:+.1f}%p"
                )
    except ImportError:
        print_report(results)
