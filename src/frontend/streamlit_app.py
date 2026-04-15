"""
내과 전문 의료 챗봇 Streamlit 프론트엔드.

탭 구성:
  1. 💬 채팅   : 질문 입력 → 챗봇 답변 vs 정답 나란히 표시 + 참조 원천 패널
  2. 📊 평가   : A/B/C 조건 비교 대시보드 (evaluator 결과 JSON 로드)
  3. ⚙️ 설정   : 모드 선택(A/B/C), API 키 입력, top_k 슬라이더
"""

from __future__ import annotations

import json
import os
import sys

import streamlit as st

# ─── 경로 설정 ──────────────────────────────────────────────────────────────
_SRC = os.path.join(os.path.dirname(__file__), "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ─── 상수 ───────────────────────────────────────────────────────────────────
PAGE_TITLE = "🏥 내과 AI 상담"
RED_FLAG_COLOR = "#FF4B4B"
GROUND_TRUTH_BG = "#F0FFF4"
CHATBOT_BG = "#F0F4FF"
MAX_HISTORY = 30


# ─── 페이지 기본 설정 ────────────────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── CSS 커스터마이징 ─────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .chat-header {
        background: linear-gradient(90deg, #4B8BF5, #2ECC71);
        color: white;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .chat-header h2 { margin: 0; }
    .chat-header p { margin: 0; font-size:0.9em; opacity:0.95 }

    .chat-bubble-bot {
        background: #F0F4FF;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
        border-left: 4px solid #4B8BF5;
    }
    .chat-bubble-gt {
        background: #F0FFF4;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
        border-left: 4px solid #2ECC71;
    }
    .chat-bubble-user {
        background: #FFF8F0;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
        border-left: 4px solid #F5A623;
        text-align: right;
    }
    .red-flag-banner {
        background: #FFF0F0;
        border: 2px solid #FF4B4B;
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 12px;
        color: #CC0000;
        font-weight: bold;
    }
    .source-card {
        background: #FAFAFA;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
        font-size: 0.85em;
    }
    .similarity-bar {
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(to right, #4B8BF5, #2ECC71);
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── 세션 상태 초기화 ─────────────────────────────────────────────────────────
def _init_session_state() -> None:
    defaults = {
        "chat_history": [],      # list[dict] — query/chatbot_answer/ground_truth/sources/red_flag/mode
        "mode": "B",             # A / B / C
        "top_k": 5,
<<<<<<< HEAD
=======
        "show_ground_truth": True,
        "show_ground_truth": True,지
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()


# ─── ChatService 懒로딩 ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="모델 및 DB 연결 중…")
def _load_chat_service(mode: str, top_k: int):
    """설정을 반영한 ChatService 인스턴스를 캐시하여 반환."""
    try:
        from config.settings import settings
        from services.chat_service import ChatService

        settings.model_mode = mode
        settings.top_k = top_k
        return ChatService(), None
    except Exception as exc:
        return None, str(exc)


def _get_service():
    svc, err = _load_chat_service(
        st.session_state["mode"],
        st.session_state["top_k"],
    )
    return svc, err


# ─── 사이드바 ─────────────────────────────────────────────────────────────────
def _render_sidebar() -> None:
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/000000/hospital.png",
            width=72,
        )
<<<<<<< HEAD
        st.title("내과봇")
        st.caption("내과 AI 상담 · 보통 몇 분 내에 응답합니다.")
        st.title("내과봇")
        st.caption("내과 AI 상담 · 보통 몇 분 내에 응답합니다.")
        st.subheader("⚙️ 생성 모드")
        mode = st.radio(
            "실험 조건을 선택하세요:",
            options=["A", "B", "C"],
            index=["A", "B", "C"].index(st.session_state["mode"]),
            format_func=lambda x: {
                "A": "A — LLM only",
                "B": "B — LLM + RAG",
                "C": "C — LLM + RAG + LoRA",
            }[x],
            horizontal=False,
        )
        if mode != st.session_state["mode"]:
            st.session_state["mode"] = mode
            # 캐시 무효화를 위해 cache 클리어
            _load_chat_service.clear()
            st.rerun()

        st.divider()

        # RAG 설정
        st.subheader("🔍 검색 설정")
        top_k = st.slider("참조 청크 수 (top-k)", 1, 10, st.session_state["top_k"])
        if top_k != st.session_state["top_k"]:
            st.session_state["top_k"] = top_k
            _load_chat_service.clear()

        st.divider()

<<<<<<< HEAD
=======
        # 표시 옵션
        st.subheader("🖥 표시 옵션")
        st.session_state["show_ground_truth"] = st.toggle(
            "정답(Ground Truth) 표시",
            value=st.session_state["show_ground_truth"],
        )
        # 표시 옵션
        st.subheader("🖥 표시 옵션")
        st.session_state["show_ground_truth"] = st.toggle(
            "정답(Ground Truth) 표시",
            value=st.session_state["show_ground_truth"],
        )

        st.divider()

            f"**현재 모드:** {st.session_state['mode']}  \n"
            f"**top-k:** {st.session_state['top_k']}  \n"
            f"**대화 수:** {len(st.session_state['chat_history'])}"
        )

<<<<<<< HEAD
        # 평가 대시보드 / 설정을 사이드바 expander로 이동
        st.divider()
        with st.expander("📊 평가 대시보드", expanded=False):
            try:
                _tab_evaluation()
            except Exception as e:
                st.warning(f"평가 대시보드를 로드할 수 없습니다: {e}")

        with st.expander("⚙️ 설정", expanded=False):
            try:
        # 평가 대시보드 / 설정을 사이드바 expander로 이동
        st.divider()
        with st.expander("📊 평가 대시보드", expanded=False):
            try:
                _tab_evaluation()
            except Exception as e:
                st.warning(f"평가 대시보드를 로드할 수 없습니다: {e}")

        with st.expander("⚙️ 설정", expanded=False):
            try:
                _tab_settings()
            except Exception as e:
                st.warning(f"설정 패널을 로드할 수 없습니다: {e}")

        f'<div class="red-flag-banner">{banner}</div>',
        unsafe_allow_html=True,
    )


def _render_answer_pair(
    chatbot_answer: str,
    ground_truth: str | None,
    show_gt: bool,
) -> None:
    """챗봇 답변과 정답을 나란히 렌더링."""
    if show_gt and ground_truth:
        col_bot, col_gt = st.columns(2)
        with col_bot:
            st.markdown("**🤖 챗봇 답변**")
            st.markdown(
                f'<div class="chat-bubble-bot">{chatbot_answer}</div>',
                unsafe_allow_html=True,
            )
        with col_gt:
            st.markdown("**✅ 실제 정답 (Ground Truth)**")
            st.markdown(
                f'<div class="chat-bubble-gt">{ground_truth}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown("**🤖 챗봇 답변**")
        st.markdown(
            f'<div class="chat-bubble-bot">{chatbot_answer}</div>',
            unsafe_allow_html=True,
        )


def _render_sources(sources: list[dict]) -> None:
    """참조 원천 데이터 청크 패널 렌더링."""
    if not sources:
        return
    with st.expander(f"📚 참조 원천 데이터 ({len(sources)}개 청크)", expanded=False):
        for i, src in enumerate(sources, 1):
            sim = src.get("similarity_score", 0.0)
            bar_width = int(sim * 100)
            spec = src.get("source_spec", "출처 불명")
            snippet = src.get("content_snippet", "")

            st.markdown(
                f"""<div class="source-card">
                <b>[{i}] {spec}</b>
                <div class="similarity-bar" style="width:{bar_width}%;"></div>
                <span style="font-size:0.8em; color:#666;">유사도: {sim:.3f}</span>
                <hr style="margin:6px 0;">
                {snippet}
                </div>""",
                unsafe_allow_html=True,
            )


def _render_chat_history() -> None:
    """저장된 대화 기록을 상단부터 렌더링."""
    for entry in st.session_state["chat_history"]:
        # 사용자 질문
        st.markdown(
            f'<div class="chat-bubble-user">🧑 {entry["query"]}</div>',
            unsafe_allow_html=True,
        )

        # Red Flag 배너
        if entry.get("red_flag_triggered"):
            _render_red_flag_banner()

        # 답변 쌍
        _render_answer_pair(
            chatbot_answer=entry["chatbot_answer"],
<<<<<<< HEAD
            ground_truth=None,
            show_gt=False,
=======
            ground_truth=entry.get("ground_truth"),
            show_gt=st.session_state["show_ground_truth"],
>>>>>>> b10249884822f0ef1ceaa52a7daa5cb87bd6a4e3
        )

        # 참조 소스
        _render_sources(entry.get("retrieved_sources", []))

        # 모드 뱃지
        mode_badge = {"A": "🔵 A", "B": "🟡 B", "C": "🟢 C"}.get(
            entry.get("mode", "A"), entry.get("mode", "A")
        )
<<<<<<< HEAD
        top_k_info = entry.get("top_k", st.session_state.get("top_k"))
        st.caption(f"생성 모드: {mode_badge}  ·  top-k: {top_k_info}")
=======
        st.caption(f"생성 모드: {mode_badge}")
>>>>>>> b10249884822f0ef1ceaa52a7daa5cb87bd6a4e3
        st.divider()


# ─── 탭 1: 채팅 ──────────────────────────────────────────────────────────────
def _tab_chat() -> None:
    st.header("💬 채팅")

    # 서비스 로드 상태 확인
    svc, err = _get_service()
    if err:
        st.warning(
            f"⚠️ 서비스 초기화 실패 (데모 모드로 실행됩니다):\n\n`{err}`",
            icon="⚠️",
        )

    # 대화 기록 렌더링
    _render_chat_history()

    # 입력창
    with st.form(key="chat_form", clear_on_submit=True):
<<<<<<< HEAD
        query = st.text_area(
            "질문을 입력하세요",
            placeholder="예: 고혈압 환자에게 베타차단제를 처방할 때 주의사항은?",
            height=120,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("📨 질문하기", use_container_width=True)

    if submitted and query.strip():
        with st.spinner("답변 생성 중…"):
            if svc is None:
                # 데모 응답 (서비스 없을 때)
                result_dict = _demo_response(query)
=======
        col_input, col_gt = st.columns([3, 2])
        with col_input:
            query = st.text_area(
                "질문을 입력하세요",
                placeholder="예: 고혈압 환자에게 베타차단제를 처방할 때 주의사항은?",
                height=90,
                label_visibility="collapsed",
            )
        with col_gt:
            ground_truth_input = st.text_area(
                "정답 (선택, 평가용)",
                placeholder="라벨링 정답을 입력하면 나란히 비교됩니다 (선택사항)",
                height=90,
            )
        submitted = st.form_submit_button("📨 질문하기", use_container_width=True)

    if submitted and query.strip():
        ground_truth = ground_truth_input.strip() or None

        with st.spinner("답변 생성 중…"):
            if svc is None:
                # 데모 응답 (서비스 없을 때)
                result_dict = _demo_response(query, ground_truth)
>>>>>>> b10249884822f0ef1ceaa52a7daa5cb87bd6a4e3
            else:
                try:
                    from domain.models.chat_result import ChatResult

                    result: ChatResult = svc.handle(
                        query=query.strip(),
                        mode=st.session_state["mode"],
<<<<<<< HEAD
                        ground_truth=None,
=======
                        ground_truth=ground_truth,
>>>>>>> b10249884822f0ef1ceaa52a7daa5cb87bd6a4e3
                    )
                    result_dict = {
                        "query": result.query,
                        "chatbot_answer": result.chatbot_answer,
<<<<<<< HEAD
=======
                        "ground_truth": result.ground_truth,
>>>>>>> b10249884822f0ef1ceaa52a7daa5cb87bd6a4e3
                        "retrieved_sources": [
                            s.model_dump() for s in result.retrieved_sources
                        ],
                        "red_flag_triggered": result.red_flag_triggered,
                        "mode": result.mode,
<<<<<<< HEAD
                        "top_k": result.top_k or st.session_state["top_k"],
=======
>>>>>>> b10249884822f0ef1ceaa52a7daa5cb87bd6a4e3
                    }
                except Exception as exc:
                    st.error(f"❌ 오류 발생: {exc}")
                    return
<<<<<<< HEAD
=======

>>>>>>> b10249884822f0ef1ceaa52a7daa5cb87bd6a4e3
        # 기록에 추가 (최대 MAX_HISTORY)
        st.session_state["chat_history"].append(result_dict)
        if len(st.session_state["chat_history"]) > MAX_HISTORY:
            st.session_state["chat_history"].pop(0)

        st.rerun()


<<<<<<< HEAD
def _demo_response(query: str) -> dict:
=======
def _demo_response(query: str, ground_truth: str | None) -> dict:
>>>>>>> b10249884822f0ef1ceaa52a7daa5cb87bd6a4e3
    """서비스 없을 때 보여주는 데모 응답."""
    return {
        "query": query,
        "chatbot_answer": (
            "⚠️ [데모 모드] 실제 LLM/ChromaDB가 연결되지 않았습니다.  \n"
            "`.env` 파일에 `OPENAI_API_KEY`를 설정하거나 Qwen 모델 경로를 지정하세요."
        ),
<<<<<<< HEAD
        "retrieved_sources": [],
        "red_flag_triggered": False,
        "mode": st.session_state["mode"],
        "top_k": st.session_state["top_k"],
=======
        "ground_truth": ground_truth,
        "retrieved_sources": [],
        "red_flag_triggered": False,
        "mode": st.session_state["mode"],
>>>>>>> b10249884822f0ef1ceaa52a7daa5cb87bd6a4e3
    }


# ─── 탭 2: 평가 대시보드 ─────────────────────────────────────────────────────
def _tab_evaluation() -> None:
    st.header("📊 평가 대시보드 (A / B / C 비교)")

    # JSON 파일 업로드 또는 기본 경로 로드
    col_upload, col_load = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "평가 결과 JSON 업로드 (`eval_results/summary.json`)",
            type=["json"],
        )
    with col_load:
        st.write("")
        st.write("")
        if st.button("🔄 기본 경로에서 로드", use_container_width=True):
            default_path = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "eval_results", "summary.json",
            )
            if os.path.exists(default_path):
                with open(default_path, "r", encoding="utf-8") as f:
                    st.session_state["eval_results"] = json.load(f)
                st.success("✅ 로드 완료")
            else:
                st.warning("기본 경로에 파일이 없습니다: `eval_results/summary.json`")

    if uploaded:
        st.session_state["eval_results"] = json.loads(uploaded.read())

    results = st.session_state.get("eval_results")

    if not results:
        st.info(
            "ℹ️ 평가 결과가 없습니다.  \n"
            "먼저 `evaluator.py`를 실행하거나, 위 업로드 버튼을 사용하세요."
        )
        _show_sample_eval()
        return

    _render_eval_results(results)


def _render_eval_results(results: list[dict]) -> None:
    """실제 평가 결과 렌더링."""
    import pandas as pd

    df = pd.DataFrame(results)

    # 요약 메트릭 카드
    st.subheader("🏆 핵심 지표 요약")
    cols = st.columns(len(results))
    for col, r in zip(cols, results):
        mode_label = {"A": "A — LLM only", "B": "B — LLM+RAG", "C": "C — LoRA"}.get(
            r["mode"], r["mode"]
        )
        col.metric(
            label=mode_label,
            value=f"EM {r['em']:.1%}",
            delta=(
                f"{(r['em'] - results[0]['em']) * 100:+.1f}%p"
                if r is not results[0]
                else None
            ),
        )

    st.divider()

    # 데이터프레임
    st.subheader("📋 상세 결과")
    df_display = df.rename(
        columns={
            "mode": "모드",
            "em": "EM",
            "rouge_l": "ROUGE-L",
            "bert_score": "BERTScore",
            "llm_judge": "LLM-Judge",
            "n_samples": "샘플 수",
        }
    )
    numeric_cols = ["EM", "ROUGE-L", "BERTScore", "LLM-Judge"]
    for c in numeric_cols:
        if c in df_display.columns:
            df_display[c] = df_display[c].apply(
                lambda v: f"{v:.4f}" if v is not None else "N/A"
            )
    st.dataframe(df_display, use_container_width=True)

    st.divider()

    # 바 차트
    st.subheader("📈 조건별 비교 차트")
    chart_metrics = [c for c in ["em", "rouge_l", "bert_score"] if c in df.columns]
    chart_df = df.set_index("mode")[chart_metrics].rename(
        columns={"em": "EM", "rouge_l": "ROUGE-L", "bert_score": "BERTScore"}
    )
    st.bar_chart(chart_df)

    # LLM-Judge 별도 차트
    if "llm_judge" in df.columns and df["llm_judge"].notna().any():
        st.subheader("🧑‍⚖️ LLM-Judge 점수 (1~5)")
        st.bar_chart(df.set_index("mode")[["llm_judge"]].rename(columns={"llm_judge": "LLM-Judge"}))

    st.divider()

    # 개선율 요약
    if len(results) >= 2:
        st.subheader("📌 개선율 요약")
        for r in results[1:]:
            em_delta = (r["em"] - results[0]["em"]) * 100
            rl_delta = (r["rouge_l"] - results[0]["rouge_l"]) * 100
            label = "RAG 추가 (A → B)" if r["mode"] == "B" else "LoRA 추가 (A → C)"
            st.markdown(
                f"**{label}**:  EM `{em_delta:+.1f}%p`  |  ROUGE-L `{rl_delta:+.1f}%p`"
            )


def _show_sample_eval() -> None:
    """샘플 평가 데이터로 미리보기 렌더링."""
    st.subheader("📋 샘플 미리보기 (가상 데이터)")
    sample = [
        {"mode": "A", "em": 0.42, "rouge_l": 0.38, "bert_score": 0.71, "llm_judge": 3.2, "n_samples": 200},
        {"mode": "B", "em": 0.57, "rouge_l": 0.53, "bert_score": 0.81, "llm_judge": 3.8, "n_samples": 200},
        {"mode": "C", "em": 0.63, "rouge_l": 0.59, "bert_score": 0.85, "llm_judge": 4.1, "n_samples": 200},
    ]
    _render_eval_results(sample)


# ─── 탭 3: 설정 ──────────────────────────────────────────────────────────────
def _tab_settings() -> None:
    st.header("⚙️ 설정")

    st.subheader("🔑 API 키")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
        help=".env 파일에 OPENAI_API_KEY=... 를 설정하는 것을 권장합니다.",
    )
    if st.button("저장 (세션 한정)", key="save_api"):
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("✅ 환경 변수에 저장되었습니다 (현재 세션만 유효).")

    st.divider()

    st.subheader("🛠 모델 설정")
    try:
        from config.settings import settings

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**model_backend**: `{settings.model_backend}`")
            st.info(f"**model_path**: `{settings.model_path}`")
        with col2:
            st.info(f"**embedding_model**: `{settings.embedding_model}`")
            st.info(f"**chroma_db_path**: `{settings.chroma_db_path}`")

        if settings.lora_adapter_path:
            st.success(f"✅ LoRA 어댑터 경로: `{settings.lora_adapter_path}`")
        else:
            st.warning("⚠️ LoRA 어댑터 경로가 설정되지 않았습니다 (`LORA_ADAPTER_PATH`).")

    except Exception as exc:
        st.warning(f"설정을 불러올 수 없습니다: `{exc}`")

    st.divider()

    st.subheader("📁 데이터 경로 확인")
    try:
        from config.settings import settings

        db_path = settings.chroma_db_path
        if os.path.exists(db_path):
            st.success(f"✅ ChromaDB 경로 존재: `{db_path}`")
        else:
            st.warning(f"⚠️ ChromaDB 경로 없음: `{db_path}`  \n`build_knowledge_base.py`를 먼저 실행하세요.")
    except Exception:
        st.warning("설정을 불러올 수 없습니다.")

    st.divider()

    st.subheader("ℹ️ 사용 안내")
    st.markdown(
        """
        **실험 조건 설명:**
        | 모드 | 설명 |
        |------|------|
        | A    | LLM 단독 답변 (RAG 없음) |
        | B    | LLM + RAG (ChromaDB 검색 활용) |
        | C    | LLM + RAG + LoRA 파인튜닝 모델 |

        **안전 정책:**
        - 🚨 자살/의식소실/흉통 등 **Red Flag** 감지 시 긴급 상담 권고 배너 표시
        - 모든 답변은 **권유형 어조** ("~하는 것이 좋습니다") 사용
        - 의학적 조언은 참고용이며, **전문의 상담을 권장**합니다

        **한계:**
        - 본 챗봇은 연구/교육 목적의 프로토타입입니다
        - 실제 진단 및 처방에 사용하지 마세요
        """
    )


# ─── 메인 ─────────────────────────────────────────────────────────────────────
def main() -> None:
    _render_sidebar()

    st.title(PAGE_TITLE)
<<<<<<< HEAD
    st.markdown(
        "<div class='chat-header'><h2>내과봇</h2><p>내과 AI 상담 · 보통 몇 분 내에 응답합니다.</p></div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # 메인: 채팅 전용
    _tab_chat()
=======
    st.caption(
        "내과 전문 의료 QA 챗봇 | Qwen2.5-7B + RAG + LoRA  |  "
        "⚠️ 본 서비스는 참고용이며, 의학적 결정은 전문의와 상담하세요."
    )
    st.divider()

    tab_chat, tab_eval, tab_settings = st.tabs(
        ["💬 채팅", "📊 평가 대시보드", "⚙️ 설정"]
    )

    with tab_chat:
        _tab_chat()

    with tab_eval:
        _tab_evaluation()

    with tab_settings:
        _tab_settings()
>>>>>>> b10249884822f0ef1ceaa52a7daa5cb87bd6a4e3


if __name__ == "__main__":
    main()

