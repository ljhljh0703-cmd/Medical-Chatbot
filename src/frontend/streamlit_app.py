"""
내과 AI 상담 Streamlit 프론트엔드.

레이아웃:
  - Step 0: 프로필 입력 (성별 · 연령대 선택)
  - Step 1: 채팅 (내과봇 문진)
  - 사이드바: 생성 모드(A/B/C), 검색 설정, 평가 대시보드, 설정
"""

from __future__ import annotations

import json
import os
import sys

import streamlit as st

# ─── 경로 설정 ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ─── 상수 ───────────────────────────────────────────────────────────────────
PAGE_TITLE = "🏥 내과 AI 챗봇 서비스"
MAX_HISTORY = 30

# 캐릭터 이미지 경로
IMG_AVATAR = os.path.join(_HERE, "Chatbot 64x64.png")
IMG_SIDEBAR = os.path.join(_HERE, "Chatbot 96x96.png")
IMG_PROFILE = os.path.join(_HERE, "chatbot 200x200.png")

GENDER_OPTIONS = ["남성", "여성"]
AGE_OPTIONS = ["10대", "20대", "30대", "40대", "50대 이상"]

GREETING_MSG = "안녕하세요! 알려줄고양입니다 🩺\n환자분의 상세한 증상을 편하게 말씀해 주세요."


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
    /* ═══════════════════════════════════════════
       폰 베젤 모의 (Phone Bezel Mockup)
    ═══════════════════════════════════════════ */

    /* 페이지 배경 어둡게 + 폰 수직 중앙 정렬 */
    .stApp {
        background: linear-gradient(150deg, #0d0d1f 0%, #1a1a3a 100%) !important;
        min-height: 100vh !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* 메인 컨텐츠를 폰 화면처럼 */
    .stMainBlockContainer,
    .main .block-container {
        max-width: 420px !important;
        width: 420px !important;
        background: #f8f9fa !important;
        border-radius: 44px !important;
        box-shadow:
            0 0 0 14px #1c1c1e,
            0 0 0 16px #3d3d3d,
            0 30px 80px rgba(0, 0, 0, 0.75) !important;
        margin: 60px auto !important;
        height: 820px !important;
        padding: 42px 20px 32px !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }

    /* 폰 상단 노치 */
    .stMainBlockContainer::before,
    .main .block-container::before {
        content: '';
        display: block;
        width: 120px;
        height: 28px;
        background: #1c1c1e;
        border-radius: 0 0 20px 20px;
        margin: -28px auto 14px;
    }

    /* 폰 하단 홈 바 */
    .stMainBlockContainer::after,
    .main .block-container::after {
        content: '';
        display: block;
        width: 120px;
        height: 5px;
        background: #888;
        border-radius: 3px;
        margin: 24px auto 0;
    }

    /* ── 앱 헤더 ── */
    .phone-app-header {
        background: linear-gradient(135deg, #4B8BF5 0%, #2ECC71 100%);
        color: white;
        padding: 14px 20px;
        border-radius: 12px;
        margin-bottom: 16px;
        text-align: center;
    }
    .phone-app-header h3 { margin: 0; font-size: 1.15em; }
    .phone-app-header p  { margin: 4px 0 0 0; font-size: 0.82em; opacity: 0.9; }

    /* ── 채팅 버블 ── */
    .chat-bubble-bot {
        background: #F0F4FF;
        border-radius: 16px 16px 16px 4px;
        padding: 14px 18px;
        margin-bottom: 8px;
        border-left: 4px solid #4B8BF5;
    }
    .chat-bubble-user {
        background: #1976D2;
        color: white;
        border-radius: 16px 16px 4px 16px;
        padding: 14px 18px;
        margin-bottom: 8px;
        text-align: right;
    }

    /* ── Red Flag 배너 ── */
    .red-flag-banner {
        background: #FFF0F0;
        border: 2px solid #FF4B4B;
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 12px;
        color: #CC0000;
        font-weight: bold;
    }

    /* ── 참조 소스 카드 ── */
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

    /* ── 프로필 선택 버튼 ── */
    .profile-section-title {
        font-size: 1.1em;
        font-weight: 600;
        color: #333;
        margin: 18px 0 10px 0;
    }

    /* h1 (st.title) 숨기기 */
    [data-testid="stMain"] h1 {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── 세션 상태 초기화 ─────────────────────────────────────────────────────────
def _init_session_state() -> None:
    defaults = {
        "step": 0,                   # 0=프로필, 1=채팅
        "patient_gender": None,      # "남성" | "여성"
        "patient_age_group": None,   # "10대" ~ "50대 이상"
        "chat_history": [],          # list[dict]
        "greeting_shown": False,     # 인사 메시지 표시 여부
        "mode": "B",                 # A / B / C
        "top_k": 5,
        "eval_results": None,        # 평가 JSON 로드 결과
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()


# ─── ChatService 로딩 ────────────────────────────────────────────────────────
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
        # 사이드바 로고
        if os.path.exists(IMG_SIDEBAR):
            st.image(IMG_SIDEBAR, width=72)
        else:
            st.image("https://img.icons8.com/fluency/96/000000/hospital.png", width=72)

        st.title("알려줄고양")
        st.caption("내과 AI 상담 · 보통 몇 분 내에 응답한다옹.")
        st.divider()

        # 환자 프로필 표시 (step 1일 때)
        if st.session_state["step"] == 1:
            gender = st.session_state.get("patient_gender", "—")
            age = st.session_state.get("patient_age_group", "—")
            st.markdown(f"🧑‍⚕️ **환자 정보**: {gender} · {age}")
            if st.button("🔄 처음부터 다시", use_container_width=True):
                st.session_state["step"] = 0
                st.session_state["patient_gender"] = None
                st.session_state["patient_age_group"] = None
                st.session_state["chat_history"] = []
                st.session_state["greeting_shown"] = False
                st.rerun()
            st.divider()

        # 모드 선택
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

        # 채팅 초기화
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state["chat_history"] = []
            st.session_state["greeting_shown"] = False
            st.rerun()

        # 현재 상태 정보
        st.divider()
        st.caption(
            f"**현재 모드:** {st.session_state['mode']}  \n"
            f"**top-k:** {st.session_state['top_k']}  \n"
            f"**대화 수:** {len(st.session_state['chat_history'])}"
        )

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


# ═══════════════════════════════════════════════════════════════════════════════
# Step 0: 프로필 입력
# ═══════════════════════════════════════════════════════════════════════════════
def _profile_setup() -> None:
    """성별 · 연령대 선택 화면."""

    # 캐릭터 + 인사말
    col_char, col_text = st.columns([1, 3])
    with col_char:
        if os.path.exists(IMG_PROFILE):
            st.image(IMG_PROFILE, width=160)
        else:
            st.markdown("## 🩺")
    with col_text:
        st.markdown(
            "### 나와 잘 어울리는 진료를 찾고 있냥?\n\n"
            "그럼 잘 왔당.\n\n"
            "내과 전문인 이 몸이 증상을 파악하고\n"
            "적절한 안내를 도와주겠다옹."
        )

    st.divider()

    # ── 성별 선택 ──
    st.markdown('<p class="profile-section-title">Q. 성별을 알려달라냥.</p>', unsafe_allow_html=True)
    gender_cols = st.columns(len(GENDER_OPTIONS))
    for i, g in enumerate(GENDER_OPTIONS):
        with gender_cols[i]:
            is_selected = st.session_state["patient_gender"] == g
            btn_type = "primary" if is_selected else "secondary"
            if st.button(
                f"{'🙋‍♂️' if g == '남성' else '🙋‍♀️'} {g}",
                key=f"gender_{g}",
                use_container_width=True,
                type=btn_type,
            ):
                st.session_state["patient_gender"] = g
                st.rerun()

    st.write("")

    # ── 연령대 선택 ──
    st.markdown('<p class="profile-section-title">Q. 연령대를 알려달라냥.</p>', unsafe_allow_html=True)
    age_cols = st.columns(3)
    for i, a in enumerate(AGE_OPTIONS):
        with age_cols[i % 3]:
            is_selected = st.session_state["patient_age_group"] == a
            btn_type = "primary" if is_selected else "secondary"
            if st.button(
                a,
                key=f"age_{a}",
                use_container_width=True,
                type=btn_type,
            ):
                st.session_state["patient_age_group"] = a
                st.rerun()

    st.write("")
    st.divider()

    # ── 문진 시작 버튼 ──
    both_selected = (
        st.session_state["patient_gender"] is not None
        and st.session_state["patient_age_group"] is not None
    )

    if both_selected:
        st.success(
            f"✅ **{st.session_state['patient_gender']}** · **{st.session_state['patient_age_group']}** 선택됨"
        )

    if st.button(
        "🩺 문진 시작하기",
        use_container_width=True,
        type="primary",
        disabled=not both_selected,
    ):
        st.session_state["step"] = 1
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: 채팅
# ═══════════════════════════════════════════════════════════════════════════════

# ─── 채팅 헬퍼 ────────────────────────────────────────────────────────────────
def _render_red_flag_banner(keywords: list[str] | None = None) -> None:
    kw_str = ", ".join(keywords) if keywords else ""
    banner = (
        "🚨 **긴급 상황이 감지되었습니다냥.**  \n"
        "즉시 119에 신고하거나 가까운 응급실을 방문하라냥.  \n"
    )
    if kw_str:
        banner += f"감지 키워드: `{kw_str}`"
    st.markdown(
        f'<div class="red-flag-banner">{banner}</div>',
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


def _render_bot_message(text: str) -> None:
    """아바타 + 말풍선으로 봇 메시지 렌더링."""
    col_avatar, col_msg = st.columns([1, 11])
    with col_avatar:
        if os.path.exists(IMG_AVATAR):
            st.image(IMG_AVATAR, width=48)
        else:
            st.markdown("🤖")
    with col_msg:
        st.markdown(
            f'<div class="chat-bubble-bot">{text}</div>',
            unsafe_allow_html=True,
        )


def _render_chat_history() -> None:
    """저장된 대화 기록을 상단부터 렌더링."""
    for entry in st.session_state["chat_history"]:
        role = entry.get("role", "user")

        if role == "bot_greeting":
            # 인사 메시지 (시스템)
            _render_bot_message(entry["content"])
            continue

        if role == "user":
            st.markdown(
                f'<div class="chat-bubble-user">{entry["query"]}</div>',
                unsafe_allow_html=True,
            )

            # Red Flag 배너
            if entry.get("red_flag_triggered"):
                _render_red_flag_banner()

            # 챗봇 답변
            _render_bot_message(entry["chatbot_answer"])

            # 참조 소스
            _render_sources(entry.get("retrieved_sources", []))

            # 모드 + top_k 뱃지
            mode_badge = {"A": "🔵 A", "B": "🟡 B", "C": "🟢 C"}.get(
                entry.get("mode", "A"), entry.get("mode", "A")
            )
            top_k_info = entry.get("top_k", st.session_state.get("top_k"))
            st.caption(f"생성 모드: {mode_badge}  ·  top-k: {top_k_info}")
            st.divider()


def _build_query_with_context(raw_query: str) -> str:
    """프로필 정보를 query 앞에 태그로 삽입."""
    gender = st.session_state.get("patient_gender", "")
    age = st.session_state.get("patient_age_group", "")
    if gender and age:
        return f"[환자 정보: {gender}, {age}] {raw_query}"
    return raw_query


def _tab_chat() -> None:
    """채팅 메인 화면."""

    # 첫 진입 시 인사 메시지 삽입
    if not st.session_state["greeting_shown"]:
        st.session_state["chat_history"].insert(0, {
            "role": "bot_greeting",
            "content": GREETING_MSG,
        })
        st.session_state["greeting_shown"] = True

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
        query = st.text_area(
            "증상을 입력하라냥",
            placeholder="예: 3일 전부터 속이 쓰리고 식후에 복통이 있어요.",
            height=100,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("📨 전송", use_container_width=True)

    if submitted and query.strip():
        enriched_query = _build_query_with_context(query.strip())

        with st.spinner("알려줄고양이 답변을 준비하고 있습니다…"):
            if svc is None:
                result_dict = _demo_response(query.strip())
            else:
                try:
                    from domain.models.chat_result import ChatResult

                    result: ChatResult = svc.handle(
                        query=enriched_query,
                        mode=st.session_state["mode"],
                        ground_truth=None,
                    )
                    result_dict = {
                        "role": "user",
                        "query": query.strip(),
                        "chatbot_answer": result.chatbot_answer,
                        "retrieved_sources": [
                            s.model_dump() for s in result.retrieved_sources
                        ],
                        "red_flag_triggered": result.red_flag_triggered,
                        "mode": result.mode,
                        "top_k": result.top_k or st.session_state["top_k"],
                    }
                except Exception as exc:
                    st.error(f"❌ 오류 발생: {exc}")
                    return

        # 기록에 추가 (최대 MAX_HISTORY)
        st.session_state["chat_history"].append(result_dict)
        if len(st.session_state["chat_history"]) > MAX_HISTORY:
            st.session_state["chat_history"].pop(1)  # 인사 메시지(0) 유지

        st.rerun()


def _demo_response(query: str) -> dict:
    """서비스 없을 때 보여주는 데모 응답."""
    return {
        "role": "user",
        "query": query,
        "chatbot_answer": (
            "⚠️ [데모 모드] 실제 LLM/ChromaDB가 연결되지 않았습니다.  \n"
            "`.env` 파일에 `OPENAI_API_KEY`를 설정하거나 Qwen 모델 경로를 지정하세요."
        ),
        "retrieved_sources": [],
        "red_flag_triggered": False,
        "mode": st.session_state["mode"],
        "top_k": st.session_state["top_k"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 평가 대시보드
# ═══════════════════════════════════════════════════════════════════════════════
def _tab_evaluation() -> None:
    st.header("📊 평가 대시보드 (A / B / C 비교)")

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
            default_path = os.path.join(_HERE, "..", "..", "eval_results", "summary.json")
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
        st.info("ℹ️ 평가 결과가 없습니다. `evaluator.py`를 실행하거나 위 업로드 버튼을 사용하세요.")
        _show_sample_eval()
        return

    _render_eval_results(results)


def _render_eval_results(results: list[dict]) -> None:
    import pandas as pd

    df = pd.DataFrame(results)

    st.subheader("🏆 핵심 지표 요약")
    cols = st.columns(len(results))
    for col, r in zip(cols, results):
        mode_label = {"A": "A — LLM only", "B": "B — LLM+RAG", "C": "C — LoRA"}.get(r["mode"], r["mode"])
        col.metric(
            label=mode_label,
            value=f"EM {r['em']:.1%}",
            delta=f"{(r['em'] - results[0]['em']) * 100:+.1f}%p" if r is not results[0] else None,
        )

    st.divider()

    st.subheader("📋 상세 결과")
    df_display = df.rename(columns={
        "mode": "모드", "em": "EM", "rouge_l": "ROUGE-L",
        "bert_score": "BERTScore", "llm_judge": "LLM-Judge", "n_samples": "샘플 수",
    })
    for c in ["EM", "ROUGE-L", "BERTScore", "LLM-Judge"]:
        if c in df_display.columns:
            df_display[c] = df_display[c].apply(lambda v: f"{v:.4f}" if v is not None else "N/A")
    st.dataframe(df_display, use_container_width=True)

    st.divider()

    st.subheader("📈 조건별 비교 차트")
    chart_metrics = [c for c in ["em", "rouge_l", "bert_score"] if c in df.columns]
    chart_df = df.set_index("mode")[chart_metrics].rename(
        columns={"em": "EM", "rouge_l": "ROUGE-L", "bert_score": "BERTScore"}
    )
    st.bar_chart(chart_df)

    if "llm_judge" in df.columns and df["llm_judge"].notna().any():
        st.subheader("🧑‍⚖️ LLM-Judge 점수 (1~5)")
        st.bar_chart(df.set_index("mode")[["llm_judge"]].rename(columns={"llm_judge": "LLM-Judge"}))

    st.divider()

    if len(results) >= 2:
        st.subheader("📌 개선율 요약")
        for r in results[1:]:
            em_d = (r["em"] - results[0]["em"]) * 100
            rl_d = (r["rouge_l"] - results[0]["rouge_l"]) * 100
            label = "RAG 추가 (A → B)" if r["mode"] == "B" else "LoRA 추가 (A → C)"
            st.markdown(f"**{label}**:  EM `{em_d:+.1f}%p`  |  ROUGE-L `{rl_d:+.1f}%p`")


def _show_sample_eval() -> None:
    st.subheader("📋 샘플 미리보기 (가상 데이터)")
    sample = [
        {"mode": "A", "em": 0.42, "rouge_l": 0.38, "bert_score": 0.71, "llm_judge": 3.2, "n_samples": 200},
        {"mode": "B", "em": 0.57, "rouge_l": 0.53, "bert_score": 0.81, "llm_judge": 3.8, "n_samples": 200},
        {"mode": "C", "em": 0.63, "rouge_l": 0.59, "bert_score": 0.85, "llm_judge": 4.1, "n_samples": 200},
    ]
    _render_eval_results(sample)


# ═══════════════════════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════════════════════
def _tab_settings() -> None:
    st.header("⚙️ 설정")

    st.subheader(" 모델 설정")
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
    st.markdown("""
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
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    _render_sidebar()

    # 휴대폰 위치 고정 CSS
    st.markdown(
        """
        <style>
        .stMainBlockContainer { margin-top: 150px !important; }
        .main .block-container { margin-top: 150px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title(PAGE_TITLE)  # CSS로 숨김 처리 (폰 앱 헤더 사용)

    # 폰 앱 헤더
    st.markdown(
        '<div class="phone-app-header">'
        "<h3>🐱 알려줄고양</h3>"
        "<p>내과 AI 챗봇 서비스 · 보통 몇 분 내에 응답한다냥.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # step 분기: 0=프로필 입력, 1=채팅
    if st.session_state["step"] == 0:
        _profile_setup()
    else:
        _tab_chat()


if __name__ == "__main__":
    main()
