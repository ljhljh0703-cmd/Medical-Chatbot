"""
내과 의료 챗봇 FastAPI 애플리케이션.

실행:
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

환경 변수(.env):
  OPENAI_API_KEY, MODEL_MODE, TOP_K, CHROMA_DB_PATH 등
"""

import sys
import os

# src/ 경로를 모듈 검색 경로에 추가 (uvicorn을 src/ 바깥에서 실행할 경우 대비)
_SRC = os.path.dirname(__file__)
_ROOT_SRC = os.path.join(_SRC, "..")
if _ROOT_SRC not in sys.path:
    sys.path.insert(0, _ROOT_SRC)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat

# ─── 앱 인스턴스 생성 ────────────────────────────────────────────────────────
app = FastAPI(
    title="내과 의료 챗봇 API",
    description=(
        "내과 전문 의료 QA 챗봇 REST API.  \n"
        "모드 A(LLM only) / B(+RAG) / C(+LoRA) 비교 지원."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 운영 환경에서는 도메인 제한 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 라우터 등록 ─────────────────────────────────────────────────────────────
app.include_router(chat.router, prefix="/chat", tags=["chat"])


# ─── 루트 엔드포인트 ──────────────────────────────────────────────────────────
@app.get("/", summary="루트")
def root():
    return {"message": "내과 의료 챗봇 API 정상 동작 중", "docs": "/docs"}

