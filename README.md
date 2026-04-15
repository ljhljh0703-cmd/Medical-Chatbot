# 내과 의료 챗봇 (Medical Chatbot)
check
> Qwen2.5-7B-Instruct 기반 RAG + LoRA 내과 전문 챗봇

---

## 개요

- **모델**: Qwen2.5-7B-Instruct (로컬), gpt-4o-mini (fallback)
- **검색**: Dense (OpenAI text-embedding-3-small) + Sparse (BM25) + Hybrid 융합
- **파인튜닝**: PEFT LoRA (r=16, alpha=32) + BitsAndBytes 4-bit 양자화
- **평가**: EM / ROUGE-L / BERTScore / LLM-as-Judge
- **서빙**: FastAPI REST API + Streamlit 프론트엔드

---

## 디렉터리 구조

```
Medical Chat Bot(내과)/
│
├── configs/                        # 설정 파일
│   ├── train_config.yaml           # LoRA 학습 하이퍼파라미터
│   └── app_config.yaml             # 서빙 환경 설정
│
├── data/                           # 데이터 (gitignore·별도 보관)
│   ├── raw/                        # 원본 JSON 데이터
│   └── processed/                  # 전처리 완료 데이터 (train/test 분리)
│
├── docs/                           # 문서 (추후 작성)
│
├── src/
│   ├── app/                        # FastAPI 애플리케이션
│   │   ├── main.py                 # 앱 진입점 · CORS 설정
│   │   ├── routers/
│   │   │   └── chat.py             # /chat 엔드포인트
│   │   └── schemas/
│   │       └── chat.py             # 요청/응답 Pydantic 스키마
│   │
│   ├── config/                     # 전역 설정
│   │   ├── settings.py             # pydantic-settings (.env 로드)
│   │   └── prompts.py              # 시스템 프롬프트 템플릿
│   │
│   ├── domain/
│   │   └── models/
│   │       └── __init__.py         # RetrievedChunk 등 도메인 모델
│   │
│   ├── evaluation/                 # 평가 모듈
│   │   ├── evaluator.py            # 평가 파이프라인 (LLM-as-Judge 포함)
│   │   ├── metrics.py              # EM · ROUGE-L · BERTScore 계산
│   │   └── report.py              # 결과 리포트 생성
│   │
│   ├── frontend/
│   │   └── streamlit_app.py        # Streamlit 3탭 UI (채팅/평가/설정)
│   │
│   ├── ingestion/                  # 지식베이스 구축 파이프라인
│   │   ├── loaders/
│   │   │   └── json_loader.py      # JSON 원본 로더
│   │   ├── preprocess/
│   │   │   └── cleaner.py          # 텍스트 정제
│   │   ├── chunking/
│   │   │   └── chunker.py          # 문서 청킹 (size·overlap 설정)
│   │   └── indexing/
│   │       └── build_knowledge_base.py  # ChromaDB 인덱싱
│   │
│   ├── llm/                        # LLM 클라이언트 · 생성 서비스
│   │   ├── generation_service.py   # 프롬프트 조립 → 응답 생성
│   │   └── model_clients/
│   │       ├── openai_client.py    # OpenAI API 클라이언트
│   │       └── qwen_client.py      # Qwen 로컬 추론 클라이언트
│   │
│   ├── observability/
│   │   └── logger.py               # 구조화 로깅 (JSON)
│   │
│   ├── retrieval/                  # 검색 모듈
│   │   ├── dense/
│   │   │   ├── embedder.py         # 텍스트 임베딩 (OpenAI)
│   │   │   ├── chroma_store.py     # ChromaDB 저장소 래퍼
│   │   │   └── dense_retriever.py  # Dense 벡터 검색
│   │   ├── sparse/
│   │   │   └── bm25_retriever.py   # BM25 희소 검색 (rank-bm25)
│   │   ├── hybrid/
│   │   │   └── fusion.py           # Weighted-Sum / RRF 하이브리드 융합
│   │   └── query/
│   │       └── normalizer.py       # 쿼리 정규화 · 확장
│   │
│   ├── safety/                     # 안전 필터
│   │   ├── safety_service.py       # 안전 파이프라인 오케스트레이터
│   │   ├── red_flag/
│   │   │   └── detector.py         # 위험·비의료 입력 감지
│   │   └── filters/
│   │       └── post_filter.py      # 응답 후처리 필터
│   │
│   ├── services/                   # 비즈니스 로직
│   │   ├── chat_service.py         # Safety→Retrieval→Generation 전체 파이프라인
│   │   └── retrieval_service.py    # 검색 서비스 (Dense/Hybrid 선택)
│   │
│   └── training/                   # 학습 모듈 (src 표준)
│       ├── __init__.py
│       ├── data_module.py          # 데이터셋 준비 · 분리 · DataCollator
│       ├── model_module.py         # 베이스 모델 로드 · LoRA 적용 · 병합
│       ├── trainer_module.py       # SFTTrainer 빌드 · 학습 실행
│       └── main_train.py           # YAML 파싱 → 학습 파이프라인 진입점
│
├── training/                       # 레거시 학습 스크립트 (shim — src/training 위임)
│   ├── prepare_dataset.py
│   ├── finetune_lora.py
│   └── merge_adapter.py
│
├── .env                            # API 키 등 환경 변수 (gitignore)
├── .gitignore
├── requirements.txt                # 의존성 패키지
├── run_train.sh                    # 학습 원클릭 실행 스크립트
└── README.md
```

---

## 빠른 시작

### 1. 환경 설정

```bash
pip install -r requirements.txt
cp .env.example .env   # OPENAI_API_KEY 입력
```

### 2. 지식베이스 구축

```bash
python src/ingestion/indexing/build_knowledge_base.py \
    --input data/raw/<파일>.json
```

### 3. 서버 실행

```bash
# FastAPI
uvicorn src.app.main:app --reload --port 8000

# Streamlit (별도 터미널)
streamlit run src/frontend/streamlit_app.py
```

### 4. LoRA 학습

```bash
# 데이터 준비
python training/prepare_dataset.py \
    --input data/raw/<파일>.json \
    --output_dir data/processed

# 학습 실행
bash run_train.sh
# 또는 직접 호출
bash run_train.sh --config configs/train_config.yaml \
                  --train_path data/processed/train.json \
                  --eval_path data/processed/test.json
```

---

## 설정 파일

| 파일 | 용도 |
|------|------|
| `configs/train_config.yaml` | LoRA 학습 전용 하이퍼파라미터 |
| `configs/app_config.yaml` | 서빙 환경 (검색·안전·로깅) |
| `src/config/settings.py` | 런타임 설정 (`.env` 로드) |

---

## 주요 의존성

| 패키지 | 용도 |
|--------|------|
| `transformers>=4.40.0` | Qwen2.5 모델 |
| `peft>=0.10.0` | LoRA 파인튜닝 |
| `trl>=0.8.0` | SFTTrainer |
| `chromadb` | 벡터 스토어 |
| `rank-bm25>=0.2.2` | BM25 희소 검색 |
| `openai` | 임베딩 + LLM-as-Judge |
| `fastapi` + `streamlit` | API + UI |

---

## 브랜치 전략

`main` — 안정 통합 브랜치 (직접 push 지양)
