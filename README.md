<!-- Medical-Chatbot 깃허브 repo용 README. 검증 사실(발표자료 PDF·AI-Hub·repo) 기반. 작가가 repo 루트 README.md로 복사. -->
# 알려줄고양 · Medical Chatbot

> 내과 도메인에 특화된 **AI 의료 자문 챗봇**. LoRA 파인튜닝 + Hybrid RAG + RedFlag 안전필터로 "안전을 코드로 설계"한 사전 문진 도우미.

![stage](https://img.shields.io/badge/stage-작동_프로토타입-blue) ![model](https://img.shields.io/badge/LLM-Qwen2.5--7B--Instruct%2BLoRA-orange) ![rag](https://img.shields.io/badge/RAG-Hybrid(BM25%2BDense)-green) ![license](https://img.shields.io/badge/data-AI--Hub_(clean--room)-lightgrey)

> ⚠️ **의료 안전 고지** — 본 챗봇은 **참고용 자문 도구**이며 의학적 진단·처방을 대체하지 않습니다. 이상 증상은 반드시 전문 의료기관을 방문하세요.

---

## 📄 포트폴리오 · 데모

- **인터랙티브 포트폴리오**: [`docs/portfolio.html`](./docs/portfolio.html) — 기능·아키텍처·평가 결과를 한눈에 (repo `docs/`에 배치 후 GitHub Pages 권장)
- **발표자료(PDF)**: [Google Drive](https://drive.google.com/file/d/1Tu9P8ALFhJbKnqfvpBHS3H_C5oGtQfgj/view)
- **데모 영상 (시나리오 4종)**:
  | 시나리오 | 시연 기능 | 링크 |
  |---|---|---|
  | 체온 40도 | 🚨 Red Flag → Emergency Exit | https://youtu.be/TfC9v9idPLc |
  | 전신 두드러기 | ⚠️ Semi Red Flag 경고 배너 | https://youtu.be/x3v56hw8stk |
  | 감기 (싱글턴) | 단일 질의 자문 | https://youtu.be/neYV6TMC-XA |
  | 복통 (멀티턴) | 🔁 다중턴 문진 → 리포트 | https://youtu.be/_FehgjfhF9g |

---

## ✨ 핵심 기능

- **🚨 RedFlag 안전장치** — 응급 키워드를 *안내문이 아닌 코드*로 탐지. `BYPASS` 13개 정규식 감지 시 LLM·RAG를 건너뛰고 119·응급실 가이드 즉시 반환(Emergency Exit). `RED_FLAG` 37개 키워드는 경고 배너 후 정상 자문 진행.
- **📚 Hybrid RAG (출처 인용)** — Dense(의미) + BM25(키워드) 검색을 α=0.5로 결합. 참조 청크·유사도까지 제시해 환각을 억제.
- **🧬 QLoRA 도메인 특화** — Qwen2.5-7B-Instruct를 내과 데이터로 **LoRA(r=16) + 4-bit 양자화(nf4)** 파인튜닝(QLoRA). PeftModel로 어댑터 동적 부착.
- **🔤 QueryNormalizer** — 일상어("머리 지끈")를 표준 의학용어("두통")로 변환해 검색 정확도 확보. *피부질환 의료상담 챗봇 질의문 언어패턴 연구*를 내과에 응용한 `SYMPTOM_PATTERNS` 설계.
- **🔁 멀티턴 문진** — 기간·악화요인·동반증상을 추가 질문으로 수집(감별진단).
- **📋 의사용 문진 요약 리포트** — 의심질병·참조 문서·인근 병원(카카오맵)을 의사 전달용으로 정리. `PostFilter`가 단정형→권유형 어미 변환 + 면책 자동 삽입.

---

## 🏗 아키텍처

```
사용자 입력
   │  Streamlit UI (:8501)        FastAPI (:8000)
   ▼
ChatService (오케스트레이터)
   ├── SafetyService     → RedFlagDetector(BYPASS 13 / RED_FLAG 37) · PostFilter
   ├── RetrievalService  → QueryNormalizer → Hybrid(Dense+BM25, α=0.5) → ChromaDB
   └── GenerationService → SYSTEM_PROMPT + RAG_CONTEXT + query → OpenAI / Qwen(+LoRA)
   ▼
ChatResult (자문 응답 + 참조 근거 + 면책)
```

## 🧰 기술 스택

| 영역 | 스택 |
|---|---|
| LLM | Qwen2.5-7B-Instruct + **QLoRA**(LoRA r=16 · 4-bit nf4) · PeftModel · OpenAI API(CoT 전처리·한↔영 의학용어 Dictionary) |
| 백엔드 | FastAPI (`:8000`) · ChatService / SafetyService / RetrievalService / GenerationService |
| 프론트엔드 | Streamlit (`:8501`) — Disclaimer → Profile → Chat → Report 4단계 |
| RAG | ChromaDB(벡터) · BM25Okapi + Kiwi 형태소 · Hybrid(α=0.5) · ko-sroberta-multitask(768d) |
| 안전 | RedFlagDetector · PostFilter |
| 외부 | 카카오맵(위치 기반 병원·응급실) |

---

## 🧠 모델 학습

- **베이스**: Qwen2.5-7B-Instruct (76.1억 파라미터) → Full Fine-Tuning 불가 → **QLoRA = LoRA(r=16, 642만 파라미터 / 전체의 0.084%) + 4-bit 양자화(`load_in_4bit: true`, `quant_type: nf4`)** 로 자원 절감. (config: `train_config.yaml`)
- **CoT 전처리**: `instruction`="주어진 임상 문제 및 환자 상태를 분석하여, 최적의 진단/처치 또는 의학적 근거를 논리적인 추론 과정과 함께 서술하시오." → 출력 `[상황·핵심 파악] → [의학적 추론·근거] → [최종 결론]` 3단 구조.
- **3-Stage 커리큘럼** (질문 유형별 점진 학습):

  | Stage | 데이터 | max_len | batch | lr | epoch |
  |---|---|---|---|---|---|
  | 1 | 단답형 | 560 | 4 | 0.002 | 2 |
  | 2 | 서술형 | 730 | 2 | 0.001 | 2 |
  | 3 | 객관식 | 840 | 2 | 0.0005 | 2 |

---

## 🗂 데이터 출처 (clean-room)

학습·RAG 데이터는 **AI-Hub** 공개 의료 데이터셋을 사용했습니다.

- [**필수의료 의학지식 데이터** (dataSetSn 71875)](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71875)
- [**전문 의학지식 데이터** (dataSetSn 71874)](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71874)
- 구축: 과학기술정보통신부 · 한국지능정보사회진흥원(NIA) / 주관: 가톨릭대학교 산학협력단 (서울성모·삼성서울·서울대·세브란스·보라매병원 참여)

**본 프로젝트 사용 범위**
- 내과 도메인 질의응답 **12,781쌍** → train/valid **10,299 / 1,248** 분할·CoT 전처리 → LoRA 학습
- 원천 의학 말뭉치(의학 교과서·학회 가이드라인·온라인 의료 정보) → 청킹(size 500 / overlap 50) → **RAG 약 27,000 chunk** (학술 논문·기타는 미사용)
- 처리 데이터(repo): `TL_내과_통합.json` · `VL_내과_통합.json`(내과 통합 train/valid) · 한↔영 의학용어 사전 `ko_en_dictionary.json`

> 🔒 본 데이터는 보건의료 데이터로 **안심존(보안구역)** 을 통해 연구·개발 목적으로 개방됩니다. **원본 데이터는 재배포하지 않으며, 본 저장소는 *설계·코드·결과*만 공개합니다.**

---

## 📊 평가 결과 — 3원 통제 비교 (valid 1,248)

> 동일 평가셋에서 변수(LoRA·RAG)를 하나씩 통제 추가해 설계 선택의 독립 기여도를 분리.

| 지표 | Mode A (LLM) | Mode B (LLM+RAG) | **Mode C (LoRA+RAG)** ✅ |
|---|---|---|---|
| 객관식 EM (995) | 0.69 | 0.67 | **0.71** |
| 단답형 EM (148) | 0.05 | 0.06 | **0.06** |
| 서술형 Rouge-L (105) | 0.12 | 0.11 | **0.14** |
| 서술형 BERT Score (105) | 0.70 | 0.69 | **0.73** |

- **LoRA Fine-tuning + Hybrid Retrieval RAG(Mode C)가 전 지표 최고.** 단 객관식 +0.02·서술형 BERT +0.03의 *소폭 우위*이며, "압도적 향상"이 아님을 정직하게 표기합니다.
- **단답형 EM이 전 모드 0.05~0.06으로 낮음** — 한↔영 의학용어 표기 차이에서 오는 *지표 한계*(채점용 한영 Dictionary 생성). 과대포장 없이 노출합니다.

> **평가 설계(repo)**: 자동지표(EM·ROUGE-L·BERTScore) 외에 **Evidence-F1**(모델이 인용한 근거 vs gold 근거 일치 — `retrieved_doc_ids` ↔ `gold_doc_ids`), **LLM-as-Judge**, Safety metrics(red-flag 검출률)까지 설계. 개선 유의성은 paired 검정(McNemar·Wilcoxon)으로 확인. (`metrics.py`)

---

## 🚀 실행 (BYOK)

> 키는 코드에 하드코딩하지 않고 **환경변수 또는 앱 내 입력(BYOK)** 으로 주입합니다. `.env`는 `.gitignore` 처리(템플릿은 `.env.example`).

```bash
# 1) 설치
pip install -r requirements.txt

# 2) 환경변수 (또는 Streamlit 앱에서 직접 입력)
cp .env.example .env          # OPENAI_API_KEY 등 설정

# 3) 지식베이스 구축 (RAG 인덱스 — ChromaDB + BM25)
python src/.../build_knowledge_base.py    # 경로는 repo 구조 참조

# 4) (선택) 도메인 학습 — QLoRA 3-Stage 커리큘럼
bash run_train.sh             # 설정: training/train_config.yaml

# 5) 앱 실행 (Streamlit)
streamlit run src/frontend/streamlit_app.py
```

> repo 핵심 모듈 — 검색·임베딩 `embedder.py` / `chroma_store.py` / `bm25_retriever.py` / `fusion.py` · 생성·학습 `model_module.py` / `trainer_module.py` / `generation_service.py` · 안전·평가 `safety_service.py` / `detector.py` / `metrics.py` · 데이터 `build_knowledge_base.py` / `chunker.py` (config: `train_config.yaml` · `app_config.yaml`).

---

## 🔭 한계 · 다음

- **단답형 EM 한계** → 의학용어 사전 확장 · Fuzzy Match 평가 도입.
- **Reranker 미적용**(현재 Hybrid fusion까지) → cross-encoder(예: bge-reranker) 추가 시 정밀도 향상 기대. (근거: Nogueira & Cho, *Passage Re-ranking with BERT*, 2019)
- **RAG 전용 평가 미실시** → RAGAS(faithfulness·context recall)로 검색/생성 단계 분리 진단 예정. (근거: Es et al., arXiv:2309.15217)
- 현재 **내과 한정** → 타과 확장 시 도메인별 데이터·안전필터 재설계 필요.

---

## 👤 팀 · 기간

- **멋쟁이사자처럼 AI 엔지니어 NLP 부트캠프** 팀 프로젝트 (4인) · 기간 약 1주일
- **이주형** — 팀장 · PM · 프론트엔드 · UX · AI 파이프라인 기획 · 검색 모듈(BM25+Hybrid) 담당 (모델 학습·백엔드 일부는 팀 공동)

---

## 📜 라이선스 · 면책

- 코드: (저장소 LICENSE 참조)
- 데이터: AI-Hub 보건의료 데이터(연구·개발 목적, 안심존 개방) — 원본 비재배포.
- **의료 면책**: 본 서비스는 참고용 자문 도구이며 의학적 진단·처방을 대체하지 않습니다. 응급 시 즉시 119 또는 가까운 응급실로.
