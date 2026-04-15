# 개발 가이드 (Development Guide)

## 프로젝트 구조

```
src/
├── app/                 # FastAPI 애플리케이션
│   ├── main.py         # 진입점, CORS 설정
│   ├── routers/        # 라우터 (엔드포인트)
│   │   └── chat.py     # /chat/* 엔드포인트
│   └── schemas/        # Pydantic 스키마
│       └── chat.py     # 요청/응답 모델
│
├── config/             # 전역 설정
│   ├── settings.py     # 환경 변수 (BaseSettings)
│   └── prompts.py      # 시스템 프롬프트
│
├── services/           # 비즈니스 로직
│   ├── chat_service.py       # 핵심 파이프라인
│   └── retrieval_service.py  # 검색 서비스
│
├── retrieval/          # 검색 모듈
│   ├── dense/          # 벡터 검색
│   │   ├── embedder.py
│   │   ├── chroma_store.py
│   │   └── dense_retriever.py
│   ├── sparse/         # BM25 검색
│   │   └── bm25_retriever.py
│   ├── hybrid/         # 하이브리드 융합
│   │   └── fusion.py
│   └── query/          # 쿼리 처리
│       └── normalizer.py
│
├── llm/                # LLM 클라이언트
│   ├── generation_service.py    # 프롬프트 + 생성
│   └── model_clients/
│       ├── qwen_client.py       # 로컬 Qwen
│       └── openai_client.py     # OpenAI API
│
├── safety/             # 안전 필터
│   ├── safety_service.py
│   ├── red_flag/
│   │   └── detector.py          # 위험 감지
│   └── filters/
│       └── post_filter.py       # 응답 후처리
│
├── evaluation/         # 평가 모듈
│   ├── evaluator.py
│   ├── metrics.py      # EM, ROUGE, BERTScore
│   └── report.py
│
├── ingestion/          # 데이터 수집 파이프라인
│   ├── loaders/
│   │   └── json_loader.py
│   ├── preprocess/
│   │   └── cleaner.py
│   ├── chunking/
│   │   └── chunker.py
│   └── indexing/
│       └── build_knowledge_base.py
│
├── training/           # 학습 모듈
│   ├── data_module.py
│   ├── model_module.py
│   ├── trainer_module.py
│   └── main_train.py
│
├── observability/      # 로깅
│   └── logger.py
│
└── domain/             # 도메인 모델
    └── models/
        └── __init__.py
```

---

## 핵심 파이프라인

### Chat Service

`src/services/chat_service.py`는 전체 요청을 처리합니다:

```
입력 쿼리
  ↓
1. 안전 검사 (Safety Check)
  ↓
2. 쿼리 정규화 (Query Normalization)
  ↓
3. 문서 검색 (Dense/Hybrid/BM25)
  ↓
4. 프롬프트 조립 (Context Assembly)
  ↓
5. LLM 응답 생성 (OpenAI/Qwen)
  ↓
6. 응답 필터링 (Post Filter)
  ↓
출력 응답
```

---

## 개발 팁

### 1. 새로운 엔드포인트 추가

`src/app/routers/chat.py`에 추가:

```python
from fastapi import APIRouter, HTTPException
from src.app.schemas.chat import QueryRequest, QueryResponse

router = APIRouter()

@router.post("/my-endpoint")
async def my_endpoint(request: QueryRequest) -> QueryResponse:
    """내 엔드포인트 설명"""
    # 로직 구현
    return QueryResponse(...)
```

### 2. 새로운 LLM 백엔드 추가

`src/llm/model_clients/my_model_client.py` 생성:

```python
class MyModelClient:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
    
    def generate(self, prompt: str, **kwargs) -> str:
        return self.model.generate(prompt, **kwargs)
```

###3. 검색 알고리즘 변경

`src/retrieval/hybrid/fusion.py`의 가중치 조정:

```python
# 현재: Dense 70%, BM25 30%
WEIGHTS = {"dense": 0.7, "bm25": 0.3}

# 변경: Dense 50%, BM25 50%
WEIGHTS = {"dense": 0.5, "bm25": 0.5}
```

### 4. 시스템 프롬프트 커스터마이징

`src/config/prompts.py`에서 수정:

```python
SYSTEM_PROMPT = """당신은 경험 많은 내과 의사입니다.
환자의 질문에 정확하고 도움이 되는 답변을 제공하세요."""
```

---

## 로컬 개발 환경 설정

### 1. 개발 의존성 설치

```bash
pip install -e .  # 현재 프로젝트를 editable 모드로 설치
pip install pytest black isort flake8  # 개발 도구
```

### 2. 코드 포매팅

```bash
# Black으로 포매팅
black src/ tests/

# isort로 import 정렬
isort src/ tests/

# flake8으로 린트
flake8 src/
```

### 3. 테스트 작성 및 실행

```bash
# 테스트 작성
# tests/test_chat_service.py 생성

# 테스트 실행
pytest tests/ -v

# 커버리지 확인
pytest tests/ --cov=src/
```

---

## 디버깅

### 로그 레벨 설정

```bash
# .env에서 설정
LOG_LEVEL=DEBUG
```

### 대화형 디버깅

```python
# 특정 위치에서 실행을 중단하고 싶다면
import pdb; pdb.set_trace()  # Python debugger

# 또는 VSCode Python Extension 사용
# run.py 생성하고 디버거 시작
```

### ChromaDB 검색 결과 확인

```python
from src.retrieval.dense.chroma_store import ChromaStore

store = ChromaStore()
results = store.search("당뇨병", top_k=5)
for r in results:
    print(r)
```

---

## 성능 최적화

### 1. 모델 양자화 (BitsAndBytes 4-bit)

```python
# model_clients/qwen_client.py에서
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config
)
```

### 2. 배치 처리

```python
# 여러 쿼리를 한 번에 처리
queries = ["감기", "독감", "폐렴"]
results = batch_search(queries, top_k=5)
```

### 3. 캐싱

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text: str):
    return embedder.encode(text)
```

---

## 배포

### Docker 빌드

```bash
# Dockerfile 작성
docker build -t medical-chatbot:latest .

# 실행
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-xxx \
  medical-chatbot:latest
```

### 환경 변수 관리

```bash
# .env.production 생성
OPENAI_API_KEY=sk-prod-key
LOG_LEVEL=INFO
MODEL_BACKEND=openai  # 프로덕션에서는 OpenAI 권장
```

---

## 문제 해결

### 모듈 임포트 오류

```bash
# Python path 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 또는 .env 파일을 통해
python -c "from src.services.chat_service import ChatService"
```

### ChromaDB 연결 실패

```python
# 벡터 DB 재구축
python src/ingestion/indexing/build_knowledge_base.py --rebuild
```

### OOM (Out of Memory) 에러

```bash
# GPU 메모리 부족 시 CPU 사용
export CUDA_VISIBLE_DEVICES=""

# 또는 배치 크기 감소
BATCH_SIZE=1
```

---

## 기여 가이드 (Contributing)

1. 새로운 브랜치 생성: `git checkout -b feature/my-feature`
2. 변경 사항 커밋: `git commit -m "Add my feature"`
3. 푸시: `git push origin feature/my-feature`
4. Pull Request 생성

---

## 추가 리소스

- [Qwen 문서](https://github.com/qwenlm/Qwen/)
- [ChromaDB 가이드](http://docs.trychroma.com/)
- [FastAPI 튜토리얼](https://fastapi.tiangolo.com/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
