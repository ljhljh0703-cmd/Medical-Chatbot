# API 문서 (API Documentation)

## FastAPI 서버

### 기본 정보

- **기본 URL**: `http://localhost:8000`
- **문서**: `http://localhost:8000/docs` (Swagger UI)
- **대체 문서**: `http://localhost:8000/redoc` (ReDoc)

---

## 엔드포인트

### 1. 채팅 쿼리 처리

#### `POST /chat/query`

의료 질문에 대한 답변을 생성합니다.

**요청 (Request)**

```json
{
  "query": "감기 증상이 있어요",
  "mode": "B",
  "stream": false,
  "use_hybrid": false,
  "top_k": 5
}
```

**파라미터**

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `query` | string | ✅ | 사용자의 의료 질문 |
| `mode` | string | ❌ | `A` (LLM), `B` (LLM+RAG), `C` (LLM+RAG+LoRA), 기본값: `C` |
| `stream` | boolean | ❌ | 스트리밍 응답, 기본값: `false` |
| `use_hybrid` | boolean | ❌ | 하이브리드 검색 사용, 기본값: `.env` 설정값 |
| `top_k` | integer | ❌ | 검색 결과 개수, 기본값: 5 |

**응답 (200 OK)**

```json
{
  "query": "감기 증상이 있어요",
  "mode": "B",
  "response": "감기는 바이러스 감염으로 인한 상기도 감염입니다...",
  "retrieved_docs": [
    {
      "chunk_id": "doc_1_0",
      "text": "감기의 정의 및 증상...",
      "score": 0.92
    }
  ],
  "model_used": "qwen",
  "processing_time_ms": 1234
}
```

**에러 응답 (400 Bad Request)**

```json
{
  "detail": "Query is too short (minimum length: 2 characters)"
}
```

---

### 2. 검색만 수행

#### `POST /chat/retrieve`

쿼리에 관련된 문서를 검색합니다.

**요청**

```json
{
  "query": "당뇨병 치료",
  "top_k": 10,
  "use_hybrid": true
}
```

**응답 (200 OK)**

```json
{
  "query": "당뇨병 치료",
  "total_retrieved": 10,
  "documents": [
    {
      "chunk_id": "doc_5_2",
      "text": "당뇨병 치료의 기본 원칙은...",
      "score": 0.95,
      "metadata": {
        "source": "TL_내과_통합.json",
        "chunk_index": 2
      }
    }
  ]
}
```

---

### 3. 건강 상태 확인

#### `GET /health`

서버 상태를 확인합니다.

**응답 (200 OK)**

```json
{
  "status": "healthy",
  "model": "qwen",
  "chroma_db": "connected",
  "openai_api": "configured"
}
```

---

### 4. 모델 정보 조회

#### `GET /models`

현재 활성화된 모델 정보를 조회합니다.

**응답 (200 OK)**

```json
{
  "backend": "qwen",
  "model_id": "Qwen/Qwen2.5-7B-Instruct",
  "mode": "C",
  "lora_adapter": "./adapters/medical_lora",
  "embedding_model": "jhgan/ko-sroberta-multitask",
  "max_tokens": 512,
  "temperature": 0.7
}
```

---

## 에러 코드

| 코드 | 설명 |
|------|------|
| `400` | 잘못된 요청 (누락된 필드, 잘못된 형식) |
| `422` | 유효성 검사 실패 |
| `500` | 서버 내부 오류 |
| `503` | 서비스 이용 불가 (모델 로딩 대기) |

---

## 사용 예제

### Python requests

```python
import requests

# 기본 쿼리
response = requests.post(
    "http://localhost:8000/chat/query",
    json={"query": "감기 증상"}
)
print(response.json())

# 하이브리드 검색 + LoRA 모드
response = requests.post(
    "http://localhost:8000/chat/query",
    json={
        "query": "당뇨병 관리",
        "mode": "C",
        "use_hybrid": True,
        "top_k": 10
    }
)
print(response.json())
```

### JavaScript/TypeScript

```javascript
// Fetch API
const response = await fetch("http://localhost:8000/chat/query", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "고혈압 증상",
    mode: "B"
  })
});

const data = await response.json();
console.log(data.response);
```

### cURL

```bash
curl -X POST http://localhost:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "심부전 치료",
    "mode": "C",
    "top_k": 5
  }' | jq .
```

---

## Rate Limiting (계획 중)

현재는 rate limiting이 없지만, 프로덕션 배포 시 다음을 고려합니다:

- 사용자당 초당 최대 10 요청
- 시간당 최대 1000 요청
- IP 기반 트로틀링

---

## 사용 팁

1. **첫 요청이 느린 이유**: 모델 로딩에 시간 소요 (초, 최대 30초)
2. **응답 품질 향상**: `mode="C"` (LoRA) + `use_hybrid=true` 권장
3. **성능 최적화**: GPU 사용 (`CUDA_VISIBLE_DEVICES` 설정)
4. **디버깅**: `LOG_LEVEL=DEBUG` 설정하고 `./logs/app.log` 확인
