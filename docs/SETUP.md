# 초기 설정 가이드 (Setup Guide)

## 사전 요구사항

- Python 3.10 이상
- pip 또는 conda
- Git
- OpenAI API 키 (선택사항, Qwen 로컬 모델 사용 시)
- GPU (권장, CUDA 12.1+ for Qwen 로컬 추론)

## 1단계: 저장소 클론 및 환경 설정

```bash
# 저장소 클론
git clone -b hahyun https://github.com/ljhljh0703-cmd/Medical-Chatbot.git
cd Medical-Chatbot

# 가상 환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

## 2단계: 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집 (API 키 입력)
# 주의: OPENAI_API_KEY는 필수 (LLM-as-Judge 평가에 필요)
```

### 필수 환경 변수

- `OPENAI_API_KEY`: OpenAI API 키 (ChatGPT, 임베딩 모델)
- `MODEL_BACKEND`: `qwen` (로컬) 또는 `openai`
- `MODEL_MODE`: `A` (LLM), `B` (LLM+RAG), `C` (LLM+RAG+LoRA)

## 3단계: 지식베이스 구축

원본 데이터에서 ChromaDB 벡터 DB로 인덱싱합니다.

```bash
# 단일 파일 인덱싱
python src/ingestion/indexing/build_knowledge_base.py \
    --input data/raw/TL_내과_통합.json \
    --output ./chroma_db

# 다중 파일 인덱싱
python src/ingestion/indexing/build_knowledge_base.py \
    --input data/raw/ \
    --output ./chroma_db
```

## 4단계: 서버 실행

### FastAPI 서버 실행

```bash
# 기본 포트 8000에서 실행
uvicorn src.app.main:app --reload --port 8000

# 출력: Uvicorn running on http://127.0.0.1:8000
```

### Streamlit 프론트엔드 실행 (별도 터미널에서)

```bash
streamlit run src/frontend/streamlit_app.py

# 출력: You can now view your Streamlit app in your browser at http://localhost:8501
```

## 5단계: API 테스트

### cURL로 테스트

```bash
curl -X POST http://localhost:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "감기 증상이 있어요",
    "mode": "B",
    "stream": false
  }'
```

### Python으로 테스트

```python
import requests

response = requests.post(
    "http://localhost:8000/chat/query",
    json={
        "query": "감기 증상이 있어요",
        "mode": "B"
    }
)
print(response.json())
```

## Troubleshooting

### CUDA / GPU 문제

```bash
# GPU 가용성 확인
python -c "import torch; print(torch.cuda.is_available())"

# CPU만 사용하도록 강제
export CUDA_VISIBLE_DEVICES=""
```

### OpenAI API 키 오류

```bash
# API 키 설정 확인
echo $OPENAI_API_KEY

# API 키 유효성 확인
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Qwen 모델 다운로드 오류

```bash
# 모델 수동 다운로드
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
  AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct'); \
  AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"
```

## LoRA 학습 (선택사항)

```bash
# 데이터 준비
python training/prepare_dataset.py \
    --input data/raw/TL_내과_통합.json

# 학습 실행
bash run_train.sh

# 또는 직접 실행
python src/training/main_train.py --config configs/train_config.yaml
```

## 다음 단계

- [API 문서](./API.md) - REST API 엔드포인트
- [개발 가이드](./DEVELOPMENT.md) - 코드 구조 및 커스터마이징
- [평가 방법](./EVALUATION.md) - 모델 성능 평가
