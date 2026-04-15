# 협력 노트

> 팀 개발 시 주의사항 및 가이드

## 브랜치 전략

- `main`: 최종 릴리스 버전
- `develop`: 개발 정상 버전
- `feature/*`: 기능 개발 (예: `feature/rag-improvement`)
- `bugfix/*`: 버그 수정 (예: `bugfix/embeddings-error`)
- `hahyun`: 팀 협업 분기

## 커밋 메시지 규칙

```
[타입] 간단한 설명

상세 설명 (필요시)

- 변경사항 1
- 변경사항 2

Fixes #123 (관련 이슈 번호, 필요시)
```

### 타입

- `feat`: 새 기능 추가
- `fix`: 버그 수정
- `refactor`: 코드 리팩토링 (기능 변화 X)
- `perf`: 성능 개선
- `docs`: 문서 수정
- `test`: 테스트 추가/수정
- `chore`: 설정, 의존성 추가 (기능 변화 X)

### 예시

```
feat: 하이브리드 검색 가중치 조정

- Dense 검색 가중치를 0.7에서 0.6으로 변경
- BM25 가중치를 0.3에서 0.4로 변경
- 검색 정확도 5% 향상 확인

Fixes #45
```

## 코드 리뷰 체크리스트

PR을 생성하기 전에 다음을 확인하세요:

- [ ] 코드가 PEP 8 스타일을 따르는가?
- [ ] 새 기능에 대한 문서가 있는가?
- [ ] 불필요한 주석은 제거했는가?
- [ ] 테스트를 추가했는가?
- [ ] 로컬에서 테스트 가능한가?
- [ ] `.env.example` 파일을 업데이트했는가 (환경 변수 추가 시)?

## 파일 수정 규칙

### 절대로 수정하면 안 되는 파일들

- `.gitignore`: 별도 논의 필요
- `requirements.txt`: 패키지 관리자 사용 (pip freeze 금지)
- `configs/`: 중요 설정 변경은 팀 공지

### 각자의 작업 영역

```
src/
├── app/          # 🟡 API 라우팅 (API담당)
├── config/       # 🔴 전역 설정 (리드만 수정)
├── services/     # 🟢 비즈니스 로직 (모두 가능)
├── retrieval/    # 🟢 검색 모듈 (모두 가능)
├── llm/          # 🟢 LLM 클라이언트 (모두 가능)
├── safety/       # 🟡 안전 필터 (Safety담당)
├── evaluation/   # 🟡 평가 (평가담당)
├── ingestion/    # 🟢 데이터 수집 (모두 가능)
└── training/     # 🟡 학습 (학습담당)
```

- 🟢 초록색: 누구나 수정 가능
- 🟡 노란색: 담당자 검토 후 수정
- 🔴 빨간색: 리드만 수정 가능

## 데이터 관리

- 원본 데이터: `data/raw/` (Git에서 제외, .gitignore)
- 처리된 데이터: `data/processed/` (Git에서 제외)
- 벡터 DB: `chroma_db/` (Git에서 제외)
- 모델: 로컬만 (HuggingFace에서 자동 다운로드)

## 시크릿 관리

- `.env` 파일: **절대 Git에 커밋하지 말 것**
- `.env.example`: 공개 가능한 샘플, Git에 커밋 ✅
- API 키: 환경 변수로만 전달

## 충돌 해결 (Merge Conflict)

```bash
# 최신 develop 받기
git fetch origin develop

# 현재 브랜치에 develop 병합
git merge origin/develop

# 충돌 파일 확인
git status

# 충돌 수정 후
git add <파일>
git commit -m "Resolve merge conflict"
```

## 정기 회의

- 주간 회의: 매주 월요일 10:00 (진행 상황 공유)
- 스탠드업: 매일 오전 팀 Slack 공지
- 리뷰: 마감 전 금요일 (최종 검수)

## 긴급 연락

- 🔴 Critical 버그: 팀 Slack 호출
- 🟡 Important: Group chat 메시지
- 🟢 Normal: Issue 등록

---

**행운을 빕니다! 팀의 성공을 응원합니다! 🚀**
