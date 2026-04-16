**Plan: 의료 지식 기반 QA (RAG + LoRA)**
간단요약: 한국어 의료(내과) 질문에 대해 검색 기반 근거를 함께 제공하면서 정확도(정답률)를 최우선 목표로 삼는 연구입니다. 하이브리드 검색(Chroma + BM25)으로 관련 문서를 회수한 뒤, LoRA로 LLM을 도메인 적응시켜 근거 활용 정확도를 높이고, 안전 필터로 유해·오도 응답을 최소화합니다.

Problem Definition

Task: Knowledge-grounded QA(의료 내과 도메인) — 객관식/단답/서술형 질문에 대해 근거 문서를 제시하며 정확한 답변 생성.
구체적 문제: 한국어 입력 질문에 대해 관련 문헌/문서에서 근거를 검색하고, 검색 결과를 바탕으로 오답·hallucination을 최소화한 정확한 답변을 생성. 객관식·단답형은 Exact Match 중심, 서술형은 내용 유사성·근거 정합성으로 평가.
중요성(응용): 임상 의사결정 보조, 의학 교육(문제 자동 채점·해설), 환자 상담 보조 등에서 잘못된 정보는 심각한 위험을 초래하므로 높은 정확도와 근거 제시가 필수적.
Background & Baseline

기존 접근 요약: Retrieval-Augmented Generation(RAG)은 dense 임베딩 기반 검색과 sparse 키워드 기반 검색을 결합하여 LLM의 지식 근거를 제공하는 표준적 방법입니다. 도메인 적응에는 LoRA/SFT가 비용 효율적입니다.
현 레포 기반 요소: Chroma 벡터스토어와 BM25 sparse 검색이 구현되어 있으며, LoRA 학습 파이프라인과 4-bit 양자화 설정이 존재합니다. (참고 파일: build_knowledge_base.py, chroma_store.py, train_config.yaml).
초기 Baseline (향후 비교군): OpenAI(클라우드) zero/few-shot, Qwen-base(로컬) zero-shot, RAG(검색 포함) + 베이스 LLM. 본 연구의 개선 대상은 RAG + LoRA 구성으로, baseline 대비 증거 정합성 및 EM 향상을 목표로 함.
Proposed Method

요약: 하이브리드 검색(Chroma dense 임베딩 + BM25 sparse)로 후보 문서를 회수한 뒤, retrieval context와 함께 LLM에 입력해 생성. 모델 측면에서는 Qwen 계열에 LoRA로 경량 파인튜닝을 적용해 도메인 표현을 학습시키고, 필요 시 4-bit 양자화로 자원 절감.
구성 요소 상세:
Retrieval: dense 임베딩은 embedder.py로 생성, 벡터는 Chroma에 저장(chroma.sqlite3). sparse 검색은 bm25_retriever.py를 사용. 두 결과를 fusion.py의 융합 방식(가중치 α, RRF 등)으로 결합.
Generation & Adaptation: Qwen(로컬)을 주 모델로 사용하고 LoRA 어댑터를 PeftModel 방식으로 동적 부착. 주요 하이퍼파라미터는 lora_r, lora_alpha, lora_dropout (참조: train_config.yaml).
Evidence Alignment: 학습 데이터에 정답과 관련 근거 passage를 명시하여 모델이 근거를 참조하도록 지도학습(예: input에 "Context: [retrieved passsages]" 포함). 추가로 evidence-consistency를 평가용 지표로 도입.
Safety/Post-processing: RedFlagDetector로 쿼리 및 응답 위험 탐지, PostFilter로 면책문구·톤 조정 및 위험 응답 차단.
왜 효과적인가: Hybrid retrieval은 키워드 민감성 및 의미적 유사성의 장점을 결합해 관련 증거 회수율을 높이고, LoRA는 소량의 도메인 데이터로 모델이 의료 표현·판단 패턴을 빠르게 학습해 근거 활용도를 증가시킵니다.
Dataset & Preprocessing

데이터 소스: 내부 코퍼스 — TL_내과_통합.json, VL_내과_통합.json. 보조 사전: ko_en_dictionary.json.
데이터 특성: 객관식·단답·서술형 혼합, 한국어 중심, 의료 전문 용어·약어 포함. 샘플별 메타(예: q_type, domain, qa_id)가 존재함.
전처리 파이프라인:
정규화: 약어 확장, 철자·표기 정규화, 한영 매핑(사전 활용).
PII 제거: 환자 식별 정보 제거 또는 익명화 절차 적용.
청크 및 인덱싱: JSONLoader → Chunker(chunker.py)로 문서 청크 생성(메타 포함) → 임베딩 생성 후 Chroma에 인덱싱(build_knowledge_base.py).
라벨 정제: 객관식/단답의 정답 표기 일관화(동의어 처리 규칙 정의). gold answer normalization 필수.
데이터 분할: stratified 방식으로 train/dev/test 분리(예: q_type·domain 기준 층화). (실험 실행은 현재 진행 상황에 맞춰 조정)
Experiment Design

목표: 정확도(정답률)를 최대화하되, 모델이 실제로 근거를 참조하는지(Evidence Alignment)를 함께 확보.
실험 조합(설계):
RAG variants: dense-only, sparse-only, hybrid (fusion 파라미터 α 조정).
LoRA variants: r·alpha·dropout 조합 실험(예: lora_r ∈ {8,16}, lora_alpha ∈ {16,32}).
Quantization 영향: 4-bit 양자화(load_in_4bit: true, quant_type: nf4) 사용 시 성능 저하 여부 검증.
Ablation study: RAG-only vs RAG+LoRA, retrieval k 변화, evidence input 유무.
평가 지표 및 검증 방법:
정량지표: Exact Match(객관식·단답), Accuracy, ROUGE-L(서술), BERTScore(서술), Evidence-F1(모델이 제시한 근거와 gold 근거의 일치성), Safety metrics(red-flag 검출 비율).
Evidence-F1 정의(제안): 모델이 인용한 passage들 중 gold evidence로 태깅된 passage의 재현율/정밀도 기반 F1. (retrieved_doc_ids vs gold_doc_ids 비교)
자동 평가 보완: llm_judge(LLM-as-Judge)로 정성적 판단 보조.
통계적 검정: paired 테스트(연속형: t-test 또는 Wilcoxon, 이진: McNemar)로 개선 유의성 확인.
실험 절차(요약):
데이터 전처리·인덱스 생성.
Baseline(베이스 LLM) 성능 측정(검증셋).
LoRA 학습(도메인 적응) → 평가.
RAG 파라미터 튜닝(검색 k, fusion α) 및 최종 평가.
인간 검토(전문가 평가)으로 자동 지표 보완.
Human Expert Evaluation (개요)

목적: 모델 출력의 정합성(정확성), 제시한 근거의 타당성, 실무적 유용성 및 안전성(임상 위험 가능성) 점검.
평가자 자격: 임상의(내과 전공 권장) 또는 임상 경험이 있는 전문가. 다수 평가자를 통한 독립 평가 및 불일치 시 중재(adjudication)를 실시.
평가지표(라벨링 스키마):
correctness: {correct / partial / incorrect} — 객관식/단답/서술 유형별 판단 기준 제공.
evidence_alignment: {support / contradict / no_evidence} — 모델이 제시한 출처가 실제 근거인지 여부.
helpfulness: 1-5 Likert(임상적 유용성).
safety_risk: {safe / minor_issue / unsafe} — 잠재적 오도 또는 해악 여부.
notes: 자유기술.
운영 원칙: 각 샘플은 복수 평가자에 의해 독립 평가되며, 불일치 시 adjudication으로 gold 라벨을 확정. IAA(Inter-Annotator Agreement)를 산출하여 라벨 품질을 검증.
Ethics & Risk Management

PII/데이터 보호: 데이터에 포함된 식별자 제거·익명화 필수, 접근 통제 및 기록 유지.
임상 적용 주의: 연구 결과는 임상 결정의 보조 수단으로 제한, 사용자-facing 서비스에는 명확한 disclaimer 적용(PostFilter) 및 전문가 검토 권고.
IRB 고려: 데이터·평가 방식에 따라 기관 심의 필요 여부 판단.
Deliverables & Reproducibility

산출물: 학습 스크립트(run_train.sh), 최종 LoRA 어댑터(아티팩트), Chroma 인덱스 스냅샷, 평가 결과(자동 지표 + 전문가 라벨), 분석 리포트.
재현성 표준: 모든 실험은 seed 고정, 사용 config 기록(train_config.yaml, app_config.yaml), 모델/인덱스/어댑터 저장 경로 명시.
Implementation Pointers (참고 파일)

데이터·청크·인덱스: build_knowledge_base.py, chunker.py
검색·임베딩: embedder.py, chroma_store.py, bm25_retriever.py, fusion.py
생성·LoRA: model_module.py, trainer_module.py, generation_service.py
안전·평가: safety_service.py, detector.py, metrics.py
