"""
학습 데이터 전처리 모듈 (Data Module)

역할:
  - 라벨링된 의료 데이터(JSONL)를 안전하게 읽어옵니다.
  - 리스트 형태의 데이터를 고속 처리에 유리한 HuggingFace Dataset 객체로 변환합니다.
  - 기존 평가 기준(Baseline)과 완벽히 일치하도록 프롬프트를 조립합니다.
  - 모델이 한 번에 읽고 학습할 수 있는 단일 'text' 컬럼을 생성합니다.

수정/반영 사항:
  - 통제된 변수 비교(Apples-to-Apples)를 위해, 최신 대화형(messages) 구조나 
    Completion-Only 마스킹 기법을 제거하고 전통적인 단일 텍스트(text) SFT 방식으로 복구했습니다.
"""

import json
from datasets import Dataset

class MedicalDataModule:
    """
    [클래스 개요]
    정제된 JSONL 형태의 의료 데이터를 불러와, AI 모델이 학습하기 가장 좋은 
    '질문-환자정보-응답(추론 포함)' 형태의 프롬프트로 전처리하는 공장(Factory) 역할을 합니다.
    """
    
    def __init__(self, tokenizer):
        # 토크나이저를 초기화 시 받아옵니다. 
        # (현재 버전에서는 단순히 텍스트만 합치지만, 추후 토큰 길이를 미리 계산하거나 
        # 잘라내는(Truncation) 고급 전처리를 도입할 때 유용하게 쓰입니다.)
        self.tokenizer = tokenizer

    # ------------------------------------------------------------
    # 1. 데이터 로드 및 포맷팅 메인 함수
    # ------------------------------------------------------------
    def get_formatted_dataset(self, data_path):
        """
        [기능] 
        디스크에 있는 파일을 읽어 메모리에 올리고, 모델 학습용 프롬프트 양식을 씌웁니다.
        """
        data_list = []
        
        # [각주 1] 인코딩 'utf-8-sig'의 중요성
        # 윈도우 환경에서 텍스트 파일을 만들면 눈에 보이지 않는 BOM(Byte Order Mark)이라는 
        # 특수 문자가 맨 앞에 붙는 경우가 많습니다. 'utf-8-sig'는 이를 알아서 걸러주어 
        # JSONDecodeError(파싱 에러)를 원천 차단합니다.
        with open(data_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                # 빈 줄이 들어왔을 때 에러가 나지 않도록 방어하는 로직입니다.
                if line.strip():
                    data_list.append(json.loads(line))
        
        # [각주 2] HuggingFace Dataset의 장점
        # 일반 파이썬 리스트보다 메모리를 훨씬 적게 차지하고, 
        # 나중에 수백만 건의 데이터를 다룰 때 병렬 처리(map) 속도가 압도적으로 빠릅니다.
        dataset = Dataset.from_list(data_list)

        # ------------------------------------------------------------
        # 2. 프롬프트 템플릿 설계 (Formatting)
        # ------------------------------------------------------------
        def formatting_prompts_func(example):
            """
            각 데이터의 조각(instruction, input, output)을 
            하나의 긴 문서(text)로 풀로 붙이듯 조립합니다.
            """
            # [각주 3] 프롬프트 엔지니어링
            # 모델에게 단순 암기가 아니라 '의학적 추론'을 유도하기 위해
            # 기존 비교 실험 모델과 100% 동일한 구분자(### 질문:, ### 응답: 등)를 사용합니다.
            text = (
                f"### 질문:\n{example['instruction']}\n\n"
                f"### 환자 정보 및 문제:\n{example['input']}\n\n"
                f"### 응답:\n{example['output']}"
            )
            return {"text": text}

        # ------------------------------------------------------------
        # 3. 전처리 실행 및 불필요한 데이터 청소
        # ------------------------------------------------------------
        # [각주 4] remove_columns의 역할
        # 모델은 오직 'text' 필드만 보고 학습합니다. 기존의 'instruction', 'input' 등의 
        # 원본 쪼가리 컬럼을 지워주지 않으면, Trainer가 "이건 무슨 데이터인지 모르겠어!"라며 
        # 에러를 뱉거나 메모리만 낭비하게 됩니다.
        return dataset.map(
            formatting_prompts_func, 
            remove_columns=dataset.column_names,
            desc="SFT용 단일 텍스트 프롬프트 포맷팅 중..."
        )