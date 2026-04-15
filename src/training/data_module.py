"""
학습 데이터 모듈.

역할:
  - 라벨링 JSON/JSONL 로드 (CoT 포맷)
  - 
  - DataCollatorForCompletionOnlyLM 마스킹 (Response 부분만 loss 계산)

"""

from datasets import load_dataset

class MedicalDataModule:
    """
    [클래스 개요]
    사람이 읽기 편한 엑셀/JSON 형태의 원본 데이터를, 인공지능이 소화할 수 있는 
    '정형화된 텍스트 덩어리(Prompt)'로 가공해 주는 데이터 전처리(Preprocessing) 전담 공장입니다.
    """

    def __init__(self, tokenizer):
        """
        [기능] 
        토크나이저(단어 사전)를 모듈 내부에 저장해 둡니다.
        
        [왜 필요한가? (Why)]
        데이터를 가공할 때, 챗봇에게 "여기서 대답이 끝났어!"라고 알려주는 특수 마침표 기호
        (EOS 토큰, End of Sequence)를 정확하게 가져다 붙이기 위해 반드시 필요합니다.
        """
        self.tokenizer = tokenizer

    def get_formatted_dataset(self, data_path):
        """
        [기능]
        지정된 경로의 데이터를 불러와서 LLM 파인튜닝의 '정석 포맷'으로 엮어낸 뒤 반환합니다.
        
        [예상 결과 (Result)]
        'instruction(지시)', 'input(증상)', 'output(진단)'으로 쪼개져 있던 데이터베이스 컬럼들이, 
        모델이 바로 읽고 학습할 수 있는 하나의 거대한 문장 컬럼("text")으로 합쳐집니다.
        """
        
        # ---------------------------------------------------------
        # 1. 고효율 데이터 로드 (Memory-Efficient Loading)
        # ---------------------------------------------------------
        # [기법] HuggingFace Datasets 라이브러리 사용
        # 일반적인 파이썬 open()이나 pandas의 read_json()을 쓰지 않는 이유:
        # 이 라이브러리는 수십 GB의 거대한 데이터도 RAM에 한 번에 올리지 않고, 
        # 디스크에서 필요한 만큼만 스트리밍(청크 단위)하듯 가져와서 메모리 터짐(OOM)을 원천 차단합니다.
        dataset = load_dataset('json', data_files={'train': data_path})['train']

        def format_func(example):
            """
            [기능] 프롬프트 템플릿 씌우기 (Prompt Formatting)
            이 함수가 모델의 '말투'와 '논리 구조'를 결정짓는 파이프라인의 심장부입니다.
            """
            texts = []
            for i in range(len(example['instruction'])):
                # [핵심 기법 1] 일관된 템플릿 구조화 (Alpaca/CoT Style)
                # 모델에게 "지시사항 -> 환자 정보 -> AI의 답변" 이라는 명확한 구획을 훈련시킵니다.
                # 이 뼈대(### 마커)가 없으면 모델은 언제 질문을 읽고 언제 답을 시작해야 할지 몰라 헤매게 됩니다.
                text = (
                    f"### Instruction:\n{example['instruction'][i]}\n\n"
                    f"### Input:\n{example['input'][i]}\n\n"
                    
                    # [핵심 기법 2] 할루시네이션(환각) 방어의 핵심: EOS 토큰 결합
                    # 답변(output)이 끝나는 지점에 반드시 특수 종료 마크(예: <|im_end|>)를 찍어줍니다.
                    # 이 처리를 누락하면, 실제 서비스에서 챗봇이 대답을 끝내고도 말을 끊지 못해
                    # "그리고... 그래서 환자는..." 하며 무한 루프 헛소리를 뱉어내게 됩니다.
                    f"### Response:\n{example['output'][i]}{self.tokenizer.eos_token}"
                )
                texts.append(text)
            
            # SFTTrainer는 학습할 때 기본적으로 "text"라는 이름의 컬럼을 찾아가므로 이름을 맞춰서 반환합니다.
            return {"text": texts}

        # ---------------------------------------------------------
        # 2. 고속 병렬 처리 및 메모리 청소 (Batching & Garbage Collection)
        # ---------------------------------------------------------
        # [기법 1] batched=True: 
        # 데이터를 for문으로 한 줄씩 굼벵이처럼 처리하지 않고, 1,000개씩 묶어서(Batch) 
        # CPU 코어를 모두 활용해 병렬 처리(Map)합니다. 전처리 속도가 수십 배 빨라집니다.
        #
        # [기법 2] remove_columns: 
        # 새로 만든 예쁜 "text" 컬럼 하나만 남기고, 기존의 지저분한 원본 컬럼(instruction, input 등)은
        # RAM에서 싹 지워버립니다. GPU로 데이터를 넘기기 전에 RAM 다이어트를 시키는 실무 필수 테크닉입니다.
        return dataset.map(format_func, batched=True, remove_columns=dataset.column_names)