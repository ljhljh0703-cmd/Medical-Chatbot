"""
Colab 환경에서 학습을 실행하는 메인 엔트리포인트.

파이프라인:
  data_module → model_module → trainer_module

실행 방법:
    # 프로젝트 루트에서
    python src/training/main_train.py --config configs/train_config.yaml

    # run_train.sh 를 통해
    bash run_train.sh
"""
import os
import sys
import yaml
import torch
import gc
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

# 시스템 경로를 추가하여 src 폴더 안의 모듈들을 에러 없이 불러오도록 설정
sys.path.append(os.getcwd())

from src.training.data_module import MedicalDataModule
from src.training.model_module import MedicalModelModule

def main():
    """
    [스크립트 개요]
    모든 환경 설정(YAML), 데이터 전처리, 모델 로드, 그리고 3단계 커리큘럼 학습을 
    순차적으로 실행하고 메모리를 관리하는 메인 오케스트레이터입니다.
    """
    
    # ---------------------------------------------------------
    # 1. 설정 파일(YAML) 로드 (Decoupling)
    # ---------------------------------------------------------
    # [왜 필요한가?] 하이퍼파라미터를 파이썬 코드 안에 하드코딩하면 나중에 실험할 때마다 코드를 고쳐야 합니다.
    # 이렇게 분리해 두면, 연구자는 코드는 건드리지 않고 YAML 파일의 숫자만 바꿔가며 수백 번의 실험을 자동화할 수 있습니다.
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # ---------------------------------------------------------
    # 2. 모델 및 토크나이저 초기화
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(config['model']['id'])
    model_module = MedicalModelModule()

    # 4-bit 양자화된 베이스 모델을 불러오고, 그 위에 LoRA 메모지를 붙입니다.
    model = model_module.load_base_model(config['model']['id'], config['quantization'])
    model = model_module.apply_lora(model, config['training'])

    data_module = MedicalDataModule(tokenizer)

    # ---------------------------------------------------------
    # 3. 커리큘럼 러닝 (Curriculum Learning) 루프
    # ---------------------------------------------------------
    # [기법 설명] 인간이 초등 -> 중등 -> 고등학교를 거치듯, AI에게도 쉬운 것부터 가르칩니다.
    # Stage 1(단답) -> Stage 2(서술, CoT) -> Stage 3(객관식 판별) 순으로 난이도를 올립니다.
    for stage, params in config['training']['stages'].items():
        print(f"\n" + "="*60)
        print(f"🏥 STAGE START: {stage.upper()}")
        print(f"📊 Strategy: MaxLength={params['max_len']}, Batch={params['batch']}, LR={params['lr']}")
        print("="*60)

        # 현재 스테이지에 맞는 데이터셋(예: 단답형.jsonl)을 불러와서 전처리합니다.
        train_ds = data_module.get_formatted_dataset(config['data'][stage])

        # ---------------------------------------------------------
        # 4. SFT (Supervised Fine-Tuning) 설정 (핵심 엔진)
        # ---------------------------------------------------------
        sft_config = SFTConfig(
            output_dir=os.path.join(config['model']['output_dir'], stage),
            
            # 물리적 배치 사이즈: 한 번에 GPU 메모리에 올릴 데이터의 개수 (보통 2~4로 아주 작게 설정)
            per_device_train_batch_size=params['batch'],
            
            # [핵심] Gradient Accumulation (기울기 누적): "메모리 할부 결제"
            # VRAM이 부족해서 배치를 2밖에 못 주지만, 이 값을 8로 설정하면 
            # 2개씩 8번(총 16개)을 계산한 뒤에야 가중치를 한 번 업데이트합니다. 실질적인 배치 사이즈를 키우는 꼼수입니다.
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
            
            learning_rate=params['lr'],
            num_train_epochs=params['epochs'],
            
            # Max Length: 입력 문장의 최대 길이. 이 길이가 길어질수록 메모리 소모량이 기하급수적(제곱)으로 늘어납니다.
            max_length=params['max_len'],
            dataset_text_field="text",
            
            # [핵심] Completion-only Loss: "정답만 채점하기"
            # 프롬프트의 질문 부분(Instruction, Input)은 모델이 외울 필요가 없습니다. 
            # 오직 모델이 스스로 생성해야 할 'Response' 부분에 대해서만 벌점(Loss)을 매기도록 하는 고급 설정입니다.
            completion_only_loss=True,
            
            # 하드웨어 가속 및 메모리 방어 설정들
            bf16=True,                             # NVIDIA 최신 GPU(Ampere 이상)에서 학습 속도를 뻥튀기해 줍니다.
            optim="paged_adamw_32bit",             # OOM(메모리 터짐)이 발생할 것 같으면 시스템 RAM으로 데이터를 잠시 대피시킵니다.
            gradient_checkpointing=True,           # VRAM을 아끼기 위해 중간 계산 결과를 버리고 필요할 때 다시 계산합니다.
            
            # [핵심] 학습 안정화 스케줄러 (Airplane Landing)
            # 초반에는 학습률을 서서히 올리며 예열하고(Warmup), 후반부에는 코사인 곡선처럼 부드럽게 학습률을 낮춰서
            # 모델이 정답(최적점)을 지나쳐 튕겨 나가는 것을 방지합니다.
            lr_scheduler_type=config['training'].get('lr_scheduler_type', 'cosine'),
            warmup_ratio=config['training'].get('warmup_ratio', 0.05),

            logging_steps=100,
            logging_strategy="steps",
            report_to="none",
            disable_tqdm=False
        )

        # ---------------------------------------------------------
        # 5. 학습 실행 및 메모리 청소
        # ---------------------------------------------------------
        trainer = SFTTrainer(model=model, args=sft_config, train_dataset=train_ds)

        print(f"💡 [{stage}] 학습 시작...")
        train_result = trainer.train()

        print(f"\n✅ {stage.upper()} 완료 | 최종 평균 Loss: {train_result.training_loss:.4f}")

        # [핵심] Garbage Collection (가비지 컬렉션)
        # 한 스테이지가 끝났을 때, 이전 학습에서 사용했던 찌꺼기 메모리(캐시)를 강제로 비워줍니다.
        # 이 과정을 생략하면 다음 스테이지로 넘어가는 순간 VRAM이 누적되어 100% 확률로 터집니다.
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 6. 최종 모델 저장
    # ---------------------------------------------------------
    # 3단계를 모두 무사히 마친 훌륭한 모델의 가중치와 단어 사전(토크나이저)을 디스크에 영구 저장합니다.
    model.save_pretrained(config['model']['final_dir'])
    tokenizer.save_pretrained(config['model']['final_dir'])
    
    print(f"\n🎉 [MISSION COMPLETE] 모든 스테이지 학습이 성공적으로 종료되었습니다.")
    print(f"📁 최종 로라(LoRA) 가중치 저장 경로: {config['model']['final_dir']}")

if __name__ == "__main__":
    main()