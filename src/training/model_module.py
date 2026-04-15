"""
모델 로드 모듈.

역할:
  - Qwen 토크나이저 + 모델 로드 (finetune_lora.py 모델 로딩 흡수)
  - BitsAndBytes 4-bit 양자화 (선택)
  - LoRA 설정 적용 (PEFT)
  - LoRA 어댑터 병합 + 저장 (merge_adapter.py 흡수)
  - train_config.yaml dict 연동

기존 training/finetune_lora.py (모델 부분) + training/merge_adapter.py 로직 통합.
"""

from __future__ import annotations

from typing import Optional


# ─── 토크나이저 ──────────────────────────────────────────────────────────────
def load_tokenizer(model_name: str, trust_remote_code: bool = True):
    """AutoTokenizer 로드. pad_token 미설정 시 eos_token으로 대체."""
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ─── 베이스 모델 로드 ────────────────────────────────────────────────────────
def load_base_model(
    model_name: str,
    torch_dtype: str = "float16",
    device_map: str = "auto",
    use_quantization: bool = True,
    trust_remote_code: bool = True,
):
    """
    베이스 모델 로드.

    use_quantization=True 이면 BitsAndBytes 4-bit 양자화 적용.
    VRAM 부족 환경(Colab T4)에서 권장.
    """
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM  # type: ignore

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.float16)

    bnb_config = None
    if use_quantization:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            print("[model_module] bitsandbytes 없음 — 양자화 비활성화")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype if bnb_config is None else None,
        device_map=device_map,
        quantization_config=bnb_config,
        trust_remote_code=trust_remote_code,
    )
    return model


# ─── LoRA 적용 ───────────────────────────────────────────────────────────────
def apply_lora(
    model,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
    bias: str = "none",
):
    """PEFT LoRA 설정 적용 후 학습 파라미터 수 출력."""
    from peft import LoraConfig, get_peft_model, TaskType  # type: ignore

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─── YAML config 통합 로드 ──────────────────────────────────────────────────
def load_model_and_tokenizer_from_config(cfg: dict):
    """
    train_config.yaml dict → (model, tokenizer).

    사용 예:
        import yaml
        cfg = yaml.safe_load(open("configs/train_config.yaml"))
        model, tokenizer = load_model_and_tokenizer_from_config(cfg)
    """
    m_cfg = cfg.get("model", {})
    q_cfg = cfg.get("quantization", {})
    l_cfg = cfg.get("lora", {})

    tokenizer = load_tokenizer(
        m_cfg["name"],
        trust_remote_code=m_cfg.get("trust_remote_code", True),
    )
    model = load_base_model(
        model_name=m_cfg["name"],
        torch_dtype=m_cfg.get("torch_dtype", "float16"),
        device_map=m_cfg.get("device_map", "auto"),
        use_quantization=q_cfg.get("enabled", True),
        trust_remote_code=m_cfg.get("trust_remote_code", True),
    )
    model = apply_lora(
        model,
        r=l_cfg.get("r", 16),
        lora_alpha=l_cfg.get("lora_alpha", 32),
        lora_dropout=l_cfg.get("lora_dropout", 0.05),
        target_modules=l_cfg.get("target_modules", None),
        bias=l_cfg.get("bias", "none"),
    )
    return model, tokenizer


# ─── 어댑터 병합 (merge_adapter.py 흡수) ────────────────────────────────────
def merge_and_save(
    base_model: str,
    adapter_path: str,
    output_dir: str,
    torch_dtype: str = "float16",
    trust_remote_code: bool = True,
) -> None:
    """
    LoRA 어댑터를 베이스 모델에 병합 후 단일 가중치로 저장.

    학습 완료 후 서빙 시 사용. CPU 로드 권장 (VRAM 절약).
    """
    import torch  # type: ignore
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    from peft import PeftModel  # type: ignore

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.float16)

    print(f"[model_module] 베이스 모델 로드: {base_model}")
    tokenizer = load_tokenizer(base_model, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )

    print(f"[model_module] 어댑터 로드: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    import os
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[model_module] 병합 완료. 저장 경로: {output_dir}")
