"""
QLoRA fine-tuning configuration for ByT5-XL Akkadian translation.
All hyperparameters centralized here.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class QLoRAConfig:
    # ============ PATHS ============
    model_path: str = "models/byt5-xl-akkadian"
    output_dir: str = "qlora/qlora_checkpoints"
    adapter_output_dir: str = "qlora/qlora_adapter_final"
    merged_output_dir: str = "qlora/merged_model"
    data_dir: str = "data"

    # ============ DATA ============
    # Safe training data (high quality)
    competition_train_path: str = "data/competition/train.csv"
    clean_v1_path: str = "data/01_safe_training/clean_v1/train_clean_v1.csv"
    phuc_oa_path: str = "data/01_safe_training/phucthaiv02_oa_sentences/train.parquet"
    michel_path: str = "data/01_safe_training/michel_oa_letters/train.csv"
    # Risky training data (PDF extracted, lower quality)
    phuc_pdf_path: str = "data/02_risky_training/phucthaiv02_pdf_extracted/train_oa_filtered.parquet"

    prepared_data_dir: str = "qlora/prepared_data"

    val_ratio: float = 0.05
    use_risky_data: bool = True   # Whether to include phuc_pdf (risky) data
    max_source_len: int = 512     # byte-level tokenization
    max_target_len: int = 384
    seed: int = 42

    # ============ QUANTIZATION ============
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # ============ LORA ============
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]
    )
    lora_bias: str = "none"

    # ============ TRAINING ============
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4   # effective batch size = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit"
    bf16: bool = True

    # ============ EVALUATION ============
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 50
    metric_for_best_model: str = "eval_chrf"
    generation_max_length: int = 256
    generation_num_beams: int = 1          # greedy for eval (faster)

    # ============ TASK PREFIX ============
    task_prefix: str = "translate Akkadian to English: "
