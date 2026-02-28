# QLoRA Fine-Tuning Plan for ByT5-XL Akkadian Translation

## Context
当前 ByT5-XL 模型 (Venom2212) 在 Kaggle 得分 38.1（Top1: 39.5）。消融实验表明后处理无法提升分数，翻译质量问题（数字格式、语法风格）需要从模型层面解决。RTX 5080 16GB 无法全参数微调 4B 模型，因此使用 QLoRA（4-bit 量化 + LoRA 适配器）进行微调。

---

## Step 1: 环境准备

安装 bitsandbytes（4-bit 量化所需）:
```bash
conda activate kaggle_deep_past
pip install bitsandbytes==0.49.2
```

验证脚本测试 NF4 量化是否正常工作。

**兼容性风险**: PyTorch nightly 2.11.0.dev + CUDA 12.8 可能与 bnb 不兼容
**降级方案**: 8-bit 量化 → fp16 LoRA（无量化，VRAM 约 13.5GB 仍可行）

---

## Step 2: 数据准备

### 数据源
| 来源 | 文件 | 行数 |
|---|---|---|
| Enriched (竞赛+出版+Michel) | `data/manwithacat_enriched/train_enriched.csv` | 3,032 |
| ORACC 额外数据 | `data/manwithacat_combined/combined_akkadian_v2_oracc.csv` | ~4,196 (去除竞赛重复) |

### 处理流程
1. 合并去重 → ~6,500 条
2. 古亚述 (OA) 数据 3x 上采样 → ~13,000-15,000 条
3. 按 `data_format_rules.md` 对 source 和 target 做格式对齐
4. 95/5 划分 train/val（~12,500 train, ~650 val）

### 预处理函数
- **Source**: 复用 notebook 中 `OptimizedPreprocessor.preprocess_input_text()` 逻辑
- **Target**: 复用 `VectorizedPostprocessor` 逻辑（gap 标准化、月份转换、分数转换等）
- 前缀: `"translate Akkadian to English: "`

创建文件: `qlora/prepare_data.py`

---

## Step 3: QLoRA 配置

### 量化配置
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```
基座模型 VRAM: ~2.0 GB

### LoRA 配置
```python
LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],  # 注意力 + FFN 投影
    bias="none",
)
```
- 可训练参数: ~40M (1.07%)（含 FFN 层后增加）
- 覆盖: encoder 36层 + decoder 12层 + cross-attention 12层 的注意力投影 + FFN 投影
- FFN 层 (`wi_0`, `wi_1`, `wo`) 对模型理解语境和输出格式提升显著
- Adapter 大小: ~80-160 MB

### 训练超参数
```python
Seq2SeqTrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,      # 有效 batch size = 8
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.06,
    optim="paged_adamw_8bit",
    bf16=True,
    gradient_checkpointing=True,
    eval_strategy="steps", eval_steps=500,
    predict_with_generate=True,
)
```

### VRAM 预算
| 组件 | 估计 VRAM |
|---|---|
| 基座模型 (NF4) | ~2.0 GB |
| LoRA 参数 + 优化器 + 梯度 | ~0.15 GB |
| 激活值 (梯度检查点, bs=2) | ~4-6 GB |
| CUDA 开销 | ~2-3 GB |
| **总计** | **~8-11 GB / 17.1 GB** |

---

## Step 4: 训练脚本

创建文件: `qlora/train_qlora.py`

核心流程:
1. 加载 tokenizer + 4-bit 量化模型
2. `prepare_model_for_kbit_training()` + `get_peft_model()`
3. 创建 Dataset（byte-level tokenization, max_source=512, max_target=256）
   - **重要**: tokenize target 时将 `pad_token_id` 替换为 `-100`，避免模型学习预测 `<pad>`
   - 示例: `labels[labels == tokenizer.pad_token_id] = -100`
4. Seq2SeqTrainer 训练，使用 **chrF++** 作为早停指标（比 BLEU 更稳定，复用 notebook 中 `_chrfpp_sentence_score` 函数）
5. 保存 adapter 到 `qlora/qlora_adapter_final/`

---

## Step 5: 超参数调优（分阶段）

| 阶段 | 实验 | 预计时间 |
|---|---|---|
| Phase 1: 验证 | 1 epoch, lr=2e-4, r=16 | ~40 min |
| Phase 2: LR 搜索 | {5e-5, 1e-4, 2e-4} × 3 epochs | ~6 h |
| Phase 3: Rank 搜索 | {8, 16, 32} × 3 epochs（用最佳 LR） | ~6 h |
| Phase 4: 消融 | ±ORACC, 3 vs 5 epochs | ~4 h |
| **总计** | | **~16-20 h GPU** |

---

## Step 6: 本地验证

1. **训练前基线**: fp16 模型在 val set 上的 GeoMean
2. **训练后评估**: 合并 adapter 后 fp16 模型在 val set 上的 GeoMean
3. **只有当新 GeoMean > 基线时才提交 Kaggle**

创建文件: `qlora/eval_qlora.py`

---

## Step 7: Kaggle 部署

### 方案 A (推荐): 本地合并后上传
```python
# merge_adapter.py
base_model = AutoModelForSeq2SeqLM.from_pretrained("models/byt5-xl-akkadian", torch_dtype=torch.bfloat16, device_map="cpu")  # bfloat16 匹配训练精度
model = PeftModel.from_pretrained(base_model, "./qlora_adapter_final")
model = model.merge_and_unload()
model.save_pretrained("./merged_model", safe_serialization=True)
```
- 合并后模型 ~8.6 GB，替换原模型上传到 Kaggle 数据集
- Notebook 只需改 model_path，无需安装额外库

### 方案 B (备选): 只上传 Adapter
- Adapter ~35-71 MB + peft wheel 上传
- Notebook 增加 `pip install peft` + `PeftModel.from_pretrained` + `merge_and_unload()`
- 无需 bitsandbytes（合并在 fp16 下完成）

创建文件: `qlora/merge_adapter.py`

---

## Step 8: 风险评估

| 风险 | 级别 | 缓解措施 |
|---|---|---|
| 灾难性遗忘 | **高** | 保守 LR(5e-5起)、低 rank(16)、早停、对比基线 |
| bitsandbytes 兼容性 | **中** | 降级到 8-bit → fp16 LoRA |
| ORACC 数据干扰 | **中** | OA 3x 上采样、消融测试 |
| 合并精度损失 | **低** | bfloat16 训练匹配推理精度 |
| 过拟合 | **低** | Dropout 0.05、weight decay 0.01、早停 |

---

## 文件结构
```
qlora/
  prepare_data.py          # 数据加载、合并、预处理
  train_qlora.py           # 主训练脚本
  eval_qlora.py            # 评估脚本
  merge_adapter.py         # 合并 adapter 到基座模型
  config.py                # 超参数集中管理
```

## 关键依赖文件
- `models/byt5-xl-akkadian/` - 基座模型 (8.6GB)
- `submission/mbr_v2_push/deep-pasta-mbr-v2.ipynb` - 预处理/后处理逻辑复用
- `data_format_rules.md` - 格式对齐规则
- `data/manwithacat_enriched/train_enriched.csv` - 主要训练数据

## 验证方式
1. 安装 bitsandbytes 后运行验证脚本确认 4-bit 量化正常
2. `prepare_data.py` 输出数据统计和样本检查
3. Phase 1 训练后检查 loss 下降和样本翻译
4. 最终 val GeoMean > 基线后合并并提交 Kaggle
