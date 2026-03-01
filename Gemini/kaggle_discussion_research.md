# Deep Past Initiative Machine Translation Challenge

## 核心数据格式与预处理规则 (来自 Pinned Posts)
这是一份从官方论坛置顶帖中提取的重要数据格式及比赛规则总结：

### 1. 间隙 (Gaps) 统一化
*   所有类型的文本破损标记（例如 `x`, `[x]`, `...`, `(break)`, `(large break)`, `(n broken lines)` 以及较早使用的 `<big_gap>`）已经被统一替换为单一标签： **`<gap>`**。
*   这项修改已经完全落实到 `train.csv`, `published.csv` 以及隐藏的测试集 `test.csv` 中。
*   **注意:** 对于与单词相连的间隙（例如 `<gap>-A-šùr`），应该保持原样，无需拆分。

### 2. 阿卡德语音译字母 (Akkadian Transliteration)
*   比赛使用 **带有变音符号的扩展字母表** 进行阿卡德语的音译。
*   **关键策略:** 绝对 **不要** 将这些带有变音符号的字母简化为基础 ASCII 字符。相反，建议将数据中任何使用 ASCII 替代的字符（例如用 `sz` 代替 `š`，用 `s,` 代替 `ṣ`，用 `t,` 代替 `ṭ`）转换回它们正确的变音符号形式，以符合比赛标准。

### 3. 命名实体 (Named Entities) 挑战
*   个人名 (PN)、地名 (GN) 和神名 (DN) 是翻译错误的主要来源。它们的音译往往不一致，并遵循旧的正字法规则。
*   **建议策略:** 在补充的 ["Old Assyrian grammars and other resources"](https://www.kaggle.com/datasets/deeppast/old-assyrian-grammars-and-other-resources/data) 数据集中提供了一个 **`onomasticon.csv`**（人名地名专名表）。强烈建议使用这个词表作为约束层或在生成后进行后处理修复。

### 4. 比赛更新相关
*   数据集已经过几次更新，修复了转录错误并提高了 `published.csv` 中的 `text_case_id` 覆盖率。
*   排行榜也已经重新评分，以反映使用 `<gap>` 标签的统一规则和数据修复。

### 5. 推荐工作流与资源
*   比赛提供官方 Discord 频道供实时交流。
*   强烈建议查看 Data 选项卡中的 "Suggested Workflow for Building Additional Training Data"（构建额外训练数据的建议工作流），以获取高级的数据增强策略。

## 评估指标与时间表 (来自 Overview Tab)

### 评估指标 (Evaluation Metric)
*   **指标:** 分数基于 **BLEU** 和 **chrF++** 的 **几何平均值 (Geometric Mean)**。
*   **计算方式:** Micro-average，即在整个语料库上汇总各项分数的充分统计量后再进行计算。 内部打分使用的是 `sacreBLEU` 库。
*   **目标:** 将测试集中每个阿卡德语音译**句子**翻译成英语，以最大化这个联合指标。

### 比赛时间表
*   **报名截止日期 (Entry Deadline):** 2026年3月16日 11:59 PM UTC (必须在此日期前接受比赛规则)。
*   **最终提交截止日期 (Final Submission Deadline):** 2026年3月23日 11:59 PM UTC。

## 提交格式要求 (来自 Data Tab)
*   **文件名:** 必须命名为 **`submission.csv`**。
*   **列:**
    *   **`id`**: 对应 `test.csv` 中的句子唯一标识符。
    *   **`translation`**: 你生成的英文翻译文本。
*   **结构约束:** 每个翻译条目必须**严格构成一个单句 (single sentence)**。注意：训练数据 (`train.csv`) 是文档级别的翻译对齐，而**测试数据 (`test.csv`) 是句子级别的翻译对齐**。
    
### 翻译处理规则 (可选翻译)
*   在测试集中，同一个输入 `id` 可能对应**多个正确的参考翻译** (反映了不同学者的解读)。训练集中官方为方便起见挑选了一个主要的翻译版本。补充数据 (`published_texts.csv`) 中包含大量可能存在冲突翻译的未经处理文本。
*   学者常常在括号 `( )` 中包含对于破损符号的"可选"或尝试性翻译。格式建议在预处理阶段妥善处理这些括号内的内容。
*   **限定词 (Determinatives):** 例如 `{ki}`, `{d}` 保留在花括号内，处理时需要考虑如何保留或过滤这些语义分类符。
