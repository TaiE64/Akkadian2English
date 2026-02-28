# Deep Past Challenge - 比赛规则与数据格式总结

> 最后更新: 2026-02-28 (已通过 Playwright 逐页验证)
> 信息来源: 比赛 Overview 页面、Discussion 置顶帖 ("A Stitch in Time Saves Nine", "Dataset Update - Mind the Gaps", "Two practical stumbling blocks", "Old Assyrian dataset updates") 以及社区讨论
>
> **数据已定稿**: Adam Anderson (Competition Host) 于 2026-02-27 确认 "There will be no further updates"，当前 train.csv 为最终版本。

---

## 0. 比赛概述

### 任务
将古亚述 (Old Assyrian) **音节转写 (syllabic transliteration)** 翻译为英文。

- **输入格式**: 音节转写 (syllabic)，例如 `KIŠIB ma-nu-ba-lúm-a-šur`
- **输出格式**: 英文翻译，例如 `Seal of Mannum-balum-Assur`
- **注意**: 比赛用的是 **syllabic** 格式，不是 normalized 格式（如 `kunuk Mannum-bālum-Aššur`）

### 评估指标
**Geometric Mean of BLEU and chrF++** (micro-average, SacreBLEU)

```
score = sqrt(BLEU × chrF++)
```

- BLEU: `sacrebleu.corpus_bleu(predictions, [references])`
- chrF++: `sacrebleu.corpus_chrf(predictions, [references], word_order=2)`
- 微平均：整个测试集一起算，不是逐句取平均

### 截止日期
2026-03-23

### 排行榜 (截至 2026-02-28, 重新评分后)

> Public LB 仅用 **34%** 测试数据，final 用另外 66%，最终排名可能大变。

| 排名 | 队伍 | 分数 |
|---|---|---|
| 1 | KE WU | 40.2 |
| 2 | Yurnero | 38.9 |
| 3 | Hrithik Reddy | 38.6 |
| 4 | vinpro | 38.2 |
| **5** | **JoJoImpact (我们)** | **38.2** |

---

## 1. 数据集概览

### ⚠️ 关键差异: 文档级 vs 句子级

> **训练数据 = 文档级 (document-level)**：train.csv 的 1,561 条是整块泥板的翻译，一条可能包含十几个句子。
> **测试数据 = 句子级 (sentence-level)**：真实测试集 ~4000 条是逐句翻译。
>
> Data 页面原文: "Note that while the training data has translations aligned at the **document** level, the test data has translations aligned at the **sentence** level."
>
> **影响**: 如果用 train.csv 微调，必须先用 `Sentences_Oare_FirstWord_LinNum.csv` 辅助拆分为句子级。第三方数据集 (clean_v1, phuc_oa 等) 已经是句子级。

### 1.1 比赛官方数据 (`data/competition/`)

| 文件 | 类型 | 说明 |
|---|---|---|
| `train.csv` | 平行语料 | 1,561 对 (文档级，含多句) |
| `test.csv` | 测试集 | 4行占位符 (真实测试集 ~4000 句) |
| `sample_submission.csv` | 模板 | 提交格式示例 |
| `Sentences_Oare_FirstWord_LinNum.csv` | **辅助工具** | 9,782 条 (帮助将 train.csv 拆分为句子级的首词/行号索引，**不是**平行语料) |
| `publications.csv` | 参考 | 554MB，学术出版物文本 |
| `published_texts.csv` | 参考 | 文本索引及元数据 |
| `OA_Lexicon_eBL.csv` | 词典 | 古亚述词汇表 |
| `eBL_Dictionary.csv` | 词典 | eBL 项目词典 |
| `bibliography.csv` | 参考 | 文献引用 |
| `resources.csv` | 参考 | 外部资源链接 |

### 1.2 当前可用训练数据 (经清洗过滤后)

| 位置 | 文件 | 去重后行数 | 说明 |
|---|---|---|---|
| `data/competition/` | `train.csv` | 1,559 | 官方金标准 (文档级) |
| `data/01_safe_training/clean_v1/` | `train_clean_v1.csv` | 2,378 | 清洗版句子级 (与 train.csv 仅 408 重叠) |
| `data/01_safe_training/phucthaiv02_oa_sentences/` | `train.parquet` | 10,545 | 高质量 OA 句子对 |
| `data/01_safe_training/michel_oa_letters/` | `train.csv` | 261 | Michel OA 书信 (列名: akkadian/english) |
| `data/02_risky_training/phucthaiv02_pdf_extracted/` | `train_oa_filtered.parquet` | 35,629 | PDF 提取经多轮过滤 (质量略低) |
| **合计 (去重后)** | | **~49,773** | 跨数据集重复仅 ~599 条 |

> **已删除的数据** (格式不匹配、质量差或非 OA):
> - `oracc_combined/`, `oracc_raw/` — 0% OA 内容
> - `enriched/` — normalized 格式 + OCR 垃圾
> - 原始未过滤的 `phuc_pdf/train.parquet` (72,653 行)

### 1.3 辅助参考数据 (`data/03_auxiliary/`)

词典、词汇表、专名表等参考材料（非训练数据）。`Sentences_Oare` 可辅助拆分 train.csv 为句子级。

---

## 2. 强制对齐项 (必须执行)

这些更改已在最终评估用的 **Test Data** 中应用。如果不照做，预测结果会因格式不匹配而扣分。

> 最新数据更新: 2026-02-25，重新评分完成: 2026-02-26

### 2.1 Gap (残缺) 标记统一

所有损坏、缺失标记全部合并为唯一的 `<gap>`。

**替换目标**:
- 单个占位符: `x`, `[x]`
- 省略号: `…` (各种长短)
- 文字标注: `(break)`, `(large break)`, `(n broken lines)`

**去重逻辑**:
- `<big_gap>` **已完全废弃** → 全部转为 `<gap>`
- 连续 gap 折叠: `<gap> <gap>`, `-<gap> <gap>`, `<big_gap> <big_gap>-` → 单个 `<gap>`
- **最终**: 不存在 `<big_gap>`，不存在连续 `<gap> <gap>`

**train.csv 当前状态**: 0 个 `<big_gap>` (已清理)

### 2.2 Determinatives (限定词) 括号替换

| 旧格式 | 新格式 |
|---|---|
| `(d)` | `{d}` |
| `(ki)` | `{ki}` |
| `(TÚG)` | `TÚG` (去括号) |

**train.csv 当前状态**: 0 个 `(d)`, 482 个 `{d}` (已转换)

**完整 determinatives 列表 (Overview 页面)**:
| 符号 | 含义 | 用法 |
|---|---|---|
| `{d}` | dingir 神 | 前置于非人类神灵 |
| `{mul}` | 星 | 前置于天体和星座 |
| `{ki}` | 地 | 后置于地名 |
| `{lu₂}` | 人 | 前置于人物和职业 |
| `{e₂}` | 房 | 前置于建筑和机构（神庙、宫殿） |
| `{uru}` | 城 | 前置于聚落（村、镇、城市） |
| `{kur}` | 国/山 | 前置于领土和山脉 |
| `{mi}` | munus 女 | 前置于女性人名 |
| `{m}` | 男 | 前置于男性人名 |
| `{geš}`/`{ĝeš}` | 木 | 前置于树木和木制品 |
| `{tug₂}` | 布 | 前置于纺织品 |
| `{dub}` | 泥板 | 前置于泥板/文书/法律记录 |
| `{id₂}` | 河 | 前置于河流和运河 |
| `{mušen}` | 鸟 | 前置于鸟类 |
| `{na₄}` | 石 | 前置于石材 |
| `{kuš}` | 皮 | 前置于动物皮革 |
| `{u₂}` | 草 | 前置于植物 |

### 2.3 浮点数截断

过长浮点数截断到**小数点后第 4 位** (不四舍五入):
- `1.3333300000000001` → `1.3333`
- `2.6666600000000003` → `2.6666`

---

## 3. 现代抄写符号处理 (Transliteration & Translation 通用)

> 来源: Overview 页面 "Formatting Suggestions for Transliterations and Translations"

### 3.0a 需移除的现代符号 (Remove)
- `!` — 学者对困难读法的确认标记
- `?` — 学者对困难读法的疑问标记
- `/` — 行分隔符
- `:` 或 `.` — 词分隔符
- `< >` — 抄写插入标记 (移除括号但**保留内部文本**)
- `˹ ˺` — 部分损坏符号标记 (从 transliteration 移除)
- `[ ]` — 完全损坏标记 (文档级移除括号，如 `[KÙ.BABBAR]` → `KÙ.BABBAR`)

### 3.0b 需替换的符号 (Replace)
- `[x]` → `<gap>`
- `…` → `<gap>` (注意: Overview 原文写 `<big_gap>`，但已被 Discussion 更新废弃)
- `[… …]` → `<gap>`
- 上标限定词 → 花括号: <sup>ki</sup> → `{ki}`
- 下标数字 → 常规数字: il₅ → il5

### 3.0c 字符格式对照表 (Overview 页面)

| 字符 | CDLI | ORACC | Unicode |
|---|---|---|---|
| á | a2 | a₂ | - |
| à | a3 | a₃ | - |
| é | e2 | e₂ | - |
| è | e3 | e₃ | - |
| š | sz | š | U+161 |
| ṣ | s, | ṣ | U+1E63 |
| ṭ | t, | ṭ | U+1E6D |
| ḫ | h | h | U+1E2B |
| Ḫ | H | H | U+1E2A |
| 0-9 | 0-9 | ₀-₉ | U+2080-U+2089 |

> 比赛数据使用 ORACC 风格 (Unicode 下标)，需转为 CDLI 风格 (普通数字)。

---

## 4. Transliterations (输入阿卡德语) 额外处理

### 4.1 字符标准化

> Overview 页面原文: "These rows of Ḫ ḫ are here to indicate that **training data (and publication data) has Ḫ ḫ but the test data has only H h**."

| 旧 | 新 | train.csv 状态 |
|---|---|---|
| `Ḫ` | `H` | 1,281 行含 `ḫ` (需处理) |
| `ḫ` | `h` | 同上 |
| `KÙ.B.` | `KÙ.BABBAR` | - |

### 4.2 下标数字转常规数字

Unicode 下标数字转为阿拉伯数字:
- `₀`→`0`, `₁`→`1`, `₂`→`2` ... `₉`→`9`

> 注意: "A Stitch in Time" 帖子标记此项为 **"Optional changes"**，但选手确认最新 train.csv 中下标数字**仍未被官方清理**（如 `il₅`, `tur₄` 等），需自行处理。

**train.csv 当前状态**: 1,173 行含下标数字 (需自行处理)

---

## 5. Translations (目标英文) 额外处理

### 5.1 需移除的杂音

- **语法标签**: `fem.`, `sing.`, `pl.`, `plural`
- **不确定性标记**: `(?)`
- **零碎符号**: `..`, 无意义的 `?`, 孤立的 `x`/`xx`, `<< >>`, `< >`
- **多选翻译**: `you / she brought` → 只保留一个 → `you brought`

### 5.2 必须保留的符号

- 双引号: `" "`
- 单引号/撇号: `'`
- 有语义的 `?` 和 `!`

### 5.3 特定词汇与单位替换

**人名占位符**:
- `PN` → `<gap>` (仅替换字面量 `PN` token，不是所有人名。Adam Anderson: "That's a literal PN token, there are some of these in Veenhof's translations from AKT 8.")

**物品前缀**:
- `-gold` → `pašallum gold`
- `-tax` → `šadduātum tax`
- `textiles` → `kutānum textiles`

**重量单位 (Shekel → Grains)**:
| 原始 | 替换 |
|---|---|
| `1 / 12 (shekel)` | `⅔ shekel 15 grains` |
| `5 / 12 shekel` | `15 grains` |
| `5 11 / 12 shekels` | `6 shekels less 15 grains` |
| `7 / 12 shekel` | `½ shekel 15 grains` |

### 5.4 月份罗马数字转阿拉伯数字

- Month I → Month 1, Month II → Month 2 ... Month XII → Month 12

---

## 6. 小数与 Unicode 分数转换

> **关键确认 (Adam Anderson, 2026-02-25)**: "I already shortened the floats, so that the conversion to fractions will be easier for you, if you choose to do so. As seen in the example at the end, **the test contains only unicode fractions (no decimals at all)**."

无论 Translation 还是 Transliteration，固定小数**必须**转为 Unicode 分数符（测试集中只有 Unicode 分数，没有小数）:

| 小数 | 分数符 |
|---|---|
| `0.5` | `½` |
| `0.25` | `¼` |
| `0.75` | `¾` |
| `0.3333` | `⅓` |
| `0.6666` | `⅔` |
| `0.1666` | `⅙` |
| `0.8333` | `⅚` |
| `0.625` | `⅝` |

> **train.csv 状态**: Source 有 0 个 Unicode 分数 + 276 个长小数; Target 有 19 个 Unicode 分数 + 195 个长小数。训练集中的小数**需要自行转换为 Unicode 分数**以匹配测试集格式。

---

## 7. 构建额外训练数据的建议工作流

> 来源: Data 选项卡 "Suggested Workflow for Building Additional Training Data"

`publications.csv` 包含近 900 个 PDF 的 OCR 输出，从中提取翻译是关键的第一步。步骤如下:

1. **定位每篇文本及其翻译**: 使用文档标识符 (ID、别名或博物馆编号) 将 transliteration 与 OCR 输出中对应的 translation 匹配
2. **统一翻译为英文**: 原始翻译可能是多种语言 (英语、法语、德语、土耳其语)，需全部转换为英文
3. **创建句子级对齐**: 将阿卡德语 transliteration 和对应的英文 translation 拆分为句子，进行逐句对齐。句子级映射是训练和评估 MT 模型最有用的格式

辅助文件:
- `Sentences_Oare_FirstWord_LinNum.csv` — 帮助在 `train.csv` 中进行句子级对齐，标注了每个句子的首词及其在泥板上的位置
- `published_texts.csv` — 文本索引，可通过 `OARE_Text_ID` 定位到 PDF 原文
- `resources.csv` — 可用于获取额外数据的外部资源列表

辅助数据集 (官方发布):
- **`deeppast/old-assyrian-grammars-and-other-resources`** (Kaggle, 1.18GB) — 包含:
  - `onomasticon.csv` — 古亚述专名表（人名/地名及拼写变体），可用于后处理修正
  - `secondary_sources.csv` — 二手文献 OCR
  - 语法书 PDF (Kouwenberg 2017/2019)、词典 (CAD)、文本版本 (Larsen 2002, ICK 4)

参考书目链接:
- https://cdli.earth/publications
- https://cdli.ox.ac.uk/wiki/abbreviations_for_assyriology

---

## 8. 已知问题与风险

### 8.1 训练数据截断问题
最新版训练数据 (V3) 中约 10% 的长文本被异常截断（如 592→138 字符）。可通过 `published_texts.csv` 的 `OARE_Text_ID` 去 PDF 找回原文。

### 8.2 训练集分数/小数格式不一致
官方训练集中分数处理不统一：同一数据集内既有 Unicode 分数符 (`⅓`) 又有长小数 (`0.3333`)。但 Adam Anderson 已确认**测试集中只有 Unicode 分数**，因此训练集中的小数应全部转换为分数。官方已将长浮点截断到 4 位以便于转换。

### 8.3 排行榜评分延迟
官方确认 Leaderboard 分数更新存在 Bug，可能显示不准确。2026-02-26 已完成基于最新数据的重新评分。

### 8.4 月份转换表有误
官方帖子中的月份名称→数字对照表存在已知错误，选手在评论区指出了修正。建议直接使用罗马数字 I-XII → 1-12 的映射，不依赖官方的月份名称表。

### 8.5 第三方数据格式风险
`train_enriched.csv` 中的 Old Assyrian 数据使用 **normalized** 转写格式，与比赛要求的 **syllabic** 格式不同。直接用于微调可能对模型有害。

### 8.6 Overview 页面与 Discussion 更新的矛盾
Overview 页面仍写着 "one for a small break `<gap>` and the other for more than one sign `<big_gap>`"，但 Discussion 更新已将 `<big_gap>` 完全废弃。**以 Discussion 更新为准**。

### 8.7 `published_texts.csv` 潜在数据源

- `note` 字段: "Notes made by specialists for **commentary or translations**" — 可能包含专家翻译，未被充分探索
- 有两列 transliteration: `transliteration_orig` (OARE 原始) 和 `transliteration` (按格式建议清洗后)，可直接用清洗版
- `AICC_translation` 字段: 链接到在线机器翻译，但官方标注 "most of these translations are very poor quality"

### 8.8 "A Stitch in Time" 帖子不包含所有变更
选手确认: "this update does not include all the changes made in the previous update"。例如 Unicode 下标数字在最新 train.csv 中仍然存在（如 `tur₄`, `il₅`），需自行参照 Overview 页面的字符对照表处理。
