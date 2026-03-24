---
name: Extract OA Transliterations and English Translations from Images
description: Extract Old Akkadian (OA) transliterations and their English translations from academic publication images, normalize formatting, join lines into sentences, and append to a CSV file.
---

# Extract OA Transliterations and English Translations

## Overview

This skill extracts Old Akkadian (OA) transliterations and their corresponding English translations from images of academic publications (e.g., AKT series). The output is a CSV file with two columns: `oa` and `english`.

## Input

- One or more images containing two-column layouts:
  - **Left column**: OA transliteration lines
  - **Right column**: English translation lines

## Output

- CSV file with header `oa,english`, where each row is a complete sentence pair.

## Extraction Rules

### 8. Replace Half-Brackets with Square Brackets

The half-bracket characters `⌈` and `⌉` should be replaced with `[` and `]`.

### 9. Replace Gap Brackets with `<gap>` in OA

In OA transliterations, brackets containing **only `x`, spaces, or `+`** represent illegible/missing signs. Replace them with `<gap>`:

| Original | Normalized |
|----------|-----------|
| `[x x x]` | `<gap>` |
| `[x+]5` | `<gap>5` |
| `[ x]` | `<gap>` |
| `[ ]` | `<gap>` |
| `[x x]-A-šur` | `<gap>-A-šur` |

> [!CAUTION]
> Do NOT replace brackets containing **readable text** like `[DUM]U`, `[a-n]a`, `[42]` — these are restorations of damaged signs.

### 10. Remove Square Brackets and Contents from English Translations

In the **English translation** column, if there are square brackets `[...]`, remove the brackets **and** everything inside them. This applies to interpolated/editorial additions in the translation.

| Original English | After Removal |
|-----------------|--------------|
| `bound on [the one of them] who` | `bound on who` |
| `within [x] weeks` | `within weeks` |
| `he will pay [the silver]` | `he will pay` |

> [!CAUTION]
> This rule applies ONLY to the **English** column. Square brackets in the **OA** column must be preserved — they indicate restored/damaged text.

### 1. Normalize Superscripts and Subscripts

There are **two different treatments** depending on the type:

#### 1a. Subscript Numbers → Convert to Normal Integers

Unicode subscript digits (`₀₁₂₃₄₅₆₇₈₉`) in OA transliterations indicate **sign variants** and must be **converted to normal integers**:

| Original | Normalized |
|----------|-----------|
| `ŠU.NIGIN₂` | `ŠU.NIGIN2` |
| `bi₄` | `bi4` |
| `il₅` | `il5` |
| `en₆` | `en6` |
| `Puzur₄` | `Puzur4` |

#### 1b. Other Superscripts/Subscripts → Remove Entirely

Non-numeric superscripts and subscripts (`d`, `ki`, `?`, `*`) must be **completely removed**:

| Original | After Removal | Removed Characters |
|----------|--------------|-------------------|
| `ᵈIŠKUR` | `IŠKUR` | `d` (superscript determinative) |
| `a-limᵏⁱ` | `a-lim` | `ki` (superscript) |
| `[TÚG ša?]` | `[TÚG ša]` | `?` (superscript) |
| `ku-ta-n[i⁷ /T[A²]]*` | `ku-ta-n[i7 /T[A2]]` | `*` removed; `⁷`,`²` converted to `7`,`2` |

### 2. Replace Dense Dots with `<gap>`

Dense dots / ellipses (`......`, `........`, `..........`) in the **English translation** represent **missing or damaged text**, NOT sentence-ending periods. Replace each cluster of dense dots with `<gap>`.

| Original English | After Replacement |
|-----------------|------------------|
| `which ..... ....,` | `which <gap>,` |
| `the agent ........` | `the agent <gap>` |
| `..............................` | `<gap>` |

> [!CAUTION]
> Do NOT treat dense dots as sentence boundaries. Only single isolated periods (`.`) mark sentence endings.

### 3. Sentence Joining by Period (`.`)

Each line in the image is typically a **fragment** of a sentence. Use the **period (`.`)** in the **English translation** column as the sentence boundary:

- Scan the English translations line by line.
- When a line ends with `.` (period), that marks the **end of the current sentence**.
- **Concatenate** all OA lines from the sentence start to this line (space-separated) → one `oa` cell.
- **Concatenate** all English lines from the sentence start to this line (space-separated) → one `english` cell.
- Lines ending with `,` (comma) or no punctuation are **mid-sentence** and should be joined with the next line(s).

### 4. Split Overly Long Sentences

If a joined sentence is **too long** (roughly more than 6-8 original lines), split it at a natural **English comma (`,`)** boundary to keep entries manageable. Both OA and English should be split at the corresponding position.

- Choose a comma that falls at a **logical clause boundary** (not in the middle of a list or name).
- Each resulting segment should be **coherent on its own** — avoid splitting mid-clause.
- Aim for roughly 3–6 original lines per segment.



In OA transliterations, if there is a **space before `:`**, remove the space. The space after `:` is kept.

| Original | Normalized |
|----------|-----------|
| `ba-áb : né-be-ri-šu` | `ba-áb: né-be-ri-šu` |
| `um-ma Hi-na-a-ma : a-na` | `um-ma Hi-na-a-ma: a-na` |

### 6. Remove Line-Break Markers (`/`)

The `/` character in OA transliterations indicates a **line break on the original tablet** where a word is split across lines. Remove all such `/` characters.

| Original | Normalized |
|----------|-----------|
| `ṣa-ru-pá-/am` | `ṣa-ru-pá-am` |
| `i-tí-/iq` | `i-tí-iq` |
| `ku-un-kà-/ma` | `ku-un-kà-ma` |
| `ITU./KAM` | `ITU.KAM` |

> [!CAUTION]
> Do NOT remove `/` in fractions like `1/3`, `5/6` — these are numeric values, not line breaks.

#### Example

Given these image lines:

```
OA                          English
1½ ma-na 2 [GÍN]           For 1 mina 32 shekels
i-ṭup-pi-im                he has been registered
lá-pi-it                   on the tablet.          ← period here = end of sentence
1 5/6 ma-na                1 5/6 mina
a-qá-ti-a t[a-dí]          you deposited as my share.  ← period = end of sentence
```

Output:

```csv
oa,english
"1½ ma-na 2 [GÍN] i-ṭup-pi-im lá-pi-it","For 1 mina 32 shekels he has been registered on the tablet."
"1 5/6 ma-na a-qá-ti-a t[a-dí]","1 5/6 mina you deposited as my share."
```

### 3. CSV Formatting

- Both `oa` and `english` values must be **double-quoted**.
- Internal double quotes should be escaped as `""`.
- Use UTF-8 encoding to preserve diacritics (e.g., `á`, `ṭ`, `š`, `ù`).

## Step-by-Step Workflow

1. **Read the image** and identify the two-column layout (OA left, English right).
2. **Transcribe** each line pair (OA + English) exactly as shown.
3. **Normalize** superscripts/subscripts to plain text.
4. **Scan English lines** for periods (`.`) to determine sentence boundaries.
5. **Join** all OA fragments and English fragments within each sentence boundary using spaces.
6. **Check the target CSV file** — read it to understand existing format and content.
7. **Append** the new sentence pairs to the CSV file, matching the existing format.

## Common OA Conventions to Recognize

| Pattern | Meaning |
|---------|---------|
| `[...]` | Restored/broken text |
| `(...)` | Scribal/editorial note |
| `<<...>>` | Erased text |
| `<...>` | Omitted by scribe |
| `IGI` | "In the presence of" / "Witnessed by" |
| `KÙ.B.` / `KÙ.GI` | Silver / Gold |
| `ŠU.NIGIN` | Total/sum |
| `É` | House/office |
| `TÚG` | Textile |
| `GÍN` | Shekel |
| `ma-na` | Mina (unit of weight) |
| `DUMU` | Son of |

### 11. Skip Seal Notations

Lines containing **seal notations** such as `seal A`, `seal B`, `seal C`, `seal D`, `seal C (upside down)`, etc. are **physical annotations** about seal impressions on the tablet, not part of the transliteration or translation. **Ignore these lines entirely** — do not include them in the OA or English output.

Also skip lines that are **seal catalogue references** in the format `CS XXXX` (e.g., `CS 1081`, `CS 1083`). These are modern catalogue numbers for seal impressions and are not part of the text.

| Skip These | Reason |
|-----------|--------|
| `seal A` | Physical seal annotation |
| `seal B` | Physical seal annotation |
| `seal C` | Physical seal annotation |
| `seal D` | Physical seal annotation |
| `seal C (upside down)` | Physical seal annotation |
| `SEAL A` | Physical seal annotation (uppercase variant) |
| `CS 1081` | Seal catalogue reference |
| `CS 1083` | Seal catalogue reference |
| `CS XXXX` (any number) | Seal catalogue reference |

### 12. Maximum Row Length — Split Long Rows

After joining lines into sentences, check if any single row is **too long**. A row is too long if the **English translation** contains more than **~4 complete sentences** (periods) or the row spans more than **~8 original image lines**.

If a row is too long, **split it into multiple rows** at natural **sentence boundaries** (periods `.` in the English column), with the OA column split at the corresponding position.

**Splitting rules:**
- Split at a period (`.`) in the English translation that falls at a natural topic/clause boundary.
- Each resulting row should contain **2–4 English sentences** and be coherent on its own.
- Both OA and English must be split at the **same logical point**.

> [!CAUTION]
> Do NOT split in the middle of a sentence. Always split at a period boundary.
