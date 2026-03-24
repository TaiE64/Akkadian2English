# Old Assyrian Extended Corpus

A streamlined dataset for Ancient Akkadian machine translation, supporting the [Deep Past Challenge](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation) competition.

## Quick Start

```python
import pandas as pd

# Load corpus (parallel + monolingual texts)
corpus = pd.read_csv('akkadian_corpus.csv')

# Get only parallel texts (with translations)
parallel = corpus[corpus['has_translation'] == True]

# Get monolingual texts (for pre-training/retrieval)
monolingual = corpus[corpus['has_translation'] == False]

# Load dictionary
dictionary = pd.read_csv('akkadian_dictionary.csv')
```

## Dataset Files

| File | Size | Records | Description |
|------|------|---------|-------------|
| `akkadian_corpus.csv` | 6.6 MB | 7,953 | All texts (parallel + monolingual) |
| `akkadian_dictionary.csv` | 3.8 MB | 39,659 | Lexicon + Logogram dictionary |

## akkadian_corpus.csv

Contains all Akkadian texts in one unified file.

### Key Columns

| Column | Description |
|--------|-------------|
| `transliteration` | Romanized Akkadian text |
| `translation` | English translation (null for monolingual) |
| `has_translation` | `True` = parallel, `False` = monolingual |
| `data_type` | `parallel` or `monolingual` |
| `genre_label` | Text genre (letter, debt note, etc.) |
| `cdli_id` | CDLI catalog number |
| `logogram_count` | Count of Sumerograms |
| `gap_count` | Count of damaged sections |

### Data Composition

| Type | Count | Use Case |
|------|-------|----------|
| Parallel | 1,561 | Training MT models |
| Monolingual | 6,392 | Pre-training, retrieval augmentation |

### Usage Examples

```python
# Filter by genre
letters = corpus[corpus['genre_label'] == 'letter']

# Get high-quality texts (no gaps)
clean = corpus[corpus['gap_count'] == 0]

# TF-IDF similarity search
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,6))
vectors = vectorizer.fit_transform(corpus['transliteration'].str.lower())
```

## akkadian_dictionary.csv

Unified dictionary combining lexicon entries and Sumerograms.

### Key Columns

| Column | Description |
|--------|-------------|
| `entry_type` | `lexicon` or `logogram` |
| `form` | Written form in texts |
| `norm` | Normalized form |
| `known_meaning` | English meaning (for logograms) |
| `train_frequency` | Occurrence count in training data |
| `in_train_data` | Boolean: appears in training? |

### Common Logograms (Sumerograms)

| Logogram | Meaning | Frequency |
|----------|---------|-----------|
| KU.BABBAR | silver | 20,778 |
| DUMU | son of | 11,454 |
| IGI | witness/before | 7,025 |
| AN.NA | tin | 3,380 |
| URUDU | copper | 2,988 |
| DINGIR | god/divine | 2,330 |

### Usage Examples

```python
# Get most common words
common = dictionary[dictionary['train_frequency'] > 100]

# Get logograms with meanings
logograms = dictionary[dictionary['entry_type'] == 'logogram']
with_meaning = logograms[logograms['known_meaning'].notna()]

# Lookup word frequency
def get_freq(word):
    match = dictionary[dictionary['form'] == word]
    return match['train_frequency'].iloc[0] if len(match) > 0 else 0
```

## Competition Tips

1. **Retrieval-based approach works well** - Test data has ~87% similarity to training
2. **Use monolingual data** for TF-IDF/BM25 retrieval augmentation
3. **Logogram dictionary** helps understand commercial terms
4. **Word frequency** indicates which lexicon entries are most relevant

## Data Sources

- OARE (Old Assyrian Research Environment)
- CDLI (Cuneiform Digital Library Initiative)
- eBL (electronic Babylonian Library)

## License

CC-BY-4.0
