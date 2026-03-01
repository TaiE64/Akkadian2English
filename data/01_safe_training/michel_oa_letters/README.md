# Michel Old Assyrian Merchant Letters

Parallel corpus of **264 Old Assyrian merchant letters** with Akkadian transliteration and English translations, extracted from Cécile Michel's scholarly publication.

## Source

**Cécile Michel, "Correspondance des marchands de Kanish au début du IIe millénaire avant J.-C."** (LAPO 19, Paris: Éditions du Cerf, 2001)

This landmark publication contains French translations of Old Assyrian merchant correspondence from the ancient trading colony of Kaniš (modern Kültepe, Turkey), dating to approximately 1950-1750 BCE.

## Relevance to Deep Past Competition

This dataset provides **additional parallel training data** for the [Deep Past Initiative Machine Translation Competition](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation). The competition's training data contains 1,561 Old Assyrian texts; this dataset adds 264 unique texts from the same period and genre.

## Data Structure

### Files

| File | Description |
|------|-------------|
| `train.csv` | 264 parallel Akkadian-English pairs |
| `statistics.json` | Dataset statistics and metadata |

### Columns in train.csv

| Column | Description |
|--------|-------------|
| `id` | OARE database identifier |
| `akkadian` | Akkadian text in scholarly transliteration |
| `english` | English translation |
| `source_ref` | Publication reference (e.g., "TC 1, 142") |
| `letter_num` | Letter number in Michel's publication |
| `genre` | Text genre classification |

## Transliteration Conventions

The Akkadian column uses standard Assyriological transliteration:

- **Logograms**: UPPERCASE (e.g., `KÙ.BABBAR` = silver)
- **Syllabic signs**: lowercase with hyphens (e.g., `um-ma` = "thus")
- **Determinatives**: Superscript markers like `{d}` for divine names
- **Damage markers**: `<gap>`, `<big_gap>` for lacunae, `[...]` for breaks
- **Special characters**: ṣ, ṭ, š for emphatic/sibilant consonants

## Translation Pipeline

1. **OCR Extraction**: French translations extracted from digitized publication pages
2. **Cross-reference**: Matched with Akkadian transliterations via OARE database aliases
3. **Machine Translation**: French → English via Google Translate API
4. **Quality Control**: 264 of 286 letters (92%) successfully matched with transliterations

## Content Overview

These letters document the activities of Old Assyrian merchants who operated a long-distance trade network between Aššur (northern Iraq) and Anatolia. Common themes include:

- **Trade goods**: Tin, textiles, silver, copper
- **Financial matters**: Debts, interest rates, contracts
- **Legal disputes**: Witnesses, testimony, colony regulations
- **Personal correspondence**: Family matters, travel arrangements

## Sample Entry

```
Akkadian: um-ma wa-ak-lúm-ma a-na kà-ri-im kà-ni-iš qí-bi-ma...
English: Thus (speaks) the waklum: say to the kārum of Kaniš...
```

## Citation

If you use this dataset, please cite the original scholarly work:

> Michel, Cécile. *Correspondance des marchands de Kanish au début du IIe millénaire avant J.-C.* Littératures anciennes du Proche-Orient 19. Paris: Éditions du Cerf, 2001.

## Limitations

- English translations are machine-translated from French (not direct Akkadian→English)
- Some texts have significant lacunae marked with `<gap>` or `<big_gap>`
- OCR artifacts may occasionally appear in the text
- Translation quality varies based on text preservation

## Related Datasets

- [ORACC Akkadian-English Parallel Corpus](https://www.kaggle.com/datasets/manwithacat/oracc-akkadian-english-parallel-corpus) - Neo-Assyrian royal inscriptions and letters (2,117 texts)

## License

CC BY-SA 4.0 - Academic use with attribution
