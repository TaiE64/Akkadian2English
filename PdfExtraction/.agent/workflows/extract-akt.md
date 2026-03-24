---
description: Extract OA transliterations from images into AKT CSV files
---

// turbo-all

## Steps

1. Read the SKILL.md file for extraction rules
2. Read the target CSV file to understand existing format and row count
3. Extract OA transliterations and English translations from the provided images
4. Apply all normalization rules (subscripts, superscripts, line-breaks, colons, gaps, etc.)
5. Join lines into sentences by period boundaries, split overly long sentences
6. Append the new rows to the target CSV file
7. Validate the CSV with Python to confirm row count and column integrity
