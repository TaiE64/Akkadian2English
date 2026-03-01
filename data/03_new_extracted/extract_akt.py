#!/usr/bin/env python3
"""
Extract transliteration-translation pairs from AKT PDF volumes.
Targets English-language volumes: AKT 6c, 6d, 6e, AKT 8.
Output: data/03_new_extracted/akt_extracted.csv

Layout notes (verified by inspection):
- AKT 6c/d/e: TWO-COLUMN layout per page.
    Left col  (x0 < page_w*0.38): Akkadian transliteration
    Right col (x0 > page_w*0.50): English translation
    Middle   (line numbers, tablet IDs) and full-width blocks (Notes/Comment) are skipped.
- AKT 8: line-by-line interleaved (Akkadian line, then English line, alternating).
"""
import re
import fitz  # pymupdf
import pandas as pd
from pathlib import Path

PDF_DIR = Path("c:/Users/29421/Desktop/kaggle_Challenge/data/04_reference/kultepe_tablets_pdf")
OUT_DIR = Path("c:/Users/29421/Desktop/kaggle_Challenge/data/03_new_extracted")
OUT_DIR.mkdir(exist_ok=True)

TARGET_PDFS = [
    "AKT 6c.pdf",
    "AKT 6d.pdf",
    "AKT 6e.pdf",
    "AKT 8 2015.pdf",
]

# ── Patterns ──────────────────────────────────────────────────────────────────

# Tablet catalogue entry like "527. kt 94/k 937" or "749. kt 94/k 912*"
TABLET_ID_RE = re.compile(r'^\d{2,4}\.\s+kt\s+\d+/[a-z]\s+\d+', re.IGNORECASE)

# Line/face notes: "l. 15:", "l.e.", "r.e.", "lo.e.", "u.e.", "15 r."
LINE_NOTE_RE = re.compile(
    r'^[Ll]r?\.\s*\d|^[Ll]r?\.e\.|^[Rr]r?\.e\.|^[Ll]o\.e\.|^[Uu]\.e\.'
    r'|^\d+\s+[rl]\.$|^\d+\'?\s+[rl]\.'
)

# Blocks to skip entirely (section headers, author names, volume headers)
SKIP_BLOCK_RE = re.compile(
    r'^(Notes?:|Comment:|Commentary|KISIB|MOGENS|KULTEPE|ANKARA|Introduction'
    r'|Contents?|Bibliography|Index|Preface|Abbreviations?|Fig\.|Plate|break)',
    re.IGNORECASE
)

# Sumerian logograms commonly found in OA transliteration
LOGOGRAMS = re.compile(
    r'\b(KÙ\.BABBAR|KÙ\.B|AN\.NA|DUMU|IGI|DUMU\.SAL|TÚG|GÍN|ma-na|MUNUSMEŠ|'
    r'GEME|UNUG|URU|DINGIR|LUGAL|GUD|UDU|NINDA|KI\.LÁM|BE\.LÍ|URUDU|AN\.NA'
    r'|ITU\.KAM|GU|GIN|SAG|DAM\.GAR|E\.GAL|TUG)\b'
)

# Akkadian syllable-hyphen pattern
AKKA_HYPH = re.compile(r'\b[a-záéíóúāēīōūšṭṣḫ]{1,5}-[a-záéíóúāēīōūšṭṣḫ]', re.IGNORECASE)

# Special OA transliteration characters
SPEC_CHARS = re.compile(r'[áéíóúāēīōūšṭṣḫŠṬṢḪÁÉÍÓÚĀĒĪŌŪ]')

# English function words
ENG_WORDS = re.compile(
    r'\b(the|of|for|his|her|its|my|our|your|their|that|this|which|who|'
    r'silver|minas|shekels|tin|textiles|talent|copper|gold|barley|'
    r'said|spoke|wrote|sent|brought|paid|gave|took|has|have|will|shall|'
    r'tablet|seal|witness|son|daughter|father|brother|house|city|'
    r'and|but|or|if|when|because|so)\b',
    re.IGNORECASE
)


def is_transliteration(text: str) -> bool:
    text = text.strip()
    if not text or len(text) < 4:
        return False
    cleaned = re.sub(r'^\d+[\'.\s]+', '', text).strip()
    if not cleaned:
        return False
    score = 0
    if LOGOGRAMS.search(text):
        score += 3
    if AKKA_HYPH.search(cleaned):
        score += 2
    if SPEC_CHARS.search(cleaned):
        score += 1
    if '-' in cleaned and len(cleaned) < 80:
        score += 1
    eng_hits = len(ENG_WORDS.findall(text))
    if eng_hits >= 3:
        score -= 3
    return score >= 2


# ── AKT 6c/d/e: two-column layout extraction ─────────────────────────────────

def extract_pairs_two_column(pdf_path: Path) -> list:
    """
    Uses block-level x-positions to separate Akkadian (left col) from
    English translation (right col).  Tablet entries delimited by TABLET_ID_RE.
    """
    doc = fitz.open(str(pdf_path))
    skip_pages = max(15, len(doc) // 8)

    pairs = []
    translit_buf = []
    transl_buf = []

    def flush():
        if translit_buf and transl_buf:
            src = ' '.join(translit_buf)
            tgt = ' '.join(transl_buf)
            if len(src) > 10 and len(tgt) > 10:
                pairs.append((src, tgt))
        translit_buf.clear()
        transl_buf.clear()

    for pg_num in range(skip_pages, len(doc)):
        page = doc[pg_num]
        page_w = page.rect.width

        # Column thresholds (validated for AKT 6c/d/e widths 504-528)
        left_max = page_w * 0.38    # transliteration: x0 < left_max
        right_min = page_w * 0.50   # translation:     x0 > right_min
        full_width_min = page_w * 0.65  # skip wide commentary blocks

        blocks = sorted(page.get_text('blocks'), key=lambda b: b[1])  # top→bottom

        for x0, y0, x1, y1, raw_text, *_ in blocks:
            text = raw_text.strip()
            if not text or len(text) < 3:
                continue

            block_w = x1 - x0
            first_line = text.split('\n')[0].strip()

            # ── Tablet boundary ─────────────────────────────────────────────
            if TABLET_ID_RE.match(first_line):
                flush()
                continue

            # ── Skip section headers (Notes:, Comment:, author names, etc.) ─
            if SKIP_BLOCK_RE.match(first_line):
                continue

            # ── Skip line/face notes ────────────────────────────────────────
            if LINE_NOTE_RE.match(first_line):
                continue

            # ── Skip full-width blocks (Notes text, Commentary paragraphs) ──
            if block_w > full_width_min:
                continue

            # ── Right column → translation ──────────────────────────────────
            if x0 > right_min:
                # Skip bare line numbers like "5", "10 r.", "lo.e."
                if re.match(r'^\d{1,3}[\'.]?\s*$', text) or LINE_NOTE_RE.match(text):
                    continue
                transl_buf.append(text.replace('\n', ' ').strip())

            # ── Left column → transliteration ───────────────────────────────
            elif x0 < left_max:
                # Skip page numbers (bare integers) and face labels
                if re.match(r'^\d{1,4}\s*$', text):
                    continue
                # Accept multi-line blocks: split into lines and filter
                for ln in text.split('\n'):
                    ln = re.sub(r'^\d+[\'.\s]+', '', ln).strip()
                    if ln and is_transliteration(ln):
                        translit_buf.append(ln)

    flush()
    return pairs


# ── AKT 8: interleaved format ─────────────────────────────────────────────────

AKT8_SKIP_RE = re.compile(
    r'^(Notes?|Comment|Introduction|Bibliography|Abbreviations?|'
    r'Concordance|Contents?|Index|Preface|PREFACE|TABLE|Fig\.|Plate|'
    r'CHAPTER|Chapter|\d+\s*$|[IVX]+\s*$)',
    re.IGNORECASE
)


def is_translation(line: str) -> bool:
    line = line.strip()
    if not line or len(line) < 5:
        return False
    eng_hits = len(ENG_WORDS.findall(line))
    if eng_hits < 1:
        return False
    if LOGOGRAMS.search(line) and AKKA_HYPH.search(line):
        return False
    return True


def extract_pairs_interleaved(text_blocks: list) -> list:
    """Line-by-line interleaved format (AKT 8)."""
    lines = []
    for block in text_blocks:
        for ln in block.split('\n'):
            ln = ln.strip()
            if ln and not AKT8_SKIP_RE.match(ln) and len(ln) > 3:
                lines.append(ln)

    pairs = []
    i = 0
    while i < len(lines) - 1:
        if is_transliteration(lines[i]) and is_translation(lines[i + 1]):
            src = re.sub(r'^\d+[\.\s]+', '', lines[i]).strip()
            tgt = lines[i + 1].strip()
            if len(src) > 5 and len(tgt) > 5:
                pairs.append((src, tgt))
            i += 2
        else:
            i += 1
    return pairs


# ── Per-PDF dispatcher ────────────────────────────────────────────────────────

def extract_pdf(pdf_path: Path) -> list:
    print(f"\nExtracting: {pdf_path.name}")
    doc = fitz.open(str(pdf_path))
    print(f"  Pages: {len(doc)}")
    skip_pages = max(15, len(doc) // 8)

    if "AKT 8" in pdf_path.name:
        # Interleaved format
        blocks = []
        for pg in range(skip_pages, len(doc)):
            text = doc[pg].get_text()
            if text.strip():
                blocks.append(text)
        pairs = extract_pairs_interleaved(blocks)
        print(f"  Format: interleaved")
    else:
        # Two-column layout (AKT 6c/d/e)
        pairs = extract_pairs_two_column(pdf_path)
        print(f"  Format: two-column")

    print(f"  Extracted: {len(pairs)} pairs")
    return [(pdf_path.name, src, tgt) for src, tgt in pairs]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_rows = []
    for pdf_name in TARGET_PDFS:
        pdf_path = PDF_DIR / pdf_name
        if not pdf_path.exists():
            print(f"MISSING: {pdf_name}")
            continue
        rows = extract_pdf(pdf_path)
        all_rows.extend(rows)

    print(f"\nTotal pairs extracted: {len(all_rows)}")

    df = pd.DataFrame(all_rows, columns=["source_pdf", "transliteration", "translation"])
    df = df[df["transliteration"].str.len() >= 10]
    df = df[df["translation"].str.len() >= 10]
    df = df.drop_duplicates(subset=["transliteration"])
    print(f"After filtering: {len(df)} pairs")

    # ── Quality spot-check (30 random samples, shown per PDF) ────────────────
    print("\n=== QUALITY SPOT-CHECK (10 random per PDF) ===")
    for pdf_name in TARGET_PDFS:
        sub = df[df["source_pdf"] == pdf_name]
        if sub.empty:
            continue
        sample = sub.sample(min(10, len(sub)), random_state=42)
        print(f"\n--- {pdf_name} ({len(sub)} pairs) ---")
        for _, row in sample.iterrows():
            print(f"  SRC: {row['transliteration'][:90]}")
            print(f"  TGT: {row['translation'][:90]}")
            print()

    out_path = OUT_DIR / "akt_extracted.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved → {out_path}")

    print("\nPer-PDF counts:")
    for name, cnt in df.groupby("source_pdf").size().items():
        print(f"  {name}: {cnt}")


if __name__ == "__main__":
    main()
