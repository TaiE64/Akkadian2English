#!/usr/bin/env python3
"""
Extract transliteration-translation pairs from AKT 6a and AKT 6b PDFs.

Uses RAWDICT extraction for character-level font/size info:
  - Times-Italic font encodes diacriticals as ASCII symbols (£→í, {→ì, etc.)
  - Subscript digits detected by font size ratio
  - Superscript alphabetic chars → determinative brackets {d}, {ki}

Layout (two-column):
  Left col  (x < ~250/270): Akkadian transliteration
  Margin    (x ~215-250):   Line numbers (5, 10, 15, ...), edge markers (e., r.)
  Right col (x >= ~250/270): English translation

Output: one row per tablet with concatenated transliteration and translation.
"""

import re
import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

PDF_DIR = Path(__file__).resolve().parent.parent / "04_reference" / "kultepe_tablets_pdf"
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(exist_ok=True)

# Per-PDF configuration
PDF_CONFIGS = {
    "AKT 6a.pdf": {
        "start_page": 51,
        "end_page": 440,
        "col_x": 250,       # translation starts at x >= 250
        "margin_min": 215,   # margin items: 215 <= x < 250
        "translit_min": 25,  # transliteration starts at x >= 25
    },
    "AKT 6b.pdf": {
        "start_page": 58,
        "end_page": 360,
        "col_x": 270,       # translation starts at x >= 270
        "margin_min": 235,   # margin items: 235 <= x < 270
        "translit_min": 40,  # transliteration starts at x >= 40
    },
}

# ── Patterns ─────────────────────────────────────────────────────────────────

# Tablet header: "1. kt 94/k 1263" or "302b. kt 94/k 1488B"
TABLET_HEADER_RE = re.compile(
    r'^\d{1,3}[a-c]?\.\s+[Kk]t\s+\d+/[a-z]\s+\d+', re.IGNORECASE
)

# Section headers like "Salim-Assur's Legal Texts"
SECTION_HEADER_RE = re.compile(r'^[A-Z][a-z]+-[A-Z]|^[IVXL]+\.\s+[A-Z]')

# Bare line numbers in the margin
LINE_NUM_RE = re.compile(r"^\d{1,3}'?\s*$")

# Edge markers
EDGE_MARKER_RE = re.compile(
    r'^(l\.?e\.?|r\.?e\.?|u\.?e\.?|lo\.?e\.?|rev\.?|le\.?e\.?|Vs\.?|Rs\.?|u\.?\s*K\.?)$',
    re.IGNORECASE,
)

# Note/comment start markers
NOTES_START_RE = re.compile(
    r'^(Notes?:|Comment(ary|s)?:|See\s|Cf\.\s|For\s+(the|this|a)\s|'
    r'The\s+(reading|text|tablet|sign|name|word|form|verb|restoration)\s|'
    r'This\s+(text|tablet|document|letter)\s)',
    re.IGNORECASE
)

# Seal annotations to strip
SEAL_ANNOTATION_RE = re.compile(
    r'\bseal\s+[A-Z]\b'
    r'|\bno\s+seal\b'
    r'|\(upside\s+down\)'
    r'|\(sideways\)'
    r'|\(erased?\)'
    r'|\(erasure\)',
    re.IGNORECASE,
)

# Bold note-line markers (e.g. "4-5.", "8.", "8-14.")
NOTE_LINE_MARKER_RE = re.compile(r'^\d{1,3}(-\d{1,3})?\.$')

# Extended English vocabulary for post-processing cleanup
ENGLISH_CONTENT_WORDS = frozenset(
    'the of for his her its my our your their a an is are was were has have had '
    'be been that this which who what when where how why and but or if so as at '
    'by in on to up not no it he she they we with from said also must may can '
    'could would should about after before between into through during against '
    'without within very more most than such each other these those some any all '
    'both only just will shall did does do '
    'silver gold tin copper textiles minas shekels talent '
    'son daughter father brother sister wife husband house city '
    'seal witness tablet letter caravan donkey bag container '
    'gave took brought paid sent wrote spoke says say told '
    'price cost total amount debt credit interest profit loss '
    'bought sold trade merchant trader representative agent '
    'journey travel road day year month time '
    'placed deposited received accepted delivered '
    'according concerning regarding behalf '
    'see below above note refers reference lines obverse reverse '
    'cf pp vol published edition copy text '.split()
)

# Akkadian special characters
AKK_SPECIAL_RE = re.compile(r'[áéíóúāēīōūšṭṣḫŠṬṢḪ]')


# ── Times-Italic font encoding map ──────────────────────────────────────────
#
# AKT 6a/6b uses Times-Italic subset fonts for transliteration. The font subsets
# replace standard ASCII glyphs with Akkadian diacritical glyphs. PyMuPDF extracts
# the Unicode codepoints (not glyphs), so we must map them back.
#
# CONFIRMED MAPPINGS (verified by rendering + cross-reference with train.csv):
#
#   PDF char   Unicode   Maps to   Evidence
#   ────────   ───────   ───────   ──────────────────────
#   s          U+0073    š         Font glyph is š (shin). Rendered PDF confirms.
#   S          U+0053    Š         Font glyph is Š (Shin). Rendered PDF confirms.
#   f          U+0066    í         AKT6 'tar-df-u-um' = train 'tár-dí-ú-um'
#   '          U+0027    (grave)   AKT6 'm'i' = train 'mì'. Combining grave accent.
#   £          U+00A3    í         i acute (i₂)
#   {          U+007B    ì         i grave (i₃)
#   }          U+007D    ú         u acute (u₂)
#   $          U+0024    ṣ         tsade
#   #          U+0023    ṭ         tet
#   &          U+0026    ā         a macron
#   !          U+0021    š/Š       context: š before lower, Š before upper
#   ·          U+00B7    -         hyphen (syllable separator)
#   «»         U+00AB/BB (remove)  editorial brackets
#
# Additionally, characters at font size < 0.75 * base_size:
#   - Digits → Unicode subscript (₂,₃,₄,₅ etc.)
#   - Alphabetic → determinative brackets: ki→{ki}, d→{d}
#   - 'v' at small size → paragraph separator (skip)
#   - 'X' at small size → gap marker → x

# Font-encoded chars that need context-based mapping
FONT_ENCODED_CHARS = frozenset('!#$&')

# 1:1 character mapping (no context needed) — applied to ALL fonts
_DIRECT_MAP = {
    '£': 'í',   # U+00A3 → í (i acute)
    '·': '-',   # U+00B7 → hyphen
    '«': '',    # editorial bracket → remove
    '»': '',    # editorial bracket → remove
}

# 1:1 mapping ONLY for italic fonts (transliteration-specific)
_ITALIC_MAP = {
    '{': 'ì',   # U+007B → ì (i grave)
    '}': 'ú',   # U+007D → ú (u acute)
    's': 'š',   # Font glyph substitution: s → š
    'S': 'Š',   # Font glyph substitution: S → Š
    'f': 'í',   # Font glyph substitution: f → í
}

# Grave accent combination: ' + vowel → grave-accented vowel
_GRAVE_MAP = {
    'a': 'à', 'e': 'è', 'i': 'ì', 'u': 'ù',
    'A': 'À', 'E': 'È', 'I': 'Ì', 'U': 'Ù',
}

# Context-based mapping for FONT_ENCODED_CHARS
_LOWER_MAP = {'!': 'š', '#': 'ṭ', '$': 'ṣ', '&': 'ā'}
_UPPER_MAP = {'!': 'Š', '#': 'Ṭ', '$': 'Ṣ', '&': 'Ā'}

# Subscript digit map: ASCII digit → Unicode subscript
_SUBSCRIPT_DIGITS = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')

# Known logogram patterns to normalize
_LOGOGRAM_FIXES = [
    (re.compile(r'KI[ŠṬṢ]IB'), 'KIŠIB'),
    (re.compile(r'AN[ŠṬṢ]E'), 'ANŠE'),
    (re.compile(r'(?<!\S)[ŠṬṢ]E(?=[\s.\-]|$)'), 'ŠE'),
    (re.compile(r'I[ŠṬṢ]TAR'), 'IŠTAR'),
    (re.compile(r'[ŠṬṢ]UNIGIN'), 'ŠUNIGIN'),
    (re.compile(r'[ŠṬṢ]À\.BA'), 'ŠÀ.BA'),
    (re.compile(r'[ŠṬṢ]U\.NIGIN'), 'ŠU.NIGIN'),
]

# ── Determinative detection (textual pattern matching) ────────────────────────
#
# AKT 6a/6b PDF encodes determinatives as regular-sized text (not superscript),
# so they cannot be detected from font size. We use text patterns instead.
#
# Rule: determinative text is DIRECTLY APPENDED to the word (no hyphen).
# Regular syllables are always hyphenated (e.g., ki-ma, da-am).

# {ki} (place): ki appended directly after a syllable-final letter
_DET_KI_RE = re.compile(r'([a-zàèìùáéíúšṣṭā])ki\b')
# {d} (divine): lowercase d directly before uppercase logogram (2+ uppercase chars)
_DET_D_RE = re.compile(r'\bd([A-ZŠṢṬ][A-ZŠṢṬ.]+)')
# {f} (female): lowercase f directly before uppercase or capitalized name
_DET_F_RE = re.compile(r'\bí([A-ZŠṢṬ][a-zšṣṭ])')  # 'f' was mapped to 'í'


# ── Rawdict character extraction ─────────────────────────────────────────────

def _build_word_text(char_tuples):
    """Build word text from (x0, y0, x1, y1, char, font_size, font_name) tuples.

    Applies font encoding fixes in order:
    1. Direct 1:1 mapping (£→í, ·→-, «»→remove) — all fonts
    2. Italic-specific glyph substitution (s→š, S→Š, f→í, {→ì, }→ú)
    3. Context-based mapping (!→š/Š, $→ṣ/Ṣ, #→ṭ/Ṭ, &→ā/Ā)
    4. Grave accent combination ('+ vowel → grave-accented vowel)
    5. Super/subscript detection by font size ratio
    """
    if not char_tuples:
        return ''

    # Get base font size (most common/largest among non-space chars)
    sizes = [t[5] for t in char_tuples if t[4].strip()]
    if not sizes:
        return ''.join(t[4] for t in char_tuples)
    base_size = max(sizes)
    threshold = base_size * 0.75

    result = []
    in_super = False
    skip_next = False

    for i, t in enumerate(char_tuples):
        if skip_next:
            skip_next = False
            continue

        ch, sz, fn = t[4], t[5], t[6]
        is_italic = 'Italic' in fn
        is_small = sz < threshold and base_size > 5.0

        # Step 1: Direct 1:1 mapping (all fonts)
        if ch in _DIRECT_MAP:
            ch = _DIRECT_MAP[ch]
            if not ch:
                continue

        # Step 2: Italic-specific glyph substitution (s→š, S→Š, f→í, {→ì, }→ú)
        if is_italic and ch in _ITALIC_MAP:
            ch = _ITALIC_MAP[ch]

        # Step 3: Context-based mapping for !#$&
        if is_italic and ch in FONT_ENCODED_CHARS:
            next_c = char_tuples[i + 1][4] if i + 1 < len(char_tuples) else ''
            prev_c = char_tuples[i - 1][4] if i > 0 else ''
            next_up = next_c.isupper() if (next_c and next_c.isalpha()) else False
            prev_up = prev_c.isupper() if (prev_c and prev_c.isalpha()) else False
            ch = _UPPER_MAP[ch] if (next_up or prev_up) else _LOWER_MAP[ch]

        # Step 4: Grave accent combination (' + vowel → àèìù)
        if is_italic and ch == "'":
            next_c = char_tuples[i + 1][4] if i + 1 < len(char_tuples) else ''
            if next_c in _GRAVE_MAP:
                # Combine ' + vowel → grave-accented vowel
                ch = _GRAVE_MAP[next_c]
                skip_next = True
            else:
                # Standalone apostrophe — keep as-is (cleaned later)
                pass

        # Step 5: Handle small-size chars (subscript/superscript)
        if is_small:
            # Check original char (before mapping) for special small-size handling
            orig_ch = t[4]
            if orig_ch == 'v':
                # Paragraph separator at small size — skip
                continue
            if orig_ch == 'X' or ch == 'x':
                # Gap marker — convert to lowercase x
                if in_super:
                    result.append('}')
                    in_super = False
                result.append('x')
                continue
            elif ch.isalpha():
                # Superscript letter → determinative: wrap in {}
                if not in_super:
                    result.append('{')
                    in_super = True
                result.append(ch)
                continue
            elif ch.isdigit():
                # Subscript digit → Unicode subscript
                if in_super:
                    result.append('}')
                    in_super = False
                result.append(ch.translate(_SUBSCRIPT_DIGITS))
                continue

        # Normal-size character
        if in_super:
            result.append('}')
            in_super = False
        result.append(ch)

    if in_super:
        result.append('}')
    return ''.join(result)


def extract_words_rawdict(page) -> list:
    """Extract words from rawdict with per-character font/size info.

    Returns list of tuples: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
    """
    rd = page.get_text('rawdict')
    words = []

    for block_no, block in enumerate(rd.get('blocks', [])):
        if 'lines' not in block:
            continue
        for line_no, line in enumerate(block['lines']):
            line_chars = []
            for span in line['spans']:
                sz = span['size']
                fn = span['font']
                for ch_info in span.get('chars', []):
                    line_chars.append((
                        ch_info['bbox'][0], ch_info['bbox'][1],
                        ch_info['bbox'][2], ch_info['bbox'][3],
                        ch_info['c'], sz, fn,
                    ))

            if not line_chars:
                continue

            # Group characters into words (split on spaces)
            word_chars = []
            word_no = 0
            for item in line_chars:
                if item[4] in (' ', '\t'):
                    if word_chars:
                        w_text = _build_word_text(word_chars)
                        if w_text.strip():
                            words.append((
                                word_chars[0][0],
                                min(wc[1] for wc in word_chars),
                                word_chars[-1][2],
                                max(wc[3] for wc in word_chars),
                                w_text, block_no, line_no, word_no,
                            ))
                        word_no += 1
                        word_chars = []
                else:
                    word_chars.append(item)

            if word_chars:
                w_text = _build_word_text(word_chars)
                if w_text.strip():
                    words.append((
                        word_chars[0][0],
                        min(wc[1] for wc in word_chars),
                        word_chars[-1][2],
                        max(wc[3] for wc in word_chars),
                        w_text, block_no, line_no, word_no,
                    ))

    return words


def fix_font_encoding(text: str) -> str:
    """Post-processing: normalize logograms and detect determinatives."""
    # 1. Normalize known logogram patterns
    for pat, repl in _LOGOGRAM_FIXES:
        text = pat.sub(repl, text)

    # 2. Detect determinatives from text patterns
    # {ki} (place): ki directly appended without hyphen
    text = _DET_KI_RE.sub(r'\1{ki}', text)
    # {d} (divine): lowercase d before uppercase logogram
    text = _DET_D_RE.sub(r'{d}\1', text)

    return text


# ── Word classification ──────────────────────────────────────────────────────

def is_akkadian_word(word: str) -> bool:
    w = word.strip().rstrip('.,;:')
    if not w:
        return False
    if '-' in w and len(w) >= 3:
        return True
    alpha = ''.join(c for c in w if c.isalpha())
    if alpha and alpha.isupper() and len(alpha) >= 2:
        return True
    if AKK_SPECIAL_RE.search(w):
        return True
    if w.startswith('[') or w.endswith(']'):
        return True
    if w.isdigit():
        return True
    if re.match(r'^\d+/\d+$', w):
        return True
    # Determinative brackets
    if '{' in w and '}' in w:
        return True
    return False


def is_english_word(word: str) -> bool:
    return word.lower().rstrip('.,;:!?()') in ENGLISH_CONTENT_WORDS


def is_margin_item(word: str) -> bool:
    w = word.strip()
    return bool(LINE_NUM_RE.match(w) or EDGE_MARKER_RE.match(w))


def _is_eng_content(word: str) -> bool:
    w = word.lower().rstrip('.,;:!?()')
    if not w:
        return False
    w = w.lstrip('(').rstrip(')')
    return w in ENGLISH_CONTENT_WORDS


def clean_english_from_translit(text: str) -> str:
    """Remove English contamination from transliteration text."""
    # 1. Remove academic references
    text = re.sub(
        r'\bREL\s+\d+\)?'
        r'|\*\d+:\d+[-–]\d+'
        r'|\*\d+:\d+'
        r'|\bp\.\s*\d+'
        r'|\bpp\.\s*\d+'
        r'|\bpl\.\b'
        r'|\bvol\.\s*\d+'
        r'|\bno\.\s*\d+'
        r'|\btext\s+no\.\s*\d+',
        '', text, flags=re.IGNORECASE
    )

    # 2. Remove parenthetical English phrases
    def _strip_eng_parens(m):
        inner = m.group(1)
        words = inner.split()
        eng_count = sum(1 for w in words if _is_eng_content(w))
        if len(words) >= 2 and eng_count >= len(words) * 0.5:
            return ''
        return m.group(0)
    text = re.sub(r'\(([^)]{8,})\)', _strip_eng_parens, text)

    # 3. Remove inline English glosses next to numbers/logograms
    text = re.sub(
        r'\b(\d+)\s+(minas?|shekels?|talents?|textiles?)\b',
        r'\1', text, flags=re.IGNORECASE
    )

    # 4. Remove runs of 2+ consecutive English words
    tokens = text.split()
    if not tokens:
        return text

    keep = [True] * len(tokens)
    run_start = None

    for i, tok in enumerate(tokens):
        if _is_eng_content(tok):
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                if i - run_start >= 2:
                    for j in range(run_start, i):
                        keep[j] = False
                run_start = None

    if run_start is not None:
        if len(tokens) - run_start >= 2:
            for j in range(run_start, len(tokens)):
                keep[j] = False

    # 5. Remove isolated English function words between Akkadian tokens
    ISOLATED_ENG = frozenset(
        'the a an and but or if so at by in on to up is are was were '
        'his her its my our your their this that these those '
        'separately belonging entrusted'.split()
    )
    for i, tok in enumerate(tokens):
        if not keep[i]:
            continue
        w = tok.lower().rstrip('.,;:!?()')
        if w in ISOLATED_ENG and not is_akkadian_word(tok):
            prev_eng = (i > 0 and keep[i-1] and _is_eng_content(tokens[i-1]))
            next_eng = (i < len(tokens)-1 and keep[i+1] and _is_eng_content(tokens[i+1]))
            if not prev_eng and not next_eng:
                keep[i] = False

    return ' '.join(tok for tok, k in zip(tokens, keep) if k)


# ── Line grouping ────────────────────────────────────────────────────────────

def group_words_into_lines(words, y_tol=3.0):
    """Group word tuples into visual lines by y-position."""
    if not words:
        return []
    ws = sorted(words, key=lambda w: (w[1], w[0]))
    lines = []
    cur = [ws[0]]
    cur_y = ws[0][1]
    for w in ws[1:]:
        if abs(w[1] - cur_y) <= y_tol:
            cur.append(w)
        else:
            lines.append(sorted(cur, key=lambda w: w[0]))
            cur = [w]
            cur_y = w[1]
    if cur:
        lines.append(sorted(cur, key=lambda w: w[0]))
    return lines


# ── Main extraction ──────────────────────────────────────────────────────────

def extract_akt6(pdf_path, config):
    doc = fitz.open(str(pdf_path))
    start = config["start_page"]
    end = min(config["end_page"], len(doc))
    col_x = config["col_x"]
    margin_min = config["margin_min"]
    translit_min = config["translit_min"]

    print(f"  Pages: {len(doc)}")
    print(f"  Scanning pages {start}–{end - 1}")
    print(f"  Column threshold: {col_x}, margin: {margin_min}")

    results = []
    current_tablet = None
    translit_buf = []
    trans_buf = []
    state = 'IDLE'  # IDLE → DESCRIPTION → COLLECTING → NOTES

    def flush():
        nonlocal current_tablet, translit_buf, trans_buf
        if current_tablet and translit_buf and trans_buf:
            src = re.sub(r'\s{2,}', ' ', ' '.join(translit_buf)).strip()
            tgt = re.sub(r'\s{2,}', ' ', ' '.join(trans_buf)).strip()

            # Apply logogram normalization
            src = fix_font_encoding(src)

            # Post-process: remove seal annotations from transliteration
            src = SEAL_ANNOTATION_RE.sub('', src).strip()
            src = re.sub(r'\s{2,}', ' ', src)

            # Remove English contamination from transliteration
            src = clean_english_from_translit(src)
            src = re.sub(r'\s{2,}', ' ', src).strip()

            # Remove leading page numbers from translation
            tgt = re.sub(r'^\d{2,3}\s+', '', tgt)

            # Quality gate: reject if src is mostly English
            src_tokens = src.split()
            akk_count = sum(1 for t in src_tokens if is_akkadian_word(t))
            eng_count = sum(1 for t in src_tokens if _is_eng_content(t))
            if len(src) >= 10 and len(tgt) >= 10 and (
                len(src_tokens) < 3 or akk_count > eng_count
            ):
                results.append({
                    'tablet_id': current_tablet,
                    'transliteration': src,
                    'translation': tgt,
                })
        translit_buf = []
        trans_buf = []

    for pg_num in range(start, end):
        page = doc[pg_num]
        raw_words = extract_words_rawdict(page)
        lines = group_words_into_lines(raw_words)

        for line_words in lines:
            if not line_words:
                continue

            texts = [w[4] for w in line_words]
            full_text = ' '.join(texts)

            # ── Tablet header ────────────────────────────────────────────
            if TABLET_HEADER_RE.match(full_text):
                flush()
                current_tablet = full_text.strip()
                state = 'DESCRIPTION'
                continue

            # ── Running page header (y < 60) ─────────────────────────────
            line_y = line_words[0][1]
            if line_y < 60:
                continue

            # ── Notes / Comment marker ───────────────────────────────────
            stripped = full_text.strip()
            if stripped in ('Notes', 'Comment', 'Commentary', 'Comments',
                           'Notes:', 'Comment:', 'Commentary:', 'Comments:'):
                state = 'NOTES'
                continue

            if state == 'COLLECTING' and NOTES_START_RE.match(stripped):
                state = 'NOTES'
                continue

            # ── Bold note-line marker at margin ──────────────────────────
            first_w = line_words[0]
            if (first_w[0] < translit_min
                    and NOTE_LINE_MARKER_RE.match(first_w[4].strip())
                    and state == 'COLLECTING'):
                state = 'NOTES'
                continue

            # ── Section headers (chapter titles) ─────────────────────────
            if len(texts) <= 6:
                all_title = all(w.istitle() or w in "'-–" or w.isupper()
                                for w in texts if w.isalpha())
                if all_title and len(full_text) > 10 and not is_akkadian_word(texts[0]):
                    continue

            # ── State-based processing ───────────────────────────────────
            if state in ('IDLE', 'NOTES'):
                continue
            if current_tablet is None:
                continue

            # Find the first content word past the margin
            content_words = [
                (w[0], w[4]) for w in line_words
                if w[0] >= translit_min and not is_margin_item(w[4])
            ]
            if not content_words:
                continue

            first_x, first_word = content_words[0]

            # ── DESCRIPTION → COLLECTING ─────────────────────────────────
            if state == 'DESCRIPTION':
                if first_x >= translit_min and first_x < col_x and is_akkadian_word(first_word):
                    state = 'COLLECTING'
                else:
                    continue

            # ── COLLECTING ───────────────────────────────────────────────
            if state == 'COLLECTING':
                # Guard: mostly English in left column → notes/desc leaked
                if not is_akkadian_word(first_word) and first_x < col_x:
                    left_words = [w[4] for w in line_words if translit_min <= w[0] < col_x]
                    eng_count = sum(1 for w in left_words if is_english_word(w))
                    if len(left_words) >= 3 and eng_count >= len(left_words) * 0.5:
                        state = 'NOTES'
                        continue

                # Split words into transliteration / translation
                left_parts = []
                right_parts = []

                for w in line_words:
                    x0, word = w[0], w[4].strip()
                    if not word:
                        continue

                    if x0 >= col_x:
                        right_parts.append(word)
                    elif x0 >= translit_min:
                        if is_margin_item(word) and x0 >= margin_min:
                            continue
                        left_parts.append(word)

                if left_parts:
                    translit_buf.append(' '.join(left_parts))
                if right_parts:
                    trans_buf.append(' '.join(right_parts))

    flush()
    print(f"  Extracted: {len(results)} tablet-level pairs")
    return results


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    all_rows = []

    for pdf_name, config in PDF_CONFIGS.items():
        pdf_path = PDF_DIR / pdf_name
        if not pdf_path.exists():
            print(f"WARNING: {pdf_path} not found, skipping")
            continue

        print(f"\nExtracting: {pdf_name}")
        rows = extract_akt6(pdf_path, config)

        for r in rows:
            r['source_pdf'] = pdf_name
        all_rows.extend(rows)

    if not all_rows:
        print("No pairs extracted!")
        return

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=['transliteration'])
    print(f"\n=== Combined Results ===")
    print(f"  Total pairs: {len(df)}")
    print(f"  Per source:")
    for src, cnt in df['source_pdf'].value_counts().items():
        print(f"    {src}: {cnt}")

    # Quality check: English leakage
    eng_re = re.compile(
        r'\b(the|of|for|his|her|silver|minas|shekels|says?|seal|son|daughter|'
        r'house|city|brought|gave|took|paid|who|which|that|this|from|with|'
        r'your|my|our|their)\b', re.IGNORECASE
    )
    leak_count = sum(1 for _, r in df.iterrows()
                     if len(eng_re.findall(r['transliteration'])) >= 3)
    print(f"  English leakage (>=3 eng words in src): {leak_count}/{len(df)}")

    # Diacritics coverage
    diacritics_re = re.compile(r'[áéíóúāēīōūšṭṣŠṬṢ]')
    has_diacritics = sum(1 for _, r in df.iterrows()
                        if diacritics_re.search(r['transliteration']))
    print(f"  Diacritics coverage: {has_diacritics}/{len(df)} ({100*has_diacritics/len(df):.1f}%)")

    # Determinative coverage
    det_re = re.compile(r'\{[a-z₀-₉]+\}')
    has_det = sum(1 for _, r in df.iterrows()
                  if det_re.search(r['transliteration']))
    print(f"  Determinatives: {has_det}/{len(df)} ({100*has_det/len(df):.1f}%)")

    # Subscript digit coverage
    sub_re = re.compile(r'[₀-₉]')
    has_sub = sum(1 for _, r in df.iterrows()
                  if sub_re.search(r['transliteration']))
    print(f"  Subscript digits: {has_sub}/{len(df)} ({100*has_sub/len(df):.1f}%)")

    # Quality spot-check
    print("\n=== QUALITY SPOT-CHECK (10 random samples) ===")
    sample = df.sample(min(10, len(df)), random_state=42)
    for _, row in sample.iterrows():
        print(f"\n  TABLET: {row['tablet_id'][:70]}")
        print(f"  SRC:    {row['transliteration'][:150]}...")
        print(f"  TGT:    {row['translation'][:150]}...")

    # Statistics
    print(f"\n=== Statistics ===")
    print(f"  Total tablet pairs : {len(df)}")
    print(f"  Avg src length     : {df['transliteration'].str.len().mean():.0f} chars")
    print(f"  Avg tgt length     : {df['translation'].str.len().mean():.0f} chars")
    print(f"  Median src length  : {df['transliteration'].str.len().median():.0f} chars")
    print(f"  Median tgt length  : {df['translation'].str.len().median():.0f} chars")

    out_path = OUT_DIR / "akt6ab_extracted.csv"
    df.to_csv(out_path, index=False, encoding='utf-8')
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
