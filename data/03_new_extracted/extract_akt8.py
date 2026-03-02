#!/usr/bin/env python3
"""
Extract transliteration-translation pairs from AKT 8 2015.pdf.

Uses WORD-LEVEL extraction (get_text('words')) for precise x-position
per word, avoiding PyMuPDF's span-merging artifacts.  Detects tablet
headers and notes sections from text patterns alone (no font info needed).

Layout (two-column):
  Left col  (x ~113): Akkadian transliteration
  Right col (x ~297): English translation
  Margin    (x < 100): Line numbers, edge markers

Output: one row per tablet with concatenated transliteration and translation.
"""

import re
import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

PDF_DIR = Path(__file__).resolve().parent.parent / "04_reference" / "kultepe_tablets_pdf"
PDF_NAME = "AKT 8 2015.pdf"
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(exist_ok=True)

START_PAGE = 48
END_PAGE = 520

# X thresholds (from word-position analysis of the PDF):
#   transliteration words: x0 in [113, 251]
#   translation words:     x0 in [297, 510]
#   clear gap:             [252, 296]
COL_X = 280
MARGIN_X = 100
TRANSLIT_MIN_X = 108

# ── Patterns ─────────────────────────────────────────────────────────────────

# Tablet header: "1. Kt 91/k 347 (1-204-91)" — always the full text of a line
TABLET_HEADER_RE = re.compile(
    r'^\d{1,3}\.\s+Kt\s+\d+/[a-z]\s+\d+', re.IGNORECASE
)

# Section headers: "I. SIX BASIC DOCUMENTS", "II. CARAVAN DOCUMENTS"
SECTION_HEADER_RE = re.compile(r'^[IVXL]+\.\s+[A-Z]')

# Bare line numbers in the margin
LINE_NUM_RE = re.compile(r"^\d{1,3}'?\s*$")

# Edge markers in the margin
EDGE_MARKER_RE = re.compile(
    r'^(l\.?e\.?|r\.?e\.?|u\.?e\.?|lo\.?e\.?|rev\.?|le\.?e\.?)$',
    re.IGNORECASE,
)

# Bold note-line markers (e.g. "4-5.", "8.", "8-14.")
NOTE_LINE_MARKER_RE = re.compile(r'^\d{1,3}(-\d{1,3})?\.$')

# Annotation phrases to strip from transliteration
SEAL_ANNOTATION_RE = re.compile(
    r'\bseal\s+[A-Z]\b'
    r'|\bno\s+seal\b'
    r'|\(upside\s+down\)'
    r'|\(sideways\)'
    r'|\(erased?\)'
    r'|\(erasure\)',
    re.IGNORECASE,
)

# Extended English vocabulary for post-processing cleanup
ENGLISH_CONTENT_WORDS = frozenset(
    'the of for his her its my our your their a an is are was were has have had '
    'be been that this which who what when where how why and but or if so as at '
    'by in on to up not no it he she they we with from said also must may can '
    'could would should about after before between into through during against '
    'without within very more most than such each other these those some any all '
    'both only just will shall did does do '
    # Content words commonly found in AKT 8 translations
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

# Akkadian special characters (including font-encoded replacements)
AKK_SPECIAL_RE = re.compile(r'[áéíóúāēīōūšṭṣḫŠṬṢḪ°¿]')


# ── SemiramisUnicode font character fix ─────────────────────────────────────
#
# The SemiramisUnicode font encodes transliteration characters as ASCII symbols.
# Italic and Regular variants use DIFFERENT mappings for the same codepoints.
# We use rawdict extraction to get per-character font info and apply the correct
# mapping per font variant.
#
# Font mappings (verified from word contexts in the PDF):
#
#   Char  Italic (transliteration)  Regular (logograms/names)  Bold (headers)
#   ────  ────────────────────────  ─────────────────────────  ──────────────
#   !     š                         ū                          š
#   #     ṭ                         ē                          ṭ
#   $     ṣ                         ī                          ṣ
#   %     ē                         Š                          Š
#   &     ā                         ā                          ā
#   "     ā                         (normal quote)             (normal quote)
#   ""    –                         šš (A""ur → Aššur)         –
#   °     ⌈ (left half-bracket for damaged signs)
#   ¿     ⌉ (right half-bracket)
#   ¡     ! (correct reading convention marker)

FONT_ENCODED_CHARS = frozenset('!#$%&')
_QUOTE_CHARS = frozenset('"\u201c\u201d')  # regular + smart double-quotes
# Direct 1:1 character replacements (no context needed)
_DIRECT_MAP = str.maketrans({'°': '⌈', '¿': '⌉', '¡': '!'})

# Lowercase (transliteration syllable) mappings
_LOWER_MAP = {'!': 'š', '#': 'ṭ', '$': 'ṣ', '%': 'ē', '&': 'ā'}
# Uppercase (logogram) mappings — all map to Š except &→ā
_UPPER_MAP = {'!': 'Š', '#': 'Š', '$': 'Š', '%': 'Š', '&': 'ā'}

# Known logogram patterns to normalize after context-based char fix
_LOGOGRAM_FIXES = [
    (re.compile(r'KI[ŠṬṢē]IB'), 'KIŠIB'),
    (re.compile(r'AN[ŠṬṢē]E'), 'ANŠE'),
    (re.compile(r'(?<!\S)[ŠṬṢē]E(?=[\s.\-]|$)'), 'ŠE'),
    (re.compile(r'I[ŠṬṢē]TAR'), 'IŠTAR'),
    (re.compile(r'[ŠṬṢē]UNIGIN'), 'ŠUNIGIN'),
    (re.compile(r'[ŠṬṢē]À\.BA'), 'ŠÀ.BA'),
    (re.compile(r'[ŠṬṢē]U\.NIGIN'), 'ŠU.NIGIN'),
    # Aššur: encoded as A!!ur, A""ur, A##ur, A&&ur, A%%ur, A$$ur
    # All produce garbled variants (AŠšur, Aāāur, AŠṭur, etc.) → normalize
    (re.compile(r'A[ŠšṭṬṣṢēĒāĀ]{2}ur'), 'Aššur'),
    # Ištar: uppercase I causes next encoded char to map as Š instead of š
    (re.compile(r'I[ŠṬṢē]tar'), 'Ištar'),
]


def _fix_encoded_word(word: str) -> str:
    """Fix font-encoded chars in a single word using case context."""
    if not any(ch in word for ch in FONT_ENCODED_CHARS | _QUOTE_CHARS) and '(' not in word:
        return word
    chars = list(word)
    result = []
    for i, ch in enumerate(chars):
        if ch in _QUOTE_CHARS:
            # Italic font: " → ā (long vowel), not š
            # (š/Š mapping only applies in Regular font for A""ur → Aššur)
            prev_alpha = chars[i - 1].isalpha() if i > 0 else False
            next_alpha = chars[i + 1].isalpha() if i + 1 < len(chars) else False
            if prev_alpha or next_alpha:
                next_c = chars[i + 1] if i + 1 < len(chars) else ''
                prev_c = chars[i - 1] if i > 0 else ''
                next_up = next_c.isupper() if (next_c and next_c.isalpha()) else False
                prev_up = prev_c.isupper() if (prev_c and prev_c.isalpha()) else False
                result.append('Ā' if (next_up or prev_up) else 'ā')
            else:
                result.append(ch)
        elif ch == '(' and i == 0:
            # '(' at word start followed by lowercase alpha → Š (e.g. (u-Sú-in → Šu-Suen)
            # But NOT if word contains ')' — that's a real parenthetical like (and)
            next_c = chars[i + 1] if i + 1 < len(chars) else ''
            if next_c and next_c.islower() and ')' not in word:
                result.append('Š')
            else:
                result.append(ch)
        elif ch == '&' and i == 0:
            # '&' at word start → Š (names: Šu-, Ša-lim, etc.)
            # Mid-word '&' → ā (long vowel) — handled by FONT_ENCODED_CHARS below
            result.append('Š')
        elif ch in FONT_ENCODED_CHARS:
            next_c = chars[i + 1] if i + 1 < len(chars) else ''
            prev_c = chars[i - 1] if i > 0 else ''
            next_up = next_c.isupper() if (next_c and next_c.isalpha()) else False
            prev_up = prev_c.isupper() if (prev_c and prev_c.isalpha()) else False
            result.append(_UPPER_MAP[ch] if (next_up or prev_up) else _LOWER_MAP[ch])
        else:
            result.append(ch)
    return ''.join(result)


def fix_font_encoding(text: str) -> str:
    """Apply context-based font char fix, then normalize known logograms."""
    # 1. Direct 1:1 replacements (°→⌈, ¿→⌉, ¡→!)
    text = text.translate(_DIRECT_MAP)
    # 2a. Fraction chars after digits: ' and & → ½
    text = re.sub(r"(\d)\s*[&']", r'\1 ½', text)
    # 2b. Pre-process: double-quote pairs between alpha → šš (A""ur → Aššur)
    text = re.sub(r'([a-zA-Z])""([a-zA-Z])', r'\1šš\2', text)
    text = re.sub(r'([a-zA-Z])\u201c\u201d([a-zA-Z])', r'\1šš\2', text)
    text = re.sub(r'([a-zA-Z])\u201d\u201c([a-zA-Z])', r'\1šš\2', text)
    # 3. Context-based symbol → transliteration char
    words = text.split(' ')
    fixed = ' '.join(_fix_encoded_word(w) for w in words)
    # 3. Normalize known logogram patterns
    for pat, repl in _LOGOGRAM_FIXES:
        fixed = pat.sub(repl, fixed)
    return fixed


def _fix_encoded_word_translation(word: str) -> str:
    """Fix font-encoded chars in translation text (conservative).

    Same logic as _fix_encoded_word but with guards to preserve English:
      - '"' only mapped when BOTH neighbors are alpha (preserves "quotes")
      - '!' not mapped at word-final position (preserves exclamation!)
      - '(' never mapped (preserves parentheticals in English)
      - '&' at word start only if NOT followed by space-like context
    """
    if not any(ch in word for ch in FONT_ENCODED_CHARS | _QUOTE_CHARS):
        return word
    chars = list(word)
    n = len(chars)
    result = []
    for i, ch in enumerate(chars):
        prev_alpha = chars[i - 1].isalpha() if i > 0 else False
        next_alpha = chars[i + 1].isalpha() if i + 1 < n else False

        if ch in _QUOTE_CHARS:
            # Single " between alpha → ā (Italic Akkadian words in translation)
            # Note: "" → šš (A""ur → Aššur) is handled by pre-processing regex
            if prev_alpha and next_alpha:
                next_c = chars[i + 1]
                prev_c = chars[i - 1]
                next_up = next_c.isupper() if next_c.isalpha() else False
                prev_up = prev_c.isupper() if prev_c.isalpha() else False
                result.append('Ā' if (next_up or prev_up) else 'ā')
            else:
                result.append(ch)
        elif ch == '!':
            # '!' only map when next char is alpha (preserves gold!, City!")
            if next_alpha:
                next_c = chars[i + 1]
                prev_c = chars[i - 1] if i > 0 else ''
                next_up = next_c.isupper() if next_c.isalpha() else False
                prev_up = prev_c.isupper() if (prev_c and prev_c.isalpha()) else False
                result.append(_UPPER_MAP['!'] if (next_up or prev_up) else _LOWER_MAP['!'])
            else:
                result.append(ch)
        elif ch == '(':
            # Never map ( in translation — too many English false positives
            result.append(ch)
        elif ch in FONT_ENCODED_CHARS:
            # #$%& — use standard case-context mapping
            next_c = chars[i + 1] if i + 1 < n else ''
            prev_c = chars[i - 1] if i > 0 else ''
            next_up = next_c.isupper() if (next_c and next_c.isalpha()) else False
            prev_up = prev_c.isupper() if (prev_c and prev_c.isalpha()) else False
            result.append(_UPPER_MAP[ch] if (next_up or prev_up) else _LOWER_MAP[ch])
        else:
            result.append(ch)
    return ''.join(result)


def fix_font_encoding_translation(text: str) -> str:
    """Conservative font fix for English translation text.

    Applies safe 1:1 replacements, then context-based fixes with guards
    to preserve English punctuation (quotes, exclamations, parentheses).
    Uses two passes to handle sequential encoded chars (A!!ur → AŠšur).
    """
    text = text.translate(_DIRECT_MAP)
    # Fraction chars after digits: ' and & → ½
    text = re.sub(r"(\d)\s*[&']", r'\1 ½', text)
    # Pre-process: double-quote pairs between alpha → šš (A""ur → Aššur)
    text = re.sub(r'([a-zA-Z])""([a-zA-Z])', r'\1šš\2', text)
    text = re.sub(r'([a-zA-Z])\u201c\u201d([a-zA-Z])', r'\1šš\2', text)
    text = re.sub(r'([a-zA-Z])\u201d\u201c([a-zA-Z])', r'\1šš\2', text)
    # Two passes: first pass maps chars with alpha neighbors, second pass
    # catches chars that were blocked by adjacent encoded chars
    words = text.split(' ')
    fixed = ' '.join(_fix_encoded_word_translation(w) for w in words)
    words2 = fixed.split(' ')
    fixed = ' '.join(_fix_encoded_word_translation(w) for w in words2)
    # Normalize known logogram patterns
    for pat, repl in _LOGOGRAM_FIXES:
        fixed = pat.sub(repl, fixed)
    return fixed


# Subscript digit map: ASCII digit → Unicode subscript
_SUBSCRIPT_DIGITS = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')


def _build_word_text(char_tuples):
    """Build word text from (x0, y0, x1, y1, char, font_size) tuples.

    Detects super/subscript characters (font_size < 0.75 * base_size):
      - Alphabetic → wrap in {}: a-lim{ki}, {d}UTU, {f}PN
      - Digits → Unicode subscript: u₄, bi₄, li₅
      - '!' → ʾ (aleph/correct reading marker, NOT font-encoded š)
      - '?' and other punctuation → kept inline as-is
    """
    if not char_tuples:
        return ''
    sizes = [t[5] for t in char_tuples if t[4].strip()]
    if not sizes:
        return ''.join(t[4] for t in char_tuples)
    base_size = max(sizes)
    threshold = base_size * 0.75

    result = []
    in_super = False
    for t in char_tuples:
        ch, sz = t[4], t[5]
        is_super = sz < threshold

        if is_super and ch.isalpha():
            # Superscript letter → determinative: wrap in {}
            if not in_super:
                result.append('{')
                in_super = True
            result.append(ch)
        elif is_super and ch.isdigit():
            # Subscript digit → Unicode subscript (u₄, bi₄)
            if in_super:
                result.append('}')
                in_super = False
            result.append(ch.translate(_SUBSCRIPT_DIGITS))
        elif is_super and ch == '!':
            # Superscript ! → ʾ (aleph / correct reading marker)
            if in_super:
                result.append('}')
                in_super = False
            result.append('ʾ')
        else:
            if in_super:
                result.append('}')
                in_super = False
            result.append(ch)

    if in_super:
        result.append('}')
    return ''.join(result)


def extract_words_rawdict(page) -> list:
    """Extract words from rawdict with per-character position data.

    Uses rawdict for precise character-level bounding boxes (better than
    get_text('words') for SemiramisUnicode font). Character encoding is
    NOT fixed here — fix_font_encoding is applied later in flush().
    Superscript alphabetic determinatives are wrapped in {} braces.
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
                for ch_info in span.get('chars', []):
                    line_chars.append((
                        ch_info['bbox'][0], ch_info['bbox'][1],
                        ch_info['bbox'][2], ch_info['bbox'][3],
                        ch_info['c'], sz,
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
                words.append((
                    word_chars[0][0],
                    min(wc[1] for wc in word_chars),
                    word_chars[-1][2],
                    max(wc[3] for wc in word_chars),
                    w_text, block_no, line_no, word_no,
                ))

    return words


# ── Word classification ──────────────────────────────────────────────────────

def is_akkadian_word(word: str) -> bool:
    """Heuristic: does this word look like Akkadian transliteration?"""
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
    return False


def is_english_word(word: str) -> bool:
    return word.lower().rstrip('.,;:!?()') in ENGLISH_CONTENT_WORDS


def is_margin_item(word: str) -> bool:
    w = word.strip()
    return bool(LINE_NUM_RE.match(w) or EDGE_MARKER_RE.match(w))


def _is_eng_content(word: str) -> bool:
    """Check if a word is English (for post-processing cleanup)."""
    w = word.lower().rstrip('.,;:!?()')
    if not w:
        return False
    # Parenthesized English phrases
    w = w.lstrip('(').rstrip(')')
    return w in ENGLISH_CONTENT_WORDS


def clean_english_from_translit(text: str) -> str:
    """Remove English contamination from transliteration text.

    1. Strip academic references (REL NN, p. NN, *NN:NN, pl.)
    2. Strip parenthetical English phrases
    3. Strip inline English glosses (e.g. "6 minas", "10 shekels")
    4. Remove runs of 2+ consecutive English words
    5. Remove isolated English function words between Akkadian tokens
    """
    # 1. Remove academic/reference patterns
    text = re.sub(
        r'\bREL\s+\d+\)?'           # REL 76)
        r'|\*\d+:\d+[-–]\d+'        # *69:1-11, *74:21-24
        r'|\*\d+:\d+'               # *69:49
        r'|\bp\.\s*\d+'             # p. 225
        r'|\bpp\.\s*\d+'            # pp. 225
        r'|\bpl\.\b'               # pl.
        r'|\bvol\.\s*\d+'           # vol. 5
        r'|\bno\.\s*\d+'            # no. 7
        r'|\btext\s+no\.\s*\d+',    # text no. 7
        '', text, flags=re.IGNORECASE
    )

    # 2. Remove parenthetical English phrases (3+ words inside parens)
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
                run_len = i - run_start
                if run_len >= 2:
                    for j in range(run_start, i):
                        keep[j] = False
                run_start = None

    # Handle trailing run
    if run_start is not None:
        run_len = len(tokens) - run_start
        if run_len >= 2:
            for j in range(run_start, len(tokens)):
                keep[j] = False

    # 5. Remove isolated English function words between Akkadian tokens
    #    (only very common ones that never appear in transliteration)
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
            # Only remove if surrounded by non-English tokens
            prev_eng = (i > 0 and keep[i-1]
                        and _is_eng_content(tokens[i-1]))
            next_eng = (i < len(tokens)-1 and keep[i+1]
                        and _is_eng_content(tokens[i+1]))
            if not prev_eng and not next_eng:
                keep[i] = False

    result = [tok for tok, k in zip(tokens, keep) if k]
    return ' '.join(result)


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

def extract_akt8(pdf_path):
    doc = fitz.open(str(pdf_path))
    print(f"  Pages: {len(doc)}")
    print(f"  Scanning pages {START_PAGE}–{min(END_PAGE, len(doc)) - 1}")

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
            # Fix SemiramisUnicode font encoding (!→š, #→ṭ, etc.)
            src = fix_font_encoding(src)
            tgt = fix_font_encoding_translation(tgt)
            # Post-process: remove seal annotations from transliteration
            src = SEAL_ANNOTATION_RE.sub('', src).strip()
            src = re.sub(r'\s{2,}', ' ', src)
            # Remove runs of consecutive English words from transliteration
            src = clean_english_from_translit(src)
            src = re.sub(r'\s{2,}', ' ', src).strip()
            # Remove leading page numbers from translation
            tgt = re.sub(r'^\d{2,3}\s+', '', tgt)
            # Final quality gate: reject if src is mostly English
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

    for pg_num in range(START_PAGE, min(END_PAGE, len(doc))):
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

            # ── Section header ───────────────────────────────────────────
            if SECTION_HEADER_RE.match(full_text.strip()):
                continue

            # ── Running page header (y < 75) ────────────────────────────
            line_y = line_words[0][1]
            if line_y < 75:
                continue

            #── Notes / Comment marker ───────────────────────────────────
            stripped = full_text.strip()
            if stripped in ('Notes', 'Comment', 'Commentary', 'Comments'):
                state = 'NOTES'
                continue
            # "Note:" or "See ..." at start of line in collecting state
            if state == 'COLLECTING' and re.match(
                r'^(Note:|See\s|Cf\.\s|For\s+(the|this|a)\s|'
                r'The\s+(reading|text|tablet|sign|name|word|form|verb)\s|'
                r'This\s+(text|tablet|document|letter)\s)',
                stripped, re.IGNORECASE
            ):
                state = 'NOTES'
                continue

            # ── Bold note-line marker at margin (e.g. "4-5. See on...") ──
            first_w = line_words[0]
            if (first_w[0] < MARGIN_X
                    and NOTE_LINE_MARKER_RE.match(first_w[4].strip())
                    and state == 'COLLECTING'):
                state = 'NOTES'
                continue

            # ── State-based processing ───────────────────────────────────
            if state in ('IDLE', 'NOTES'):
                continue
            if current_tablet is None:
                continue

            # Find the first content word past the margin
            content_words = [
                (w[0], w[4]) for w in line_words
                if w[0] >= MARGIN_X and not is_margin_item(w[4])
            ]
            if not content_words:
                continue

            first_x, first_word = content_words[0]

            # ── DESCRIPTION → COLLECTING ─────────────────────────────────
            if state == 'DESCRIPTION':
                if first_x >= TRANSLIT_MIN_X and is_akkadian_word(first_word):
                    state = 'COLLECTING'
                else:
                    continue

            # ── COLLECTING ───────────────────────────────────────────────
            if state == 'COLLECTING':
                # Guard: text starting at x < transliteration indent
                if first_x < TRANSLIT_MIN_X:
                    continue

                # Guard: mostly English in left column → notes/desc leaked
                if not is_akkadian_word(first_word):
                    left_words = [w[4] for w in line_words
                                  if MARGIN_X <= w[0] < COL_X]
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

                    if x0 >= COL_X:
                        right_parts.append(word)
                    elif x0 >= MARGIN_X:
                        if is_margin_item(word) and x0 < TRANSLIT_MIN_X:
                            continue
                        left_parts.append(word)
                    # x0 < MARGIN_X: skip line numbers / edge markers

                if left_parts:
                    translit_buf.append(' '.join(left_parts))
                if right_parts:
                    trans_buf.append(' '.join(right_parts))

    flush()
    print(f"  Extracted: {len(results)} tablet-level pairs")
    return results


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    pdf_path = PDF_DIR / PDF_NAME
    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} not found")
        return

    print(f"\nExtracting: {PDF_NAME}")
    rows = extract_akt8(pdf_path)

    df = pd.DataFrame(rows)
    if df.empty:
        print("  No pairs extracted!")
        return

    df = df.drop_duplicates(subset=['transliteration'])
    print(f"  After dedup: {len(df)} pairs")

    # ── Quality check: English leakage ────────────────────────────────────────
    eng_re = re.compile(
        r'\b(the|of|for|his|her|silver|minas|shekels|says?|seal|son|daughter|'
        r'house|city|brought|gave|took|paid|who|which|that|this|from|with|'
        r'your|my|our|their)\b', re.IGNORECASE
    )
    leak_count = sum(1 for _, r in df.iterrows()
                     if len(eng_re.findall(r['transliteration'])) >= 3)
    print(f"  English leakage (>=3 eng words in src): {leak_count}/{len(df)}")

    # ── Quality spot-check ────────────────────────────────────────────────────
    print("\n=== QUALITY SPOT-CHECK (10 random samples) ===")
    sample = df.sample(min(10, len(df)), random_state=42)
    for _, row in sample.iterrows():
        print(f"\n  TABLET: {row['tablet_id'][:70]}")
        print(f"  SRC:    {row['transliteration'][:120]}...")
        print(f"  TGT:    {row['translation'][:120]}...")

    # ── Statistics ────────────────────────────────────────────────────────────
    print(f"\n=== Statistics ===")
    print(f"  Total tablet pairs : {len(df)}")
    print(f"  Avg src length     : {df['transliteration'].str.len().mean():.0f} chars")
    print(f"  Avg tgt length     : {df['translation'].str.len().mean():.0f} chars")
    print(f"  Median src length  : {df['transliteration'].str.len().median():.0f} chars")
    print(f"  Median tgt length  : {df['translation'].str.len().median():.0f} chars")

    out_path = OUT_DIR / "akt8_extracted.csv"
    df.to_csv(out_path, index=False, encoding='utf-8')
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
