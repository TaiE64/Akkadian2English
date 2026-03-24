#!/usr/bin/env python3
"""
Inspect AKT 5 2008.pdf layout: fonts, x-positions, page structure.
Helps calibrate extract_akt5.py parameters.
"""

import sys, re
import fitz
from pathlib import Path
from collections import Counter, defaultdict

PDF_PATH = Path(__file__).resolve().parent.parent / "04_reference" / "kultepe_tablets_pdf" / "AKT 5 2008.pdf"

def inspect_page(doc, page_idx, show_words=True):
    page = doc[page_idx]
    print(f"\n{'='*70}")
    print(f"PAGE {page_idx+1} (0-indexed: {page_idx})")
    print(f"  Size: {page.rect}")

    # Collect font names
    fonts = set()
    rd = page.get_text('rawdict')
    for block in rd.get('blocks', []):
        if 'lines' not in block:
            continue
        for line in block['lines']:
            for span in line['spans']:
                fonts.add(span.get('font', '?'))

    print(f"  Fonts: {sorted(fonts)}")

    # Get words
    words = page.get_text('words')
    if not words:
        print("  (no words)")
        return

    # X position distribution
    x_vals = [w[0] for w in words]
    print(f"  X range: {min(x_vals):.1f} – {max(x_vals):.1f}")

    # Bucket x values into 20px bins
    buckets = Counter(int(x // 20) * 20 for x in x_vals)
    print("  X distribution (bucket: count):")
    for bucket in sorted(buckets):
        bar = '#' * min(buckets[bucket], 40)
        print(f"    x~{bucket:4d}: {buckets[bucket]:3d}  {bar}")

    if show_words:
        print(f"\n  First 40 words (x0, y0, text):")
        for w in sorted(words, key=lambda w: (w[1], w[0]))[:40]:
            print(f"    x={w[0]:6.1f} y={w[1]:6.1f}  '{w[4]}'")


def scan_tablet_headers(doc, start_page, end_page, max_pages=20):
    """Find tablet header patterns in first N pages."""
    print(f"\n{'='*70}")
    print(f"SCANNING HEADERS: pages {start_page+1}–{min(end_page, start_page+max_pages)}")
    for pg_idx in range(start_page, min(end_page, start_page + max_pages)):
        page = doc[pg_idx]
        words = page.get_text('words')
        if not words:
            continue
        # Group into lines
        lines = defaultdict(list)
        for w in words:
            y_key = round(w[1] / 3) * 3
            lines[y_key].append(w)
        for y_key in sorted(lines):
            line_words = sorted(lines[y_key], key=lambda w: w[0])
            text = ' '.join(w[4] for w in line_words)
            # Look for potential tablet headers (numbers at start, "Kt", etc.)
            if re.search(r'\bKt\b', text, re.IGNORECASE):
                print(f"  p{pg_idx+1}: '{text[:100]}'")
            elif re.match(r'^\d{1,3}\.', text.strip()):
                print(f"  p{pg_idx+1} (num?): '{text[:100]}'")


def main():
    if not PDF_PATH.exists():
        print(f"ERROR: {PDF_PATH} not found")
        sys.exit(1)

    doc = fitz.open(str(PDF_PATH))
    print(f"AKT 5 2008.pdf: {len(doc)} pages total")

    # Inspect pages 41-44 (0-indexed = user's page 42-45)
    for pg in [41, 42, 43, 44, 45, 50, 55]:
        if pg < len(doc):
            inspect_page(doc, pg, show_words=(pg <= 43))

    # Scan for tablet headers starting at page 41
    scan_tablet_headers(doc, 41, min(len(doc), 100))

    doc.close()


if __name__ == '__main__':
    main()
