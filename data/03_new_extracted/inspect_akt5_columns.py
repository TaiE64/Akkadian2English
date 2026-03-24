#!/usr/bin/env python3
"""
精确测量 AKT 5 各内容页的双列 x 坐标分布。
只扫描 tablet header 后面的内容页（非注释页）。
"""

import re, fitz
from pathlib import Path
from collections import defaultdict

PDF_PATH = Path(__file__).resolve().parent.parent / "04_reference" / "kultepe_tablets_pdf" / "AKT 5 2008.pdf"

# 从 inspect_akt5.py 输出中已知 tablet header 出现页（1-indexed）
# 取这些页的下一页（即实际内容页）来测量 x 分布
TABLET_HEADER_RE = re.compile(r'^Kt\s+\d+/[a-z]\s+\d+', re.IGNORECASE)

def get_line_texts(page):
    words = page.get_text('words')
    if not words:
        return []
    lines = defaultdict(list)
    for w in words:
        y_key = round(w[1] / 3) * 3
        lines[y_key].append(w)
    result = []
    for y_key in sorted(lines):
        line_ws = sorted(lines[y_key], key=lambda w: w[0])
        result.append(line_ws)
    return result

def inspect_content_page(doc, pg_idx):
    page = doc[pg_idx]
    words = page.get_text('words')
    if not words:
        return

    # 收集字体
    fonts = set()
    rd = page.get_text('rawdict')
    for block in rd.get('blocks', []):
        if 'lines' not in block:
            continue
        for line in block['lines']:
            for span in line['spans']:
                fonts.add(span.get('font', '?'))

    print(f"\n--- Page {pg_idx+1} ---  fonts: {sorted(fonts)}")

    # 过滤掉页眉页脚（y < 70 或 y > 730）
    content_words = [w for w in words if 70 <= w[1] <= 730]
    if not content_words:
        return

    # X 分布（20px 桶）
    from collections import Counter
    x_vals = [w[0] for w in content_words]
    buckets = Counter(int(x // 20) * 20 for x in x_vals)
    print(f"  X range: {min(x_vals):.0f} – {max(x_vals):.0f}")
    print("  X distribution (x_start: count | bar):")
    for bucket in sorted(buckets):
        bar = '#' * min(buckets[bucket], 50)
        print(f"    {bucket:4d}: {buckets[bucket]:3d} {bar}")

    # 显示前20行内容（按 y 排序）
    print("  Sample lines (first 20):")
    lines = get_line_texts(page)
    for line_ws in lines[:20]:
        text = ' '.join(w[4] for w in line_ws)
        x0 = line_ws[0][0]
        print(f"    x={x0:6.1f}  '{text[:90]}'")


def main():
    doc = fitz.open(str(PDF_PATH))
    print(f"AKT 5: {len(doc)} pages")

    # 扫描第 41-100 页（0-indexed），找 tablet header 并检查后续内容页
    content_pages_inspected = 0
    i = 41
    while i < min(len(doc), 120) and content_pages_inspected < 6:
        page = doc[i]
        lines = get_line_texts(page)
        for line_ws in lines:
            text = ' '.join(w[4] for w in line_ws)
            if TABLET_HEADER_RE.match(text.strip()):
                print(f"\n{'='*60}")
                print(f"TABLET HEADER on page {i+1}: '{text.strip()[:80]}'")
                # 检查这一页和下一页的 x 分布
                inspect_content_page(doc, i)
                if i + 1 < len(doc):
                    inspect_content_page(doc, i + 1)
                content_pages_inspected += 1
                break
        i += 1

    doc.close()


if __name__ == '__main__':
    main()
