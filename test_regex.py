import pandas as pd
import re

def test_regex():
    # Setup exactly what the VectorizedPostprocessor does
    empty_fallback = ""
    def postprocess(texts):
        s = pd.Series(texts)
        s = s.fillna("").astype(str)

        forbidden_chars = "()—–<>⌈⌋⌊+ʾ" # keeping '' and ""
        forbidden_trans = str.maketrans("", "", forbidden_chars)

        s = s.str.replace(r'\(d\)', '{d}', regex=True)
        s = s.str.replace(r'\(ki\)', '{ki}', regex=True)
        s = s.str.replace(r'\(TÚG\)', 'TÚG', regex=True)

        s = s.str.replace(r'(?<!\w)fem\.\s*', '', regex=True)
        s = s.str.replace(r'(?<!\w)sing\.\s*', '', regex=True)
        s = s.str.replace(r'(?<!\w)pl\.\s*', '', regex=True)
        s = s.str.replace(r'(?<!\w)plural(?!\w)\s*', '', regex=True)
        s = s.str.replace(r'\(\?\)', '', regex=True)
        s = s.str.replace(r'<<\s*>>', '', regex=True)
        s = s.str.replace(r'(?<!gap)(?<!<)<(?!gap)(?!<)\s*>', '', regex=True)

        s = s.str.replace("<gap>", "\x00GAP\x00", regex=False)
        s = s.str.translate(forbidden_trans)
        s = s.str.replace("\x00GAP\x00", " <gap> ", regex=False)

        # specific words replacements
        s = s.str.replace(r'(?<!\w)-gold(?!\w)', 'pašallum gold', regex=True)
        s = s.str.replace(r'(?<!\w)-tax(?!\w)', 'šadduātum tax', regex=True)
        s = s.str.replace(r'(?<!kutānum )(?<!\w)textiles(?!\w)', 'kutānum textiles', regex=True)

        # explicit fixes we added
        s = s.str.replace(r'1\s*/\s*12\s*\(?shekel\)?', '⅔ shekel 15 grains', regex=True)
        s = s.str.replace(r'(?<=[a-zA-Z])\s*/\s*[a-zA-Z]+', '', regex=True)
        s = s.str.replace(r'(?<!\w)-?textiles(?!\w)', 'textiles', regex=True)
        s = s.str.replace(r'kutānu-textiles', 'kutānu textiles', regex=True)

        # fractions
        s = s.str.replace(r'(?<!\d)0\.5(?!\d)', '½', regex=True)

        # whitespace clean
        s = s.str.replace(r'\s+', ' ', regex=True).str.strip()

        return s.tolist()

    tests = [
        # Quotes and marks
        ("He said 'hello' fem.", "He said 'hello'"),
        ('"this is a test"', '"this is a test"'),
        # Optional words
        ("you / she brought", "you brought"),
        # Fractions
        ("0.5 shekels", "½ shekels"),
        ("1 / 12 shekel", "⅔ shekel 15 grains"),
        ("1 / 12 (shekel)", "⅔ shekel 15 grains"),
        # specific removals
        ("fem. sing. pl. plural (?)", ""),
        ("here is a << >> text < >", "here is a text"),
        # words
        ("-gold from there", "pašallum gold from there"),
        ("-tax of mine", "šadduātum tax of mine"),
        ("textiles are great", "kutānum textiles are great"),
        # Host specific comment:
        ("31 kutānu-textiles of Iddin-Suen, 30 kutānu-textiles of Ah-šalim", "31 kutānu textiles of Iddin-Suen, 30 kutānu textiles of Ah-šalim"),
        ("14 textiles import-tax", "14 kutānum textiles import šadduātum tax"), # import-tax becomes import šadduātum tax? Wait, host says: "14 textiles import-tax" -> "14 textiles import-tax" in his BEFORE / AFTER
    ]

    results = postprocess([t[0] for t in tests])

    print("\n--- TEST RESULTS ---")
    for (orig, expected), res in zip(tests, results):
        if res == expected:
            print(f"[PASS] {orig[:40]:<40} -> {res}")
        else:
            print(f"[FAIL] {orig[:40]:<40}")
            print(f"       Expected: {expected}")
            print(f"       Got     : {res}")

if __name__ == '__main__':
    test_regex()
