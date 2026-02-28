import re

# We will just test pure string replacement exactly like the notebook does using pandas .str.replace
def test_regex():
    tests = [
        # Quotes and marks (Keep quotes and apostrophes)
        ("He said 'hello' fem.", "He said 'hello' "),
        ('"this is a test"', '"this is a test"'),
        
        # Optional words (you / she -> you)
        ("you / she brought", "you brought"),
        
        # Fractions
        ("0.5 shekels", "½ shekels"),
        ("1 / 12 shekel", "⅔ shekel 15 grains"),
        ("1 / 12 (shekel)", "⅔ shekel 15 grains"),
        
        # specific removals
        ("fem. sing. pl. plural (?)", "    "),
        ("here is a << >> text < >", "here is a text "),
        
        # words
        ("-gold from there", "pašallum gold from there"),
        ("-tax of mine", "šadduātum tax of mine"),
        ("textiles are great", "kutānum textiles are great"),
        
        # Host specific comment:
        ("31 kutānu-textiles of Iddin-Suen", "31 kutānu textiles of Iddin-Suen"),
        ("14 textiles import-tax", "14 kutānum textiles import šadduātum tax"), 
    ]

    print("\\n--- TEST RESULTS ---")
    for orig, expected in tests:
        s = orig
        
        # 1. Grammar parens
        s = re.sub(r'(?<!\w)fem\.\s*', '', s)
        s = re.sub(r'(?<!\w)sing\.\s*', '', s)
        s = re.sub(r'(?<!\w)pl\.\s*', '', s)
        s = re.sub(r'(?<!\w)plural(?!\w)\s*', '', s)
        s = re.sub(r'\(\?\)', '', s)
        s = re.sub(r'<<\s*>>', '', s)
        s = re.sub(r'(?<!gap)(?<!<)<(?!gap)(?!<)\s*>', '', s)

        # Quotes are KEPT now!
        # s = re.sub(_QUOTES_RE, "", s) # removed
        
        # 2. specific words replacements
        s = re.sub(r'(?<!\w)-gold(?!\w)', 'pašallum gold', s)
        s = re.sub(r'(?<!\w)-tax(?!\w)', 'šadduātum tax', s)
        # Note: the original notebook had r'(?<!kutānum )(?<!\w)textiles(?!\w)' -> 'kutānum textiles'
        s = re.sub(r'(?<!kutānum )(?<!\w)textiles(?!\w)', 'kutānum textiles', s)

        # 3. specific math / fractions
        s = re.sub(r'1\s*/\s*12\s*\(?shekel\)?', '⅔ shekel 15 grains', s)
        s = re.sub(r'(?<=[a-zA-Z])\s*/\s*[a-zA-Z]+', '', s)
        
        # 4. the NEW textile fix
        s = re.sub(r'(?<!\w)-?textiles(?!\w)', 'textiles', s)
        s = re.sub(r'kutānu-textiles', 'kutānu textiles', s)

        # Fractions
        s = re.sub(r'(?<!\d)0\.5(?!\d)', '½', s)

        if s.strip() == expected.strip():
            print(f"[PASS] {orig[:40]:<40} -> {s.strip()}")
        else:
            print(f"[FAIL] {orig[:40]:<40}")
            print(f"       Expected: {expected.strip()}")
            print(f"       Got     : {s.strip()}")

if __name__ == '__main__':
    test_regex()
