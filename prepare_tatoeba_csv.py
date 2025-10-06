import pandas as pd

# STEP 1: Set your file paths (edit as needed)
en_path = r"C:\Users\Jay\Documents\GitHub\NLP\tatoeba_en_ko.txt\Tatoeba.en-ko.en"
ko_path = r"C:\Users\Jay\Documents\GitHub\NLP\tatoeba_en_ko.txt\Tatoeba.en-ko.ko"

# STEP 2: Read files
with open(en_path, "r", encoding="utf-8") as f_en:
    en_lines = f_en.readlines()

with open(ko_path, "r", encoding="utf-8") as f_ko:
    ko_lines = f_ko.readlines()

# STEP 3: Check alignment
assert len(en_lines) == len(ko_lines), "Mismatch in number of lines!"

# STEP 4: Create DataFrame
df = pd.DataFrame({
    "src": [s.strip() for s in en_lines],
    "tgt": [t.strip() for t in ko_lines]
})

# STEP 5: Clean and convert
df = df[df["src"].notnull() & df["tgt"].notnull()]
df["src"] = df["src"].astype(str)
df["tgt"] = df["tgt"].astype(str)
df = df[df["src"].str.strip() != ""]
df = df[df["tgt"].str.strip() != ""]

# STEP 6: Filter by sentence length (optional)
df = df[(df["src"].str.split().str.len() > 2) & (df["src"].str.split().str.len() < 80)]
df = df[(df["tgt"].str.split().str.len() > 2) & (df["tgt"].str.split().str.len() < 80)]

# STEP 7: Save to CSV
df.to_csv("eng_kor_data.csv", index=False, encoding="utf-8")
print(f"Saved {len(df)} aligned English–Korean sentence pairs to eng_kor_data.csv")
