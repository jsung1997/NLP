from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
import sacrebleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Load test data
df = pd.read_csv("eng_kor_data.csv")
df = df[df["src"].notnull() & df["tgt"].notnull()]
df = df[df["src"].str.strip() != ""]
df = df[df["tgt"].str.strip() != ""]

# Load the student model
model_dir = "./student_nllb_enko_distilled"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

# Set source and target language
tokenizer.src_lang = "eng_Latn"
tokenizer.tgt_lang = "kor_Hang"

# Get forced_bos_token_id manually
forced_bos_token_id = tokenizer("", return_tensors="pt")["input_ids"][0][0].item()

# Example sentence
sentence = "This is never going to end."
inputs = tokenizer(sentence, return_tensors="pt").to(device)


# Evaluation loop
predictions = []
references = []

for i, row in df.sample(n=100).iterrows():  # Use a sample of 100 for speed; increase if needed
    input_ids = tokenizer(row["src"], return_tensors="pt").to(device)
    output_ids = model.generate(
        **input_ids,
        forced_bos_token_id=forced_bos_token_id,
        max_length=128,
        no_repeat_ngram_size=3
    )
    pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    ref = row["tgt"].strip()

    predictions.append(pred)
    references.append(ref)

    print(f"🔹 EN: {row['src']}")
    print(f"🔁 Predicted: {pred}")
    print(f"✅ Ground Truth: {ref}\n")

# Compute sacreBLEU
bleu = sacrebleu.corpus_bleu(predictions, [references])
print(f"\n🎯 SacreBLEU score: {bleu.score:.2f}")

# Generate prediction
outputs = model.generate(
    **inputs,
    forced_bos_token_id=forced_bos_token_id,
    max_length=128
)
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print("🔁 Predicted:", result)
