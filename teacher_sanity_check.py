from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load 1.3B teacher model

# model_name = "facebook/nllb-200-1.3B"

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

# ✅ Set the proper tokenizer languages
tokenizer.src_lang = "eng_Latn"
tokenizer.tgt_lang = "kor_Hang"

# ✅ Tokenize input without any language prefix
text = "Today is June 18th and it is Muiriel's birthday."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cuda")

# ✅ Get correct BOS token ID for the decoder
forced_bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tgt_lang)

# ✅ Generate
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_length=128,
        num_beams=5,
        early_stopping=True,
        forced_bos_token_id=forced_bos_token_id,
    )

    translated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(text)
print("🔁 Translated:", translated)
