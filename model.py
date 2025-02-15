from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_path = r"C:\Users\HP\Downloads\final\LaMini-T5-738M"

# Load locally to verify
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("Model loaded successfully!")
