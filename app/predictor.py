# app/predictor.py
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
MODEL_PATH = "model/best_model"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def classify_article(text: str):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
    pred = int(torch.argmax(outputs.logits, dim=1).item())
    label = "Fake" if pred==1 else "Real"
    return {"label": label, "pred": pred, "probs": probs}
