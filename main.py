# main.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from app.predictor import classify_article
from app.retriever import get_context
from app.explainer import explain_with_llm

app = FastAPI()

class ArticleRequest(BaseModel):
    text: str

@app.post("/analyze")
def analyze(req: ArticleRequest):
    text = req.text
    cls = classify_article(text)
    contexts = get_context(text, k=3)
    explanation = explain_with_llm(text, contexts)
    return {
        "classifier": cls,
        "contexts": contexts,
        "explanation": explanation
    }

if __name__ == "__main__":
    # CLI mode
    text = input("Paste article text: ")
    cls = classify_article(text)
    print("Classifier:", cls)
    contexts = get_context(text, k=3)
    for c in contexts: print("->", c['meta'], c['text'][:200])
    print("LLM explanation:")
    print(explain_with_llm(text, contexts))
