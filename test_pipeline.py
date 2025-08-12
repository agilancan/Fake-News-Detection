from app.predictor import classify_article
from app.retriever import get_context
from app.explainer import explain_with_llm

article = "Anonymous post claims vaccine X reverses aging in humans."

print("Running classifier...")
cls = classify_article(article)
print(cls)

print("Retrieving contexts...")
contexts = get_context(article)
for i,c in enumerate(contexts):
    print(i, c['meta'], c['text'][:200])

print("Getting explanation from LLM...")
exp = explain_with_llm(article, contexts)
print(exp)
