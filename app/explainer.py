# app/explainer.py
import os
import openai
from dotenv import load_dotenv
import os
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def format_sources_context(ctx_list):
    blocks = []
    for i, r in enumerate(ctx_list, start=1):
        blocks.append(f"{i}. ({r['meta']['doc_id']}) {r['text'][:400]}...")
    return "\n\n".join(blocks)

PROMPT_TEMPLATE = """You are an evidence-based fact-checking assistant.

User article:
{article}

Trusted evidence snippets:
{evidence}

Task:
- Decide whether the article is likely REAL or FAKE.
- Explain the reasons succinctly referencing the evidence.
- If possible, provide a suggested correction/summary.

Answer in JSON with keys: verdict, confidence (0-100), explanation, suggested_correction.
"""

def explain_with_llm(article: str, contexts: list):
    evidence = format_sources_context(contexts)
    prompt = PROMPT_TEMPLATE.format(article=article, evidence=evidence)
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=400
    )
    text = resp.choices[0].message['content'].strip()
    return text
