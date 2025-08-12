# model/train_classifier.py
from datasets import load_dataset
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import sklearn.metrics as metrics


# --- DEBUG: confirm TrainingArguments is correct at import time ---
print("\n[DEBUG-IMPORT] TrainingArguments source:", TrainingArguments.__module__)
print("[DEBUG-IMPORT] TrainingArguments class name:", TrainingArguments.__name__)
import transformers
print("[DEBUG-IMPORT] transformers.__version__ =", transformers.__version__)
print()

# --- CONFIG ---
MODEL_NAME = "roberta-base"      # or "distilroberta-base" if you need faster
OUTPUT_DIR = "model/best_model"
DATA_PATH = "data/raw/fake_or_real_news.csv"

import os
print("Current working directory:", os.getcwd())
print("Looking for data at:", os.path.abspath(DATA_PATH))

# --- load dataset ---
import pandas as pd
df = pd.read_csv(DATA_PATH)  # expect at least ['title','text','label']
df['text'] = df['title'].fillna('') + ". " + df['text'].fillna('')
df = df.rename(columns={'label':'labels'})  # labels: 0 real, 1 fake (adjust if needed)

print(f"Loaded {len(df)} rows from {DATA_PATH}")
print(df.head(3))  # Show a few rows for sanity check
print(df['labels'].value_counts())  # Confirm labels distribution


df['labels'] = df['labels'].map({'REAL': 0, 'FAKE': 1})



# simple split
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['labels'])

# convert to datasets
from datasets import Dataset
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
def preprocess(ex):
    return tokenizer(ex['text'], truncation=True, padding='max_length', max_length=128)

train_ds = train_ds.map(preprocess, batched=True)
val_ds = val_ds.map(preprocess, batched=True)

train_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
val_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        'accuracy': metrics.accuracy_score(p.label_ids, preds),
        'f1': metrics.f1_score(p.label_ids, preds)
    }

# --- DEBUG: confirm TrainingArguments is still correct before creating it ---
print("[DEBUG-PRE-ARGS] TrainingArguments source:", TrainingArguments.__module__)
print("[DEBUG-PRE-ARGS] TrainingArguments class name:", TrainingArguments.__name__)
print()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    use_cpu=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Train dataset size: {len(train_ds)}")
print(f"Validation dataset size: {len(val_ds)}")
