# app/predictor.py
import os
from dotenv import load_dotenv
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from google.cloud import storage
import torch

# Load .env first so GOOGLE_APPLICATION_CREDENTIALS is available
load_dotenv()

# Ensure absolute path for credentials
creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not os.path.isabs(creds_path):
    creds_path = os.path.abspath(creds_path)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

print(f"Using GCP credentials from: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")


MODEL_PATH = "model/best_model"
BUCKET_NAME = "assignment_2_bucket_rodrigo"
GCS_FOLDER = "model/best_model"

def download_model_from_gcs(bucket_name, source_folder, destination_folder):
    """Download all files from a GCS folder to local destination."""
    os.makedirs(destination_folder, exist_ok=True)
    client = storage.Client()  # This now sees the credentials
    blobs = client.list_blobs(bucket_name, prefix=source_folder)

    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        dest_path = os.path.join(destination_folder, os.path.basename(blob.name))
        if not os.path.exists(dest_path):
            blob.download_to_filename(dest_path)
            print(f"Downloaded {blob.name} to {dest_path}")

if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
    print("Downloading model from GCS...")
    download_model_from_gcs(BUCKET_NAME, GCS_FOLDER, MODEL_PATH)
else:
    print("Model already exists locally, skipping download.")

tokenizer = RobertaTokenizer.from_pretrained("rodrangal/my_roberta_model")
model = RobertaForSequenceClassification.from_pretrained("rodrangal/my_roberta_model")
model.eval()

def classify_article(text: str):
    inputs = tokenizer(
        text, truncation=True, padding=True,
        return_tensors="pt", max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
    pred = int(torch.argmax(outputs.logits, dim=1).item())
    label = "Fake" if pred == 1 else "Real"
    return {"label": label, "pred": pred, "probs": probs}
