# 📰 Fake News Detection - Backend

This is the **backend** for our Fake News Detection application, powered by a fine-tuned transformer model and an integrated **Retrieval-Augmented Generation (RAG)** system.  
The backend handles **fake news classification**, **trusted source retrieval** from Politifact, Snopes, and Wikipedia, and serves results to the frontend (deployed on Vercel).  

---

## 🚀 Features

### 1. **Fake News Classification (LLM)**
- Fine-tuned **RoBERTa-base** transformer model for binary classification (Real / Fake) => https://huggingface.co/FacebookAI/roberta-base.
- Trained on a labeled dataset of news articles => https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?resource=download.
- Preprocessing includes title + article text concatenation.

### 2. **Trusted Source Retrieval (RAG)**
- Scrapes and stores verified article snippets from:
  - ✅ [Politifact](https://www.politifact.com/)
  - ✅ [Snopes](https://www.snopes.com/)
  - ✅ [Wikipedia](https://www.wikipedia.org/) (verified topics)
- Embeds text using `sentence-transformers` and stores vectors in **FAISS**.
- Metadata includes article source, snippet text, and URL.
- Retrieval system fetches most relevant verified sources for user queries.

### 3. **Cloud Deployment**
- **FAISS index** (`.faiss`) and metadata (`.pkl`) stored on **Google Cloud Storage (GCS)**.
- Hugging Face Hub hosts the trained model for scalable inference => https://huggingface.co/rodrangal/my_roberta_model.
- Backend deployed via **Render**, connected to frontend via REST API => Backend URL: https://fake-news-detect-cx8p.onrender.com, Frontend URL: https://fake-news-detect.vercel.app

---

## 🏗 Architecture

```plaintext
[ User ] → [ Frontend (Vercel) ] → [ Backend (Render) ]
                                ↳ [ Hugging Face Model Hosting ]
                                ↳ [ GCS: FAISS + Metadata ]

```
### 4. **Project Structure**
```
.
├── data/
│   ├── raw/                # Original dataset
│   └── vector_store/       # FAISS + metadata
├── model/
│   ├── train_classifier.py # Model fine-tuning script
│   └── inference.py        # Model inference logic
├── rag/
│   ├── scrape_sources.py   # Politifact, Snopes, Wikipedia scrapers
│   ├── build_faiss.py      # Create FAISS index + metadata
│   └── query_faiss.py      # Retrieve relevant trusted sources
├── app.py                  # Flask/FastAPI backend API
├── requirements.txt
└── README.md
```

### 5. **Setup & Installation**

1️⃣ Clone Repository
- git clone https://github.com/agilancan/Fake-News-Detection.git
- cd Fake-News-Detection

2️⃣ Create Virtual Environment
- python -m venv venv
- source venv/bin/activate  # Mac/Linux
- venv\Scripts\activate     # Windows

3️⃣ Install Dependencies
- pip install -r requirements.txt

4️⃣ Environment Variables

Create a .env file:
- GCP_BUCKET_NAME=your-bucket
- GCP_CREDENTIALS_PATH=path/to/service_account.json
- HF_MODEL_REPO=your-username/fake-news-model

🖥 Running the Backend Locally
- python app.py

The API will be available at http://127.0.0.1:8000

Endpoints:
- POST /predict → Classify news article as Real/Fake
- POST /retrieve → Retrieve related verified sources
- POST /rag → Combined classification + retrieval

☁️ Cloud Deployment

Model Hosting
- Model pushed to Hugging Face Hub for external inference.

Vector Store Hosting
- FAISS index + metadata uploaded to Google Cloud Storage.
- Backend downloads and loads index at startup.

API Hosting
- Backend deployed on Render.
- Connected to frontend hosted on Vercel.

### 6. **Development Team**
- Backend, Model, RAG & Infrastructure => Agilan Sivakumaran, Rodrigo Rangel-alvarado
- Frontend, Intergration & Deployment => Thedyson eduard Luzon, Yash Patel

### 7. **Development Team**
```
| Component     | Choice             | Reason                               |
| ------------- | ------------------ | ------------------------------------ |
|   Model       | RoBERTa-base       | High accuracy for NLP classification |
|   Vector DB   | FAISS              | Fast, lightweight, easy to deploy    |
|   Cloud       | GCS + Hugging Face | Scalability, separation of concerns  |
|   Backend     | FastAPI/Flask      | Lightweight, async-ready API layer   |
```

### 8. **Demo Video**
[![Watch the video](https://img.youtube.com/vi/i7ZSjIbGSLk/hqdefault.jpg)](https://youtu.be/i7ZSjIbGSLk)
