<div align="center">

# 📄 PaperTrail
### A Transparent, Beginner-Friendly RAG Research Paper Assistant

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/LLM%20Inference-Groq%20Cloud-f55036?logo=speedtest&logoColor=white)](https://groq.com/)
[![Sentence Transformers](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-yellow)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![NumPy](https://img.shields.io/badge/Similarity-Pure%20NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<p align="center">
  <b>Upload research papers in PDF format, ask natural language questions, and receive accurate answers strictly grounded in the paper with verifiable <code>[Page X]</code> citations.</b>
</p>

[Key Features](#-key-features) •
[Why PaperTrail?](#-why-papertrail) •
[Architecture & Pipeline](#-architecture--how-it-works) •
[The Two-Model Setup](#-the-two-model-setup-explained) •
[Quickstart](#-quickstart-guide) •
[Code Walkthrough](#-code-walkthrough) •
[Testing Checklist](#-manual-testing-checklist)

</div>

---

## 💡 Why PaperTrail?

Most modern Retrieval-Augmented Generation (RAG) tutorials hide everything behind monolithic orchestration frameworks like **LangChain** or **LlamaIndex** and external vector databases like Chroma or Pinecone. While convenient for production, this creates a **"black box"** for beginners learning Machine Learning and NLP.

**PaperTrail strips away the abstractions:**
- ❌ **No LangChain or LlamaIndex:** Every step of the RAG pipeline is implemented directly in plain Python.
- ❌ **No external vector database:** Embeddings remain in-memory and similarity search is computed using pure **NumPy** matrix dot products.
- ✅ **100% Transparent:** You can read, debug, and inspect every single step — from PDF parsing to chunking, vector normalization, cosine ranking, and prompt formatting.
- ✅ **Verifiable Evidence:** An expandable evidence inspector lets you view the exact text chunks, similarity scores, and page numbers retrieved for each question.

---

## ✨ Key Features

- **📑 Page-Preserving Text Extraction:** Extracts text page-by-page using `pypdf`, retaining 1-indexed page metadata across all chunks for grounded citations.
- **🛡️ Scanned & Empty PDF Guard:** Detects image-only or corrupt PDFs with no selectable text and shows actionable error warnings.
- **🧩 Overlapping Text Chunking:** Configurable sliding-window chunking (default: 1,000 characters with 150 characters of overlap) prevents semantic loss across boundaries.
- **⚡ Local Vector Embeddings:** Encodes chunks into 384-dimensional dense vectors using Hugging Face's `all-MiniLM-L6-v2` locally via `sentence-transformers`.
- **📐 Mathematical Cosine Similarity:** Normalizes embeddings so that cosine similarity is computed directly via fast NumPy dot products (`u · v`).
- **🚀 Blazing-Fast Answer Generation:** Queries Groq's high-speed inference engine with strict prompts to enforce grounded answers and prevent hallucinations.
- **🔍 Retrieved Evidence Viewer:** Expandable UI card in Streamlit displaying retrieved chunk text, source page numbers, and cosine similarity scores.
- **🔒 Session State Isolation:** Caches models with `@st.cache_resource` and automatically recalculates embeddings only when a new PDF is uploaded.

---

## 🧠 Architecture & How It Works

```
                                 OFFLINE / INGESTION STAGE
                                 
  [ Uploaded PDF ] 
         │
         ▼
  [ pypdf: Page Text Extractor ]  ──>  Extracts text per page, preserving 1-indexed page numbers
         │
         ▼
  [ Text Chunker ]               ──>  Sliding window (1,000 chars, 150 overlap) with page metadata
         │
         ▼
  [ all-MiniLM-L6-v2 ]           ──>  Local model generates normalized 384-D chunk embeddings
         │
         ▼
  [ In-Memory Matrix (NumPy) ]   ──>  Stored in session state (shape: [N, 384])
  
─────────────────────────────────────────────────────────────────────────────────────────────
                                 QUERY & RETRIEVAL STAGE
                                 
  [ User Question ]
         │
         ▼
  [ all-MiniLM-L6-v2 ]           ──>  Encodes question into normalized 384-D query vector
         │
         ▼
  [ NumPy Dot Product ]          ──>  cosine_similarity = np.dot(chunk_embeddings, query_vector)
         │
         ▼
  [ Top-K Ranker ]               ──>  Retrieves top 3 most similar chunks + page metadata
  
─────────────────────────────────────────────────────────────────────────────────────────────
                                    GENERATION STAGE
                                    
  [ Retrieved Chunks + Question ]
         │
         ▼
  [ Strict Grounded Prompt ]     ──>  Treats paper text as untrusted reference context
         │
         ▼
  [ Groq Chat Completions API ]  ──>  Fast LLM inference (e.g. openai/gpt-oss-120b, qwen/qwen3.8-27b)
         │
         ▼
  [ Grounded Answer ]            ──>  Answer with verifiable [Page X] citations
```

---

## 🤖 The Two-Model Setup Explained

A common point of confusion for beginners is why PaperTrail uses **two different models**:

| Component | Model Name | Where It Runs | What It Does |
| :--- | :--- | :--- | :--- |
| **Embedding Model** | `all-MiniLM-L6-v2` | **Locally on your CPU/GPU** | Converts text into 384 numbers (vectors). It **cannot write text or converse**; it only measures semantic similarity. |
| **Generative LLM** | `openai/gpt-oss-120b` or `qwen/qwen3.8-27b` | **In the Cloud (Groq API)** | Reads the retrieved chunks and writes a fluent, grounded answer citing `[Page X]`. |

```text
User Question ──> [ all-MiniLM-L6-v2 ] ──> Finds Top 3 Chunks ──> [ Groq LLM ] ──> Final Answer
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Partha-zzz/Papertrail.git
cd Papertrail
```

### 2. Set Up a Virtual Environment

**On Windows (PowerShell):**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

**On macOS / Linux (bash/zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Your Groq API Key
1. Get a free API key from the [Groq Console](https://console.groq.com/keys).
2. Create your `.env` configuration file from the template:
   ```bash
   # Windows PowerShell
   cp .env.example .env

   # macOS / Linux
   cp .env.example .env
   ```
3. Open `.env` and insert your API key:
   ```env
   GROQ_API_KEY=gsk_your_actual_key_here
   GROQ_MODEL=openai/gpt-oss-120b
   ```

### 5. Launch PaperTrail
```bash
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501`.

---

## 🔍 Code Walkthrough

The entire application logic is contained in [`app.py`](app.py) for maximum educational clarity:

```text
PaperTrail/
├── app.py              # Main RAG pipeline and Streamlit web application
├── requirements.txt    # Project dependencies
├── .env.example        # Environment variable template
├── .env                # Local secrets (never committed to Git)
├── .gitignore          # Ignores .venv, .env, and caches
└── README.md           # Documentation
```

### Core Pipeline Functions:

| Function in `app.py` | One-Sentence Summary |
| :--- | :--- |
| `load_environment()` | Loads configuration variables and credentials from the `.env` file into the environment. |
| `get_embedding_model()` | Loads and caches the `SentenceTransformer` model in memory so it is only loaded once. |
| `extract_text_from_pdf()` | Extracts text page-by-page from an uploaded PDF while preserving 1-indexed page numbers. |
| `chunk_text()` | Splits page text into smaller, overlapping chunks while preserving original page metadata. |
| `generate_chunk_embeddings()` | Generates normalized vector embeddings for a list of document chunks using the embedding model. |
| `retrieve_relevant_chunks()` | Finds and ranks the top-k most relevant chunks for a question using cosine similarity. |
| `generate_answer_with_groq()` | Sends the retrieved evidence and user question to the Groq API to generate a grounded answer. |
| `main()` | Orchestrates the Streamlit UI, file upload lifecycle, and question-answering workflow. |

---

## 📐 The Mathematics: Why Dot Product Equals Cosine Similarity

Cosine similarity measures the angular distance between two vectors:

$$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$

When vectors are **unit-normalized** (L2 norm = 1.0, so $\|\vec{u}\| = 1$ and $\|\vec{v}\| = 1$):

$$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \vec{u} \cdot \vec{v}$$

In PaperTrail, we pass `normalize_embeddings=True` to `sentence-transformers`. This allows us to compute cosine similarity across all $N$ chunks in one single NumPy matrix-vector multiplication:

```python
# chunk_embeddings is (N, 384), question_embedding is (384,)
# similarity_scores is (N,) representing cosine similarity for each chunk
similarity_scores = np.dot(chunk_embeddings, question_embedding)
```

---

## ⚙️ Configuration Options

You can adjust parameters at the top of [`app.py`](app.py) or via `.env`:

| Parameter | Default | Location | Description |
| :--- | :--- | :--- | :--- |
| `CHUNK_SIZE` | `1000` | `app.py` | Maximum characters per chunk (~150-200 words). |
| `CHUNK_OVERLAP` | `150` | `app.py` | Overlap characters between chunks to preserve sentence context. |
| `TOP_K_RETRIEVAL`| `3` | `app.py` | Number of top evidence chunks sent to the LLM. |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | `app.py` | Hugging Face model ID for local embeddings. |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | `.env` | Model ID on Groq (`openai/gpt-oss-120b`, `qwen/qwen3.8-27b`). |

---

## ✅ Manual Testing Checklist

- [x] **Startup Test:** Run `streamlit run app.py` and confirm the UI renders cleanly with the sidebar settings.
- [x] **API Key Validation:** Verify that missing API keys display a warning in the sidebar and block queries gracefully.
- [x] **PDF Text Extraction:** Upload an arXiv paper and verify the page and chunk count in the green success notification.
- [x] **Scanned PDF Handling:** Upload an image-only PDF and confirm that PaperTrail alerts you that no text was found.
- [x] **Semantic Retrieval:** Check the *"🔍 View Retrieved Evidence"* drawer to verify that retrieved chunks match the query context.
- [x] **Citation Accuracy:** Verify that the generated answer includes `[Page X]` tags corresponding directly to the evidence chunks.
- [x] **Hallucination Fallback:** Ask an out-of-scope question (e.g. *"What is the capital of France?"*) and verify the assistant answers: *"The answer cannot be determined from the uploaded paper."*

---

## 🛠️ Troubleshooting & FAQ

<details>
<summary><b>1. Error: <code>ModuleNotFoundError: No module named '...'</code></b></summary>
Ensure your virtual environment is activated before starting Streamlit:
```powershell
.venv\Scripts\Activate.ps1   # Windows PowerShell
source .venv/bin/activate    # macOS / Linux
```
</details>

<details>
<summary><b>2. Error: <code>groq.NotFoundError: The model does not exist</code></b></summary>
The model configured in <code>.env</code> might not be enabled for your Groq tier. Update <code>.env</code> to one of the verified active models:
```env
GROQ_MODEL=openai/gpt-oss-120b
# or
GROQ_MODEL=qwen/qwen3.8-27b
```
</details>

<details>
<summary><b>3. Why is the first query slow?</b></summary>
On the first run, <code>sentence-transformers</code> downloads the <code>all-MiniLM-L6-v2</code> weights (~90 MB) into your local Hugging Face cache. Subsequent runs load instantaneously from disk.
</details>

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). Built for learning and educational purposes.
