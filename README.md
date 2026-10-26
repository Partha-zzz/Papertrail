# PaperTrail 📄 — A Beginner-Friendly RAG Research Paper Assistant

PaperTrail is a lightweight, educational Retrieval-Augmented Generation (RAG) application built with Python and Streamlit. It allows you to upload research papers in PDF format, ask questions about their content, and receive accurate answers strictly grounded in the paper with verifiable **[Page X]** citations.

This project is built from scratch without complex abstractions (no LangChain or LlamaIndex) so you can clearly understand how each stage of a RAG pipeline works under the hood.

---

## 📑 Table of Contents
1. [Core Features](#core-features)
2. [Technology Stack](#technology-stack)
3. [Understanding the RAG Pipeline](#understanding-the-rag-pipeline)
4. [Key Concepts Explained](#key-concepts-explained)
5. [Code Walkthrough (`app.py`)](#code-walkthrough-apppy)
6. [Installation & Setup (Windows & VS Code)](#installation--setup-windows--vs-code)
7. [Groq API Configuration](#groq-api-configuration)
8. [Running the Application](#running-the-application)
9. [Manual Testing Checklist](#manual-testing-checklist)
10. [Troubleshooting & Common Errors](#troubleshooting--common-errors)

---

## 🌟 Core Features

- **PDF Text Extraction with Page Preservation:** Extracts text page-by-page using `pypdf`, retaining 1-indexed page numbers so every cited fact can be verified.
- **Empty PDF & Scanned Detection:** Gracefully detects image-only or scanned PDFs that lack selectable text, alerting the user with an actionable message.
- **Configurable Text Chunking:** Splits text into overlapping sliding-window chunks (default: 1,000 characters with 150 characters of overlap) to capture coherent ideas.
- **Local Embedding Generation:** Uses Hugging Face's `all-MiniLM-L6-v2` via `sentence-transformers` locally to turn text into normalized 384-dimensional semantic vectors.
- **In-Memory Cosine Similarity Search:** Computes cosine similarity scores using pure NumPy dot products without introducing any vector database.
- **Grounded LLM Answer Generation:** Calls the official Groq API (e.g. `openai/gpt-oss-120b` or `qwen/qwen3.8-27b`) with strict instructions to answer *only* from retrieved evidence and to cite sources like `[Page 4]`.
- **Untrusted Input Boundaries:** Treats document contents as untrusted reference data, protecting against prompt injection.
- **Transparent Evidence Inspection:** Displays an expandable viewer showing the exact retrieved chunks, their source page numbers, and their similarity scores.
- **Efficient State Management:** Caches embedding models and reuses embeddings across Streamlit reruns unless a different PDF is uploaded.

---

## 🛠️ Technology Stack

| Technology | Purpose | Why We Use It |
| :--- | :--- | :--- |
| **Python 3.10 - 3.12** | Core language | Clean syntax and wide ML library support. |
| **Streamlit** | Web Interface | Simple, reactive UI in pure Python without writing HTML/JS. |
| **pypdf** | PDF Processing | Lightweight extraction of text and metadata per page. |
| **sentence-transformers** | Embeddings | Generates local semantic vectors using `all-MiniLM-L6-v2`. |
| **NumPy** | Similarity Math | Computes vector dot products and ranks top-k chunks. |
| **Groq Python SDK** | LLM Generation | Ultra-fast inference using open-weights models (e.g., Llama 3.3). |
| **python-dotenv** | Configuration | Safely loads API keys and settings from a local `.env` file. |

---

## 🧠 Understanding the RAG Pipeline

Retrieval-Augmented Generation (RAG) is a technique that combines an external knowledge base with a Large Language Model (LLM). Instead of relying solely on what the model learned during pretraining, RAG retrieves relevant excerpts from your documents and feeds them to the LLM as reference context.

```
+-----------------------------------------------------------------------------------------+
|                                  OFFLINE / INGESTION PHASE                              |
|                                                                                         |
|  [Uploaded PDF] ---> [pypdf: Extract Text per Page] ---> [Chunking with Page Metadata]  |
|                                                                      |                  |
|                                                                      v                  |
|                               [In-Memory Embeddings] <--- [all-MiniLM-L6-v2 Model]      |
+-----------------------------------------------------------------------------------------+
                                                                       |
+----------------------------------------------------------------------|------------------+
|                                    QUERY / RETRIEVAL PHASE           v                  |
|                                                                                         |
|  User Question ---> [all-MiniLM-L6-v2 Model] ---> [Question Vector]  |                  |
|                                                          |           |                  |
|                                                          v           v                  |
|                                      [NumPy Dot Product Cosine Similarity Search]       |
|                                                          |                              |
|                                                          v                              |
|                                               [Top-K Relevant Chunks]                   |
+-----------------------------------------------------------------------------------------+
                                                           |
+----------------------------------------------------------|------------------------------+
|                                      GENERATION PHASE    v                              |
|                                                                                         |
|  [System Prompt + Evidence Chunks + Question] ---> [Groq Chat API: Llama 3.3]           |
|                                                          |                              |
|                                                          v                              |
|                                           [Answer with [Page X] Citations]              |
+-----------------------------------------------------------------------------------------+
```

---

## 📖 Key Concepts Explained

### 1. What is Chunking and Why is it Necessary?
Large language models have finite context windows, and embedding models (like `all-MiniLM-L6-v2`) have a maximum sequence length (typically 256 or 512 tokens).
- If you embed an entire 30-page research paper as one single vector, all nuances and specific facts are averaged out and lost.
- By splitting the paper into smaller chunks (e.g., 1,000 characters), each chunk represents a focused concept or paragraph.
- **Overlap** (e.g., 150 characters) ensures that if an important sentence falls right across a chunk boundary, it is preserved in at least one chunk without breaking context.

### 2. What are Embeddings?
An embedding is a numerical representation of text as a vector of floating-point numbers in a high-dimensional space (384 dimensions for `all-MiniLM-L6-v2`).
- Words and sentences with similar semantic meaning are mapped close to each other in this space.
- For example, *"convolutional neural network architecture"* and *"deep CNN model layers"* will have vectors pointing in nearly the same direction, even if they share few identical words.

### 3. What is Cosine Similarity and How Does Dot Product Work?
Cosine similarity measures the angle between two vectors:
$$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$
- When vectors are **normalized** to unit length ($\|\vec{u}\| = 1$ and $\|\vec{v}\| = 1$), the denominator equals 1.
- The formula simplifies directly to the **dot product**:
$$\text{Cosine Similarity} = \vec{u} \cdot \vec{v}$$
- In NumPy, for a matrix of $N$ chunk embeddings and a question vector $\vec{q}$, the dot product `np.dot(chunk_embeddings, question_embedding)` calculates all $N$ similarities simultaneously in milliseconds.

### 4. What is Top-K Retrieval?
Top-K retrieval selects the $K$ chunks (default: 3) with the highest cosine similarity scores. These chunks represent the most relevant evidence in the paper to answer the question.

### 5. Grounded Generation & Hallucination Prevention
To prevent hallucinations (the LLM making up plausible-sounding false answers):
- We instruct the LLM through a strict system prompt to answer *only* from the retrieved chunks.
- If the answer is not present in the evidence, the LLM is instructed to state that the answer cannot be determined from the paper.
- The LLM is required to explicitly cite the supporting page numbers (e.g., `[Page 4]`).
- *Note:* RAG significantly minimizes hallucination, but no LLM is 100% immune; providing expandable raw evidence allows you to verify every claim.

---

## 🔍 Code Walkthrough (`app.py`)

Here is how data flows through the functions in [app.py](file:///d:/Projects/PaperTrail/app.py):

| Function | What It Does | Input -> Output |
| :--- | :--- | :--- |
| `load_environment()` | Reads configuration from `.env` and environment variables. | None -> `(api_key, model_name)` |
| `get_embedding_model()` | Loads and caches the `SentenceTransformer` model in memory. | None -> `SentenceTransformer` instance |
| `extract_text_from_pdf()` | Extracts text page-by-page from an uploaded PDF while tracking page numbers. | `UploadedFile` -> `[{"page_number": int, "text": str}]` |
| `chunk_text()` | Splits page text into overlapping segments with page numbers. | `pages_data` -> `[{"page_number": int, "text": str}]` |
| `generate_chunk_embeddings()` | Computes normalized vector embeddings for all chunks. | `chunks, model` -> `np.ndarray` of shape `(N, 384)` |
| `retrieve_relevant_chunks()` | Encodes the user's question and ranks chunks by cosine similarity. | `question, chunks, embeddings, model, top_k` -> `[{"text", "page_number", "similarity_score"}]` |
| `generate_answer_with_groq()` | Delimits evidence and prompts the Groq LLM to answer with citations. | `question, retrieved_chunks, api_key, model_name` -> `str` (answer text) |
| `main()` | Orchestrates UI state, uploads, queries, and displays results in Streamlit. | Streamlit App Lifecycle |

---

## 💻 Installation & Setup (Windows & VS Code)

### Prerequisites
- Windows 10/11
- Python 3.10, 3.11, or 3.12 (Python 3.12 recommended)
- VS Code (with Python extension)
- Git (optional)

### Step 1: Open Project in VS Code
Open VS Code, press ``Ctrl + ` `` to open the integrated PowerShell terminal, and navigate to the project directory:
```powershell
cd d:\Projects\PaperTrail
```

### Step 2: Create a Virtual Environment
Create an isolated Python virtual environment named `.venv`:
```powershell
# If you have Python 3.12 installed via Python launcher:
py -3.12 -m venv .venv

# Or using standard python command:
python -m venv .venv
```

### Step 3: Activate the Virtual Environment
```powershell
# In PowerShell:
.venv\Scripts\Activate.ps1

# (If PowerShell blocks script execution, run this once: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser)
```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🔑 Groq API Configuration

1. Visit the [Groq Console](https://console.groq.com/keys) and sign in (free tier available).
2. Click **Create API Key**, copy the key.
3. In the project root, copy `.env.example` to create a `.env` file:
   ```powershell
   cp .env.example .env
   ```
4. Open `.env` and fill in your key:
   ```env
   GROQ_API_KEY=gsk_your_actual_groq_api_key_here
   GROQ_MODEL=openai/gpt-oss-120b
   ```

> [!NOTE]
> Supported models on Groq include:
> - `openai/gpt-oss-120b` (Recommended: high accuracy and grounded reasoning)
> - `qwen/qwen3.8-27b` (Fast, lightweight alternative)

---

## 🚀 Running the Application

With your virtual environment active, start the Streamlit server:

```powershell
streamlit run app.py
```

Streamlit will launch locally in your browser at `http://localhost:8501`.

---

## ✅ Manual Testing Checklist

Follow this checklist to verify your PaperTrail application:

- [ ] **Application Startup:** Run `streamlit run app.py` and confirm the UI loads without syntax or import errors.
- [ ] **Missing API Key Notice:** If `.env` is empty, check that the sidebar warns about a missing API key and the app gracefully informs you if you try to ask a question.
- [ ] **PDF Upload & Text Extraction:** Upload a standard research paper PDF (e.g., an arXiv paper) and verify the green success message displays the correct page count and chunk count.
- [ ] **Scanned/Empty PDF Handling:** Upload an empty PDF or image-only scanned document and confirm the error message: *"No extractable text was found in this PDF..."* appears.
- [ ] **Single-File Session Isolation:** Upload Paper A, verify chunks, then upload Paper B. Verify that Paper B's chunks completely replace Paper A's in session state.
- [ ] **Retrieval Precision:** In the expandable *"View Retrieved Evidence"* dropdown, check that the top 3 chunks correlate directly to your question and display valid page numbers and similarity scores between 0 and 1.
- [ ] **Grounded Citations:** Verify that the Groq answer contains `[Page X]` citations corresponding directly to the evidence displayed in the dropdown.
- [ ] **Out-of-Scope Query:** Ask a question completely unrelated to the paper (e.g., *"What is the recipe for chocolate cake?"*). Verify the model responds that the answer cannot be determined from the paper.

---

## 🔧 Troubleshooting & Common Errors

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named '...'` | Virtual environment is not activated or dependencies are missing. | Run `.venv\Scripts\Activate.ps1` and `pip install -r requirements.txt`. |
| `groq.AuthenticationError: Invalid API Key` | The API key in `.env` is missing, miscopied, or revoked. | Check your `.env` file, ensure no trailing spaces, and generate a new key at [Groq Console](https://console.groq.com/keys). |
| `ExecutionPolicy` error in PowerShell | Windows blocks unapproved scripts by default. | Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`. |
| `No extractable text found in this PDF` | The PDF contains scanned raster images instead of digital text. | Use a PDF with digital selectable text (e.g., from arXiv.org, IEEE, or PubMed). |
| First run is slow | The sentence-transformers embedding model (`all-MiniLM-L6-v2`, ~90MB) downloads on first use. | Wait ~30 seconds for the one-time download to complete; subsequent runs will load instantly from local cache. |

---

## 📜 License
MIT License. Built for educational and learning purposes.
