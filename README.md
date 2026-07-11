# 🐉 Fire & Blood - Production RAG Chatbot

An end-to-end **Production-Ready Retrieval-Augmented Generation (RAG)** application that allows users to ask natural language questions about George R.R. Martin's **Fire & Blood** book.

The application combines **OpenAI Embeddings**, **ChromaDB**, **Hybrid Retrieval (Semantic + BM25)**, **Query Rewriting**, and **GPT-4o-mini** to generate accurate, context-aware answers grounded only in the uploaded document.

---

# 🚀 Features

- 📄 PDF Document Ingestion
- ✂️ Intelligent Text Chunking
- 🔢 OpenAI Embedding Generation
- 🗄️ ChromaDB Vector Database
- 🔍 Semantic Search
- 🔎 BM25 Keyword Search
- ⚡ Hybrid Retrieval (Semantic + Keyword)
- 🔄 Query Rewriting
- 💬 Conversational RAG Support
- 🤖 GPT-4o-mini Answer Generation
- 📚 Source Chunk Display
- 📊 Retrieval Performance Metrics
- 📝 Application Logging
- 🎨 Streamlit User Interface
- 🏗️ Modular Production Architecture

---

# 🏗️ Project Architecture

```
                        User Question
                              │
                              ▼
                     Query Rewriting
                              │
                              ▼
                  Hybrid Retriever
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
    Semantic Retrieval         BM25 Retrieval
        (ChromaDB)             (Keyword Search)
             │                         │
             └────────────┬────────────┘
                          ▼
                  Merge Retrieved Chunks
                          │
                          ▼
                   Prompt Construction
                          │
                          ▼
                    GPT-4o-mini LLM
                          │
                          ▼
                 Context-Aware Response
                          │
                          ▼
              Answer + Sources + Metrics
```

---

# 📂 Project Structure

```
GEN_AI_PDF_CHATBOT/

│
├── app.py
│
├── config.py
│
├── data/
│     └── Fire_and_Blood.pdf
│
├── ingestion/
│     ├── pdf_loader.py
│     └── chunker.py
│
├── embedding/
│     └── embedding_generator.py
│
├── vectordb/
│     └── chroma_manager.py
│
├── retrieval/
│     ├── semantic_retriever.py
│     ├── bm25_retriever.py
│     └── hybrid_retriever.py
│
├── llm/
│     ├── llm_manager.py
│     ├── prompt_builder.py
│     └── query_rewriter.py
│
├── pipeline/
│     └── indexing_pipeline.py
│
├── services/
│     └── rag_service.py
│
├── models/
│     ├── embedded_chunk.py
│     └── retrieved_chunk.py
│
├── utils/
│     └── logger.py
│
├── logs/
│
├── chroma_db/
│
└── requirements.txt
```

---

# ⚙️ Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Database | ChromaDB |
| Keyword Retrieval | BM25 (rank-bm25) |
| UI | Streamlit |
| Tokenization | tiktoken |
| Logging | Python Logging |
| PDF Processing | PyMuPDF |
| Retrieval Strategy | Hybrid Retrieval |

---

# 🔄 End-to-End Workflow

## Step 1 — PDF Loading

The application loads the Fire & Blood PDF and extracts text page by page.

↓

## Step 2 — Chunking

Large pages are split into overlapping chunks for better semantic retrieval.

↓

## Step 3 — Embedding Generation

Each chunk is converted into vector embeddings using:

```
text-embedding-3-small
```

↓

## Step 4 — Indexing

The application builds:

- ChromaDB Vector Store
- BM25 Keyword Index

↓

## Step 5 — User Query

The user asks a natural language question.

↓

## Step 6 — Query Rewriting

The original question is rewritten into a standalone, retrieval-friendly query.

↓

## Step 7 — Hybrid Retrieval

The system performs:

- Semantic Search (Vector Similarity)
- Keyword Search (BM25)

The retrieved results are merged before prompt generation.

↓

## Step 8 — Prompt Building

The prompt contains:

- Conversation History
- Retrieved Context
- Original Question
- Rewritten Question

↓

## Step 9 — LLM Response

GPT-4o-mini generates an answer using **only the retrieved document context**.

↓

## Step 10 — Display Results

The application displays:

- Final Answer
- Source Chunks
- Retrieval Metrics
- Similarity Score
- Retrieval Time
- Generation Time

---

# 📊 Performance Metrics

The application displays:

- Number of Retrieved Chunks
- Retrieval Time
- Generation Time
- Similarity Score
- Query Rewriting Information

---

# 📸 Sample Questions

```
Who killed Rhaenyra?

Who was Viserys I?

Who rode the dragon Vermithor?

Why did the Dance of the Dragons begin?

Who was the heir after Viserys?

Tell me about Aegon II.
```

---

# 📦 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/fire-and-blood-rag.git

cd fire-and-blood-rag
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```
OPENAI_API_KEY=your_openai_api_key
```

Run the application

```bash
streamlit run app.py
```

---

# 🔐 Environment Variables

```
OPENAI_API_KEY=your_api_key
```

---

# 📈 Future Enhancements

- Cross Encoder Re-ranking
- Multi PDF Support
- Metadata Filtering
- Streaming Responses
- FastAPI Backend
- Docker Deployment
- Authentication
- Evaluation Framework
- Cloud Deployment (Azure / AWS)

---

# 🎯 Learning Outcomes

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG)
- Vector Embeddings
- Semantic Search
- Hybrid Retrieval
- Prompt Engineering
- Query Rewriting
- Context Injection
- ChromaDB
- Streamlit Application Development
- Production-Style Project Architecture
- Logging & Performance Monitoring

---

# 👨‍💻 Author

**Nagaraj K**

Generative AI Engineer

### Skills

- Python
- Generative AI
- OpenAI
- LangChain
- Retrieval-Augmented Generation (RAG)
- Prompt Engineering
- ChromaDB
- Streamlit
- Machine Learning
- LLM Applications

---

# ⭐ If you found this project useful

Please consider giving the repository a ⭐ on GitHub.
