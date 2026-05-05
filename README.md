---
# StayEasy RAG — Vacation Rental Q&A Chatbot

> Ask any question about StayEasy's policies, pricing, and features — get answers backed by the exact source document and page.

---

## The Story Behind This

This was my first RAG project, built while learning from Ed Donner's LLM Engineering course (Week 5). The goal was simple: take a set of company documents and make them conversational.

It taught me the fundamentals — chunking, embeddings, vector search, retrieval, generation. Everything I later improved in BenefitsAI (hybrid search, reranking, RAGAS evaluation) started with the lessons from building this.

---

## How It Works

```
9 markdown documents → markdown-aware chunking → Sentence Transformer embeddings → ChromaDB → GPT-4o-mini answer
```

### Chunking Strategy
Unlike naive fixed-size chunking, this uses **markdown-aware hierarchical splitting**:
1. Split by H2 headings first — keeps sections together
2. If section > 800 chars, split at H3 boundaries
3. If still too large, split at paragraph breaks

Each chunk stores: source filename, heading path, chunk ID, text. This means retrieval results tell you exactly where in the document the answer came from.

### Retrieval
- Embed the question using `all-MiniLM-L6-v2` (SentenceTransformers, runs locally)
- Query ChromaDB for top 5 most similar chunks
- Pass chunks as context to GPT-4o-mini

### Generation
- GPT-4o-mini generates the answer from retrieved chunks only
- Answer is grounded — if it's not in the documents, it says so

---

## Evaluation

Built a custom evaluation pipeline (`evaluate.py`) with an LLM-as-judge approach across 10 test cases covering all 9 source documents.

| Metric | Score |
|--------|-------|
| Retrieval Accuracy | 10/10 (100%) |
| Mean Reciprocal Rank (MRR) | 0.95 |
| Avg Answer Relevance | 5.0 / 5 |
| Avg Answer Correctness | 4.9 / 5 |
| Avg Faithfulness | 5.0 / 5 |
| **Overall Score** | **98.6 / 100** |

**How scoring works:** Five equally weighted components (20% each) — MRR, relevance, correctness, faithfulness, retrieval hit rate. GPT-4o-mini acts as judge for the answer quality metrics.

---

## What I Learned Here vs BenefitsAI

| | StayEasy | BenefitsAI |
|--|---------|-----------|
| Retrieval | Vector search only | Hybrid (BM25 + vector) |
| Reranking | None | Cross-encoder reranking |
| Chunking | Markdown-aware, 800 chars | Fixed 400 chars with overlap |
| Evaluation | Custom LLM judge | RAGAS framework |
| Eval score | 98.6/100 | 0.95/1.0 |
| Multi-user | Shared DB | Per-session isolated DB |

StayEasy scored higher on evaluation — but it's a controlled domain (9 known documents, 10 test cases). BenefitsAI handles arbitrary user-uploaded PDFs, which is a harder problem.

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Documents | 9 markdown files (company policies, pricing, host/guest guides) |
| Chunking | Custom markdown-aware splitter |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) — local, free |
| Vector DB | ChromaDB |
| LLM | GPT-4o-mini (OpenAI) |
| UI | Gradio |
| Evaluation | Custom LLM-as-judge pipeline |

---

## Running Locally

```bash
git clone https://github.com/archana-gurimitkala/stayeasy-rag
cd stayeasy-rag
pip install -r requirements.txt

# Add your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env

# Ingest documents into ChromaDB
python ingest.py

# Run the app
python app.py
# or CLI mode:
python answer.py
```

---

## Files

| File | Purpose |
|------|---------|
| `ingest.py` | Chunks documents, creates embeddings, stores in ChromaDB |
| `answer.py` | CLI chat interface |
| `app.py` | Gradio web UI with chat + evaluation tabs |
| `evaluate.py` | Runs 10-question evaluation, saves results to JSON |
| `data/` | 9 markdown documents (company, pricing, policies, etc.) |
| `evaluation_results.json` | Last evaluation run results |

---

Built by [Archana Gurimitkala](https://github.com/archana-gurimitkala)

*This was the starting point. [BenefitsAI](https://github.com/archana-gurimitkala/-benefits-copilot-) is where it went next.*
