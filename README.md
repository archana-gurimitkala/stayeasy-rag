# StayEasy RAG

A chatbot that answers questions about a vacation rental company using RAG (Retrieval-Augmented Generation).

## About This Project

This was my hands-on project for learning RAG, and it was the week things really clicked for me around how retrieval actually works. I built a chatbot for a fictional vacation rental company — it reads through company documents, converts them into embeddings, stores them in ChromaDB, and retrieves the right context before answering with GPT. Building it from scratch made the whole "retrieval before generation" idea finally make sense.

The idea: Create fake company documents, store them in a vector database, and let users ask questions. The system finds relevant info and generates answers.

## How It Works

1. Created 9 documents for a fake company "StayEasy" (like Airbnb)
2. Split documents into small chunks
3. Converted chunks to embeddings using Sentence Transformers
4. Stored everything in ChromaDB
5. When user asks a question - search for similar chunks, send to GPT, get answer

## Example

```
User: "What is the cancellation policy?"

System finds relevant chunks from cancellation.md
        ↓
Sends to GPT with context
        ↓
Answer: "StayEasy has 3 policies: Flexible, Moderate, and Strict..."
```

## Files

```
data/               # 9 fake company documents
├── company.md
├── for_guests.md
├── for_hosts.md
├── pricing_fees.md
├── cancellation.md
├── payments.md
├── trust_safety.md
├── superhost.md
└── faqs.md

ingest.py           # Load and embed documents
answer.py           # Answer questions (terminal)
app.py              # Gradio web interface
evaluate.py         # Test the system
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Add your OpenAI key
echo "OPENAI_API_KEY=your-key" > .env

# Load documents into database
python ingest.py

# Run the chatbot
python answer.py

# Or use web interface
python app.py
```

## What I Learned

- How RAG works (retrieve then generate)
- How embeddings convert text to numbers
- How vector databases search by similarity
- How to build a chatbot with Gradio

## Tools

- Sentence Transformers (embeddings)
- ChromaDB (vector database)
- OpenAI API (GPT for answers)
- Gradio (web UI)

---

Built while learning from Ed Donner's LLM Engineering Course (Week 5 - RAG)
