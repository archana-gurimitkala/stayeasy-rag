"""
answer.py - Query the RAG system and get answers

Run this after ingest.py:
    python answer.py
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

# Load environment variables
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "stayeasy_docs"
TOP_K = 5  # Number of chunks to retrieve


# ============================================================
# STEP 1: LOAD VECTOR DATABASE
# ============================================================

def load_vector_store():
    """Load the existing ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection


# ============================================================
# STEP 2: RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve(question, collection, embedding_model, top_k=TOP_K):
    """Find the most relevant chunks for a question."""

    # Embed the question
    question_embedding = embedding_model.encode(question).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    # Extract chunks and metadata
    retrieved_chunks = []
    for i in range(len(results["documents"][0])):
        retrieved_chunks.append({
            "text": results["documents"][0][i],
            "filename": results["metadatas"][0][i]["filename"],
            "distance": results["distances"][0][i]
        })

    return retrieved_chunks


# ============================================================
# STEP 3: GENERATE ANSWER WITH LLM
# ============================================================

def generate_answer(question, chunks):
    """Send question + context to OpenAI and get answer."""

    # Build context from retrieved chunks
    context = "\n\n---\n\n".join([chunk["text"] for chunk in chunks])

    # Create prompt
    prompt = f"""You are a helpful customer support assistant for StayEasy, a vacation rental platform.

INSTRUCTIONS:
- Answer the question based ONLY on the context provided below.
- Be specific and direct. Lead with the most important fact that answers the question.
- Include specific numbers, timeframes, and requirements from the context.
- If the answer is not in the context, say "I don't have information about that."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    # Call OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful StayEasy customer support assistant. Answer questions directly and precisely, prioritizing the most specific facts from the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content


# ============================================================
# STEP 4: RAG PIPELINE (RETRIEVE + GENERATE)
# ============================================================

def ask(question, collection, embedding_model):
    """Full RAG pipeline: retrieve context, then generate answer."""

    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print("="*50)

    # Retrieve relevant chunks
    print("\n[Retrieving relevant information...]")
    chunks = retrieve(question, collection, embedding_model)

    # Show what was retrieved
    print(f"\nFound {len(chunks)} relevant chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  {i+1}. {chunk['filename']} (distance: {chunk['distance']:.4f})")

    # Generate answer
    print("\n[Generating answer...]")
    answer = generate_answer(question, chunks)

    print(f"\nAnswer: {answer}")
    return answer


# ============================================================
# MAIN: INTERACTIVE CHAT
# ============================================================

def main():
    print("=" * 50)
    print("StayEasy RAG - Customer Support Assistant")
    print("=" * 50)
    print("Type 'quit' to exit.\n")

    # Load embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load vector store
    print("Loading vector database...")
    collection = load_vector_store()
    print(f"Loaded {collection.count()} chunks\n")

    # Interactive loop
    while True:
        question = input("\nYou: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not question:
            continue

        ask(question, collection, embedding_model)


if __name__ == "__main__":
    main()
