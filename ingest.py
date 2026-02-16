"""
ingest.py - Load documents, chunk them, embed them, store in ChromaDB

Run this once (or when documents change):
    python ingest.py
"""

import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

DATA_FOLDER = "data"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "stayeasy_docs"

# Chunking settings
CHUNK_SIZE = 800       # max characters per chunk (only splits large sections)


# ============================================================
# STEP 2: LOAD DOCUMENTS
# ============================================================

def load_documents(folder_path):
    """Read all markdown files from the data folder."""
    documents = []

    for file_path in Path(folder_path).glob("*.md"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append({
                "content": content,
                "filename": file_path.name
            })
            print(f"Loaded: {file_path.name}")

    return documents


# ============================================================
# STEP 3: CHUNK DOCUMENTS (MARKDOWN-AWARE)
# ============================================================

def split_by_headings(text):
    """Split markdown text into sections based on ## headings.

    Keeps H2 sections intact (including H3 subsections).
    Only splits a H2 section into H3-level chunks if it exceeds CHUNK_SIZE.
    """
    lines = text.split("\n")
    h1_title = ""
    h2_sections = []
    current_h2_heading = ""
    current_h2_lines = []

    def flush_h2():
        content = "\n".join(current_h2_lines).strip()
        if content:
            h2_sections.append({
                "heading": current_h2_heading.strip(),
                "content": content
            })

    for line in lines:
        if line.startswith("# ") and not line.startswith("## "):
            # H1 heading - capture title, start collecting
            flush_h2()
            h1_title = line.lstrip("# ").strip()
            current_h2_heading = h1_title
            current_h2_lines = [line]
        elif line.startswith("## ") and not line.startswith("### "):
            # H2 heading - start new section (includes all H3 children)
            flush_h2()
            h2_name = line.lstrip("# ").strip()
            current_h2_heading = f"{h1_title} > {h2_name}" if h1_title else h2_name
            current_h2_lines = [line]
        else:
            current_h2_lines.append(line)

    flush_h2()

    # Now split any oversized H2 sections at H3 boundaries
    final_sections = []
    for section in h2_sections:
        if len(section["content"]) <= CHUNK_SIZE:
            final_sections.append(section)
        else:
            # Split this H2 section at H3 boundaries
            sub_lines = section["content"].split("\n")
            current_h3_heading = section["heading"]
            current_h3_lines = []

            def flush_h3():
                content = "\n".join(current_h3_lines).strip()
                if content:
                    final_sections.append({
                        "heading": current_h3_heading.strip(),
                        "content": content
                    })

            for sline in sub_lines:
                if sline.startswith("### "):
                    flush_h3()
                    h3_name = sline.lstrip("# ").strip()
                    current_h3_heading = f"{section['heading']} > {h3_name}"
                    current_h3_lines = [sline]
                else:
                    current_h3_lines.append(sline)

            flush_h3()

    return final_sections


def chunk_documents(documents):
    """Chunk all documents by markdown sections."""
    all_chunks = []

    for doc in documents:
        sections = split_by_headings(doc["content"])

        for i, section in enumerate(sections):
            text = section["content"]
            heading = section["heading"]

            # If section is too large, split it further
            if len(text) > CHUNK_SIZE:
                # Split on blank lines within the section
                paragraphs = text.split("\n\n")
                current_chunk = ""
                sub_id = 0
                for para in paragraphs:
                    if current_chunk and len(current_chunk) + len(para) > CHUNK_SIZE:
                        all_chunks.append({
                            "text": current_chunk.strip(),
                            "filename": doc["filename"],
                            "chunk_id": f"{i}_{sub_id}",
                            "heading": heading
                        })
                        sub_id += 1
                        current_chunk = para + "\n\n"
                    else:
                        current_chunk += para + "\n\n"
                if current_chunk.strip():
                    all_chunks.append({
                        "text": current_chunk.strip(),
                        "filename": doc["filename"],
                        "chunk_id": f"{i}_{sub_id}",
                        "heading": heading
                    })
            else:
                all_chunks.append({
                    "text": text,
                    "filename": doc["filename"],
                    "chunk_id": str(i),
                    "heading": heading
                })

    return all_chunks


# ============================================================
# STEP 4: CREATE EMBEDDINGS & STORE IN CHROMADB
# ============================================================

def create_vector_store(chunks):
    """Embed chunks and store in ChromaDB."""

    # Load embedding model (runs locally, free)
    print("\nLoading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Initialize ChromaDB
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection if it exists (fresh start)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except:
        pass

    # Create new collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "StayEasy documentation"}
    )

    # Prepare data for ChromaDB
    texts = [chunk["text"] for chunk in chunks]
    ids = [f"{chunk['filename']}_{chunk['chunk_id']}" for chunk in chunks]
    metadatas = [{"filename": chunk["filename"], "chunk_id": str(chunk["chunk_id"]), "heading": chunk.get("heading", "")} for chunk in chunks]

    # Create embeddings
    print(f"Creating embeddings for {len(texts)} chunks...")
    embeddings = embedding_model.encode(texts).tolist()

    # Add to ChromaDB
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    print(f"Stored {len(texts)} chunks in ChromaDB")
    return collection


# ============================================================
# MAIN: RUN THE PIPELINE
# ============================================================

def main():
    print("=" * 50)
    print("StayEasy RAG - Document Ingestion")
    print("=" * 50)

    # Step 1: Load documents
    print("\n[Step 1] Loading documents...")
    documents = load_documents(DATA_FOLDER)
    print(f"Loaded {len(documents)} documents")

    # Step 2: Chunk documents by markdown sections
    print("\n[Step 2] Chunking documents by markdown sections...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Show chunk breakdown
    for chunk in chunks:
        print(f"  [{chunk['filename']}] {chunk['heading'][:60]} ({len(chunk['text'])} chars)")

    # Step 3: Embed and store
    print("\n[Step 3] Creating embeddings and storing in ChromaDB...")
    collection = create_vector_store(chunks)

    print("\n" + "=" * 50)
    print("Ingestion complete!")
    print(f"Vector database saved to: {CHROMA_PATH}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
