# StayEasy RAG - Customer Support Assistant

A Retrieval-Augmented Generation (RAG) system for StayEasy vacation rental platform that provides intelligent customer support by answering questions based on company documentation.

## 🚀 Features

- **Document-based Q&A**: Answers questions using StayEasy's official documentation
- **Vector Search**: Uses ChromaDB for efficient semantic search
- **Interactive CLI**: Command-line interface for querying the system
- **Web UI**: Gradio-based web interface for easy interaction
- **Evaluation System**: Built-in evaluation framework to test system performance

## 📋 Prerequisites

- Python 3.9+
- OpenAI API key

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd stayeasy_rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## 📖 Usage

### Step 1: Ingest Documents

First, load and process the documentation files:

```bash
python ingest.py
```

This will:
- Load all markdown files from the `data/` folder
- Chunk the documents into smaller pieces
- Generate embeddings using sentence transformers
- Store everything in ChromaDB

### Step 2: Query the System

#### Option A: Command Line Interface

```bash
python answer.py
```

Then type your questions interactively. Type `quit` to exit.

#### Option B: Web Interface (Gradio)

```bash
python app.py
```

This will start a web server. Open the URL shown in the terminal (usually `http://127.0.0.1:7860`) in your browser.

### Step 3: Evaluate (Optional)

Run the evaluation script to test the system's performance:

```bash
python evaluate.py
```

Results will be saved to `evaluation_results.json`.

## 📁 Project Structure

```
stayeasy_rag/
├── data/                  # Source documentation files
│   ├── company.md
│   ├── faqs.md
│   ├── for_guests.md
│   ├── for_hosts.md
│   └── ...
├── chroma_db/             # Vector database (auto-generated)
├── screenshots/           # Project screenshots
├── ingest.py              # Document ingestion script
├── answer.py              # CLI query interface
├── app.py                 # Gradio web interface
├── evaluate.py            # Evaluation script
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in git)
└── README.md              # This file
```

## 🔧 Configuration

Key settings in the code:

- `CHROMA_PATH`: Path to ChromaDB storage (default: `"chroma_db"`)
- `COLLECTION_NAME`: ChromaDB collection name (default: `"stayeasy_docs"`)
- `TOP_K`: Number of chunks to retrieve (default: `3-5`)
- `CHUNK_SIZE`: Document chunk size (default: `500` characters)
- `CHUNK_OVERLAP`: Overlap between chunks (default: `100` characters)

## 📸 Screenshots

Add your project screenshots to the `screenshots/` folder. Some ideas:
- Web interface in action
- Example Q&A interactions
- System architecture diagram
- Performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is for educational/demonstration purposes.

## 🙏 Acknowledgments

- StayEasy for the documentation
- OpenAI for GPT models
- ChromaDB for vector storage
- Sentence Transformers for embeddings
