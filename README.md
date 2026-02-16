# StayEasy RAG - Customer Support Assistant

Ever wondered how to build a smart chatbot that actually knows your company's documentation? This project shows you how! It's a customer support assistant for StayEasy (a vacation rental platform) that can answer questions by searching through company documents and giving accurate answers.

## What is RAG?

RAG stands for **Retrieval-Augmented Generation**. Think of it like this:
- Your documents are stored in a smart database
- When someone asks a question, the system finds the most relevant parts
- Then it uses AI to write a helpful answer based on what it found

It's like having a super-smart assistant who never forgets your company policies!

## What Can This Do?

✨ **Answer questions** - Ask anything about StayEasy policies, fees, cancellation rules, etc.

🔍 **Find the right info** - Automatically searches through all your documents to find relevant answers

💬 **Two ways to use it** - Try it in your terminal or use the friendly web interface

📊 **See how well it works** - Built-in evaluation tools show you how accurate the answers are

## What You'll Need

Before you start, make sure you have:
- **Python 3.9 or newer** (check with `python --version`)
- **An OpenAI API key** (get one at [platform.openai.com](https://platform.openai.com/api-keys))

## Getting Started

### 1. Get the Code

First, download this project to your computer:

```bash
git clone https://github.com/archana-gurimitkala/stayeasy-rag.git
cd stayeasy-rag
```

### 2. Install the Required Tools

Install all the Python packages needed:

```bash
pip install -r requirements.txt
```

This will install things like ChromaDB (for storing documents), OpenAI (for AI), and Gradio (for the web interface).

### 3. Add Your API Key

Create a file called `.env` in the project folder and add your OpenAI API key:

```
OPENAI_API_KEY=your_actual_api_key_here
```

**Important:** Don't share your API key! The `.env` file is already set to be ignored by git, so it won't be uploaded to GitHub.

## How to Use It

### Step 1: Load Your Documents

Before you can ask questions, you need to load all the documents into the system:

```bash
python ingest.py
```

What this does:
- Reads all the markdown files from the `data/` folder
- Breaks them into smaller, manageable pieces
- Converts them into a format the AI can understand
- Saves everything in a database

You'll see it loading files and creating chunks. This might take a minute or two!

### Step 2: Ask Questions!

Now the fun part - you can ask questions in two ways:

#### Option A: Command Line (Terminal)

```bash
python answer.py
```

Then just type your questions! For example:
- "When was StayEasy founded?"
- "What is the cancellation policy?"
- "How much is the host service fee?"

Type `quit` when you're done.

#### Option B: Web Interface (Easier!)

```bash
python app.py
```

This opens a nice web page in your browser. You'll see:
- A chat box where you can type questions
- Answers appear right away
- You can even see which documents were used to answer your question!

The web address will be shown in your terminal (usually `http://127.0.0.1:7860`).

### Step 3: Check How Well It Works (Optional)

Want to see how accurate the system is? Run the evaluation:

```bash
python evaluate.py
```

This tests the system with 10 sample questions and shows you:
- How often it finds the right document
- How accurate the answers are
- Whether it makes things up (we call this "hallucination")

Results are saved to `evaluation_results.json` if you want to look at them later.

## What's Inside This Project?

```
stayeasy_rag/
├── data/                  # All the StayEasy documentation files
│   ├── company.md         # Company info
│   ├── faqs.md            # Frequently asked questions
│   ├── for_guests.md      # Info for guests
│   ├── for_hosts.md       # Info for hosts
│   └── ...                # More docs
├── chroma_db/             # The database (created automatically)
├── screenshots/           # Pictures showing how it works
├── ingest.py              # Script to load documents
├── answer.py              # Command-line version
├── app.py                 # Web interface version
├── evaluate.py            # Testing script
├── requirements.txt       # List of needed packages
└── README.md              # This file!
```

## Customizing It

Want to change how it works? Here are some settings you can tweak:

- **`TOP_K`**: How many document chunks to look at (default: 5) - more chunks = more context but slower
- **`CHUNK_SIZE`**: How big each piece of text is (default: 500 characters)
- **`CHUNK_OVERLAP`**: How much overlap between chunks (default: 100 characters) - helps keep context

You can find these in the Python files and adjust them to your needs.

## See It in Action

Check out the `screenshots/` folder to see:
- The web interface in action
- Example questions and answers
- Performance metrics showing how well it works
- Before/after improvements

## Want to Contribute?

Found a bug or have an idea? Here's how to help:

1. Fork this repository (make your own copy)
2. Create a new branch for your changes
3. Make your improvements
4. Submit a pull request

We'd love to see what you build!

## Credits

Thanks to:
- **StayEasy** for the documentation
- **OpenAI** for the GPT models that power the answers
- **ChromaDB** for storing and searching documents
- **Sentence Transformers** for understanding text meaning

## Questions?

If something doesn't work or you're confused, feel free to open an issue on GitHub. We're here to help!

---

**Note:** This project is for learning and demonstration purposes. Feel free to use it as a starting point for your own RAG projects!
