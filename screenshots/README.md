# Screenshots

This folder contains screenshots demonstrating the StayEasy RAG (Retrieval-Augmented Generation) system in action.

---

## 1. Web Interface (Gradio UI)

![Web Interface](./Screenshot%202026-02-15%20at%208.33.07%20PM.png)

**Description:**
This screenshot shows the StayEasy RAG Customer Support Assistant web interface built with Gradio. The interface demonstrates:

- **Chat Interface (Left Panel):** Interactive Q&A interface where users can ask questions about StayEasy
  - Example question: "When was StayEasy founded?" → Answer: "StayEasy was founded in 2021."
  - Example question: "What is the cancellation policy for flexible bookings?" → Detailed answer with refund policies
- **Retrieved Sources Panel (Right Panel):** Shows the source documents retrieved by the RAG system
  - Displays document filenames (e.g., `faqs.md`, `cancellation.md`)
  - Shows section paths and distance scores
  - Demonstrates transparency in the retrieval process
- **System Status:** Shows "Vector database: 57 chunks loaded"
- **Tabs:** Chat and Evaluation tabs for different interaction modes

This demonstrates the RAG system's ability to provide accurate answers while showing the source documents used.

---

## 2. Evaluation Metrics Table

![Evaluation Metrics](./Screenshot%202026-02-15%20at%208.34.48%20PM.png)

**Description:**
This screenshot displays a comprehensive evaluation metrics table showing the performance of the RAG system:

- **Retrieval Accuracy:** 100% (20/20 contribution) - The system always fetches the correct source document
- **MRR (Mean Reciprocal Rank):** 0.95 (19/20 contribution) - Excellent ranking performance
- **Average Relevance:** 5.00/5 (20/20 contribution) - Perfect relevance scores
- **Average Correctness:** 4.90/5 (19.6/20 contribution) - Highly accurate answers
- **Average Faithfulness:** 5.00/5 (20/20 contribution) - No hallucination, all answers grounded in context
- **Overall Score:** 98.6/100

This table demonstrates the system's high-quality performance across all evaluation dimensions.

---

## 3. Performance Improvement Report

![Performance Improvement](./Screenshot%202026-02-14%20at%205.28.36%20PM.png)

**Description:**
This screenshot shows the performance improvement report documenting the optimization journey:

**Changes Made:**
- **`ingest.py` - Markdown-aware chunking:**
  - Replaced character-based splitting with heading-based chunking at `##` (H2) boundaries
  - Each H2 section (including H3 children) stays as one chunk
  - Only splits to H3 level if section exceeds 800 characters
  - Adds section heading path as metadata
  - Reduced from 132 over-fragmented chunks to 57 well-sized chunks

- **`answer.py` - Better prompt + retrieval:**
  - Increased `TOP_K` from 3 to 5 for more context
  - Improved prompt: instructs LLM to be specific, lead with key facts, include numbers/timeframes
  - Lowered temperature from 0.7 to 0.3 for more precise answers

- **`evaluate.py` - Synced with answer.py:**
  - Updated `TOP_K` to 5 and synced prompt/system message

**Results:**
| Metric             | Before | After |
| :----------------- | :----- | :---- |
| Overall Score      | 96.5   | 99.5  |
| Retrieval Accuracy | 100%   | 100%  |
| Avg Relevance      | 4.60   | 5.00  |
| Avg Correctness   | 4.60   | 4.90  |
| Avg Faithfulness   | 4.40   | 5.00  |

This demonstrates the iterative improvement process and the impact of optimization techniques.

---

## 4. Evaluation Results Summary

![Evaluation Results](./Screenshot%202026-02-14%20at%205.20.10%20PM.png)

**Description:**
This screenshot shows the initial evaluation results summary:

- **Overall RAG System Score:** **96.5/100**
- **Metric Breakdown:**
  - **Retrieval Accuracy:** 10/10 (100%) - "always fetched the right source document"
  - **Answer Relevance:** 4.80/5 - Answers are highly relevant to questions
  - **Answer Correctness:** 4.60/5 - Answers match expected content
  - **Faithfulness:** 4.90/5 - "(almost no hallucination)" - Answers are well-grounded in retrieved context

This shows the baseline performance before optimizations, demonstrating the system's strong initial performance.

---

## Summary

These screenshots showcase:
- ✅ Interactive web interface for end users
- ✅ Comprehensive evaluation metrics (Recall@k, MRR, Relevance, Correctness, Faithfulness)
- ✅ Performance optimization process and results
- ✅ System transparency with source document retrieval

The StayEasy RAG system demonstrates high-quality performance with 98.6-99.5/100 scores across comprehensive evaluation metrics.
