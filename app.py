"""
app.py - Gradio UI for StayEasy RAG system

Run:
    python app.py
"""

import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import gradio as gr

load_dotenv()

# ============================================================
# CONFIG
# ============================================================

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "stayeasy_docs"
TOP_K = 5

# ============================================================
# LOAD MODELS (once at startup)
# ============================================================

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading vector database...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)
print(f"Loaded {collection.count()} chunks")


# ============================================================
# RAG FUNCTIONS
# ============================================================

def retrieve(question, top_k=TOP_K):
    """Find the most relevant chunks for a question."""
    question_embedding = embedding_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
    )
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "filename": results["metadatas"][0][i]["filename"],
            "heading": results["metadatas"][0][i].get("heading", ""),
            "distance": results["distances"][0][i],
        })
    return chunks


def generate_answer(question, chunks):
    """Send question + context to OpenAI and get answer."""
    context = "\n\n---\n\n".join([chunk["text"] for chunk in chunks])

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

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful StayEasy customer support assistant. Answer questions directly and precisely, prioritizing the most specific facts from the provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content


# ============================================================
# CHAT TAB
# ============================================================

def chat_respond(message, history):
    """Handle a chat message: retrieve chunks, generate answer, return sources separately."""
    if not message.strip():
        return "", history, ""

    # Retrieve relevant chunks
    chunks = retrieve(message)

    # Generate answer
    answer = generate_answer(message, chunks)

    # Build sources panel (shown on the right)
    sources = []
    for i, chunk in enumerate(chunks):
        sources.append(
            f"### {i+1}. {chunk['filename']}\n"
            f"**Section:** {chunk['heading']}\n\n"
            f"**Distance:** {chunk['distance']:.4f}\n\n"
            f"```\n{chunk['text'][:300]}{'...' if len(chunk['text']) > 300 else ''}\n```"
        )
    sources_md = "\n\n---\n\n".join(sources)

    # Gradio 6.x uses messages format
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return "", history, sources_md


# ============================================================
# EVALUATION TAB
# ============================================================

TEST_CASES = [
    {
        "question": "When was StayEasy founded and by whom?",
        "expected_answer": "StayEasy was founded in 2021 by Maria Rodriguez and James Chen in San Francisco.",
        "expected_source": "company.md",
    },
    {
        "question": "What is the guest service fee?",
        "expected_answer": "The guest service fee is 10-14% of the booking subtotal.",
        "expected_source": "pricing_fees.md",
    },
    {
        "question": "What is the host service fee percentage?",
        "expected_answer": "The host service fee is 3% of the booking subtotal.",
        "expected_source": "for_hosts.md",
    },
    {
        "question": "What are the Superhost requirements?",
        "expected_answer": "Superhosts need 10+ completed bookings, 90%+ response rate, less than 1% cancellation rate, and 4.8+ star rating.",
        "expected_source": "superhost.md",
    },
    {
        "question": "What is the cancellation policy for flexible bookings?",
        "expected_answer": "Flexible policy gives full refund if cancelled 24+ hours before check-in, 50% refund if less than 24 hours.",
        "expected_source": "cancellation.md",
    },
    {
        "question": "How much property damage protection do hosts get?",
        "expected_answer": "Hosts get $1,000,000 in property damage protection.",
        "expected_source": "for_hosts.md",
    },
    {
        "question": "When do hosts receive their payout?",
        "expected_answer": "Host payouts are released 24 hours after guest check-in, with 3-5 business days processing.",
        "expected_source": "payments.md",
    },
    {
        "question": "What payment methods are accepted?",
        "expected_answer": "Credit cards (Visa, Mastercard, Amex, Discover), debit cards, PayPal, Apple Pay, Google Pay, and gift cards.",
        "expected_source": "payments.md",
    },
    {
        "question": "What is the emergency phone number?",
        "expected_answer": "The urgent safety line is 1-800-782-9111.",
        "expected_source": "trust_safety.md",
    },
    {
        "question": "How many photos are required for a listing?",
        "expected_answer": "A minimum of 10 high-quality photos are required.",
        "expected_source": "for_hosts.md",
    },
]


def llm_judge(question, expected_answer, actual_answer, context_chunks):
    """Use LLM to score answer quality."""
    context = "\n\n---\n\n".join([c["text"] for c in context_chunks])

    prompt = f"""You are an evaluation judge for a RAG system. Score the following on a scale of 1-5.

QUESTION: {question}

EXPECTED ANSWER: {expected_answer}

ACTUAL ANSWER: {actual_answer}

RETRIEVED CONTEXT:
{context}

Score these three metrics (1=worst, 5=best):

1. **Answer Relevance**: Does the actual answer address the question?
2. **Answer Correctness**: Does the actual answer match the expected answer in meaning?
3. **Faithfulness**: Is the actual answer supported by the retrieved context (no hallucination)?

Respond in this exact JSON format only, no other text:
{{"answer_relevance": <1-5>, "answer_correctness": <1-5>, "faithfulness": <1-5>}}"""

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
    )

    try:
        scores = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        scores = {"answer_relevance": 0, "answer_correctness": 0, "faithfulness": 0}

    return scores


def run_evaluation(progress=gr.Progress()):
    """Run the full evaluation and return results as markdown + dataframe."""
    results = []
    retrieval_hits = 0
    all_reciprocal_ranks = []
    all_relevance = []
    all_correctness = []
    all_faithfulness = []
    total = len(TEST_CASES)

    for i, test in enumerate(progress.tqdm(TEST_CASES, desc="Evaluating questions")):
        # Retrieve
        chunks = retrieve(test["question"])
        source_files = [c["filename"] for c in chunks]
        hit = test["expected_source"] in source_files
        retrieval_hits += 1 if hit else 0

        # Compute reciprocal rank
        if hit:
            rank = source_files.index(test["expected_source"]) + 1
            rr = 1.0 / rank
        else:
            rank = 0
            rr = 0.0
        all_reciprocal_ranks.append(rr)

        # Generate answer
        actual_answer = generate_answer(test["question"], chunks)

        # Judge
        scores = llm_judge(test["question"], test["expected_answer"], actual_answer, chunks)
        all_relevance.append(scores["answer_relevance"])
        all_correctness.append(scores["answer_correctness"])
        all_faithfulness.append(scores["faithfulness"])

        results.append({
            "Q#": i + 1,
            "Question": test["question"],
            "Expected": test["expected_answer"],
            "Actual": actual_answer,
            "Source Hit": "HIT" if hit else "MISS",
            "Rank": rank,
            "RR": f"{rr:.2f}",
            "Relevance": f"{scores['answer_relevance']}/5",
            "Correctness": f"{scores['answer_correctness']}/5",
            "Faithfulness": f"{scores['faithfulness']}/5",
        })

    # Calculate summary
    avg_relevance = sum(all_relevance) / total
    avg_correctness = sum(all_correctness) / total
    avg_faithfulness = sum(all_faithfulness) / total
    mrr = sum(all_reciprocal_ranks) / total
    overall = (
        mrr * 20
        + (avg_relevance / 5) * 20
        + (avg_correctness / 5) * 20
        + (avg_faithfulness / 5) * 20
        + (retrieval_hits / total) * 20
    )

    # Build summary markdown
    summary = f"""## Evaluation Results

| Metric | Score |
|--------|-------|
| **Overall Score** | **{overall:.1f}/100** |
| Retrieval Accuracy | {retrieval_hits}/{total} ({retrieval_hits/total*100:.0f}%) |
| MRR (Mean Reciprocal Rank) | {mrr:.4f} |
| Avg Relevance | {avg_relevance:.2f}/5 |
| Avg Correctness | {avg_correctness:.2f}/5 |
| Avg Faithfulness | {avg_faithfulness:.2f}/5 |
"""

    # Build detailed results table
    table_rows = []
    for r in results:
        table_rows.append([
            r["Q#"],
            r["Question"],
            r["Source Hit"],
            r["Rank"],
            r["RR"],
            r["Relevance"],
            r["Correctness"],
            r["Faithfulness"],
            r["Actual"][:150] + "..." if len(r["Actual"]) > 150 else r["Actual"],
        ])

    return summary, table_rows


# ============================================================
# BUILD GRADIO UI
# ============================================================

with gr.Blocks(title="StayEasy RAG") as demo:
    gr.Markdown("# StayEasy RAG - Customer Support Assistant")
    gr.Markdown(f"Vector database: **{collection.count()} chunks** loaded")

    with gr.Tabs():
        # ---- Chat Tab ----
        with gr.TabItem("Chat"):
            with gr.Row():
                # Left side: Chat
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="StayEasy Assistant",
                        height=500,
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask a question about StayEasy...",
                            label="Your question",
                            scale=9,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat")

                # Right side: Retrieved Sources
                with gr.Column(scale=2):
                    gr.Markdown("### Retrieved Sources")
                    sources_display = gr.Markdown(
                        value="*Ask a question to see retrieved sources here.*",
                    )

            # Example questions
            gr.Examples(
                examples=[
                    "When was StayEasy founded?",
                    "What is the guest service fee?",
                    "When do hosts receive their payout?",
                    "What are the Superhost requirements?",
                    "What is the cancellation policy for flexible bookings?",
                    "What payment methods are accepted?",
                ],
                inputs=msg,
            )

            # Event handlers — now output sources_display too
            msg.submit(chat_respond, [msg, chatbot], [msg, chatbot, sources_display])
            send_btn.click(chat_respond, [msg, chatbot], [msg, chatbot, sources_display])
            clear_btn.click(
                lambda: ([], "", "*Ask a question to see retrieved sources here.*"),
                None,
                [chatbot, msg, sources_display],
            )

        # ---- Evaluation Tab ----
        with gr.TabItem("Evaluation"):
            gr.Markdown("### Run evaluation on 10 test questions to measure RAG quality")

            eval_btn = gr.Button("Run Evaluation", variant="primary")

            eval_summary = gr.Markdown(label="Summary")
            eval_table = gr.Dataframe(
                headers=["Q#", "Question", "Source Hit", "Rank", "RR", "Relevance", "Correctness", "Faithfulness", "Answer"],
                label="Detailed Results",
                wrap=True,
            )

            eval_btn.click(run_evaluation, outputs=[eval_summary, eval_table])

if __name__ == "__main__":
    demo.launch()
