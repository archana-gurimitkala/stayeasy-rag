"""
evaluate.py - Evaluate the RAG system's performance

Metrics:
  1. Retrieval Quality - Did we fetch the right documents?
  2. Answer Relevance  - Is the answer relevant to the question?
  3. Faithfulness       - Is the answer grounded in the retrieved context (no hallucination)?

Run:
    python evaluate.py
"""

import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

load_dotenv()

# ============================================================
# CONFIG
# ============================================================

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "stayeasy_docs"
TOP_K = 5

# ============================================================
# TEST DATASET - Questions with expected answers & source files
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


# ============================================================
# RETRIEVAL (same as answer.py)
# ============================================================

def retrieve(question, collection, embedding_model, top_k=TOP_K):
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
            "distance": results["distances"][0][i],
        })
    return chunks


# ============================================================
# LLM-BASED EVALUATION (using GPT-4o-mini as judge)
# ============================================================

def llm_judge(question, expected_answer, actual_answer, context_chunks):
    """Use LLM to score answer relevance and faithfulness."""

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


# ============================================================
# GENERATE ANSWER (same as answer.py)
# ============================================================

def generate_answer(question, chunks):
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
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message.content


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    print("=" * 60)
    print("  StayEasy RAG - Evaluation")
    print("=" * 60)

    # Load models & data
    print("\nLoading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading vector database...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Loaded {collection.count()} chunks\n")

    # Metrics accumulators
    total = len(TEST_CASES)
    retrieval_hits = 0
    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0
    all_reciprocal_ranks = []
    all_relevance = []
    all_correctness = []
    all_faithfulness = []

    results = []

    for i, test in enumerate(TEST_CASES):
        print(f"\n{'─' * 60}")
        print(f"  [{i+1}/{total}] {test['question']}")
        print(f"{'─' * 60}")

        # Step 1: Retrieve
        chunks = retrieve(test["question"], collection, embedding_model)
        source_files = [c["filename"] for c in chunks]

        # Check if expected source was retrieved + compute reciprocal rank
        hit = test["expected_source"] in source_files
        retrieval_hits += 1 if hit else 0

        # Calculate recall@k for k=1, 3, 5
        if len(source_files) > 0 and test["expected_source"] in source_files[:1]:
            recall_at_1 += 1
        if len(source_files) > 0 and test["expected_source"] in source_files[:3]:
            recall_at_3 += 1
        if len(source_files) > 0 and test["expected_source"] in source_files[:5]:
            recall_at_5 += 1

        if hit:
            rank = source_files.index(test["expected_source"]) + 1  # 1-indexed
            reciprocal_rank = 1.0 / rank
        else:
            rank = 0
            reciprocal_rank = 0.0
        all_reciprocal_ranks.append(reciprocal_rank)

        print(f"  Retrieved: {source_files}")
        print(f"  Expected source: {test['expected_source']} → {'HIT' if hit else 'MISS'} (rank: {rank}, RR: {reciprocal_rank:.2f})")

        # Step 2: Generate answer
        actual_answer = generate_answer(test["question"], chunks)
        print(f"  Expected: {test['expected_answer']}")
        print(f"  Actual:   {actual_answer}")

        # Step 3: LLM Judge
        scores = llm_judge(
            test["question"],
            test["expected_answer"],
            actual_answer,
            chunks,
        )
        all_relevance.append(scores["answer_relevance"])
        all_correctness.append(scores["answer_correctness"])
        all_faithfulness.append(scores["faithfulness"])

        print(f"  Scores → Relevance: {scores['answer_relevance']}/5  "
              f"Correctness: {scores['answer_correctness']}/5  "
              f"Faithfulness: {scores['faithfulness']}/5")

        results.append({
            "question": test["question"],
            "expected_source": test["expected_source"],
            "retrieved_sources": source_files,
            "retrieval_hit": hit,
            "rank": rank,
            "reciprocal_rank": reciprocal_rank,
            "expected_answer": test["expected_answer"],
            "actual_answer": actual_answer,
            "scores": scores,
        })

    # ============================================================
    # SUMMARY
    # ============================================================

    avg_relevance = sum(all_relevance) / total
    avg_correctness = sum(all_correctness) / total
    avg_faithfulness = sum(all_faithfulness) / total
    mrr = sum(all_reciprocal_ranks) / total
    
    # Calculate recall@k percentages
    recall_at_1_pct = (recall_at_1 / total) * 100
    recall_at_3_pct = (recall_at_3 / total) * 100
    recall_at_5_pct = (recall_at_5 / total) * 100

    print(f"\n\n{'=' * 60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total test cases:      {total}")
    print(f"\n  RETRIEVAL METRICS:")
    print(f"  ──────────────────────")
    print(f"  Recall@1:              {recall_at_1}/{total} ({recall_at_1_pct:.1f}%)")
    print(f"  Recall@3:              {recall_at_3}/{total} ({recall_at_3_pct:.1f}%)")
    print(f"  Recall@5:              {recall_at_5}/{total} ({recall_at_5_pct:.1f}%)")
    print(f"  MRR (Mean Reciprocal Rank): {mrr:.4f}")
    print(f"\n  ANSWER QUALITY METRICS:")
    print(f"  ──────────────────────")
    print(f"  Avg Answer Relevance:  {avg_relevance:.2f}/5")
    print(f"  Avg Answer Correctness:{avg_correctness:.2f}/5")
    print(f"  Avg Faithfulness:      {avg_faithfulness:.2f}/5")
    print(f"{'=' * 60}")

    # Overall score (now includes MRR)
    overall = (
        mrr * 20
        + (avg_relevance / 5) * 20
        + (avg_correctness / 5) * 20
        + (avg_faithfulness / 5) * 20
        + (retrieval_hits / total) * 20
    )
    print(f"\n  OVERALL SCORE: {overall:.1f}/100")
    print(f"{'=' * 60}")

    # Save results to file
    with open("evaluation_results.json", "w") as f:
        json.dump({
            "summary": {
                "total_cases": total,
                "retrieval_metrics": {
                    "recall_at_1": f"{recall_at_1}/{total} ({recall_at_1_pct:.1f}%)",
                    "recall_at_3": f"{recall_at_3}/{total} ({recall_at_3_pct:.1f}%)",
                    "recall_at_5": f"{recall_at_5}/{total} ({recall_at_5_pct:.1f}%)",
                    "mrr": round(mrr, 4),
                },
                "answer_quality_metrics": {
                    "avg_relevance": round(avg_relevance, 2),
                    "avg_correctness": round(avg_correctness, 2),
                    "avg_faithfulness": round(avg_faithfulness, 2),
                },
                "overall_score": round(overall, 1),
            },
            "details": results,
        }, f, indent=2)

    print(f"\n  Detailed results saved to: evaluation_results.json")


if __name__ == "__main__":
    main()
