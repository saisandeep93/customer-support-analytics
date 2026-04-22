import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'support.db')
CHROMA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_langchain')

# ─────────────────────────────────────────────
# SHARED COMPONENTS
# ─────────────────────────────────────────────

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_llm():
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )

# ─────────────────────────────────────────────
# STEP 0: INDEXING
# Load docs from SQLite, chunk with LangChain's
# RecursiveCharacterTextSplitter, embed and store
# in a LangChain Chroma vectorstore.
# ─────────────────────────────────────────────

def index_knowledge_base() -> int:
    print("Indexing knowledge base (LangChain)...")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT doc_id, title, content, category FROM knowledge_base_documents"
    ).fetchall()
    conn.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )

    documents: list[Document] = []
    for row in rows:
        # Prepend title so every chunk carries document context
        full_text = f"{row['title']}\n\n{row['content']}"
        chunks = splitter.split_text(full_text)
        for idx, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "category": row["category"],
                    "chunk_index": idx,
                }
            ))

    embeddings = get_embeddings()

    # Delete existing collection so re-runs don't duplicate
    if os.path.exists(CHROMA_PATH):
        existing = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name="knowledge_base_lc",
        )
        existing.delete_collection()
        print(f"  Cleared existing collection")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="knowledge_base_lc",
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"  Indexed {len(documents)} chunks from {len(rows)} documents")
    print(f"  Vector store saved to: {os.path.abspath(CHROMA_PATH)}")
    return len(documents)

# ─────────────────────────────────────────────
# STEP 1 + 2: RETRIEVAL
# ─────────────────────────────────────────────

def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings(),
        collection_name="knowledge_base_lc",
        collection_metadata={"hnsw:space": "cosine"},
    )

def retrieve_relevant_chunks(query: str, n_results: int = 3) -> list[dict]:
    """Returns list of dicts with text, similarity, doc_id, title, category."""
    vectorstore = get_vectorstore()
    # similarity_search_with_score returns (doc, cosine_distance) — lower is closer.
    # Convert to similarity score matching the original pipeline: similarity = 1 - distance.
    results = vectorstore.similarity_search_with_score(query, k=n_results)

    chunks = []
    for doc, distance in results:
        chunks.append({
            "text": doc.page_content,
            "similarity": round(1 - distance, 4),
            "doc_id": doc.metadata["doc_id"],
            "title": doc.metadata["title"],
            "category": doc.metadata["category"],
        })
    return chunks

# ─────────────────────────────────────────────
# STEP 3: PROMPT ASSEMBLY
# ChatPromptTemplate keeps system and human
# messages cleanly separated.
# ─────────────────────────────────────────────

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a customer support agent for QuickBite, a QSR food delivery brand. "
        "Your job is to resolve customer complaints based on company policy."
    ),
    (
        "human",
        """CUSTOMER PROFILE:
- Name: {name}
- Segment: {segment}
- Lifetime orders: {total_orders}
- Complaints in last 10 orders: {complaint_count}
- Average order value: Rs {avg_value}

RELEVANT POLICY DOCUMENTS:
{context}

CUSTOMER MESSAGE:
"{customer_query}"

INSTRUCTIONS:
1. Read the policy documents carefully
2. Apply the resolution that matches the customer's segment
3. Be empathetic and professional
4. State clearly what action you are taking (refund/replacement/voucher)
5. Do not make up policies not present in the documents above
6. Keep your response concise — 3 to 4 sentences maximum

YOUR RESPONSE:"""
    ),
])

def assemble_prompt_vars(
    customer_query: str,
    retrieved_chunks: list[dict],
    customer_profile: dict,
) -> dict:
    context_sections = []
    for i, chunk in enumerate(retrieved_chunks):
        context_sections.append(
            f"POLICY DOCUMENT {i+1} — {chunk['title']} "
            f"(relevance: {chunk['similarity']}):\n{chunk['text']}"
        )

    return {
        "name": customer_profile.get("name", "Customer"),
        "segment": customer_profile.get("segment_label", "standard").upper(),
        "total_orders": customer_profile.get("total_orders_lifetime", 0),
        "complaint_count": customer_profile.get("complaint_count_last_10", 0),
        "avg_value": f"{customer_profile.get('avg_order_value', 0):.0f}",
        "context": "\n\n".join(context_sections),
        "customer_query": customer_query,
    }

# ─────────────────────────────────────────────
# STEP 4: LLM GENERATION
# ─────────────────────────────────────────────

def generate_response(prompt_vars: dict) -> tuple[str, int, int, int]:
    """Returns (response_text, input_tokens, output_tokens, latency_ms)."""
    llm = get_llm()
    chain = PROMPT_TEMPLATE | llm

    start = datetime.now()
    ai_message = chain.invoke(prompt_vars)
    latency_ms = int((datetime.now() - start).total_seconds() * 1000)

    response_text = ai_message.content

    # Extract token usage from response_metadata
    usage = ai_message.response_metadata.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    return response_text, input_tokens, output_tokens, latency_ms

# ─────────────────────────────────────────────
# COMPLETE RAG PIPELINE
# ─────────────────────────────────────────────

def run_rag_pipeline(
    customer_query: str,
    customer_profile: dict,
    n_chunks: int = 3,
) -> dict:
    retrieved_chunks = retrieve_relevant_chunks(customer_query, n_results=n_chunks)

    prompt_vars = assemble_prompt_vars(customer_query, retrieved_chunks, customer_profile)

    response_text, input_tokens, output_tokens, latency_ms = generate_response(prompt_vars)

    return {
        "response": response_text,
        "retrieved_chunks": retrieved_chunks,
        "retrieved_doc_ids": [c["doc_id"] for c in retrieved_chunks],
        "retrieval_scores": [c["similarity"] for c in retrieved_chunks],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "latency_ms": latency_ms,
    }

# ─────────────────────────────────────────────
# TEST: Three customer profiles — high_value,
# standard, and at_risk — matching rag_pipeline.py
# ─────────────────────────────────────────────

def test_rag_pipeline():
    index_knowledge_base()

    test_cases = [
        {
            "query": "My food arrived completely cold. The burger was freezing.",
            "profile": {
                "name": "Priya Sharma",
                "segment_label": "high_value",
                "total_orders_lifetime": 45,
                "complaint_count_last_10": 0,
                "avg_order_value": 520,
            },
        },
        {
            "query": "I ordered chicken nuggets but they are missing from my order.",
            "profile": {
                "name": "Rahul Verma",
                "segment_label": "standard",
                "total_orders_lifetime": 8,
                "complaint_count_last_10": 1,
                "avg_order_value": 210,
            },
        },
        {
            "query": "I was charged twice for my order. Please refund.",
            "profile": {
                "name": "Amit Kumar",
                "segment_label": "at_risk",
                "total_orders_lifetime": 12,
                "complaint_count_last_10": 4,
                "avg_order_value": 180,
            },
        },
    ]

    for i, test in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i+1}: {test['profile']['name']} "
              f"({test['profile']['segment_label'].upper()})")
        print(f"{'='*60}")
        print(f"QUERY: {test['query']}")

        result = run_rag_pipeline(test["query"], test["profile"])

        print(f"\nRETRIEVED DOCUMENTS:")
        for chunk in result["retrieved_chunks"]:
            print(f"  - [{chunk['doc_id']}] {chunk['title']} "
                  f"(similarity: {chunk['similarity']})")

        print(f"\nRESPONSE:")
        print(result["response"])

        print(f"\nMETRICS:")
        print(f"  Input tokens:  {result['input_tokens']}")
        print(f"  Output tokens: {result['output_tokens']}")
        print(f"  Total tokens:  {result['total_tokens']}")
        print(f"  Latency:       {result['latency_ms']}ms")


if __name__ == "__main__":
    test_rag_pipeline()
