import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
import anthropic
from observability import log_interaction

load_dotenv()

sys.path.append(os.path.dirname(__file__))
from deterministic_workflow import run_deterministic_workflow, index_faqs
from rag_pipeline import run_rag_pipeline, index_knowledge_base
from queries import get_customer_profile
from agent import run_agent

# ─────────────────────────────────────────────
# TIER DEFINITIONS
# Three tiers, each with a cost and
# capability profile
# ─────────────────────────────────────────────

TIER_SIMPLE = "simple"        # FAQ lookup, no LLM
TIER_STANDARD = "standard"    # RAG pipeline, Haiku
TIER_COMPLEX = "complex"      # Agent, Sonnet

# ─────────────────────────────────────────────
# SECOND-LEVEL CLASSIFIER
# Within COMPLEX queries, determines whether
# the agent is needed or RAG is sufficient.
# Signals for AGENT:
#   - Mentions previous complaint
#   - Multiple simultaneous issues
#   - Explicit escalation request
#   - Emotional language
#   - Retention risk signals
# ─────────────────────────────────────────────

AGENT_SIGNALS = [
    "speak to a manager",
    "speak to human",
    "escalate",
    "last week",
    "before",
    "again",
    "same problem",
    "same issue",
    "multiple",
    "also",
    "and also",
    "unacceptable",
    "ridiculous",
    "terrible",
    "worst",
    "never ordering again",
    "loyal customer",
]

def needs_agent(query: str) -> bool:
    """
    Lightweight signal-based check for
    whether a complex query needs the agent
    vs the RAG pipeline.
    Returns True if agent signals detected.
    """
    query_lower = query.lower()
    return any(signal in query_lower for signal in AGENT_SIGNALS)

# ─────────────────────────────────────────────
# MASTER ROUTER
# Single entry point for all queries.
# Routes to the right tier based on
# query classification.
# ─────────────────────────────────────────────

def route_query(query: str, mobile_number: str) -> dict:
    """
    The master routing function.

    Flow:
    1. Run deterministic classifier (SIMPLE vs COMPLEX)
    2. If SIMPLE: return FAQ result
    3. If COMPLEX: check agent signals
    4. If agent signals present: run agent
    5. If no agent signals: run RAG pipeline
    """
    start_time = datetime.now()

    # Step 1: Run deterministic workflow
    # This classifies SIMPLE vs COMPLEX
    # and handles FAQ lookup if SIMPLE
    deterministic_result = run_deterministic_workflow(query)

    if deterministic_result["routed_to"] == "deterministic":
        # SIMPLE query — FAQ answered it
        return {
            "tier": TIER_SIMPLE,
            "response": deterministic_result["response"],
            "faq_matched": deterministic_result.get("matched_question"),
            "similarity": deterministic_result.get("similarity"),
            "classifier_tokens": deterministic_result["classifier_tokens"],
            "total_tokens": deterministic_result["classifier_tokens"],
            "latency_ms": deterministic_result["latency_ms"],
            "routed_to": "faq_lookup"
        }

    if deterministic_result["routed_to"] == "escalate":
        # SIMPLE but no FAQ match
        return {
            "tier": TIER_SIMPLE,
            "response": deterministic_result["response"],
            "classifier_tokens": deterministic_result["classifier_tokens"],
            "total_tokens": deterministic_result["classifier_tokens"],
            "latency_ms": deterministic_result["latency_ms"],
            "routed_to": "faq_no_match"
        }

    # COMPLEX query — now decide: RAG or Agent?
    if needs_agent(query):
        # Step 4: Agent required
        print(f"  Routing to: AGENT (complex signals detected)")
        agent_result = run_agent(
            customer_query=query,
            mobile_number=mobile_number
        )
        total_ms = int(
            (datetime.now() - start_time).total_seconds() * 1000
        )
        return {
            "tier": TIER_COMPLEX,
            "response": agent_result["response"],
            "tools_called": agent_result["tools_called"],
            "iterations": agent_result["iterations"],
            "classifier_tokens": deterministic_result["classifier_tokens"],
            "total_tokens": (
                deterministic_result["classifier_tokens"] +
                agent_result["total_tokens"]
            ),
            "latency_ms": total_ms,
            "routed_to": "agent"
        }

    else:
        # Step 5: Standard RAG pipeline
        print(f"  Routing to: RAG PIPELINE (standard complaint)")

        # Fetch customer profile for RAG prompt
        customer_profile = get_customer_profile(mobile_number)
        if not customer_profile:
            customer_profile = {
                "name": "Customer",
                "segment_label": "standard",
                "total_orders_lifetime": 0,
                "complaint_count_last_10": 0,
                "avg_order_value": 0
            }

        rag_result = run_rag_pipeline(
            customer_query=query,
            customer_profile=customer_profile
        )

        total_ms = int(
            (datetime.now() - start_time).total_seconds() * 1000
        )

        return {
            "tier": TIER_STANDARD,
            "response": rag_result["response"],
            "retrieved_docs": rag_result["retrieved_doc_ids"],
            "retrieval_scores": rag_result["retrieval_scores"],
            "classifier_tokens": deterministic_result["classifier_tokens"],
            "total_tokens": (
                deterministic_result["classifier_tokens"] +
                rag_result["total_tokens"]
            ),
            "latency_ms": total_ms,
            "routed_to": "rag_pipeline"
        }

# ─────────────────────────────────────────────
# TEST: Run all three tiers through master router
# ─────────────────────────────────────────────

def test_master_router():

    # Get real mobile numbers
    import sqlite3
    db_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'support.db'
    )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    customer = conn.execute("""
        SELECT c.mobile_number, c.name
        FROM customers c
        JOIN customer_segments cs ON c.mobile_number = cs.mobile_number
        WHERE cs.segment_label = 'standard'
        LIMIT 1
    """).fetchone()
    conn.close()

    mobile = customer["mobile_number"]
    name = customer["name"]

    test_cases = [
        {
            "label": "TIER 1 — Simple FAQ",
            "query": "What time do you open?",
        },
        {
            "label": "TIER 2 — Standard RAG complaint",
            "query": "My food arrived cold. I want a refund.",
        },
        {
            "label": "TIER 3 — Complex agent required",
            "query": (
                "I complained last week about cold food and nothing "
                "was done. Same issue again. I want to escalate this."
            ),
        }
    ]

    print("\n" + "="*60)
    print("MASTER ROUTER TEST")
    print(f"Customer: {name} ({mobile})")
    print("="*60)

    for test in test_cases:
        print(f"\n{'─'*60}")
        print(f"TEST: {test['label']}")
        print(f"QUERY: {test['query']}")
        print(f"{'─'*60}")

        result = route_query(test["query"], mobile)
        log_interaction(test["query"], mobile, result)

        print(f"TIER:       {result['tier'].upper()}")
        print(f"ROUTED TO:  {result['routed_to']}")
        print(f"TOKENS:     {result['total_tokens']}")
        print(f"LATENCY:    {result['latency_ms']}ms")
        print(f"RESPONSE:   {result['response'][:120]}...")

if __name__ == "__main__":
    # Ensure indexes are built before routing
    print("Initialising indexes...")
    index_knowledge_base()
    index_faqs()
    print("Indexes ready.\n")

    test_master_router()