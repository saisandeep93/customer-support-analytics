import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'support.db')

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn

# ─────────────────────────────────────────────
# CORE LOGGING FUNCTION
# Called after every interaction in main.py
# Writes one row to interaction_logs
# ─────────────────────────────────────────────

def log_interaction(
    query_text: str,
    mobile_number: str,
    routing_result: dict,
    ticket_id: str = None
) -> int:
    """
    Logs a complete interaction to the database.
    Returns the log_id for future reference
    (e.g. when updating with customer feedback).
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Extract fields from routing result
    # Handle all three tiers gracefully
    retrieved_doc_ids = json.dumps(
        routing_result.get("retrieved_docs", [])
    )
    retrieval_scores = json.dumps(
        routing_result.get("retrieval_scores", [])
    )
    tools_called = json.dumps(
        routing_result.get("tools_called", [])
    )

    # Separate input/output tokens where available
    # Agent returns total only so we split 80/20 estimate
    total_tokens = routing_result.get("total_tokens", 0)
    tokens_input = routing_result.get("input_tokens", int(total_tokens * 0.8))
    tokens_output = routing_result.get("output_tokens", int(total_tokens * 0.2))

    escalated = 1 if (
        "escalate_to_human" in routing_result.get("tools_called", []) or
        routing_result.get("routed_to") == "escalate"
    ) else 0

    cursor.execute('''
        INSERT INTO interaction_logs (
            ticket_id,
            query_text,
            retrieved_doc_ids,
            retrieval_scores,
            tools_called,
            llm_response,
            tokens_used_input,
            tokens_used_output,
            latency_ms,
            resolution_status,
            human_feedback_score,
            escalated_to_human,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        ticket_id,
        query_text,
        retrieved_doc_ids,
        retrieval_scores,
        tools_called,
        routing_result.get("response", ""),
        tokens_input,
        tokens_output,
        routing_result.get("latency_ms", 0),
        routing_result.get("routed_to", "unknown"),
        None,  # human_feedback_score — null until customer rates
        escalated,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ))

    log_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return log_id

# ─────────────────────────────────────────────
# FEEDBACK UPDATE
# Called when customer submits satisfaction score
# Updates the log row with their rating
# ─────────────────────────────────────────────

def update_feedback(log_id: int, score: int):
    """
    Updates a log row with customer feedback.
    Score is 1-5 (1=very unsatisfied, 5=very satisfied).
    Called after customer rates the interaction.
    """
    if score not in range(1, 6):
        raise ValueError("Score must be between 1 and 5")

    conn = get_connection()
    conn.execute(
        'UPDATE interaction_logs SET human_feedback_score = ? WHERE log_id = ?',
        (score, log_id)
    )
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────
# EVALUATION METRICS
# Six metrics computed from interaction_logs
# These power the Layer 7 dashboard
# ─────────────────────────────────────────────

def get_resolution_rate() -> dict:
    """
    What percentage of interactions resolved
    without escalation to a human agent.
    """
    conn = get_connection()
    cursor = conn.cursor()

    result = cursor.execute('''
        SELECT
            COUNT(*) as total_interactions,
            SUM(CASE WHEN escalated_to_human = 0 THEN 1 ELSE 0 END) as auto_resolved,
            SUM(CASE WHEN escalated_to_human = 1 THEN 1 ELSE 0 END) as escalated,
            ROUND(
                100.0 * SUM(CASE WHEN escalated_to_human = 0 THEN 1 ELSE 0 END)
                / COUNT(*), 1
            ) as resolution_rate_pct
        FROM interaction_logs
    ''').fetchone()

    conn.close()
    return dict(result)

def get_routing_distribution() -> list:
    """
    How queries are distributed across tiers.
    Shows cost optimisation effectiveness.
    """
    conn = get_connection()
    cursor = conn.cursor()

    results = cursor.execute('''
        SELECT
            resolution_status as routed_to,
            COUNT(*) as interaction_count,
            ROUND(AVG(latency_ms), 0) as avg_latency_ms,
            ROUND(AVG(tokens_used_input + tokens_used_output), 0) as avg_tokens,
            ROUND(
                100.0 * COUNT(*) /
                (SELECT COUNT(*) FROM interaction_logs), 1
            ) as pct_of_total
        FROM interaction_logs
        GROUP BY resolution_status
        ORDER BY interaction_count DESC
    ''').fetchall()

    conn.close()
    return [dict(r) for r in results]

def get_retrieval_quality() -> dict:
    """
    Average similarity scores for RAG interactions.
    Low scores signal knowledge base gaps.
    """
    conn = get_connection()
    cursor = conn.cursor()

    rows = cursor.execute('''
        SELECT retrieval_scores
        FROM interaction_logs
        WHERE retrieval_scores != '[]'
        AND retrieval_scores IS NOT NULL
    ''').fetchall()

    conn.close()

    if not rows:
        return {"avg_top_score": 0, "interactions_with_retrieval": 0}

    top_scores = []
    for row in rows:
        scores = json.loads(row["retrieval_scores"])
        if scores:
            top_scores.append(max(scores))

    return {
        "avg_top_similarity_score": round(sum(top_scores) / len(top_scores), 4),
        "interactions_with_retrieval": len(top_scores),
        "high_confidence_retrievals": sum(1 for s in top_scores if s >= 0.5),
        "low_confidence_retrievals": sum(1 for s in top_scores if s < 0.35)
    }

def get_cost_analysis() -> dict:
    """
    Token consumption and estimated cost by tier.
    Uses Claude Haiku and Sonnet pricing.
    """
    conn = get_connection()
    cursor = conn.cursor()

    rows = cursor.execute('''
        SELECT
            resolution_status as routed_to,
            SUM(tokens_used_input) as total_input_tokens,
            SUM(tokens_used_output) as total_output_tokens,
            COUNT(*) as interaction_count,
            ROUND(AVG(tokens_used_input + tokens_used_output), 0) as avg_tokens
        FROM interaction_logs
        GROUP BY resolution_status
    ''').fetchall()

    conn.close()

    # Claude pricing per million tokens (approximate)
    HAIKU_INPUT_PRICE = 0.80   # $0.80 per million input tokens
    HAIKU_OUTPUT_PRICE = 4.00  # $4.00 per million output tokens
    SONNET_INPUT_PRICE = 3.00  # $3.00 per million input tokens
    SONNET_OUTPUT_PRICE = 15.00 # $15.00 per million output tokens

    results = []
    for row in rows:
        r = dict(row)
        # Agent interactions use Sonnet, others use Haiku
        if r["routed_to"] == "agent":
            cost = (
                (r["total_input_tokens"] / 1_000_000 * SONNET_INPUT_PRICE) +
                (r["total_output_tokens"] / 1_000_000 * SONNET_OUTPUT_PRICE)
            )
        else:
            cost = (
                (r["total_input_tokens"] / 1_000_000 * HAIKU_INPUT_PRICE) +
                (r["total_output_tokens"] / 1_000_000 * HAIKU_OUTPUT_PRICE)
            )
        r["estimated_cost_usd"] = round(cost, 6)
        results.append(r)

    return results

def get_latency_distribution() -> dict:
    """
    P50, P95, P99 latency across all interactions.
    Identifies tail latency issues.
    """
    conn = get_connection()
    cursor = conn.cursor()

    latencies = [
        row[0] for row in cursor.execute(
            'SELECT latency_ms FROM interaction_logs ORDER BY latency_ms'
        ).fetchall()
    ]
    conn.close()

    if not latencies:
        return {"p50": 0, "p95": 0, "p99": 0, "avg": 0}

    n = len(latencies)
    return {
        "p50_ms": latencies[int(n * 0.50)],
        "p95_ms": latencies[int(n * 0.95)],
        "p99_ms": latencies[int(n * 0.99)],
        "avg_ms": round(sum(latencies) / n, 0),
        "total_interactions": n
    }

def get_feedback_summary() -> dict:
    """
    Customer satisfaction scores where available.
    Null scores are excluded from calculation.
    """
    conn = get_connection()
    cursor = conn.cursor()

    result = cursor.execute('''
        SELECT
            COUNT(*) as total_with_feedback,
            ROUND(AVG(human_feedback_score), 2) as avg_score,
            SUM(CASE WHEN human_feedback_score >= 4 THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN human_feedback_score <= 2 THEN 1 ELSE 0 END) as negative
        FROM interaction_logs
        WHERE human_feedback_score IS NOT NULL
    ''').fetchone()

    conn.close()
    return dict(result)

def get_all_metrics() -> dict:
    """
    Runs all six evaluation metrics.
    Called by the dashboard to refresh data.
    """
    return {
        "resolution_rate": get_resolution_rate(),
        "routing_distribution": get_routing_distribution(),
        "retrieval_quality": get_retrieval_quality(),
        "cost_analysis": get_cost_analysis(),
        "latency_distribution": get_latency_distribution(),
        "feedback_summary": get_feedback_summary()
    }

# ─────────────────────────────────────────────
# TEST: Generate interactions and compute metrics
# ─────────────────────────────────────────────

def test_observability():
    import sys
    sys.path.append(os.path.dirname(__file__))
    from main import route_query, index_knowledge_base, index_faqs
    from rag_pipeline import index_knowledge_base
    from deterministic_workflow import index_faqs

    # Get a real mobile number
    conn = get_connection()
    customers = conn.execute('''
        SELECT c.mobile_number, c.name, cs.segment_label
        FROM customers c
        JOIN customer_segments cs ON c.mobile_number = cs.mobile_number
        LIMIT 3
    ''').fetchall()
    conn.close()

    print("Generating test interactions and logging them...")
    print("="*60)

    test_queries = [
        "What time do you open?",
        "My food arrived cold. I want a refund.",
        "How long does delivery take?",
        "Do you have veg options?",
        "My order has items missing.",
        "I complained last week and nothing was done. Escalate this.",
        "Can I pay with UPI?",
        "My food was cold again. Same problem as before.",
    ]

    log_ids = []
    for i, query in enumerate(test_queries):
        customer = customers[i % len(customers)]
        mobile = customer["mobile_number"]

        print(f"\nQuery {i+1}: {query[:50]}...")
        result = route_query(query, mobile)
        log_id = log_interaction(query, mobile, result)
        log_ids.append(log_id)
        print(f"  Routed to: {result['routed_to']} | "
              f"Tokens: {result['total_tokens']} | "
              f"Logged as: log_id={log_id}")

    # Simulate customer feedback for some interactions
    print("\nSimulating customer feedback...")
    feedback_map = {
        log_ids[0]: 5,  # Store hours — happy
        log_ids[1]: 4,  # Cold food RAG — satisfied
        log_ids[2]: 5,  # Delivery time — happy
        log_ids[5]: 3,  # Escalation — neutral
        log_ids[7]: 2,  # Repeat cold food — frustrated
    }
    for log_id, score in feedback_map.items():
        update_feedback(log_id, score)
        print(f"  log_id={log_id} rated {score}/5")

    # Compute and display all metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)

    metrics = get_all_metrics()

    print("\n1. RESOLUTION RATE")
    r = metrics["resolution_rate"]
    print(f"   Total interactions:  {r['total_interactions']}")
    print(f"   Auto-resolved:       {r['auto_resolved']}")
    print(f"   Escalated:           {r['escalated']}")
    print(f"   Resolution rate:     {r['resolution_rate_pct']}%")

    print("\n2. ROUTING DISTRIBUTION")
    for row in metrics["routing_distribution"]:
        print(f"   {row['routed_to']:20} | "
              f"Count: {row['interaction_count']:3} | "
              f"Avg tokens: {row['avg_tokens']:6} | "
              f"{row['pct_of_total']}%")

    print("\n3. RETRIEVAL QUALITY")
    rq = metrics["retrieval_quality"]
    print(f"   Avg top similarity:        {rq['avg_top_similarity_score']}")
    print(f"   High confidence (>=0.5):   {rq['high_confidence_retrievals']}")
    print(f"   Low confidence (<0.35):    {rq['low_confidence_retrievals']}")

    print("\n4. COST ANALYSIS")
    for row in metrics["cost_analysis"]:
        print(f"   {row['routed_to']:20} | "
              f"Interactions: {row['interaction_count']} | "
              f"Est. cost: ${row['estimated_cost_usd']}")

    print("\n5. LATENCY DISTRIBUTION")
    ld = metrics["latency_distribution"]
    print(f"   P50:  {ld['p50_ms']}ms")
    print(f"   P95:  {ld['p95_ms']}ms")
    print(f"   P99:  {ld['p99_ms']}ms")
    print(f"   Avg:  {ld['avg_ms']}ms")

    print("\n6. FEEDBACK SUMMARY")
    fs = metrics["feedback_summary"]
    print(f"   Interactions with feedback: {fs['total_with_feedback']}")
    print(f"   Average score:              {fs['avg_score']}/5")
    print(f"   Positive (4-5):             {fs['positive']}")
    print(f"   Negative (1-2):             {fs['negative']}")

if __name__ == "__main__":
    test_observability()